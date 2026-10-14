import json
import logging
import os
import posixpath
import re
import tempfile
import threading
import traceback
import uuid
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterable, BinaryIO

from mlflow.entities.file_info import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadPart,
)
from mlflow.entities.trace_data import TraceData
from mlflow.environment_variables import (
    MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM,
    MLFLOW_ARTIFACT_CHECKSUM_ENABLED,
    MLFLOW_ARTIFACT_CHECKSUM_MAX_ATTEMPTS,
    MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY,
)
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataNotFound,
)
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.artifact.checksum import (
    ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME,
    DIGEST_MISMATCH,
    MANIFEST_STATE_MALFORMED,
    MANIFEST_STATE_MISSING,
    MANIFEST_STATE_PRESENT,
    MANIFEST_STATE_UNAVAILABLE,
    SIZE_MISMATCH,
    VERIFICATION_STATUS_FAILED,
    VERIFICATION_STATUS_SKIPPED,
    VERIFICATION_STATUS_VERIFIED,
    ArtifactChecksumMismatchException,
    ArtifactChecksumReport,
    ArtifactManifestMalformed,
    FileChecksumFailure,
    build_manifest,
    compute_file_digest,
    is_manifest_file,
    is_missing_artifact_error,
    parse_manifest,
    serialize_manifest,
)
from mlflow.tracing.utils.artifact_utils import TRACE_DATA_FILE_NAME
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_artifacts_logging_queue import (
    AsyncArtifactsLoggingQueue,
)
from mlflow.utils.file_utils import (
    ArtifactProgressBar,
    create_tmp_dir,
    relative_path_to_artifact_path,
)
from mlflow.utils.os import is_windows
from mlflow.utils.uri import validate_path_within_directory
from mlflow.utils.validation import bad_path_message, path_not_unique

if TYPE_CHECKING:
    from mlflow.entities.span import Span

# Constants used to determine max level of parallelism to use while uploading/downloading artifacts.
# Max threads to use for parallelism.
_NUM_MAX_THREADS = 20
# Max threads per CPU
_NUM_MAX_THREADS_PER_CPU = 2
assert _NUM_MAX_THREADS >= _NUM_MAX_THREADS_PER_CPU
assert _NUM_MAX_THREADS_PER_CPU > 0
# Default number of CPUs to assume on the machine if unavailable to fetch it using os.cpu_count()
_NUM_DEFAULT_CPUS = _NUM_MAX_THREADS // _NUM_MAX_THREADS_PER_CPU
_logger = logging.getLogger(__name__)
# Chunk size for streaming artifact uploads and downloads (1 MB).
ARTIFACT_STREAM_CHUNK_SIZE = 1024 * 1024

# Precompiled regex for sanitizing Windows-invalid characters in path components (Windows only).
# Backslash is included: artifact paths are POSIX-style (only "/" separates components), so a
# literal "\" inside a component is part of the name and is invalid in a Windows filename.
_WINDOWS_INVALID_CHARS_REGEX = re.compile(r'[<>:"|?*\\]') if is_windows() else None


def _truncate_error(err: str, max_length: int = 10_000) -> str:
    if len(err) <= max_length:
        return err
    half = max_length // 2
    return err[:half] + "\n\n*** Error message is too long, truncated ***\n\n" + err[-half:]


def _retry_with_new_creds(try_func, creds_func, orig_creds=None):
    """
    Attempt the try_func with the original credentials (og_creds) if provided, or by generating the
    credentials using creds_func. If the try_func throws, then try again with new credentials
    provided by creds_func.
    """
    try:
        first_creds = creds_func() if orig_creds is None else orig_creds
        return try_func(first_creds)
    except Exception as e:
        _logger.info(
            f"Failed to complete request, possibly due to credential expiration (Error: {e})."
            " Refreshing credentials and trying again..."
        )
        new_creds = creds_func()
        return try_func(new_creds)


def _is_object_key_within_path(object_key: str, path: str) -> bool:
    """
    Return whether an object-store key belongs to a path.

    Prefix listings can include sibling keys such as ``foobar`` when querying ``foo``, so require
    either an exact match or a ``/`` boundary. An empty path represents the repository root.
    """
    if not path:
        return True
    if path.endswith("/"):
        return object_key.startswith(path)
    return object_key == path or object_key.startswith(f"{path}/")


def _sanitize_path_component_for_windows(component: str) -> str:
    """
    Sanitize a path component by replacing Windows-invalid characters with underscores.

    Windows does not allow these characters in filenames: < > : " | ? * \\
    (Note: / is the POSIX path separator and is handled separately by splitting the path.
    A literal \\ inside a component is not a separator here and is sanitized as invalid.)

    This function does not handle Windows reserved names (CON, PRN, AUX, NUL, COM1-9,
    LPT1-9, CONIN$, CONOUT$). Artifact paths with these names are rare in practice, but
    would cause issues if encountered on Windows.

    Args:
        component: A single path component (filename or directory name).

    Returns:
        The sanitized path component with invalid characters replaced by underscores.
    """
    # A None regex means we're not on Windows, so leave the component untouched.
    if _WINDOWS_INVALID_CHARS_REGEX is None:
        return component
    return _WINDOWS_INVALID_CHARS_REGEX.sub("_", component)


def _sanitize_path_for_windows(path: str) -> str:
    """
    Sanitize a POSIX-style path by replacing Windows-invalid characters in each component.

    This function only performs sanitization on Windows. On other platforms, the path
    is returned unchanged.

    Args:
        path: A POSIX-style path (using / as separator).

    Returns:
        The sanitized path with invalid characters replaced, or the original path if not
        on Windows or if the path is empty.
    """
    if not is_windows() or not path:
        return path

    components = path.split("/")
    sanitized_components = [_sanitize_path_component_for_windows(comp) for comp in components]
    # Rejoin with "/" rather than posixpath.join, which would drop a leading empty component and
    # turn an absolute path like "/etc/passwd" into a relative one on Windows (it stays absolute
    # on POSIX). Preserving it keeps the containment check in _create_download_destination honest.
    return "/".join(sanitized_components)


@developer_stable
class ArtifactRepository:
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        self.artifact_uri = artifact_uri
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        # Limit the number of threads used for artifact uploads/downloads. Use at most
        # constants._NUM_MAX_THREADS threads or 2 * the number of CPU cores available on the
        # system (whichever is smaller)
        self.thread_pool = self._create_thread_pool()
        # Serializes read-modify-write of the checksum manifest so that concurrent uploads
        # (e.g. multi-file uploads on a thread pool) cannot overwrite each other's entries.
        self._artifact_manifest_lock = threading.RLock()
        # Per-thread collection of files uploaded inside a serial log_artifacts() batch,
        # so the manifest is written once instead of once per file.
        self._artifact_checksum_state = threading.local()
        # Set by log_artifacts() implementations that fan out through a different thread
        # pool; suppresses per-file manifest writes in their worker threads.
        self._artifact_checksum_suppressed = False

        def log_artifact_handler(filename, artifact_path=None, artifact=None):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, filename)
                if artifact is not None:
                    # User should already have installed PIL to log a PIL image
                    from PIL import Image

                    if isinstance(artifact, Image.Image):
                        artifact.save(tmp_path)
                self.log_artifact(tmp_path, artifact_path)

        self._async_logging_queue = AsyncArtifactsLoggingQueue(log_artifact_handler)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"artifact_uri={self.artifact_uri!r}, "
            f"tracking_uri={self.tracking_uri!r}, "
            f"registry_uri={self.registry_uri!r}"
            f")"
        )

    def _create_thread_pool(self):
        return ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix=f"Mlflow{self.__class__.__name__}"
        )

    def flush_async_logging(self):
        """
        Flushes the async logging queue, ensuring that all pending logging operations have
        completed.
        """
        if self._async_logging_queue._is_activated:
            self._async_logging_queue.flush()

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        Args:
            local_file: Path to artifact to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
        """

    def _log_artifact_async(self, filename, artifact_path=None, artifact=None):
        """
        Asynchronously log a local file as an artifact, optionally taking an ``artifact_path`` to
        place it within the run's artifacts. Run artifacts can be organized into directory, so you
        can place the artifact in the directory this way. Cleanup tells the function whether to
        cleanup the local_file after running log_artifact, since it could be a Temporary
        Directory.

        Args:
            filename: Filename of the artifact to be logged.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact: The artifact to be logged.

        Returns:
            An :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance
            that represents future for logging operation.
        """

        if not self._async_logging_queue.is_active():
            self._async_logging_queue.activate()

        return self._async_logging_queue.log_artifacts_async(
            filename=filename, artifact_path=artifact_path, artifact=artifact
        )

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        Args:
            local_dir: Directory of local artifacts to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifacts.
        """

    @abstractmethod
    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        Args:
            path: Relative source path that contains desired artifacts.

        Returns:
            List of artifacts as FileInfo listed directly under path.
        """

    def _is_directory(self, artifact_path):
        listing = self.list_artifacts(artifact_path)
        return len(listing) > 0

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        """
        Creates a local filesystem location to be used as a destination for downloading the artifact
        specified by `src_artifact_path`. The destination location is a subdirectory of the
        specified `dst_local_dir_path`, which is determined according to the structure of
        `src_artifact_path`. For example, if `src_artifact_path` is `dir1/file1.txt`, then the
        resulting destination path is `<dst_local_dir_path>/dir1/file1.txt`. Local directories are
        created for the resulting destination location if they do not exist.

        Note: This method normalizes path separators and validates that the resulting destination
        stays within `dst_local_dir_path` on all platforms. On Windows it additionally sanitizes
        characters that are invalid in filenames on that platform.

        Args:
            src_artifact_path: A relative, POSIX-style path referring to an artifact stored
                within the repository's artifact root location. `src_artifact_path` should be
                specified relative to the repository's artifact root location.
            dst_local_dir_path: The absolute path to a local filesystem directory in which the
                local destination path will be contained. The local destination path may be
                contained in a subdirectory of `dst_root_dir` if `src_artifact_path` contains
                subdirectories.

        Returns:
            The absolute path to a local filesystem location to be used as a destination
            for downloading the artifact specified by `src_artifact_path`.
        """
        src_artifact_path = src_artifact_path.rstrip("/")  # Ensure correct dirname for trailing '/'
        src_artifact_path = _sanitize_path_for_windows(src_artifact_path)
        # os.path.normpath() converts forward slashes (/) to OS-specific separators (\\ on Windows)
        if dirpath := posixpath.dirname(src_artifact_path):
            local_dir_path = os.path.join(dst_local_dir_path, os.path.normpath(dirpath))
        else:
            local_dir_path = dst_local_dir_path
        local_file_path = os.path.join(dst_local_dir_path, os.path.normpath(src_artifact_path))

        validate_path_within_directory(dst_local_dir_path, local_file_path)

        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        return local_file_path

    def get_local_path(self, artifact_path: str) -> str | None:
        """
        Return a safe absolute filesystem path for the specified artifact when it is already
        directly accessible by the server.

        Backends that cannot safely expose a local filesystem path should return ``None`` and rely
        on ``download_artifacts()`` to materialize the artifact locally. Backends that support
        local paths should raise their usual exceptions for missing or invalid artifacts rather than
        silently returning ``None``.
        """
        return None

    def _iter_artifacts_recursive(self, path):
        dir_content = [
            file_info
            for file_info in self.list_artifacts(path)
            # prevent infinite loop, sometimes the dir is recursively included
            if file_info.path not in [".", path]
        ]
        # Empty directory
        if not dir_content:
            yield FileInfo(path=path, is_dir=True, file_size=None)
            return

        for file_info in dir_content:
            if file_info.is_dir:
                yield from self._iter_artifacts_recursive(file_info.path)
            else:
                yield file_info

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        if dst_path:
            dst_path = os.path.abspath(dst_path)
            if not os.path.exists(dst_path):
                raise MlflowException(
                    message=(
                        "The destination path for downloaded artifacts does not"
                        f" exist! Destination path: {dst_path}"
                    ),
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif not os.path.isdir(dst_path):
                raise MlflowException(
                    message=(
                        "The destination path for downloaded artifacts must be a directory!"
                        f" Destination path: {dst_path}"
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            dst_path = create_tmp_dir()

        def _plan_download(src_artifact_path, dst_local_dir_path):
            dst_local_file_path = self._create_download_destination(
                src_artifact_path=src_artifact_path, dst_local_dir_path=dst_local_dir_path
            )
            future = self.thread_pool.submit(
                self._download_file,
                remote_file_path=src_artifact_path,
                local_path=dst_local_file_path,
            )
            return future, dst_local_file_path

        # Submit download tasks. Each future maps to (remote path, local destination)
        # so that post-transfer checksum verification can hash the landed files.
        futures = {}
        if artifact_path in ("", None):
            root_listing = self.list_artifacts(artifact_path)
            if not root_listing:
                _logger.info(
                    "No artifacts found to download at %s. Returning destination path.",
                    self.artifact_uri,
                )
                return dst_path
            is_dir = True
        else:
            is_dir = self._is_directory(artifact_path)

        if is_dir:
            for file_info in self._iter_artifacts_recursive(artifact_path):
                if file_info.is_dir:  # Empty directory
                    sanitized_path = _sanitize_path_for_windows(file_info.path)
                    normalized_path = os.path.join(dst_path, os.path.normpath(sanitized_path))
                    # A malicious listing with ".." segments or an absolute path must not create
                    # directories outside the destination.
                    validate_path_within_directory(dst_path, normalized_path)
                    os.makedirs(normalized_path, exist_ok=True)
                else:
                    fut, local_destination = _plan_download(file_info.path, dst_path)
                    futures[fut] = (file_info.path, local_destination)
        else:
            fut, local_destination = _plan_download(artifact_path, dst_path)
            futures[fut] = (artifact_path, local_destination)

        # Wait for downloads to complete and collect failures
        failed_downloads = {}
        tracebacks = {}
        with ArtifactProgressBar.files(desc="Downloading artifacts", total=len(futures)) as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                    pbar.update()
                except Exception as e:
                    path = futures[f][0]
                    failed_downloads[path] = e
                    tracebacks[path] = traceback.format_exc()

        if failed_downloads:
            if _logger.isEnabledFor(logging.DEBUG):
                template = "##### File {path} #####\n{error}\nTraceback:\n{traceback}\n"
            else:
                template = "##### File {path} #####\n{error}"

            failures = "\n".join(
                template.format(path=path, error=error, traceback=tracebacks[path])
                for path, error in failed_downloads.items()
            )
            error_codes = {
                e.error_code
                for e in failed_downloads.values()
                if isinstance(e, MlflowException) and e.error_code != "INTERNAL_ERROR"
            }
            error_code = (
                ErrorCode.Value(error_codes.pop()) if len(error_codes) == 1 else INTERNAL_ERROR
            )
            raise MlflowException(
                message=(
                    "The following failures occurred while downloading one or more"
                    f" artifacts from {self.artifact_uri}:\n{_truncate_error(failures)}"
                ),
                error_code=error_code,
            )

        # All transfers completed; compare the landed bytes against the checksum
        # manifest before reporting success. No-op when verification is disabled.
        self._verify_or_skip(
            {remote_path: local_path for remote_path, local_path in futures.values()}
        )

        if artifact_path:
            sanitized_path = _sanitize_path_for_windows(artifact_path)
            return os.path.join(dst_path, os.path.normpath(sanitized_path))
        return dst_path

    @abstractmethod
    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        Args:
            remote_file_path: Source path to the remote file, relative to the root
                directory of the artifact repository.
            local_path: The path to which to save the downloaded file.
        """

    # ------------------------------------------------------------------
    # Optional content digest (checksum) verification
    # ------------------------------------------------------------------
    #
    # When MLFLOW_ARTIFACT_CHECKSUM_ENABLED is set, uploads maintain a manifest
    # artifact (``mlflow-artifact-manifest.json`` at the repository root) mapping
    # repository-relative file paths to size and content digest. Downloads verify
    # each file against the manifest. All hooks are no-ops when verification is
    # disabled, so transfers behave exactly as before.

    def _artifact_checksum_enabled(self) -> bool:
        return MLFLOW_ARTIFACT_CHECKSUM_ENABLED.get()

    @staticmethod
    def _artifact_remote_path(local_file, artifact_path=None) -> str:
        name = os.path.basename(local_file)
        return posixpath.join(artifact_path, name) if artifact_path else name

    @contextmanager
    def _collect_artifact_checksums(self):
        """Collect source files of a serial batch upload for a single manifest write.

        ``_record_uploaded_artifact`` calls inside the context append their files to
        the yielded mapping instead of each performing a manifest read-modify-write.
        The manifest is written once when the context exits without an exception.
        """
        if not self._artifact_checksum_enabled():
            yield {}
            return
        previous = getattr(self._artifact_checksum_state, "batch_sources", None)
        sources = previous if previous is not None else {}
        self._artifact_checksum_state.batch_sources = sources
        try:
            yield sources
            if previous is None and sources:
                self._record_artifact_sources(dict(sources))
        finally:
            self._artifact_checksum_state.batch_sources = previous

    @contextmanager
    def _suppress_artifact_checksums(self):
        """Suppress per-file manifest writes for batch uploads using separate pools."""
        previous = self._artifact_checksum_suppressed
        self._artifact_checksum_suppressed = True
        try:
            yield
        finally:
            self._artifact_checksum_suppressed = previous

    def _record_uploaded_artifact(self, local_file, artifact_path=None) -> None:
        """Record a single successfully uploaded file in the checksum manifest."""
        if not self._artifact_checksum_enabled():
            return
        if getattr(self._artifact_checksum_state, "writing_manifest", False):
            return
        if self._artifact_checksum_suppressed:
            return
        remote_path = self._artifact_remote_path(local_file, artifact_path)
        if is_manifest_file(remote_path):
            _logger.warning(
                "Not recording artifact %r in the checksum manifest: the file name is "
                "reserved for the manifest itself.",
                remote_path,
            )
            return
        batch = getattr(self._artifact_checksum_state, "batch_sources", None)
        if batch is not None:
            batch[remote_path] = os.path.abspath(local_file)
            return
        self._record_artifact_sources({remote_path: os.path.abspath(local_file)})

    def _record_local_dir_artifacts(self, local_dir, artifact_path=None) -> None:
        """Record all files of an uploaded local directory in the checksum manifest."""
        if not self._artifact_checksum_enabled():
            return
        sources = {}
        for dirpath, _, filenames in os.walk(local_dir):
            for name in filenames:
                src = os.path.join(dirpath, name)
                rel = relative_path_to_artifact_path(os.path.relpath(src, local_dir))
                remote_path = posixpath.join(artifact_path, rel) if artifact_path else rel
                if is_manifest_file(remote_path):
                    # Never let a user-supplied file with the reserved manifest name
                    # masquerade as the checksum manifest.
                    _logger.warning(
                        "Not recording artifact %r in the checksum manifest: the file name "
                        "is reserved for the manifest itself.",
                        remote_path,
                    )
                    continue
                sources[remote_path] = os.path.abspath(src)
        self._record_artifact_sources(sources)

    def _fetch_checksum_manifest(self) -> tuple[dict | None, str, str | None]:
        """Download and parse the checksum manifest.

        Returns:
            A ``(manifest, state, detail)`` tuple where state is one of
            ``present``/``missing``/``malformed``/``unavailable``.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME)
            try:
                self._download_file(ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME, local_path)
            except Exception as e:
                if is_missing_artifact_error(e):
                    return None, MANIFEST_STATE_MISSING, None
                return None, MANIFEST_STATE_UNAVAILABLE, _truncate_error(str(e))
            try:
                with open(local_path, "rb") as f:
                    raw = f.read()
                return parse_manifest(raw), MANIFEST_STATE_PRESENT, None
            except ArtifactManifestMalformed as e:
                return None, MANIFEST_STATE_MALFORMED, e.reason

    def _load_existing_manifest_for_upload(self) -> dict | None:
        """Fetch an existing manifest for merging, tolerating its absence or corruption."""
        manifest, state, detail = self._fetch_checksum_manifest()
        if state == MANIFEST_STATE_PRESENT:
            return manifest
        if state == MANIFEST_STATE_MALFORMED:
            _logger.warning(
                "The existing artifact checksum manifest at %s is corrupt or invalid (%s); "
                "it will be replaced.",
                self.artifact_uri,
                detail,
            )
        elif state == MANIFEST_STATE_UNAVAILABLE:
            _logger.warning(
                "Unable to read the existing artifact checksum manifest at %s (%s); "
                "writing a new manifest.",
                self.artifact_uri,
                detail,
            )
        return None

    def _upload_checksum_manifest(self, manifest: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME)
            with open(local_path, "wb") as f:
                f.write(serialize_manifest(manifest))
            previous = getattr(self._artifact_checksum_state, "writing_manifest", False)
            self._artifact_checksum_state.writing_manifest = True
            try:
                self.log_artifact(local_path)
            finally:
                self._artifact_checksum_state.writing_manifest = previous

    def _record_artifact_sources(self, sources: dict[str, str]) -> None:
        """Merge ``remote path -> local source file`` entries into the checksum manifest.

        Degrades to a warning (or raises, depending on
        ``MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY``) when the manifest cannot be
        written. Invalid configuration values are always raised before the degrade
        policy is applied.
        """
        if not sources or not self._artifact_checksum_enabled():
            return
        # Fail at configuration parsing time for invalid algorithm/policy values.
        algorithm = MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.get()
        policy = MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.get()
        try:
            with self._artifact_manifest_lock:
                existing = self._load_existing_manifest_for_upload()
                if existing is not None and existing["algorithm"] != algorithm:
                    # A repository has one manifest and therefore one algorithm; keep
                    # digesting new files with the algorithm already in use.
                    _logger.warning(
                        "Configured checksum algorithm %r differs from the existing manifest "
                        "algorithm %r at %s; continuing with %r.",
                        algorithm,
                        existing["algorithm"],
                        self.artifact_uri,
                        existing["algorithm"],
                    )
                    algorithm = existing["algorithm"]
                files = dict(existing["files"]) if existing else {}
                for remote_path, local_file in sources.items():
                    if is_manifest_file(remote_path):
                        continue
                    size, digest = compute_file_digest(local_file, algorithm)
                    files[remote_path] = {"size": size, "digest": digest}
                manifest = build_manifest(
                    algorithm,
                    {path: (entry["size"], entry["digest"]) for path, entry in files.items()},
                )
                self._upload_checksum_manifest(manifest)
        except Exception as e:
            if policy == "error":
                raise
            _logger.warning(
                "Failed to update the artifact checksum manifest at %s (%s); proceeding "
                "without checksum verification for this upload.",
                self.artifact_uri,
                _truncate_error(str(e)),
            )

    def _log_checksum_report(self, report: ArtifactChecksumReport) -> None:
        if report.status == VERIFICATION_STATUS_VERIFIED:
            _logger.info("Artifact checksum verification passed (%s).", report.format_summary())
        else:
            _logger.info("Artifact checksum verification skipped (%s).", report.format_summary())
        if report.unverified_files:
            _logger.info(
                "The following downloaded file(s) have no checksum manifest entry and were "
                "not verified: %s",
                ", ".join(report.unverified_files),
            )

    def _redownload_files(self, remote_to_local: dict[str, str]) -> None:
        """Re-download only the specified files (used after checksum mismatches)."""
        futures = {
            self.thread_pool.submit(
                self._download_file, remote_file_path=remote_path, local_path=local_path
            ): remote_path
            for remote_path, local_path in remote_to_local.items()
        }
        errors = {}
        for future in as_completed(futures):
            remote_path = futures[future]
            try:
                future.result()
            except Exception as e:
                errors[remote_path] = e
        if errors:
            details = "; ".join(f"{path}: {err}" for path, err in sorted(errors.items()))
            raise MlflowException(
                "Failed to re-download file(s) that did not pass checksum verification: "
                f"{_truncate_error(details)}"
            )

    def _verify_files_against_manifest(
        self,
        manifest: dict,
        remote_to_local: dict[str, str],
        max_attempts: int | None = None,
    ) -> ArtifactChecksumReport:
        """Verify local files against a parsed manifest, re-downloading failed files.

        Multipart uploads/downloads are verified as whole files: the digest is
        computed over the fully assembled local file, so a single conclusion covers
        all chunks.

        Args:
            max_attempts: Total verification attempts; failed files are re-downloaded
                between attempts. Defaults to the configured value. In-place
                verification (no bytes were transferred) passes 1 since re-downloading
                a file onto itself cannot repair it.
        """
        algorithm = manifest["algorithm"]
        entries = manifest["files"]
        if max_attempts is None:
            max_attempts = max(1, MLFLOW_ARTIFACT_CHECKSUM_MAX_ATTEMPTS.get())
        else:
            max_attempts = max(1, max_attempts)
        pending = dict(remote_to_local)
        failures: dict[str, FileChecksumFailure] = {}
        checked: list[str] = []
        unverified: list[str] = []

        for attempt in range(1, max_attempts + 1):
            current_failures: dict[str, FileChecksumFailure] = {}
            for remote_path, local_path in sorted(pending.items()):
                entry = entries.get(remote_path)
                if entry is None:
                    if attempt == 1:
                        unverified.append(remote_path)
                    continue
                if attempt == 1:
                    checked.append(remote_path)
                try:
                    actual_size, actual_digest = compute_file_digest(local_path, algorithm)
                except OSError as e:
                    current_failures[remote_path] = FileChecksumFailure(
                        path=remote_path,
                        expected_size=entry["size"],
                        actual_size=None,
                        expected_digest=entry["digest"],
                        actual_digest=None,
                        reason=f"unreadable local file: {e}",
                    )
                    continue
                if actual_size != entry["size"]:
                    current_failures[remote_path] = FileChecksumFailure(
                        path=remote_path,
                        expected_size=entry["size"],
                        actual_size=actual_size,
                        expected_digest=entry["digest"],
                        actual_digest=actual_digest,
                        reason=SIZE_MISMATCH,
                    )
                elif actual_digest != entry["digest"]:
                    current_failures[remote_path] = FileChecksumFailure(
                        path=remote_path,
                        expected_size=entry["size"],
                        actual_size=actual_size,
                        expected_digest=entry["digest"],
                        actual_digest=actual_digest,
                        reason=DIGEST_MISMATCH,
                    )

            if not current_failures:
                failures = {}
                break

            failures = current_failures
            if attempt < max_attempts:
                _logger.warning(
                    "Checksum verification failed for %d file(s) at %s; re-downloading only "
                    "the failed file(s) (attempt %d/%d): %s",
                    len(current_failures),
                    self.artifact_uri,
                    attempt + 1,
                    max_attempts,
                    ", ".join(sorted(current_failures)),
                )
                self._redownload_files(
                    {path: remote_to_local[path] for path in current_failures}
                )
                pending = {path: remote_to_local[path] for path in current_failures}

        report = ArtifactChecksumReport(
            status=VERIFICATION_STATUS_FAILED if failures else VERIFICATION_STATUS_VERIFIED,
            manifest_state=MANIFEST_STATE_PRESENT,
            algorithm=algorithm,
            checked_files=sorted(checked),
            unverified_files=sorted(unverified),
            failures=[failures[path] for path in sorted(failures)],
        )
        self._log_checksum_report(report)
        if failures:
            raise ArtifactChecksumMismatchException(report)
        return report

    def _verify_or_skip(
        self,
        remote_to_local: dict[str, str],
        max_attempts: int | None = None,
    ) -> ArtifactChecksumReport | None:
        """Verify downloaded/local files, implementing missing-manifest skip semantics."""
        if not self._artifact_checksum_enabled() or not remote_to_local:
            return None
        manifest, state, detail = self._fetch_checksum_manifest()
        if state == MANIFEST_STATE_MALFORMED:
            # Reported separately from file content mismatches: a broken manifest
            # cannot be used to judge any individual file.
            raise ArtifactManifestMalformed(detail)
        if state == MANIFEST_STATE_MISSING:
            _logger.info(
                "No artifact checksum manifest found at %s; downloaded artifact(s) are treated "
                "as unverified. Artifacts written without a manifest (e.g. legacy artifacts) "
                "are not an error.",
                self.artifact_uri,
            )
            report = ArtifactChecksumReport(
                status=VERIFICATION_STATUS_SKIPPED,
                manifest_state=MANIFEST_STATE_MISSING,
                unverified_files=sorted(remote_to_local),
            )
            self._log_checksum_report(report)
            return report
        if state == MANIFEST_STATE_UNAVAILABLE:
            message = (
                f"Unable to read the artifact checksum manifest from {self.artifact_uri}: "
                f"{detail}. Downloaded artifact(s) are treated as unverified."
            )
            if MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.get() == "error":
                raise MlflowException(message)
            _logger.warning(message)
            report = ArtifactChecksumReport(
                status=VERIFICATION_STATUS_SKIPPED,
                manifest_state=MANIFEST_STATE_UNAVAILABLE,
                detail=detail,
                unverified_files=sorted(remote_to_local),
            )
            self._log_checksum_report(report)
            return report

        files = {
            remote_path: local_path
            for remote_path, local_path in remote_to_local.items()
            if not is_manifest_file(remote_path)
        }
        return self._verify_files_against_manifest(manifest, files, max_attempts=max_attempts)

    def delete_artifacts(self, artifact_path=None):
        """
        Delete the artifacts at the specified location.
        Supports the deletion of a single file or of a directory. Deletion of a directory
        is recursive.

        Args:
            artifact_path: Path of the artifact to delete.
        """

    @property
    def max_workers(self) -> int:
        """Compute the number of workers to use for multi-threading."""
        num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
        return min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)

    def download_trace_data_to_file(self, dst_path: Path) -> Path:
        """
        Download the trace data file to the specified local path without parsing it.

        Args:
            dst_path: Local filesystem path to write the trace data file to.

        Returns:
            The ``dst_path`` argument, for convenience.

        Raises:
            - `MlflowTraceDataNotFound`: The trace data is not found.
        """
        try:
            self._download_file(TRACE_DATA_FILE_NAME, dst_path)
        except Exception as e:
            raise MlflowTraceDataNotFound(artifact_path=TRACE_DATA_FILE_NAME) from e
        return dst_path

    def download_trace_data(self) -> dict[str, Any]:
        """
        Download the trace data.

        Returns:
            The trace data as a dictionary.

        Raises:
            - `MlflowTraceDataNotFound`: The trace data is not found.
            - `MlflowTraceDataCorrupted`: The trace data is corrupted.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, TRACE_DATA_FILE_NAME)
            self.download_trace_data_to_file(temp_file)
            return try_read_trace_data(temp_file)

    def download_archived_trace_data(self) -> TraceData:
        """
        Download archived trace data from ``traces.pb``.

        Returns:
            The archived trace data. If the archived payload is missing, returns an empty
            ``TraceData`` so callers can continue surfacing trace metadata.
        """
        from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, TRACE_ARCHIVAL_FILENAME)
            try:
                self._download_file(TRACE_ARCHIVAL_FILENAME, temp_file)
            except Exception:
                _logger.warning(
                    "Archived trace payload is missing for path=%s; returning empty spans.",
                    TRACE_ARCHIVAL_FILENAME,
                    exc_info=_logger.isEnabledFor(logging.DEBUG),
                )
                return TraceData(spans=[])
            return TraceData(spans=_try_read_trace_data_pb(temp_file))

    def download_trace_attachment_to_file(self, path: str, dst_path: Path) -> Path:
        """
        Download a trace attachment to the specified local path.

        Args:
            path: The attachment identifier (UUID).
            dst_path: Local filesystem path to write the attachment to.

        Returns:
            The ``dst_path`` argument, for convenience.
        """
        _validate_attachment_path(path)
        self._download_file(posixpath.join("attachments", path), dst_path)
        return dst_path

    def download_trace_attachment(self, path: str) -> bytes:
        _validate_attachment_path(path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, path)
            self.download_trace_attachment_to_file(path, temp_file)
            return temp_file.read_bytes()

    def upload_trace_data(self, trace_data: str) -> None:
        """
        Upload the trace data.

        Args:
            trace_data: The json-serialized trace data to upload.
        """
        with write_local_temp_trace_data_file(trace_data) as temp_file:
            self.log_artifact(temp_file)

    def upload_archived_trace_data(self, trace_data: TraceData) -> None:
        """
        Upload archived trace data as OTLP protobuf to ``traces.pb``.

        Args:
            trace_data: The archived trace data as a ``TraceData`` object.
        """
        from mlflow.exceptions import MlflowTraceArchivalMalformedTrace
        from mlflow.tracing.otel.otel_archival import spans_to_traces_data_pb

        if not isinstance(trace_data, TraceData):
            raise MlflowException.invalid_parameter_value(
                "Archived trace data must be a TraceData object."
            )
        try:
            data = spans_to_traces_data_pb(trace_data.spans)
        except (MlflowException, TypeError, ValueError) as e:
            raise MlflowTraceArchivalMalformedTrace(str(e)) from e
        self.upload_archived_trace_data_bytes(data)

    def upload_archived_trace_data_bytes(self, data: bytes) -> None:
        """
        Upload serialized archived trace data bytes to ``traces.pb``.

        Backends can override this hook to avoid the default temp-file staging path when their
        storage SDK supports in-memory uploads or more efficient multipart transfer primitives.
        Overriding is useful for remote object stores where direct byte uploads can reduce local
        disk I/O and let the backend apply transport-specific optimizations.
        """
        with _write_local_temp_trace_data_pb_file(data) as temp_file:
            self.log_artifact(temp_file)

    def upload_attachment(self, attachment_id: str, content_bytes: bytes) -> None:
        _validate_attachment_path(attachment_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, attachment_id)
            temp_file.write_bytes(content_bytes)
            self.log_artifact(temp_file, artifact_path="attachments")


@contextmanager
def write_local_temp_trace_data_file(trace_data: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir, TRACE_DATA_FILE_NAME)
        temp_file.write_text(trace_data, encoding="utf-8")
        yield temp_file


@contextmanager
def _write_local_temp_trace_data_pb_file(data: bytes):
    from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir, TRACE_ARCHIVAL_FILENAME)
        temp_file.write_bytes(data)
        yield temp_file


def try_read_trace_data(trace_data_path):
    if not os.path.exists(trace_data_path):
        raise MlflowTraceDataNotFound(artifact_path=trace_data_path)
    with open(trace_data_path, encoding="utf-8") as f:
        data = f.read()
    if not data:
        raise MlflowTraceDataNotFound(artifact_path=trace_data_path)
    try:
        return json.loads(data)
    except json.decoder.JSONDecodeError as e:
        raise MlflowTraceDataCorrupted(artifact_path=trace_data_path) from e


def _try_read_trace_data_pb(trace_data_path) -> list["Span"]:
    from mlflow.tracing.otel.otel_archival import traces_data_pb_to_spans

    if not os.path.exists(trace_data_path):
        raise MlflowTraceDataNotFound(artifact_path=trace_data_path)
    data = Path(trace_data_path).read_bytes()
    if not data:
        raise MlflowTraceDataCorrupted(artifact_path=trace_data_path)
    try:
        return traces_data_pb_to_spans(data)
    except Exception as e:
        raise MlflowTraceDataCorrupted(artifact_path=trace_data_path) from e


class MultipartUploadMixin(ABC):
    @abstractmethod
    def create_multipart_upload(
        self, local_file: str, num_parts: int, artifact_path: str | None = None
    ) -> CreateMultipartUploadResponse:
        """
        Initiate a multipart upload and retrieve the pre-signed upload URLS and upload id.

        Args:
            local_file: Path of artifact to upload.
            num_parts: Number of parts to upload. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """

    @abstractmethod
    def complete_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        parts: list[MultipartUploadPart],
        artifact_path: str | None = None,
    ) -> None:
        """
        Complete a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            parts: A list containing the metadata of each part that has been uploaded.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """

    @abstractmethod
    def abort_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        artifact_path: str | None = None,
    ) -> None:
        """
        Abort a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """


class MultipartDownloadMixin(ABC):
    """
    Mixin that defines the API for artifact repositories that support multipart
    download (MPD), i.e. generating presigned URLs for direct download from
    cloud storage via the MPD flow.
    """

    @abstractmethod
    def get_download_presigned_url(self, artifact_path, expiration=300):
        """
        Generate a presigned URL for downloading an artifact directly from cloud storage.

        Args:
            artifact_path: Relative path to the artifact within the artifact URI.
            expiration: Time in seconds for the presigned URL to remain valid (default: 300).

        Returns:
            PresignedDownloadUrlResponse containing the presigned URL, headers, and file size.
        """


class PresignedUploadMixin(ABC):
    """
    Mixin that defines the API for artifact repositories that support presigned
    URL uploads, i.e. generating presigned URLs for direct upload to cloud storage.
    """

    @abstractmethod
    def create_presigned_upload_url(self, artifact_path, expiration=900):
        """
        Generate a presigned URL for uploading an artifact directly to cloud storage.

        Args:
            artifact_path: Relative path within the run's artifact directory
                          (e.g. "models/model.pkl").
            expiration: URL expiration time in seconds (default: 900).

        Returns:
            CreatePresignedUploadResponse with presigned_url and headers.
        """


class StreamUploadMixin(ABC):
    """
    Mixin that defines the API for artifact repositories that support streaming
    uploads directly from a binary stream, avoiding an intermediate temp-file copy
    in the server handler.
    """

    @abstractmethod
    def log_artifact_from_stream(
        self,
        stream: BinaryIO,
        artifact_file_name: str,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log artifact contents from a binary stream.

        Args:
            stream: Readable binary stream containing the artifact contents.
            artifact_file_name: Artifact filename to log. Any directory components are
                ignored; use ``artifact_path`` to specify the destination directory.
            artifact_path: Directory within the run's artifact directory in which to log
                the artifact.
        """

    @abstractmethod
    async def log_artifact_from_async_stream(
        self,
        chunks: AsyncIterable[bytes],
        artifact_file_name: str,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log artifact contents from an async chunk stream.

        Args:
            chunks: Async iterable yielding binary chunks containing the artifact contents.
            artifact_file_name: Artifact filename to log. Any directory components are
                ignored; use ``artifact_path`` to specify the destination directory.
            artifact_path: Directory within the run's artifact directory in which to log
                the artifact.
        """


def verify_artifact_path(artifact_path):
    if artifact_path and path_not_unique(artifact_path):
        raise MlflowException(
            f"Invalid artifact path: '{artifact_path}'. {bad_path_message(artifact_path)}"
        )


# Attachment IDs are auto-generated as UUID4 by Attachment.__init__.
# Strict UUID validation doubles as path traversal prevention.
def _validate_attachment_path(path: str) -> None:
    try:
        parsed = uuid.UUID(path)
        if str(parsed) != path:
            raise ValueError("Non-canonical UUID format")
    except (ValueError, AttributeError, TypeError):
        # error_code is INVALID_PARAMETER_VALUE but this is an attribute/type validation failure
        raise MlflowException(
            f"Invalid attachment path: '{path}'. Attachment path must be a valid UUID.",
            error_code=INVALID_PARAMETER_VALUE,
            error_class="ATTRIBUTE_NOT_FOUND",
        )
