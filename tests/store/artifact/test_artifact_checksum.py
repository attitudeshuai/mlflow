"""Tests for optional artifact content digest (checksum) verification."""

import contextlib
import hashlib
import json
import logging
import os
import posixpath
import threading

import pytest

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM,
    MLFLOW_ARTIFACT_CHECKSUM_ENABLED,
    MLFLOW_ARTIFACT_CHECKSUM_MAX_ATTEMPTS,
    MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.checksum import (
    ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME,
    ArtifactChecksumMismatchException,
    ArtifactManifestMalformed,
    build_manifest,
    compute_file_digest,
    is_manifest_file,
    is_missing_artifact_error,
    parse_manifest,
    serialize_manifest,
)
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.utils.file_utils import path_to_local_file_uri

_REPO_LOGGER = "mlflow.store.artifact.artifact_repo"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@contextlib.contextmanager
def capture_log_records(logger_name: str, level: int = logging.INFO):
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    lg = logging.getLogger(logger_name)
    handler = _Handler()
    handler.setLevel(level)
    lg.addHandler(handler)
    previous_level = lg.level
    lg.setLevel(level)
    try:
        yield records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(previous_level)


def log_messages(records):
    return "\n".join(r.getMessage() for r in records)


class InMemoryArtifactRepository(ArtifactRepository):
    """Minimal leaf repository storing objects in memory."""

    def __init__(self, artifact_uri: str = "memory://root"):
        super().__init__(artifact_uri)
        self.objects: dict[str, bytes] = {}
        # key -> number of subsequent downloads that write truncated bytes
        self.truncated_downloads: dict[str, int] = {}

    @staticmethod
    def _key(local_file, artifact_path):
        name = os.path.basename(local_file)
        return posixpath.join(artifact_path, name) if artifact_path else name

    def log_artifact(self, local_file, artifact_path=None):
        key = self._key(local_file, artifact_path)
        with open(local_file, "rb") as f:
            self.objects[key] = f.read()
        self._record_uploaded_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        for root, _, files in os.walk(local_dir):
            for name in files:
                src = os.path.join(root, name)
                rel = os.path.relpath(src, local_dir).replace(os.sep, "/")
                key = posixpath.join(artifact_path, rel) if artifact_path else rel
                with open(src, "rb") as f:
                    self.objects[key] = f.read()
                parent = posixpath.dirname(key)
                self._record_uploaded_artifact(src, parent or None)

    def list_artifacts(self, path=None):
        if path and path in self.objects:
            # A path referencing a single file returns an empty listing.
            return []
        prefix = f"{path}/" if path else ""
        infos = []
        seen_dirs = set()
        for key in sorted(self.objects):
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            if "/" in rest:
                directory = rest.split("/", 1)[0]
                if directory not in seen_dirs:
                    seen_dirs.add(directory)
                    infos.append(FileInfo(prefix + directory, True, None))
            else:
                infos.append(FileInfo(key, False, len(self.objects[key])))
        return infos

    def _download_file(self, remote_file_path, local_path):
        if remote_file_path not in self.objects:
            raise MlflowException(
                f"no such object: {remote_file_path}", error_code=RESOURCE_DOES_NOT_EXIST
            )
        data = self.objects[remote_file_path]
        remaining = self.truncated_downloads.get(remote_file_path, 0)
        if remaining > 0:
            self.truncated_downloads[remote_file_path] = remaining - 1
            data = data[: max(1, len(data) // 2)]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)


class InMemoryCloudArtifactRepository(CloudArtifactRepository):
    """CloudArtifactRepository that "uploads parts" by concatenating chunks in memory."""

    def __init__(self):
        super().__init__("memory://root")
        self.objects: dict[str, bytes] = {}

    @staticmethod
    def _key(local_file, artifact_path):
        name = os.path.basename(local_file)
        return posixpath.join(artifact_path, name) if artifact_path else name

    def log_artifact(self, local_file, artifact_path=None):
        key = self._key(local_file, artifact_path)
        self._upload_to_cloud(None, local_file, key)
        self._record_uploaded_artifact(local_file, artifact_path)

    def _get_write_credential_infos(self, remote_file_paths):
        return [None for _ in remote_file_paths]

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        with open(src_file_path, "rb") as f:
            data = f.read()
        # Simulate a multipart upload: assemble the object from two byte ranges.
        mid = len(data) // 2
        assembled = bytearray()
        assembled.extend(data[:mid])
        assembled.extend(data[mid:])
        self.objects[artifact_file_path] = bytes(assembled)

    def list_artifacts(self, path=None):
        if path and path in self.objects:
            return []
        prefix = f"{path}/" if path else ""
        return [
            FileInfo(key, False, len(data))
            for key, data in sorted(self.objects.items())
            if key.startswith(prefix) and "/" not in key[len(prefix) :]
        ]

    def _get_read_credential_infos(self, remote_file_paths):
        return [None for _ in remote_file_paths]

    def _download_from_cloud(self, remote_file_path, local_path):
        with open(local_path, "wb") as f:
            f.write(self.objects[remote_file_path])


def write_source_tree(root, files: dict[str, bytes]) -> str:
    for rel, content in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
    return root


def read_manifest(repo: InMemoryArtifactRepository) -> dict:
    return parse_manifest(repo.objects[ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME])


@pytest.fixture
def checksum_enabled(monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ENABLED.name, "true")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_invalid_algorithm_name_rejected_when_config_read(monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.name, "not-a-real-algorithm")
    with pytest.raises(ValueError, match="digest algorithm"):
        MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.get()


def test_invalid_unsupported_policy_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.name, "explode")
    with pytest.raises(ValueError, match="must be one of"):
        MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.get()


def test_algorithm_config_accepts_guaranteed_algorithms(monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.name, "sha512")
    assert MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.get() == "sha512"


# ----------------------------------------------------------------------
# Manifest parsing
# ----------------------------------------------------------------------


def test_parse_manifest_roundtrip(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    size, digest = compute_file_digest(str(f), "sha256")
    manifest = build_manifest("sha256", {"f": (size, digest)})
    parsed = parse_manifest(serialize_manifest(manifest))
    assert parsed["version"] == 1
    assert parsed["algorithm"] == "sha256"
    assert parsed["files"]["f"] == {"size": 3, "digest": digest}


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        json.dumps({"version": 2, "algorithm": "sha256", "files": {}}).encode(),
        json.dumps({"version": 1, "algorithm": "md4-not-real", "files": {}}).encode(),
        json.dumps({"version": 1, "algorithm": "sha256", "files": {"a": "oops"}}).encode(),
        json.dumps(
            {"version": 1, "algorithm": "sha256", "files": {"a": {"size": -1}}}
        ).encode(),
    ],
)
def test_parse_manifest_rejects_malformed(raw):
    with pytest.raises(ArtifactManifestMalformed):
        parse_manifest(raw)


def test_is_manifest_file():
    assert is_manifest_file(ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME)
    assert is_manifest_file(f"dir/{ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME}")
    assert not is_manifest_file("dir/other.json")
    assert not is_manifest_file(None)


def test_is_missing_artifact_error():
    assert is_missing_artifact_error(
        MlflowException("x", error_code=RESOURCE_DOES_NOT_EXIST)
    )
    assert is_missing_artifact_error(FileNotFoundError())
    assert not is_missing_artifact_error(MlflowException("x"))
    assert not is_missing_artifact_error(ValueError("boom"))


# ----------------------------------------------------------------------
# Upload manifest behavior
# ----------------------------------------------------------------------


def test_disabled_by_default_writes_no_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"hello")
    root = tmp_path / "root"
    root.mkdir()
    repo = LocalArtifactRepository(path_to_local_file_uri(str(root)))
    repo.log_artifacts(str(src))
    assert not (root / ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME).exists()
    listing = {f.path for f in repo.list_artifacts()}
    assert listing == {"a.txt"}


def test_enabled_records_manifest_and_merges_subsequent_uploads(
    tmp_path, checksum_enabled
):
    root = tmp_path / "root"
    root.mkdir()
    repo = LocalArtifactRepository(path_to_local_file_uri(str(root)))

    src1 = tmp_path / "src1"
    write_source_tree(str(src1), {"a.txt": b"hello", "sub/b.bin": b"\x00" * 100})
    repo.log_artifacts(str(src1))

    manifest = json.loads((root / ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME).read_text())
    assert manifest["version"] == 1
    assert manifest["algorithm"] == "sha256"
    assert set(manifest["files"]) == {"a.txt", "sub/b.bin"}
    assert manifest["files"]["a.txt"]["size"] == 5
    assert (
        manifest["files"]["a.txt"]["digest"]
        == hashlib.sha256(b"hello").hexdigest()
    )

    # A second batch plus a singular upload merges entries instead of overwriting.
    src2 = tmp_path / "src2"
    write_source_tree(str(src2), {"c.txt": b"more"})
    repo.log_artifacts(str(src2))
    extra = tmp_path / "d.txt"
    extra.write_bytes(b"singular")
    repo.log_artifact(str(extra), artifact_path="nested")

    merged = json.loads((root / ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME).read_text())
    assert set(merged["files"]) == {
        "a.txt",
        "sub/b.bin",
        "c.txt",
        "nested/d.txt",
    }
    assert merged["files"]["nested/d.txt"]["digest"] == hashlib.sha256(b"singular").hexdigest()


def test_configured_algorithm_is_used_for_new_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ENABLED.name, "true")
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.name, "sha512")
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "src"
    write_source_tree(str(src), {"a": b"x"})
    repo = LocalArtifactRepository(path_to_local_file_uri(str(root)))
    repo.log_artifacts(str(src))
    manifest = json.loads((root / ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME).read_text())
    assert manifest["algorithm"] == "sha512"
    assert manifest["files"]["a"]["digest"] == hashlib.sha512(b"x").hexdigest()


def test_existing_manifest_algorithm_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ENABLED.name, "true")
    repo = InMemoryArtifactRepository()
    first = tmp_path / "first"
    write_source_tree(str(first), {"a": b"1"})
    repo.log_artifacts(str(first))
    assert read_manifest(repo)["algorithm"] == "sha256"

    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_ALGORITHM.name, "sha512")
    second = tmp_path / "second"
    write_source_tree(str(second), {"b": b"2"})
    with capture_log_records(_REPO_LOGGER) as records:
        repo.log_artifacts(str(second))
        assert "differs from the existing manifest algorithm" in log_messages(records)
    manifest = read_manifest(repo)
    assert manifest["algorithm"] == "sha256"
    assert manifest["files"]["b"]["digest"] == hashlib.sha256(b"2").hexdigest()


def test_reserved_manifest_filename_is_not_recorded(tmp_path, checksum_enabled):
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    write_source_tree(
        str(src),
        {"a": b"1", f"sub/{ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME}": b"user content"},
    )
    with capture_log_records(_REPO_LOGGER, logging.WARNING) as records:
        repo.log_artifacts(str(src))
        assert "reserved" in log_messages(records)
    manifest = read_manifest(repo)
    assert ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME not in manifest["files"]
    assert f"sub/{ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME}" not in manifest["files"]
    assert "a" in manifest["files"]


def test_batch_upload_with_root_manifest_name_leaves_valid_manifest(
    tmp_path, checksum_enabled
):
    # The manifest is written last in a batch, so it must win over a user file
    # that happens to use the reserved root-level manifest name.
    root = tmp_path / "root"
    root.mkdir()
    repo = LocalArtifactRepository(path_to_local_file_uri(str(root)))
    src = tmp_path / "src"
    write_source_tree(
        str(src),
        {"a": b"1", ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME: b"user content"},
    )
    repo.log_artifacts(str(src))
    manifest = json.loads((root / ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME).read_text())
    assert ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME not in manifest["files"]
    assert "a" in manifest["files"]


def test_concurrent_singular_uploads_do_not_clobber_manifest(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    num_threads = 20
    barrier = threading.Barrier(num_threads)

    def upload(i):
        src = tmp_path / f"f{i}"
        src.write_bytes(f"content-{i}".encode())
        barrier.wait()
        repo.log_artifact(str(src))

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest = read_manifest(repo)
    assert len(manifest["files"]) == num_threads
    for i in range(num_threads):
        assert manifest["files"][f"f{i}"]["digest"] == (
            hashlib.sha256(f"content-{i}".encode()).hexdigest()
        )


def test_manifest_write_failure_degrades_to_warning(tmp_path, checksum_enabled):
    class ManifestWriteFailureRepo(InMemoryArtifactRepository):
        def log_artifact(self, local_file, artifact_path=None):
            if os.path.basename(local_file) == ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME:
                raise MlflowException("manifest writes are forbidden")
            super().log_artifact(local_file, artifact_path)

    src = tmp_path / "src"
    write_source_tree(str(src), {"a": b"1"})

    repo = ManifestWriteFailureRepo()
    with capture_log_records(_REPO_LOGGER, logging.WARNING) as records:
        repo.log_artifacts(str(src))
        assert "Failed to update the artifact checksum manifest" in log_messages(records)
    # the data file itself was uploaded
    assert repo.objects["a"] == b"1"

    # With the strict policy, the upload surfaces the error.
    os.environ[MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.name] = "error"
    try:
        strict_repo = ManifestWriteFailureRepo()
        with pytest.raises(MlflowException, match="forbidden"):
            strict_repo.log_artifacts(str(src))
    finally:
        os.environ.pop(MLFLOW_ARTIFACT_CHECKSUM_UNSUPPORTED_POLICY.name, None)


# ----------------------------------------------------------------------
# Download verification behavior
# ----------------------------------------------------------------------


def test_clean_download_verifies(tmp_path, checksum_enabled):
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    write_source_tree(str(src), {"a.txt": b"hello", "sub/b.bin": os.urandom(2048)})
    repo.log_artifacts(str(src))

    dst = tmp_path / "dst"
    dst.mkdir()
    with capture_log_records(_REPO_LOGGER) as records:
        result = repo.download_artifacts("", dst_path=str(dst))
        messages = log_messages(records)
    assert result == str(dst)
    assert "verification passed" in messages
    assert (dst / "sub" / "b.bin").read_bytes()


def test_corrupted_file_fails_download_with_file_level_report(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    good = os.urandom(4096)
    write_source_tree(str(src), {"ok.txt": b"fine", "bad.bin": good})
    repo.log_artifacts(str(src))

    # Corrupt the stored object (truncation on the storage side).
    repo.objects["bad.bin"] = good[:100]

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(ArtifactChecksumMismatchException) as exc_info:
        repo.download_artifacts("", dst_path=str(dst))

    report = exc_info.value.report
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.path == "bad.bin"
    assert failure.expected_size == len(good)
    assert failure.actual_size == 100
    assert failure.expected_digest == hashlib.sha256(good).hexdigest()
    assert failure.actual_digest == hashlib.sha256(good[:100]).hexdigest()
    assert "ok.txt" in report.checked_files
    # Manifest corruption is a different exception class; this is purely a file mismatch.
    assert not isinstance(exc_info.value, ArtifactManifestMalformed)


def test_retransmitted_file_passes_overall_verification(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    write_source_tree(str(src), {"a.bin": os.urandom(4096)})
    repo.log_artifacts(str(src))

    # The first transfer lands truncated; the re-download of just this file succeeds.
    repo.truncated_downloads["a.bin"] = 1

    dst = tmp_path / "dst"
    dst.mkdir()
    with capture_log_records(_REPO_LOGGER) as records:
        result = repo.download_artifacts("", dst_path=str(dst))
        messages = log_messages(records)
    assert result == str(dst)
    assert "re-downloading only the failed file" in messages
    assert "verification passed" in messages


def test_failed_retransmission_eventually_reports_failure(
    tmp_path, monkeypatch, checksum_enabled
):
    monkeypatch.setenv(MLFLOW_ARTIFACT_CHECKSUM_MAX_ATTEMPTS.name, "2")
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    data = os.urandom(4096)
    write_source_tree(str(src), {"a.bin": data})
    repo.log_artifacts(str(src))
    repo.objects["a.bin"] = data[: 50]  # permanently truncated

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(ArtifactChecksumMismatchException) as exc_info:
        repo.download_artifacts("", dst_path=str(dst))
    assert exc_info.value.report.failures[0].path == "a.bin"


def test_missing_manifest_is_reported_as_unverified_not_error(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    repo.objects["legacy.txt"] = b"old artifact"
    dst = tmp_path / "dst"
    dst.mkdir()
    with capture_log_records(_REPO_LOGGER) as records:
        result = repo.download_artifacts("legacy.txt", dst_path=str(dst))
        messages = log_messages(records)
    assert result == os.path.join(str(dst), "legacy.txt")
    assert "No artifact checksum manifest found" in messages
    assert "unverified" in messages


def test_malformed_manifest_is_reported_separately_from_file_corruption(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    repo.objects["a"] = b"hello"
    repo.objects[ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME] = b"{not valid json"
    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(ArtifactManifestMalformed, match="corrupt or has an invalid format"):
        repo.download_artifacts("a", dst_path=str(dst))


def test_single_file_without_manifest_entry_is_unverified(
    tmp_path, checksum_enabled
):
    repo = InMemoryArtifactRepository()
    src = tmp_path / "src"
    write_source_tree(str(src), {"listed.txt": b"x"})
    repo.log_artifacts(str(src))
    # An extra file exists in storage but is not covered by the manifest.
    repo.objects["extra.txt"] = b"not in manifest"
    dst = tmp_path / "dst"
    dst.mkdir()
    with capture_log_records(_REPO_LOGGER) as records:
        repo.download_artifacts("extra.txt", dst_path=str(dst))
        assert "extra.txt" in log_messages(records)
        assert "not verified" in log_messages(records)


# ----------------------------------------------------------------------
# Local repository in-place downloads (dst_path=None)
# ----------------------------------------------------------------------


def test_local_in_place_download_verifies(tmp_path, checksum_enabled):
    root = tmp_path / "root"
    root.mkdir()
    src = tmp_path / "src"
    write_source_tree(str(src), {"a.txt": b"hello"})
    repo = LocalArtifactRepository(path_to_local_file_uri(str(root)))
    repo.log_artifacts(str(src))

    assert repo.download_artifacts("a.txt") == os.path.join(str(root), "a.txt")

    (root / "a.txt").write_bytes(b"corrupt")
    with pytest.raises(ArtifactChecksumMismatchException):
        repo.download_artifacts("a.txt")


def test_runs_repo_does_not_swallow_checksum_failure(
    tmp_path, monkeypatch, checksum_enabled
):
    leaf = InMemoryArtifactRepository()
    src = tmp_path / "src"
    data = os.urandom(1024)
    write_source_tree(str(src), {"a.bin": data})
    leaf.log_artifacts(str(src))
    leaf.objects["a.bin"] = data[:10]  # truncate in storage

    proxy = RunsArtifactRepository.__new__(RunsArtifactRepository)
    proxy.artifact_uri = "runs:/run-id/a.bin"
    proxy.tracking_uri = None
    proxy.registry_uri = None
    proxy.repo = leaf
    # No associated logged model; avoid a tracking client lookup.
    monkeypatch.setattr(proxy, "_get_logged_model_artifact_repo", lambda *a, **k: None)

    dst = tmp_path / "dst"
    dst.mkdir()
    with pytest.raises(ArtifactChecksumMismatchException):
        proxy.download_artifacts("a.bin", dst_path=str(dst))


def test_runs_repo_still_handles_missing_artifacts(tmp_path, monkeypatch):
    leaf = InMemoryArtifactRepository()
    proxy = RunsArtifactRepository.__new__(RunsArtifactRepository)
    proxy.artifact_uri = "runs:/run-id/missing.bin"
    proxy.tracking_uri = None
    proxy.registry_uri = None
    proxy.repo = leaf
    monkeypatch.setattr(proxy, "_get_logged_model_artifact_repo", lambda *a, **k: None)

    dst = tmp_path / "dst"
    dst.mkdir()
    # Pre-existing behavior: a missing artifact in both run/model raises a generic
    # "please ensure the path is correct" error, not an integrity error.
    with pytest.raises(MlflowException, match="please ensure that the path is correct"):
        proxy.download_artifacts("missing.bin", dst_path=str(dst))


# ----------------------------------------------------------------------
# Cloud batch upload and multipart whole-file conclusion
# ----------------------------------------------------------------------


def test_cloud_batch_records_whole_file_manifest_and_download_verifies(
    tmp_path, monkeypatch, checksum_enabled
):
    monkeypatch.setenv("MLFLOW_ENABLE_MULTIPART_DOWNLOAD", "false")
    repo = InMemoryCloudArtifactRepository()
    src = tmp_path / "src"
    big = os.urandom(3 * 1024 * 1024)
    write_source_tree(str(src), {"big.bin": big, "small.txt": b"hi"})
    repo.log_artifacts(str(src))

    manifest = parse_manifest(repo.objects[ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME])
    assert manifest["files"]["big.bin"] == {
        "size": len(big),
        "digest": hashlib.sha256(big).hexdigest(),
    }
    assert manifest["files"]["small.txt"]["digest"] == hashlib.sha256(b"hi").hexdigest()

    # A clean assembled object verifies as a whole.
    dst = tmp_path / "dst"
    dst.mkdir()
    assert repo.download_artifacts("", dst_path=str(dst)) == str(dst)

    # A truncated assembled object (a missing tail chunk) fails on the whole file.
    repo.objects["big.bin"] = big[: len(big) - 17]
    dst2 = tmp_path / "dst2"
    dst2.mkdir()
    with pytest.raises(ArtifactChecksumMismatchException) as exc_info:
        repo.download_artifacts("big.bin", dst_path=str(dst2))
    assert exc_info.value.report.failures[0].path == "big.bin"
