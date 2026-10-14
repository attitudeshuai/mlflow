"""Content digest (checksum) verification for artifact transfers.

When checksum verification is enabled, an upload computes a content digest for
every file and writes a manifest artifact at a fixed location in the repository.
Downloads verify each file against the manifest and report failures at file
granularity.

The manifest is versioned JSON::

    {
        "version": 1,
        "algorithm": "sha256",
        "files": {
            "relative/path": {"size": 123, "digest": "<hexdigest>"}
        }
    }

The manifest only provides integrity verification; it is not signed and does
not protect against malicious writers.
"""

import errno
import hashlib
import json
import os
from dataclasses import dataclass, field

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    RESOURCE_DOES_NOT_EXIST,
)

# Chunk size for streaming file contents while computing digests (1 MB).
_CHECKSUM_STREAM_CHUNK_SIZE = 1024 * 1024

#: File name of the checksum manifest, stored at the repository root.
ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME = "mlflow-artifact-manifest.json"

#: Current manifest format version.
ARTIFACT_CHECKSUM_MANIFEST_VERSION = 1

# Manifest states reported by download verification.
MANIFEST_STATE_PRESENT = "present"
MANIFEST_STATE_MISSING = "missing"
MANIFEST_STATE_MALFORMED = "malformed"
MANIFEST_STATE_UNAVAILABLE = "unavailable"

# Verification outcomes.
VERIFICATION_STATUS_VERIFIED = "verified"
VERIFICATION_STATUS_SKIPPED = "skipped"
VERIFICATION_STATUS_FAILED = "failed"

SIZE_MISMATCH = "size_mismatch"
DIGEST_MISMATCH = "digest_mismatch"


class ArtifactManifestMalformed(MlflowException):
    """Raised when the checksum manifest exists but is corrupt or has an invalid format.

    This is deliberately distinct from file content mismatches: a broken manifest
    cannot be used to judge any individual file and must be reported separately.
    """

    def __init__(self, reason: str):
        super().__init__(
            message=(
                f"The artifact checksum manifest '{ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME}' is "
                f"corrupt or has an invalid format and could not be used for verification: "
                f"{reason}"
            ),
            error_code=INTERNAL_ERROR,
        )
        self.reason = reason


class ArtifactChecksumMismatchException(MlflowException):
    """Raised when one or more downloaded files fail checksum verification after retries."""

    def __init__(self, report: "ArtifactChecksumReport"):
        self.report = report
        super().__init__(
            message=report.format_failures(),
            error_code=INTERNAL_ERROR,
        )


def new_digester(algorithm: str):
    """Return a fresh hashlib object for ``algorithm``.

    ``usedforsecurity=False`` is passed when supported (required for FIPS builds
    when using e.g. md5); algorithms that do not accept the keyword fall back.
    """
    try:
        return hashlib.new(algorithm, usedforsecurity=False)
    except TypeError:
        return hashlib.new(algorithm)


def compute_file_digest(local_file: str, algorithm: str) -> tuple[int, str]:
    """Compute ``(size, hexdigest)`` of a local file by streaming its contents."""
    digester = new_digester(algorithm)
    size = 0
    with open(local_file, "rb") as f:
        while chunk := f.read(_CHECKSUM_STREAM_CHUNK_SIZE):
            digester.update(chunk)
            size += len(chunk)
    return size, digester.hexdigest()


def build_manifest(algorithm: str, files: dict[str, tuple[int, str]]) -> dict:
    """Build a manifest object from a mapping of ``path -> (size, digest)``."""
    return {
        "version": ARTIFACT_CHECKSUM_MANIFEST_VERSION,
        "algorithm": algorithm,
        "files": {
            path: {"size": int(size), "digest": digest}
            for path, (size, digest) in sorted(files.items())
        },
    }


def serialize_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_manifest(raw: bytes | str) -> dict:
    """Parse and validate a serialized manifest.

    Raises:
        ArtifactManifestMalformed: The payload is not valid JSON or does not match
            the expected manifest structure.
    """
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtifactManifestMalformed(f"manifest is not valid JSON ({e})") from e

    if not isinstance(manifest, dict):
        raise ArtifactManifestMalformed("top-level value must be a JSON object")

    version = manifest.get("version")
    if version != ARTIFACT_CHECKSUM_MANIFEST_VERSION:
        raise ArtifactManifestMalformed(
            f"unsupported manifest version {version!r}; expected "
            f"{ARTIFACT_CHECKSUM_MANIFEST_VERSION}"
        )

    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ArtifactManifestMalformed("'algorithm' must be a non-empty string")
    if algorithm not in hashlib.algorithms_available:
        raise ArtifactManifestMalformed(f"unsupported digest algorithm {algorithm!r}")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactManifestMalformed("'files' must be an object")

    normalized = {}
    for path, entry in files.items():
        if not isinstance(path, str) or not path:
            raise ArtifactManifestMalformed("file paths must be non-empty strings")
        if not isinstance(entry, dict):
            raise ArtifactManifestMalformed(f"entry for {path!r} must be an object")
        size = entry.get("size")
        digest = entry.get("digest")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactManifestMalformed(
                f"entry for {path!r} has an invalid 'size': {size!r}"
            )
        if not isinstance(digest, str) or not digest:
            raise ArtifactManifestMalformed(
                f"entry for {path!r} has an invalid 'digest': {digest!r}"
            )
        normalized[path] = {"size": size, "digest": digest}

    return {"version": version, "algorithm": algorithm, "files": normalized}


def is_manifest_file(artifact_path: str | None) -> bool:
    """Return whether a repository-relative path refers to the checksum manifest."""
    return bool(artifact_path) and os.path.basename(artifact_path.replace("\\", "/")) == (
        ARTIFACT_CHECKSUM_MANIFEST_FILE_NAME
    )


def is_missing_artifact_error(exc: Exception) -> bool:
    """Best-effort detection of a "remote file does not exist" error across backends.

    Artifact repositories surface missing files through different exception types
    (MlflowException with RESOURCE_DOES_NOT_EXIST, FileNotFoundError, paramiko/FTP
    protocol errors). Anything that does not clearly indicate absence returns False
    so callers can treat it as a real failure.
    """
    if isinstance(exc, MlflowException):
        # MlflowException stores error_code as the enum *name* (a string).
        if exc.error_code in (
            RESOURCE_DOES_NOT_EXIST,
            "RESOURCE_DOES_NOT_EXIST",
        ):
            return True
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOENT:
        return True
    # requests.HTTPError / urllib errors expose a response with a 404 status.
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    # paramiko raises IOError without a useful errno; ftplib error_perm starts "550".
    message = str(exc).lower()
    if isinstance(exc, OSError) and "no such file" in message:
        return True
    if "550" in message and "no such file" in message:
        return True
    return False


@dataclass
class FileChecksumFailure:
    """File-level verification failure details."""

    path: str
    expected_size: int | None
    actual_size: int | None
    expected_digest: str | None
    actual_digest: str | None
    reason: str

    def format_line(self) -> str:
        lines = [f"  Path: {self.path} ({self.reason})"]
        lines.append(f"    Expected size:   {self.expected_size}")
        lines.append(f"    Actual size:     {self.actual_size}")
        lines.append(f"    Expected digest: {self.expected_digest}")
        lines.append(f"    Actual digest:   {self.actual_digest}")
        return "\n".join(lines)


@dataclass
class ArtifactChecksumReport:
    """Result of verifying downloaded artifacts against a checksum manifest."""

    status: str
    manifest_state: str
    algorithm: str | None = None
    checked_files: list[str] = field(default_factory=list)
    unverified_files: list[str] = field(default_factory=list)
    failures: list[FileChecksumFailure] = field(default_factory=list)
    detail: str | None = None

    def format_failures(self) -> str:
        header = (
            "Artifact integrity verification failed. "
            f"{len(self.failures)} file(s) failed checksum verification "
            "(expected vs actual digests shown):\n"
        )
        return header + "\n".join(f.format_line() for f in self.failures)

    def format_summary(self) -> str:
        parts = [
            f"status={self.status}",
            f"manifest={self.manifest_state}",
            f"checked={len(self.checked_files)}",
            f"unverified={len(self.unverified_files)}",
            f"failed={len(self.failures)}",
        ]
        if self.algorithm:
            parts.append(f"algorithm={self.algorithm}")
        return ", ".join(parts)
