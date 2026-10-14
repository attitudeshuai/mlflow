"""Cross-process write locks for the filesystem tracking backend.

The locks are advisory and backed by a single byte of a lock file:

* On POSIX, ``fcntl.flock`` is used. The kernel releases the lock automatically
  when the holder process exits (including after ``SIGKILL``).
* On Windows, ``msvcrt.locking`` acquires an exclusive one-byte region lock.

A contender never blocks inside the OS call; it retries a non-blocking acquire
until the configured timeout expires, so a stuck holder produces an explicit
error instead of an indefinite hang.

The lock file additionally stores JSON metadata (holder pid, host, acquisition
time) starting at byte 1; byte 0 is the locked region. Because Windows enforces
mandatory region locks, contenders read the metadata at offset 1 and never touch
the locked byte. The metadata makes a lock left behind by a crashed process
identifiable, so it can be reclaimed once it is older than the configurable
stale-lock grace period (with a log line recording the reclaim).

The same process is allowed to reacquire a lock it already holds on the same
thread (reentrancy); distinct threads and processes contend through the OS lock.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS,
    MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS,
    MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY,
    MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
)

_logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    import msvcrt
else:
    import fcntl

LOCK_POLICY_ERROR = "error"
LOCK_POLICY_DISABLE = "disable"
_LOCK_POLICIES = (LOCK_POLICY_ERROR, LOCK_POLICY_DISABLE)

_POLL_INTERVAL_SECONDS = 0.1

#: Lock files start with one padding byte carrying the OS lock; the JSON
#: metadata begins after it so contenders can read it on Windows without
#: touching the locked region.
_META_PREFIX = b" "


class LockUnsupportedError(Exception):
    """Raised when the filesystem cannot provide the required lock semantics."""


@dataclass(frozen=True)
class FileStoreLockConfig:
    enabled: bool
    timeout_seconds: float
    stale_grace_seconds: float
    unsupported_policy: str


def _read_float_env(env_var) -> float:
    try:
        return float(env_var.get())
    except (TypeError, ValueError) as e:
        raise MlflowException(
            f"Invalid value {env_var.get_raw()!r} for {env_var.name}: expected a number. {e}",
            INVALID_PARAMETER_VALUE,
        )


def load_lock_config() -> FileStoreLockConfig:
    """Load and validate cross-process lock configuration.

    Invalid timeout, grace period or policy values fail here, at configuration
    parsing time, rather than at the first locking operation.
    """
    enabled = MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED.get()
    timeout = _read_float_env(MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS)
    stale_grace = _read_float_env(MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS)
    policy = str(MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY.get()).strip().lower()

    if timeout <= 0:
        raise MlflowException(
            f"Invalid value {timeout!r} for {MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS.name}: "
            "lock wait timeout must be a positive number of seconds.",
            INVALID_PARAMETER_VALUE,
        )
    if stale_grace < 0:
        raise MlflowException(
            f"Invalid value {stale_grace!r} for "
            f"{MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS.name}: "
            "stale lock grace period must be zero or a positive number of seconds.",
            INVALID_PARAMETER_VALUE,
        )
    if policy not in _LOCK_POLICIES:
        raise MlflowException(
            f"Invalid value {policy!r} for {MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY.name}: "
            f"must be one of {list(_LOCK_POLICIES)}.",
            INVALID_PARAMETER_VALUE,
        )
    return FileStoreLockConfig(
        enabled=bool(enabled),
        timeout_seconds=timeout,
        stale_grace_seconds=stale_grace,
        unsupported_policy=policy,
    )


def _try_lock(fd: int) -> None:
    """Acquire a non-blocking exclusive lock on byte 0.

    Raises OSError if another open file description currently holds it.
    """
    os.lseek(fd, 0, os.SEEK_SET)
    if IS_WINDOWS:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    if IS_WINDOWS:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


class CrossProcessLock:
    """An exclusive, reentrant (per thread), cross-process lock on a lock file.

    Args:
        lock_file: Path of the lock file. Created on demand alongside its parent
            directory.
        timeout_seconds: Maximum time to wait for the lock before raising.
        stale_grace_seconds: Age after which leftover metadata is treated as a
            crashed-holder remnant and the acquisition is logged as a reclaim.
        object_description: Human-readable name of the protected object (e.g.
            ``"run '<id>'"``), included in timeout and reclaim messages.
    """

    # Keyed by lock file path; counts nested acquisitions on the same thread.
    _reentrancy = threading.local()

    def __init__(
        self,
        lock_file: str,
        timeout_seconds: float,
        stale_grace_seconds: float,
        object_description: str,
    ):
        self._lock_file = os.path.abspath(lock_file)
        self._timeout = timeout_seconds
        self._stale_grace = stale_grace_seconds
        self._object_description = object_description
        self._fd: int | None = None

    @classmethod
    def _held_locks(cls) -> dict[str, int]:
        held = getattr(cls._reentrancy, "held", None)
        if held is None:
            held = {}
            cls._reentrancy.held = held
        return held

    def _read_metadata(self) -> dict[str, Any] | None:
        """Read holder metadata without touching the locked byte 0."""
        fd = None
        try:
            fd = os.open(self._lock_file, os.O_RDONLY)
            os.lseek(fd, 1, os.SEEK_SET)
            raw = os.read(fd, 65536)
        except OSError:
            return None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if not raw:
            return None
        try:
            meta = json.loads(raw.decode("utf-8"))
            return meta if isinstance(meta, dict) else None
        except (ValueError, UnicodeDecodeError):
            return None

    @staticmethod
    def _metadata_age(meta: dict[str, Any] | None) -> float | None:
        if not meta:
            return None
        try:
            return max(0.0, time.time() - float(meta["acquired_at"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _write_metadata(self, fd: int) -> None:
        meta = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": time.time(),
            "object": self._object_description,
        }
        payload = _META_PREFIX + json.dumps(meta, separators=(",", ":")).encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)

    def _timeout_error(self, waited: float, meta: dict[str, Any] | None) -> MlflowException:
        holder = "an unknown process"
        if meta:
            age = self._metadata_age(meta)
            age_text = f" {age:.1f}s ago" if age is not None else ""
            holder = (
                f"pid {meta.get('pid')} on host {meta.get('host')}, which acquired it{age_text}"
            )
        return MlflowException(
            f"Timed out after {waited:.1f}s waiting for the cross-process write lock of "
            f"{self._object_description} (lock file: {self._lock_file}). The lock is held by "
            f"{holder}. Another process may still be updating this object or have crashed "
            "while holding the lock; increase the configured wait timeout "
            "(MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS) or retry later.",
            INTERNAL_ERROR,
        )

    def acquire(self) -> None:
        held = self._held_locks()
        if self._lock_file in held:
            held[self._lock_file] += 1
            return

        os.makedirs(os.path.dirname(self._lock_file), exist_ok=True)
        deadline = time.monotonic() + self._timeout
        stale_warning_emitted = False
        while True:
            fd = None
            try:
                fd = os.open(self._lock_file, os.O_RDWR | os.O_CREAT, 0o644)
                _try_lock(fd)
            except OSError:
                if fd is not None:
                    os.close(fd)
                meta = self._read_metadata()
                age = self._metadata_age(meta)
                if age is not None and age > self._stale_grace and not stale_warning_emitted:
                    stale_warning_emitted = True
                    _logger.warning(
                        "Write lock of %s has been held for %.1fs, longer than the stale lock "
                        "grace period of %.1fs (holder pid %s on host %s, lock file: %s). "
                        "Waiting for the lock to be released or expire before retrying.",
                        self._object_description,
                        age,
                        self._stale_grace,
                        (meta or {}).get("pid"),
                        (meta or {}).get("host"),
                        self._lock_file,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._timeout_error(self._timeout, meta)
                time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
                continue

            # The OS lock is ours. A non-empty file means the previous holder did
            # not remove it (e.g. it was killed). Reclaim it once it has aged past
            # the grace period.
            previous = self._read_metadata()
            age = self._metadata_age(previous)
            if (
                previous
                and previous.get("pid") != os.getpid()
                and age is not None
                and age > self._stale_grace
            ):
                _logger.warning(
                    "Recovering stale write lock of %s (lock file: %s): last held by pid %s "
                    "on host %s %.1fs ago, exceeding the stale lock grace period of %.1fs. "
                    "The previous holder is assumed to have crashed; the lock has been "
                    "reclaimed.",
                    self._object_description,
                    self._lock_file,
                    previous.get("pid"),
                    previous.get("host"),
                    age,
                    self._stale_grace,
                )
            self._fd = fd
            try:
                self._write_metadata(fd)
            except OSError:
                # Do not leak the OS lock (e.g. on a full disk) while propagating.
                self._fd = None
                try:
                    _unlock(fd)
                finally:
                    os.close(fd)
                raise
            held[self._lock_file] = 1
            return

    def release(self) -> None:
        held = self._held_locks()
        count = held.get(self._lock_file, 0)
        if count > 1:
            held[self._lock_file] = count - 1
            return
        held.pop(self._lock_file, None)

        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            _logger.debug("Failed to release lock %s", self._lock_file, exc_info=True)
        try:
            os.close(fd)
        except OSError:
            pass
        # Best effort: leaving the file behind is safe because stale files are
        # reclaimed after the grace period. Removal may legitimately fail on
        # Windows while contenders still have the file open.
        try:
            os.unlink(self._lock_file)
        except OSError:
            pass

    def __enter__(self) -> "CrossProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def probe_locking_supported(lock_dir: str, probe_file_name: str = ".probe.lock") -> None:
    """Verify that lock_dir supports the required locking semantics.

    Raises LockUnsupportedError when a lock cannot be acquired on the backing
    filesystem (e.g. a read-only mount or a network filesystem without lock
    support).
    """
    probe_path = os.path.join(lock_dir, probe_file_name)
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError as e:
        raise LockUnsupportedError(f"Cannot create lock directory '{lock_dir}': {e}") from e

    fd = None
    try:
        fd = os.open(probe_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            _try_lock(fd)
        except OSError as e:
            raise LockUnsupportedError(
                f"Failed to acquire an exclusive file lock in '{lock_dir}': {e}"
            ) from e
        _unlock(fd)

        if not IS_WINDOWS:
            # POSIX byte-range locks go through lockd/NLM on network filesystems
            # and fail with ENOLCK/ENOSYS when locking is unsupported.
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
                fcntl.lockf(fd, fcntl.LOCK_UN, 1)
            except OSError as e:
                if e.errno in (
                    errno.ENOLCK,
                    errno.ENOSYS,
                    getattr(errno, "EOPNOTSUPP", -1),
                    errno.EINVAL,
                ):
                    raise LockUnsupportedError(
                        f"Filesystem backing '{lock_dir}' does not support file locks: {e}"
                    ) from e
                raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(probe_path)
        except OSError:
            pass
