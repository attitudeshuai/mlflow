"""
Run liveness: heartbeat configuration, stale-run detection, and query-side hydration.

While a run is active, the client periodically writes the current timestamp to the
``mlflow.lastHeartbeatTime`` run tag (see :func:`AbstractStore.record_run_heartbeat
<mlflow.store.tracking.abstract_store.AbstractStore.record_run_heartbeat>`). The stores use the
freshness of that timestamp on read to distinguish runs whose process is still alive from
"zombie" runs that stay in the RUNNING state because the process was killed or lost power before
calling the end-run API.

Detection is computed entirely from persisted metadata, so it keeps working across server
restarts and behaves identically for the file and SQL backends. No new run status is introduced:

- ``mark`` policy (default): stale runs keep the RUNNING status but are flagged via
  ``RunInfo.suspected_zombie`` (and the ``mlflow.suspectedZombie`` tag) in get/search results.
- ``fail`` policy: stale runs are automatically transitioned to FAILED, their end time is set to
  the last heartbeat time, and a finalization reason is recorded on the run.

Runs that never wrote a heartbeat (old clients / historical data) are left untouched unless the
operator opts into a (typically wider) legacy detection window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import (
    MLFLOW_ENABLE_RUN_HEARTBEAT,
    MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS,
    MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS,
    MLFLOW_RUN_STALE_POLICY,
    MLFLOW_RUN_STALE_TIMEOUT_SECONDS,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_LAST_HEARTBEAT_TIME,
    MLFLOW_RUN_SUSPECTED_ZOMBIE,
)
from mlflow.utils.time import get_current_time_millis

if TYPE_CHECKING:
    from mlflow.entities.run import Run

_logger = logging.getLogger(__name__)

MILLIS_PER_SECOND = 1000

#: Stale runs are only flagged in get/search results, their status is never modified.
STALE_POLICY_MARK = "mark"
#: Stale runs are automatically transitioned to FAILED with a recorded finalization reason.
STALE_POLICY_FAIL = "fail"
_VALID_STALE_POLICIES = (STALE_POLICY_MARK, STALE_POLICY_FAIL)


class LivenessAction(str, Enum):
    """Outcome of evaluating a single run's liveness."""

    #: Run is not eligible for detection (not RUNNING, scheduled, terminated, ...).
    SKIP = "skip"
    #: Heartbeat is fresh. A stale marker, if any is left over, must be cleared.
    HEALTHY = "healthy"
    #: Run is stale and should only be flagged in query results.
    MARK_STALE = "mark_stale"
    #: Run is stale and should be automatically finalized as FAILED.
    FAIL_STALE = "fail_stale"


@dataclass(frozen=True)
class RunLivenessConfig:
    """Resolved liveness configuration.

    All time windows are expressed in milliseconds. ``legacy_timeout_ms`` is ``None`` when
    detection is disabled for runs that never recorded a heartbeat.
    """

    heartbeat_enabled: bool
    heartbeat_interval_ms: int
    stale_timeout_ms: int
    stale_policy: str
    legacy_timeout_ms: int | None


@dataclass(frozen=True)
class LivenessVerdict:
    """Decision produced by :func:`judge_run_liveness`."""

    action: LivenessAction
    #: End time to use when finalizing the run (last heartbeat time, or "now" for legacy runs).
    end_time_ms: int | None = None


def _require_positive_int(value: int, env_var_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{env_var_name} must be a positive integer (in seconds), but got {value!r}."
        )
    return value


def get_heartbeat_interval_seconds() -> int:
    """Read and validate the client heartbeat interval (a positive integer number of seconds)."""
    return _require_positive_int(
        MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.get(),
        MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name,
    )


def get_run_liveness_config() -> RunLivenessConfig:
    """Read and validate liveness-related environment variables.

    Validation happens here (at configuration parse time), so invalid values raise
    ``ValueError`` before any heartbeat or detection logic runs.
    """
    heartbeat_enabled = MLFLOW_ENABLE_RUN_HEARTBEAT.get()
    interval_s = get_heartbeat_interval_seconds()
    stale_timeout_s = _require_positive_int(
        MLFLOW_RUN_STALE_TIMEOUT_SECONDS.get(),
        MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name,
    )
    if stale_timeout_s < interval_s:
        raise ValueError(
            f"{MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name} ({stale_timeout_s}s) must be at least "
            f"{MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name} ({interval_s}s), otherwise a single "
            "delayed heartbeat could be mistaken for a dead run."
        )

    policy = (MLFLOW_RUN_STALE_POLICY.get() or "").strip().lower()
    if policy not in _VALID_STALE_POLICIES:
        raise ValueError(
            f"{MLFLOW_RUN_STALE_POLICY.name} must be one of {list(_VALID_STALE_POLICIES)}, "
            f"but got {MLFLOW_RUN_STALE_POLICY.get()!r}."
        )

    legacy_s = MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.get()
    if not isinstance(legacy_s, int) or isinstance(legacy_s, bool) or legacy_s < 0:
        raise ValueError(
            f"{MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name} must be a non-negative integer "
            f"(in seconds), 0 disables detection of heartbeat-less runs, but got {legacy_s!r}."
        )
    if 0 < legacy_s < stale_timeout_s:
        raise ValueError(
            f"{MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name} ({legacy_s}s) must be 0 (disabled) "
            f"or at least {MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name} ({stale_timeout_s}s); the "
            "fallback window for runs without heartbeats is expected to be wider."
        )

    return RunLivenessConfig(
        heartbeat_enabled=heartbeat_enabled,
        heartbeat_interval_ms=interval_s * MILLIS_PER_SECOND,
        stale_timeout_ms=stale_timeout_s * MILLIS_PER_SECOND,
        stale_policy=policy,
        legacy_timeout_ms=None if legacy_s == 0 else legacy_s * MILLIS_PER_SECOND,
    )


def parse_heartbeat_time(tags: dict[str, str]) -> int | None:
    """Return the last heartbeat timestamp recorded on the run, or ``None`` if absent/malformed."""
    raw = tags.get(MLFLOW_LAST_HEARTBEAT_TIME)
    if raw is None:
        return None
    try:
        heartbeat_ms = int(raw)
    except (TypeError, ValueError):
        _logger.debug("Ignoring malformed heartbeat tag value %r", raw, exc_info=True)
        return None
    return heartbeat_ms if heartbeat_ms > 0 else None


def judge_run_liveness(
    run: "Run",
    config: RunLivenessConfig,
    now_ms: int | None = None,
) -> LivenessVerdict:
    """Decide what should happen to ``run`` given its heartbeat freshness.

    The function is pure (apart from reading the run) so the same decision is shared by the
    file and SQL stores and can be unit tested without a backend.
    """
    info = run.info
    if info.lifecycle_stage != LifecycleStage.ACTIVE:
        return LivenessVerdict(LivenessAction.SKIP)
    # Only RUNNING runs are considered. SCHEDULED runs keep their existing semantics, and
    # terminated statuses (FINISHED/FAILED/KILLED) obviously need no liveness handling.
    if info.status != "RUNNING":
        return LivenessVerdict(LivenessAction.SKIP)

    now_ms = get_current_time_millis() if now_ms is None else now_ms
    tags = run.data.tags if run.data else {}
    heartbeat_ms = parse_heartbeat_time(tags)

    if heartbeat_ms is None:
        # Runs without any heartbeat come from old clients or are freshly created. Only judge
        # them when the operator explicitly enabled a (wider) legacy window. Using the start
        # time as the reference also guarantees a run that has not written its first heartbeat
        # yet is never misjudged.
        if config.legacy_timeout_ms is None:
            return LivenessVerdict(LivenessAction.SKIP)
        if now_ms - info.start_time < config.legacy_timeout_ms:
            return LivenessVerdict(LivenessAction.SKIP)
        return _stale_verdict(config, end_time_ms=now_ms)

    if now_ms - heartbeat_ms <= config.stale_timeout_ms:
        return LivenessVerdict(LivenessAction.HEALTHY)

    # The run is stale. In "fail" mode, the last heartbeat time is the best-known liveness time
    # and is used as the end time.
    return _stale_verdict(config, end_time_ms=heartbeat_ms)


def _stale_verdict(
    config: RunLivenessConfig, end_time_ms: int | None
) -> LivenessVerdict:
    if config.stale_policy == STALE_POLICY_FAIL:
        return LivenessVerdict(LivenessAction.FAIL_STALE, end_time_ms=end_time_ms)
    return LivenessVerdict(LivenessAction.MARK_STALE)


def hydrate_run_liveness(run: "Run") -> "Run":
    """Populate ``RunInfo`` liveness fields from the run's persisted tags.

    This runs on read for every store, including the REST store where tags are the only
    transport for liveness metadata. It never raises: malformed data simply degrades to
    "no heartbeat".
    """
    try:
        tags = run.data.tags if run.data else {}
        heartbeat_ms = parse_heartbeat_time(tags)
        suspected_zombie = tags.get(MLFLOW_RUN_SUSPECTED_ZOMBIE) is not None
        run.info._set_liveness(heartbeat_ms, suspected_zombie)
    except Exception:
        _logger.debug("Failed to hydrate run liveness for run", exc_info=True)
    return run
