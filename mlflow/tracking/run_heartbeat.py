"""Background thread that records run heartbeats while an MLflow run is active.

The heartbeat write path is intentionally independent of metric/parameter logging (its own
store call / HTTP request) so it cannot block or be blocked by training-side logging. Every
error on the heartbeat path is swallowed and logged: a failed heartbeat only degrades liveness
protection, it must never interrupt the training main flow. When the process is killed, the
daemon thread dies with it and heartbeats stop, which is exactly what stale-run detection on
the store side relies on.
"""

from __future__ import annotations

import logging
import threading

from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)


class RunHeartbeat:
    """Periodically record heartbeats for a run on a daemon thread.

    Args:
        run_id: ID of the run to heartbeat.
        interval_seconds: Heartbeat interval in seconds.
        tracking_uri: Tracking URI to use. ``None`` resolves the currently configured store.
    """

    def __init__(self, run_id: str, interval_seconds: float, tracking_uri: str | None = None):
        self._run_id = run_id
        if not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0:
            raise ValueError(
                f"Heartbeat interval must be a positive number of seconds, got "
                f"{interval_seconds!r}."
            )
        self._interval_seconds = float(interval_seconds)
        self._tracking_uri = tracking_uri
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the daemon heartbeat thread; the first heartbeat is written from the thread."""
        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True,
            name=f"RunHeartbeat-{self._run_id[:8]}",
        )
        self._thread.start()

    def _monitor(self) -> None:
        # Record the first heartbeat right away so a run that dies shortly after creation leaves
        # a liveness timestamp; runs without any heartbeat are never judged as stale (unless the
        # operator enables the wider legacy window). The write happens off the caller's thread
        # so run startup is never blocked by heartbeat I/O.
        self._beat()
        while not self._shutdown_event.wait(self._interval_seconds):
            self._beat()

    def _beat(self) -> None:
        try:
            from mlflow.tracking.client import MlflowClient

            MlflowClient(self._tracking_uri).record_run_heartbeat(
                self._run_id, get_current_time_millis()
            )
        except Exception:
            # Heartbeat failures are non-fatal by contract: only log, never propagate into the
            # training main flow.
            _logger.debug(
                "Failed to record heartbeat for run %s", self._run_id, exc_info=True
            )

    def finish(self) -> None:
        """Stop the heartbeat thread. Safe to call multiple times."""
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 5)
            self._thread = None
