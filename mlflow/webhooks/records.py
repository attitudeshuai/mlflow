"""Persisted delivery recording for webhook deliveries.

This module contains the pieces used only when delivery recording is enabled:

- :func:`summarize_payload` extracts an identifier-only JSON summary of an event
  payload (full event content is never persisted).
- :class:`RecordingRetry` hooks urllib3's per-attempt retry hook so that every
  HTTP attempt (including in-request retries) is observed.
- :class:`AttemptRecorder` persists attempt rows as they happen and marks the
  final delivery status. All store failures degrade to log lines: they never
  affect the actual HTTP delivery.
"""

import json
import logging
import time
from typing import Any, Callable

import requests
from urllib3.util.retry import Retry

from mlflow.entities.webhook_delivery import WebhookDeliveryAttemptResult

_logger = logging.getLogger(__name__)

#: Payload keys that identify the resource the event refers to. Only these
#: whitelisted keys are persisted as the delivery summary; no full payload
#: content (tags, descriptions, values, ...) is retained.
_PAYLOAD_SUMMARY_KEYS = (
    "name",
    "version",
    "key",
    "alias",
    "budget_policy_id",
    "target_scope",
    "target_value",
)


def summarize_payload(payload: Any) -> str:
    """Return a stable JSON string containing only identifier fields."""
    summary = {}
    if isinstance(payload, dict):
        summary = {key: payload[key] for key in _PAYLOAD_SUMMARY_KEYS if key in payload}
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def summary_to_payload(summary: str | None) -> dict[str, Any]:
    """Parse a persisted summary back into the redelivery request ``data``."""
    if not summary:
        return {}
    try:
        parsed = json.loads(summary)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class _AttemptRecorder:
    """Persists attempt rows for one delivery, tolerating store failures."""

    def __init__(self, store, delivery_id: str, workspace: str | None):
        self._store = store
        self._delivery_id = delivery_id
        self._workspace = workspace
        self._attempt_number = 0
        self._last_attempt_started = int(time.time() * 1000)
        # True when the last HTTP attempt has already been recorded through the
        # retry hook (i.e. urllib3 exhausted retries while raising/returning).
        self.terminal_attempt_recorded = False
        self.last_error_message: str | None = None

    def _persist_attempt(
        self,
        result: WebhookDeliveryAttemptResult,
        completed_at: int,
        http_status: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._attempt_number += 1
        started_at = self._last_attempt_started
        self._last_attempt_started = completed_at
        if error_message:
            self.last_error_message = error_message
        try:
            self._store.add_webhook_delivery_attempt(
                delivery_id=self._delivery_id,
                attempt_number=self._attempt_number,
                started_timestamp=started_at,
                completed_timestamp=completed_at,
                result=result,
                http_status=http_status,
                error_type=error_type,
                error_message=error_message,
                workspace=self._workspace,
            )
        except Exception:
            _logger.error(
                "Failed to persist webhook delivery attempt for delivery %s",
                self._delivery_id,
                exc_info=True,
            )

    def record_retry_hook(
        self, response: Any | None = None, error: Exception | None = None
    ) -> None:
        """Called from ``RecordingRetry.increment`` after each failed attempt."""
        completed_at = int(time.time() * 1000)
        self._record_failure(completed_at, response=response, error=error)

    def mark_terminal_recorded(self) -> None:
        """The just-recorded hook attempt was the final attempt."""
        self.terminal_attempt_recorded = True

    def _record_failure(
        self,
        completed_at: int,
        response: Any | None = None,
        error: Exception | None = None,
    ) -> None:
        if response is not None:
            self._persist_attempt(
                WebhookDeliveryAttemptResult.HTTP_ERROR,
                completed_at,
                http_status=getattr(response, "status", None),
                error_message=f"Received HTTP {getattr(response, 'status', '?')} response",
            )
        elif error is not None:
            self._persist_attempt(
                WebhookDeliveryAttemptResult.REQUEST_ERROR,
                completed_at,
                error_type=type(error).__name__,
                error_message=str(error) or repr(error),
            )

    def record_final_response(self, response: Any) -> None:
        """Record the final attempt from the caller-observed HTTP response."""
        if self.terminal_attempt_recorded:
            return
        completed_at = int(time.time() * 1000)
        status_code = response.status_code
        if status_code < 400:
            self._persist_attempt(
                WebhookDeliveryAttemptResult.SUCCESS,
                completed_at,
                http_status=status_code,
            )
        else:
            self._persist_attempt(
                WebhookDeliveryAttemptResult.HTTP_ERROR,
                completed_at,
                http_status=status_code,
                error_message=f"Received HTTP {status_code} response",
            )

    def record_final_exception(self, error: Exception) -> None:
        """Record the final attempt when the request raised, unless already done."""
        if self.terminal_attempt_recorded:
            return
        completed_at = int(time.time() * 1000)
        self._persist_attempt(
            WebhookDeliveryAttemptResult.REQUEST_ERROR,
            completed_at,
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
        )


class RecordingRetry(Retry):
    """A urllib3 ``Retry`` that observes every failed attempt via a callback.

    ``Retry`` clones itself through ``new(**self.__dict__)`` between attempts,
    so the callback is carried across clones explicitly.
    """

    def __init__(self, *args, attempt_callback: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._attempt_callback = attempt_callback

    def new(self, **kw):
        kw.setdefault("attempt_callback", self._attempt_callback)
        return super().new(**kw)

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
        callback = self._attempt_callback
        if callback is not None:
            try:
                callback(response=response, error=error)
            except Exception:
                _logger.error("Webhook delivery attempt callback failed", exc_info=True)

        try:
            return super().increment(
                method=method,
                url=url,
                response=response,
                error=error,
                _pool=_pool,
                _stacktrace=_stacktrace,
            )
        except Exception:
            # urllib3 exhausted retries while processing this attempt, so the
            # attempt just recorded by the hook is the terminal one.
            if callback is not None:
                try:
                    # Mark on the live recorder object (same callback across clones)
                    recorder = getattr(callback, "__self__", None)
                    mark = getattr(recorder, "mark_terminal_recorded", None)
                    if mark is not None:
                        mark()
                except Exception:
                    _logger.error("Failed to mark terminal delivery attempt", exc_info=True)
            raise
