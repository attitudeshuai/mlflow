"""Configuration for webhook delivery recording and automatic redelivery.

All settings are read from environment variables (see
``mlflow.environment_variables``). Invalid values raise ``ValueError`` while
this module is imported / when :func:`get_webhook_delivery_config` is called,
i.e. at configuration parsing time rather than at first use.
"""

from dataclasses import dataclass

from mlflow.environment_variables import (
    MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED,
    MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED,
    MLFLOW_WEBHOOK_DELIVERY_RECORD_RETENTION,
    MLFLOW_WEBHOOK_REDELIVERY_BACKOFF,
    MLFLOW_WEBHOOK_REDELIVERY_BATCH_SIZE,
    MLFLOW_WEBHOOK_REDELIVERY_INTERVAL_SECONDS,
    MLFLOW_WEBHOOK_REDELIVERY_MAX_ATTEMPTS,
    MLFLOW_WEBHOOK_REDELIVERY_POLL_INTERVAL_SECONDS,
)

#: Backoff strategy name for a fixed interval between redeliveries.
FIXED = "fixed"

#: Backoff strategy name for exponential (``interval * 2 ** count``) intervals.
EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class WebhookDeliveryConfig:
    """Parsed webhook delivery recording/redelivery configuration."""

    record_enabled: bool
    record_retention: int
    automatic_redelivery_enabled: bool
    redelivery_max_attempts: int
    redelivery_interval_seconds: int
    redelivery_backoff: str
    redelivery_poll_interval_seconds: int
    redelivery_batch_size: int

    @property
    def automatic_redelivery_active(self) -> bool:
        """Whether the automatic redelivery scheduler should run."""
        return self.record_enabled and self.automatic_redelivery_enabled


def _load_webhook_delivery_config() -> WebhookDeliveryConfig:
    config = WebhookDeliveryConfig(
        record_enabled=MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED.get(),
        record_retention=MLFLOW_WEBHOOK_DELIVERY_RECORD_RETENTION.get(),
        automatic_redelivery_enabled=MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED.get(),
        redelivery_max_attempts=MLFLOW_WEBHOOK_REDELIVERY_MAX_ATTEMPTS.get(),
        redelivery_interval_seconds=MLFLOW_WEBHOOK_REDELIVERY_INTERVAL_SECONDS.get(),
        redelivery_backoff=MLFLOW_WEBHOOK_REDELIVERY_BACKOFF.get(),
        redelivery_poll_interval_seconds=(
            MLFLOW_WEBHOOK_REDELIVERY_POLL_INTERVAL_SECONDS.get()
        ),
        redelivery_batch_size=MLFLOW_WEBHOOK_REDELIVERY_BATCH_SIZE.get(),
    )

    if config.automatic_redelivery_enabled and not config.record_enabled:
        raise ValueError(
            "MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED=true requires "
            "MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED=true: automatic redelivery "
            "can only run when delivery records are persisted"
        )

    return config


# Validate eagerly so that a server process fails fast at import time instead
# of discovering the misconfiguration on the first webhook event.
_WEBHOOK_DELIVERY_CONFIG = _load_webhook_delivery_config()


def get_webhook_delivery_config() -> WebhookDeliveryConfig:
    """Read and validate the current webhook delivery configuration.

    Environment variables are re-read on every call so that tests and embedded
    scenarios can override them, but invalid values always raise.
    """
    return _load_webhook_delivery_config()


def compute_next_retry_at(
    now_millis: int,
    redelivery_count: int,
    interval_seconds: int,
    backoff: str,
) -> int:
    """Compute the next automatic-redelivery timestamp (milliseconds).

    ``fixed`` repeats the base interval; ``exponential`` multiplies it by
    ``2 ** redelivery_count``.
    """
    if backoff == FIXED:
        delay_seconds = interval_seconds
    else:
        delay_seconds = interval_seconds * (2**redelivery_count)
    return now_millis + delay_seconds * 1000
