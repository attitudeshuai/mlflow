import pytest

from mlflow.webhooks.config import get_webhook_delivery_config

POSITIVE_INT_ENV_VARS = [
    "MLFLOW_WEBHOOK_DELIVERY_RECORD_RETENTION",
    "MLFLOW_WEBHOOK_REDELIVERY_MAX_ATTEMPTS",
    "MLFLOW_WEBHOOK_REDELIVERY_INTERVAL_SECONDS",
    "MLFLOW_WEBHOOK_REDELIVERY_POLL_INTERVAL_SECONDS",
    "MLFLOW_WEBHOOK_REDELIVERY_BATCH_SIZE",
]


@pytest.mark.parametrize("env_var", POSITIVE_INT_ENV_VARS)
@pytest.mark.parametrize("bad_value", ["0", "-1", "abc"])
def test_invalid_positive_int_config_fails_at_parse_time(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValueError, match=env_var):
        get_webhook_delivery_config()


def test_invalid_backoff_value_fails_at_parse_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_WEBHOOK_REDELIVERY_BACKOFF", "linear")

    with pytest.raises(ValueError, match="MLFLOW_WEBHOOK_REDELIVERY_BACKOFF"):
        get_webhook_delivery_config()


@pytest.mark.parametrize("backoff", ["fixed", "exponential", "FIXED", " Exponential "])
def test_valid_backoff_values_accepted(
    monkeypatch: pytest.MonkeyPatch, backoff: str
) -> None:
    monkeypatch.setenv("MLFLOW_WEBHOOK_REDELIVERY_BACKOFF", backoff)

    config = get_webhook_delivery_config()
    assert config.redelivery_backoff == backoff.strip().lower()


def test_automatic_redelivery_without_recording_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED", "false")

    with pytest.raises(ValueError, match="MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED"):
        get_webhook_delivery_config()


def test_automatic_redelivery_with_recording_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED", "true")

    config = get_webhook_delivery_config()
    assert config.record_enabled is True
    assert config.automatic_redelivery_enabled is True
    assert config.automatic_redelivery_active is True


def test_default_config_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in [
        "MLFLOW_WEBHOOK_DELIVERY_RECORD_ENABLED",
        "MLFLOW_WEBHOOK_DELIVERY_RECORD_RETENTION",
        "MLFLOW_WEBHOOK_AUTOMATIC_REDELIVERY_ENABLED",
        "MLFLOW_WEBHOOK_REDELIVERY_MAX_ATTEMPTS",
        "MLFLOW_WEBHOOK_REDELIVERY_INTERVAL_SECONDS",
        "MLFLOW_WEBHOOK_REDELIVERY_BACKOFF",
        "MLFLOW_WEBHOOK_REDELIVERY_POLL_INTERVAL_SECONDS",
        "MLFLOW_WEBHOOK_REDELIVERY_BATCH_SIZE",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    config = get_webhook_delivery_config()
    assert config.record_enabled is False
    assert config.record_retention == 100000
    assert config.automatic_redelivery_enabled is False
    assert config.automatic_redelivery_active is False
    assert config.redelivery_max_attempts == 5
    assert config.redelivery_interval_seconds == 300
    assert config.redelivery_backoff == "exponential"
    assert config.redelivery_poll_interval_seconds == 60
    assert config.redelivery_batch_size == 100
