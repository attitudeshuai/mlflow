from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.script import ScriptDirectory

from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.model_registry.dbmodels.models import (
    SqlWebhookDelivery,
    SqlWebhookDeliveryAttempt,
)
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase

REVISION = "e2f4a6b8c0d1"
PREVIOUS_REVISION = "c3d6457b6d8a"

DELIVERY_COLUMNS = {
    "workspace",
    "delivery_id",
    "webhook_id",
    "webhook_name",
    "webhook_url",
    "entity",
    "action",
    "payload_summary",
    "status",
    "trigger",
    "root_delivery_id",
    "parent_delivery_id",
    "redelivery_count",
    "failure_reason",
    "creation_timestamp",
    "completed_timestamp",
    "next_retry_at",
}

ATTEMPT_COLUMNS = {
    "delivery_id",
    "attempt_number",
    "started_timestamp",
    "completed_timestamp",
    "result",
    "http_status",
    "error_type",
    "error_message",
}

DELIVERY_INDEXES = {
    "idx_webhook_deliveries_webhook_id",
    "idx_webhook_deliveries_event",
    "idx_webhook_deliveries_status",
    "idx_webhook_deliveries_creation_ts",
    "idx_webhook_deliveries_root",
    "idx_webhook_deliveries_workspace",
    "idx_webhook_deliveries_next_retry_at",
}


def _prepare_database(tmp_path: Path):
    db_path = tmp_path / "webhook_delivery_migration.sqlite"
    db_url = f"sqlite:///{db_path}"
    engine = sa.create_engine(db_url)
    InitialBase.metadata.create_all(engine)
    config = _get_alembic_config(db_url)
    command.upgrade(config, PREVIOUS_REVISION)
    return engine, config, db_url


def test_migration_adds_delivery_tables(tmp_path: Path):
    engine, config, _ = _prepare_database(tmp_path)

    command.upgrade(config, REVISION)

    inspector = sa.inspect(engine)
    tables = set(inspector.get_table_names())
    assert SqlWebhookDelivery.__tablename__ in tables
    assert SqlWebhookDeliveryAttempt.__tablename__ in tables

    delivery_columns = {c["name"] for c in inspector.get_columns("webhook_deliveries")}
    assert DELIVERY_COLUMNS.issubset(delivery_columns)

    attempt_columns = {
        c["name"] for c in inspector.get_columns("webhook_delivery_attempts")
    }
    assert ATTEMPT_COLUMNS.issubset(attempt_columns)

    delivery_indexes = {index["name"] for index in inspector.get_indexes("webhook_deliveries")}
    assert DELIVERY_INDEXES.issubset(delivery_indexes)


def test_migration_downgrade_drops_tables_and_reupgrade(tmp_path: Path):
    engine, config, _ = _prepare_database(tmp_path)
    command.upgrade(config, REVISION)

    command.downgrade(config, PREVIOUS_REVISION)
    inspector = sa.inspect(engine)
    tables = set(inspector.get_table_names())
    assert SqlWebhookDelivery.__tablename__ not in tables
    assert SqlWebhookDeliveryAttempt.__tablename__ not in tables

    command.upgrade(config, REVISION)
    inspector = sa.inspect(engine)
    tables = set(inspector.get_table_names())
    assert SqlWebhookDelivery.__tablename__ in tables
    assert SqlWebhookDeliveryAttempt.__tablename__ in tables


def test_new_revision_extends_webhook_chain_head():
    # The repository has multiple pre-existing migration heads; assert that the
    # new revision is a head and linearly succeeds the webhook-chain head.
    import mlflow.store.db_migrations.versions.e2f4a6b8c0d1_add_webhook_delivery_tables as migration

    assert migration.revision == REVISION
    assert migration.down_revision == PREVIOUS_REVISION
    assert Path(migration.__file__).exists()

    alembic_config = _get_alembic_config("sqlite://")
    script_directory = ScriptDirectory.from_config(alembic_config)
    assert REVISION in script_directory.get_heads()
    revision = script_directory.get_revision(REVISION)
    assert revision.down_revision == PREVIOUS_REVISION
