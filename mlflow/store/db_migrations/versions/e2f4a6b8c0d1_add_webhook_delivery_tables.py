"""add webhook_deliveries and webhook_delivery_attempts tables

Create Date: 2026-10-14 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.model_registry.dbmodels.models import (
    SqlWebhookDelivery,
    SqlWebhookDeliveryAttempt,
)

# revision identifiers, used by Alembic.
revision = "e2f4a6b8c0d1"
down_revision = "c3d6457b6d8a"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlWebhookDelivery.__tablename__,
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column("delivery_id", sa.String(length=36), nullable=False),
        sa.Column("webhook_id", sa.String(length=256), nullable=False),
        sa.Column("webhook_name", sa.String(length=256), nullable=True),
        sa.Column("webhook_url", sa.String(length=500), nullable=True),
        sa.Column("entity", sa.String(length=50), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("payload_summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("trigger", sa.String(length=20), nullable=False),
        sa.Column("root_delivery_id", sa.String(length=36), nullable=False),
        sa.Column("parent_delivery_id", sa.String(length=36), nullable=True),
        sa.Column("redelivery_count", sa.Integer(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("completed_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("next_retry_at", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("delivery_id", name="webhook_delivery_pk"),
    )

    op.create_index(
        "idx_webhook_deliveries_webhook_id",
        SqlWebhookDelivery.__tablename__,
        ["webhook_id"],
    )
    op.create_index(
        "idx_webhook_deliveries_event",
        SqlWebhookDelivery.__tablename__,
        ["entity", "action"],
    )
    op.create_index(
        "idx_webhook_deliveries_status",
        SqlWebhookDelivery.__tablename__,
        ["status"],
    )
    op.create_index(
        "idx_webhook_deliveries_creation_ts",
        SqlWebhookDelivery.__tablename__,
        ["creation_timestamp"],
    )
    op.create_index(
        "idx_webhook_deliveries_root",
        SqlWebhookDelivery.__tablename__,
        ["root_delivery_id"],
    )
    op.create_index(
        "idx_webhook_deliveries_workspace",
        SqlWebhookDelivery.__tablename__,
        ["workspace"],
    )
    op.create_index(
        "idx_webhook_deliveries_next_retry_at",
        SqlWebhookDelivery.__tablename__,
        ["next_retry_at"],
    )

    op.create_table(
        SqlWebhookDeliveryAttempt.__tablename__,
        sa.Column("delivery_id", sa.String(length=36), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("started_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("completed_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("result", sa.String(length=20), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("error_type", sa.String(length=200), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["delivery_id"],
            [f"{SqlWebhookDelivery.__tablename__}.delivery_id"],
            ondelete="cascade",
        ),
        sa.PrimaryKeyConstraint(
            "delivery_id", "attempt_number", name="webhook_delivery_attempt_pk"
        ),
    )


def downgrade():
    op.drop_table(SqlWebhookDeliveryAttempt.__tablename__)

    for index_name in (
        "idx_webhook_deliveries_next_retry_at",
        "idx_webhook_deliveries_workspace",
        "idx_webhook_deliveries_root",
        "idx_webhook_deliveries_creation_ts",
        "idx_webhook_deliveries_status",
        "idx_webhook_deliveries_event",
        "idx_webhook_deliveries_webhook_id",
    ):
        op.drop_index(index_name, table_name=SqlWebhookDelivery.__tablename__)

    op.drop_table(SqlWebhookDelivery.__tablename__)
