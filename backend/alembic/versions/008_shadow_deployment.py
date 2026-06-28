"""Add shadow_predictions table for Phase 4C Shadow Deployment Framework.

Revision ID: 008
Revises: 007
Create Date: 2026-06-25

Phase 4C — Shadow Deployment Framework.
Records every V2 champion prediction for future challenger comparison.
Status is WAITING_FOR_CHALLENGER until a V5 challenger model is registered.

Read-only with respect to production models — champion registry NOT modified.
ENABLE_SCHEDULED_RETRAINING remains False.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("request_id", sa.String(100), nullable=False, unique=True),
        sa.Column(
            "quote_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        # Champion identification
        sa.Column("champion_model_name", sa.String(100), nullable=False),
        sa.Column("champion_version", sa.String(20), nullable=False),
        # Challenger identification (null until a challenger is registered)
        sa.Column("challenger_model_name", sa.String(100), nullable=True),
        sa.Column("challenger_version", sa.String(20), nullable=True),
        # Prediction payloads
        sa.Column(
            "champion_prediction_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "challenger_prediction_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        # Comparison metrics (null until challenger runs)
        sa.Column("premium_difference", sa.Numeric(14, 2), nullable=True),
        sa.Column("risk_difference", sa.String(20), nullable=True),
        sa.Column("prediction_latency_ms", sa.Integer, nullable=True),
        # Status and result
        sa.Column("status", sa.String(30), nullable=False, server_default="WAITING_FOR_CHALLENGER"),
        sa.Column("comparison_result", sa.String(30), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.CheckConstraint(
            "status IN ('WAITING_FOR_CHALLENGER', 'PENDING', 'COMPLETED', 'FAILED')",
            name="ck_shadow_status",
        ),
        sa.CheckConstraint(
            "comparison_result IS NULL OR comparison_result IN "
            "('MATCH', 'DIVERGENT', 'CHALLENGER_HIGHER', 'CHALLENGER_LOWER')",
            name="ck_shadow_comparison_result",
        ),
    )
    op.create_index("ix_shadow_predictions_created_at", "shadow_predictions", ["created_at"])
    op.create_index("ix_shadow_predictions_status", "shadow_predictions", ["status"])
    op.create_index("ix_shadow_predictions_quote_id", "shadow_predictions", ["quote_id"])
    op.create_index("ix_shadow_predictions_request_id", "shadow_predictions", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_shadow_predictions_request_id", table_name="shadow_predictions")
    op.drop_index("ix_shadow_predictions_quote_id", table_name="shadow_predictions")
    op.drop_index("ix_shadow_predictions_status", table_name="shadow_predictions")
    op.drop_index("ix_shadow_predictions_created_at", table_name="shadow_predictions")
    op.drop_table("shadow_predictions")
