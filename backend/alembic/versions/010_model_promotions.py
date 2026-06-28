"""Phase 5B — Champion Promotion Engine tables.

model_promotions  : one row per promotion evaluation / attempt
rollback_history  : one row per rollback event (auto or manual)

Revision ID: 010
Revises: 009
Create Date: 2026-06-25

Safety constraints:
  - champion_registry.json NOT automatically modified by migration
  - No existing tables altered
  - Fully reversible (downgrade drops both tables)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── model_promotions ──────────────────────────────────────────────────────
    op.create_table(
        "model_promotions",
        sa.Column(
            "promotion_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Training run that produced the challengers
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("training_runs.run_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        # Specific candidate models being promoted (nullable if same pair)
        sa.Column("frequency_candidate_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("severity_candidate_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Human approval references (approval_requests.request_id)
        sa.Column("frequency_approval_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("severity_approval_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Snapshot of registry state before promotion
        sa.Column("old_frequency_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("old_severity_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # New registry state after successful promotion
        sa.Column("new_frequency_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_severity_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Full evaluation detail
        sa.Column("evaluation_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gates_passed", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("gates_failed", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Lifecycle
        sa.Column("status", sa.String(30), nullable=False, server_default="PENDING"),
        sa.Column("promoted_by", sa.String(200), nullable=True),
        sa.Column("promoted_at", sa.DateTime, nullable=True),
        sa.Column("promotion_duration_seconds", sa.Numeric(10, 2), nullable=True),
        sa.Column("backup_path", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        # Audit timestamps
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime, nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING','EVALUATING','APPROVED','REJECTED','PROMOTING','ACTIVE','ROLLED_BACK','FAILED')",
            name="ck_model_promotions_status",
        ),
    )
    op.create_index("ix_model_promotions_run_id", "model_promotions", ["run_id"])
    op.create_index("ix_model_promotions_status", "model_promotions", ["status"])
    op.create_index("ix_model_promotions_created_at", "model_promotions", ["created_at"])

    # ── rollback_history ──────────────────────────────────────────────────────
    op.create_table(
        "rollback_history",
        sa.Column(
            "rollback_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "promotion_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_promotions.promotion_id", ondelete="CASCADE"),
            nullable=False,
        ),
        # What was restored
        sa.Column("restored_frequency_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("restored_severity_champion", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Why / how
        sa.Column("rollback_reason", sa.String(1000), nullable=True),
        sa.Column(
            "rollback_trigger",
            sa.String(50),
            nullable=False,
            server_default="manual",
        ),
        # Outcome
        sa.Column("rollback_status", sa.String(20), nullable=False, server_default="SUCCESS"),
        sa.Column("rollback_duration_seconds", sa.Numeric(10, 2), nullable=True),
        sa.Column("rolled_back_by", sa.String(200), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.CheckConstraint(
            "rollback_trigger IN ('health_check','load_failure','prediction_failure','registry_failure','manual','auto')",
            name="ck_rollback_trigger",
        ),
        sa.CheckConstraint(
            "rollback_status IN ('SUCCESS','FAILED','PARTIAL')",
            name="ck_rollback_status",
        ),
    )
    op.create_index("ix_rollback_history_promotion_id", "rollback_history", ["promotion_id"])
    op.create_index("ix_rollback_history_rolled_back_at", "rollback_history", ["rolled_back_at"])


def downgrade() -> None:
    op.drop_index("ix_rollback_history_rolled_back_at", table_name="rollback_history")
    op.drop_index("ix_rollback_history_promotion_id", table_name="rollback_history")
    op.drop_table("rollback_history")

    op.drop_index("ix_model_promotions_created_at", table_name="model_promotions")
    op.drop_index("ix_model_promotions_status", table_name="model_promotions")
    op.drop_index("ix_model_promotions_run_id", table_name="model_promotions")
    op.drop_table("model_promotions")
