"""Add training_runs and candidate_models tables for Phase 5A V5 Training Engine.

Revision ID: 009
Revises: 008
Create Date: 2026-06-25

Phase 5A — Enterprise V5 Training Engine.
Records every challenger training run and per-algorithm candidate models.

Safety constraints:
  - champion_registry NOT modified
  - ENABLE_SCHEDULED_RETRAINING remains False
  - No production inference path touched
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── training_runs — one row per orchestrated run ───────────────────────────
    op.create_table(
        "training_runs",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("run_tag", sa.String(100), nullable=False, unique=True),
        sa.Column("triggered_by", sa.String(100), nullable=False, server_default="admin"),
        sa.Column("triggered_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="RUNNING"),
        sa.Column("dataset_version", sa.String(50), nullable=True),
        sa.Column("dataset_rows", sa.Integer, nullable=True),
        sa.Column("feature_count", sa.Integer, nullable=True),
        sa.Column("maturity_days", sa.Integer, nullable=False, server_default="90"),
        sa.Column("train_rows", sa.Integer, nullable=True),
        sa.Column("val_rows", sa.Integer, nullable=True),
        sa.Column("test_rows", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'VALIDATION_FAILED')",
            name="ck_training_runs_status",
        ),
    )
    op.create_index("ix_training_runs_triggered_at", "training_runs", ["triggered_at"])
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_index("ix_training_runs_run_tag", "training_runs", ["run_tag"])

    # ── candidate_models — one row per (algorithm × model_type) per run ────────
    op.create_table(
        "candidate_models",
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("training_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("model_type", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="TRAINING"),
        sa.Column("started_at", sa.DateTime, nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("duration_seconds", sa.Numeric(10, 2), nullable=True),
        sa.Column("train_rows", sa.Integer, nullable=True),
        sa.Column("val_rows", sa.Integer, nullable=True),
        sa.Column("test_rows", sa.Integer, nullable=True),
        sa.Column(
            "hyperparameters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "frequency_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "severity_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "feature_importance",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.CheckConstraint(
            "algorithm IN ('xgboost', 'catboost', 'lightgbm')",
            name="ck_candidate_algorithm",
        ),
        sa.CheckConstraint(
            "model_type IN ('frequency', 'severity')",
            name="ck_candidate_model_type",
        ),
        sa.CheckConstraint(
            "status IN ('TRAINING', 'COMPLETED', 'FAILED')",
            name="ck_candidate_status",
        ),
    )
    op.create_index("ix_candidate_models_run_id", "candidate_models", ["run_id"])
    op.create_index("ix_candidate_models_algorithm", "candidate_models", ["algorithm"])
    op.create_index("ix_candidate_models_status", "candidate_models", ["status"])
    op.create_index("ix_candidate_models_model_type", "candidate_models", ["model_type"])


def downgrade() -> None:
    op.drop_index("ix_candidate_models_model_type", table_name="candidate_models")
    op.drop_index("ix_candidate_models_status", table_name="candidate_models")
    op.drop_index("ix_candidate_models_algorithm", table_name="candidate_models")
    op.drop_index("ix_candidate_models_run_id", table_name="candidate_models")
    op.drop_table("candidate_models")

    op.drop_index("ix_training_runs_run_tag", table_name="training_runs")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_index("ix_training_runs_triggered_at", table_name="training_runs")
    op.drop_table("training_runs")
