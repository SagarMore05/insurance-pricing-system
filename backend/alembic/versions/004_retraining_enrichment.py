"""Add prediction_id to agent_runs and HITL enrichment columns to model_feedback.

Revision ID: 004
Revises: 003
Create Date: 2026-06-15

Phase 1:  agent_runs.prediction_id  — links each underwriting run to the
          Prediction record it produced, enabling the retraining pipeline to
          join HITL review outcomes back to labelled training samples.

Phase 2:  model_feedback HITL columns — store human reviewer decision,
          agent decision, and fraud signal counts alongside each feedback
          record for governance metric computation.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── agent_runs: link to the Prediction produced by the underwriting run ──
    op.add_column(
        "agent_runs",
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("predictions.prediction_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_agent_runs_prediction_id", "agent_runs", ["prediction_id"])

    # ── model_feedback: HITL enrichment columns ──────────────────────────────
    op.add_column(
        "model_feedback",
        sa.Column(
            "agent_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_runs.run_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "model_feedback",
        sa.Column("human_decision", sa.String(20), nullable=True),  # 'approved' | 'rejected'
    )
    op.add_column(
        "model_feedback",
        sa.Column("agent_decision", sa.String(20), nullable=True),  # 'approve' | 'flag' | 'refer'
    )
    op.add_column(
        "model_feedback",
        sa.Column("fraud_signal_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "model_feedback",
        sa.Column("high_severity_count", sa.Integer(), nullable=True),
    )
    op.create_index("ix_model_feedback_agent_run_id", "model_feedback", ["agent_run_id"])
    op.create_index("ix_model_feedback_human_decision", "model_feedback", ["human_decision"])


def downgrade() -> None:
    op.drop_index("ix_model_feedback_human_decision", table_name="model_feedback")
    op.drop_index("ix_model_feedback_agent_run_id", table_name="model_feedback")
    op.drop_column("model_feedback", "high_severity_count")
    op.drop_column("model_feedback", "fraud_signal_count")
    op.drop_column("model_feedback", "agent_decision")
    op.drop_column("model_feedback", "human_decision")
    op.drop_column("model_feedback", "agent_run_id")
    op.drop_index("ix_agent_runs_prediction_id", table_name="agent_runs")
    op.drop_column("agent_runs", "prediction_id")
