"""Add HITL columns to agent_runs

Revision ID: 003
Revises: 002
Create Date: 2026-06-14 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extend status CHECK constraint to include 'awaiting_review'
    op.drop_constraint("ck_agent_runs_status", "agent_runs", type_="check")
    op.create_check_constraint(
        "ck_agent_runs_status",
        "agent_runs",
        "status IN ('running', 'awaiting_review', 'completed', 'failed')",
    )

    # Human-in-the-loop review columns
    op.add_column("agent_runs", sa.Column("review_outcome", sa.String(20), nullable=True))
    op.add_column("agent_runs", sa.Column("reviewed_at", sa.DateTime(), nullable=True))
    op.add_column("agent_runs", sa.Column("reviewer_note", sa.Text(), nullable=True))
    op.add_column("agent_runs", sa.Column("result_json", postgresql.JSONB(), nullable=True))

    # Index for queue queries
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_status", "agent_runs")
    op.drop_column("agent_runs", "result_json")
    op.drop_column("agent_runs", "reviewer_note")
    op.drop_column("agent_runs", "reviewed_at")
    op.drop_column("agent_runs", "review_outcome")

    op.drop_constraint("ck_agent_runs_status", "agent_runs", type_="check")
    op.create_check_constraint(
        "ck_agent_runs_status",
        "agent_runs",
        "status IN ('running', 'completed', 'failed')",
    )
