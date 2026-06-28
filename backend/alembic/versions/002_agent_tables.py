"""Agent tables for Analytics and Underwriting agents

Revision ID: 002
Revises: 001
Create Date: 2026-06-10 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("agent_type", sa.String(50), nullable=False),
        sa.Column("intent", sa.String(50), nullable=True),
        sa.Column("input_text", sa.String(), nullable=False),
        sa.Column("output_text", sa.String(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_agent_runs_status",
        ),
        sa.CheckConstraint(
            "agent_type IN ('analytics', 'underwriting')",
            name="ck_agent_runs_type",
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_agent_runs_session_id", "agent_runs", ["session_id"])
    op.create_index("ix_agent_runs_agent_type", "agent_runs", ["agent_type"])
    op.create_index("ix_agent_runs_started_at", "agent_runs", ["started_at"])

    op.create_table(
        "agent_tool_calls",
        sa.Column("call_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("tool_input", postgresql.JSONB(), nullable=True),
        sa.Column("tool_output", sa.String(), nullable=True),
        sa.Column("called_at", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.run_id"]),
        sa.PrimaryKeyConstraint("call_id"),
    )
    op.create_index("ix_agent_tool_calls_run_id", "agent_tool_calls", ["run_id"])
    op.create_index("ix_agent_tool_calls_tool_name", "agent_tool_calls", ["tool_name"])


def downgrade() -> None:
    op.drop_table("agent_tool_calls")
    op.drop_table("agent_runs")
