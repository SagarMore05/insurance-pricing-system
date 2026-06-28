"""Add approval_requests and approval_audit_logs tables for human-in-the-loop model governance.

Revision ID: 007
Revises: 006
Create Date: 2026-06-25

Phase 4B — Human Approval Workflow.
Adds enterprise-grade approval tables for gating model promotion decisions.
All approval actions are recorded in an immutable audit log.
Does NOT touch champion registry, pricing engine, or scheduler.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── approval_requests ─────────────────────────────────────────────────────
    op.create_table(
        "approval_requests",
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("model_type", sa.String(20), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("submitted_by", sa.String(100), nullable=False),
        sa.Column(
            "submitted_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("reviewed_by", sa.String(100), nullable=True),
        sa.Column("reviewed_at", sa.DateTime, nullable=True),
        sa.Column("reviewer_note", sa.Text, nullable=True),
        sa.Column(
            "model_card",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("recommendation", sa.String(50), nullable=True),
        sa.CheckConstraint(
            "model_type IN ('frequency', 'severity')",
            name="ck_approval_model_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_approval_status",
        ),
    )
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])
    op.create_index(
        "ix_approval_requests_model_type", "approval_requests", ["model_type"]
    )
    op.create_index(
        "ix_approval_requests_submitted_at", "approval_requests", ["submitted_at"]
    )

    # ── approval_audit_logs ───────────────────────────────────────────────────
    op.create_table(
        "approval_audit_logs",
        sa.Column(
            "log_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("approval_requests.request_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("actor", sa.String(100), nullable=True),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column(
            "event_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_approval_audit_logs_request_id", "approval_audit_logs", ["request_id"]
    )
    op.create_index(
        "ix_approval_audit_logs_created_at", "approval_audit_logs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_approval_audit_logs_created_at", table_name="approval_audit_logs")
    op.drop_index("ix_approval_audit_logs_request_id", table_name="approval_audit_logs")
    op.drop_table("approval_audit_logs")

    op.drop_index("ix_approval_requests_submitted_at", table_name="approval_requests")
    op.drop_index("ix_approval_requests_model_type", table_name="approval_requests")
    op.drop_index("ix_approval_requests_status", table_name="approval_requests")
    op.drop_table("approval_requests")
