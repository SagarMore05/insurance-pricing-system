"""Allow engine_cc = 0 to support Electric Vehicle customers.

Revision ID: 005
Revises: 004
Create Date: 2026-06-24

P0-003 Fix: The vehicles.engine_cc column had a CHECK constraint engine_cc > 0,
which prevented Electric Vehicle records from being stored (EVs have cc = 0).
This migration relaxes the constraint to engine_cc >= 0.
"""
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_vehicles_engine_cc", "vehicles", type_="check")
    op.create_check_constraint(
        "ck_vehicles_engine_cc",
        "vehicles",
        "engine_cc >= 0",
    )


def downgrade() -> None:
    op.drop_constraint("ck_vehicles_engine_cc", "vehicles", type_="check")
    op.create_check_constraint(
        "ck_vehicles_engine_cc",
        "vehicles",
        "engine_cc > 0",
    )
