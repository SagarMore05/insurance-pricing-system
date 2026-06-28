"""Add V4 feature columns for full persistence and V5 retrain support.

Revision ID: 006
Revises: 005
Create Date: 2026-06-24

P1-005 Fix: The database only stored 12 of the 31 raw V4 features needed
by InsurancePreprocessorV4. Missing features meant:
  - 19 V4 inputs were computed at inference time and then discarded
  - The retraining pipeline could only reconstruct V1 features from the DB
  - V5 online retraining from production data was structurally impossible

This migration adds nullable columns to customers (3), vehicles (7),
driving_profiles (9), and predictions (1 JSONB) — 20 columns in total.

All new columns are nullable so existing V1-originated rows remain valid.
The V1 quote route is unaffected. The V2 quote route persistence code is
updated separately to write these columns on each new quote.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── customers: V4 personal attributes ────────────────────────────────────
    op.add_column("customers", sa.Column("occupation", sa.String(30), nullable=True))
    op.add_column("customers", sa.Column("marital_status", sa.String(20), nullable=True))
    op.add_column("customers", sa.Column("annual_income_band", sa.String(20), nullable=True))
    op.create_index("ix_customers_occupation", "customers", ["occupation"])

    # ── vehicles: V4 vehicle attributes + lookup-derived specs ───────────────
    op.add_column("vehicles", sa.Column("fuel_type", sa.String(20), nullable=True))
    op.add_column("vehicles", sa.Column("vehicle_usage_type", sa.String(20), nullable=True))
    op.add_column("vehicles", sa.Column("parking_type", sa.String(20), nullable=True))
    op.add_column("vehicles", sa.Column("vehicle_safety_rating", sa.SmallInteger(), nullable=True))
    op.add_column("vehicles", sa.Column("airbags_count", sa.SmallInteger(), nullable=True))
    op.add_column("vehicles", sa.Column("vehicle_body_style", sa.String(20), nullable=True))
    op.add_column("vehicles", sa.Column("repair_cost_band", sa.String(20), nullable=True))
    op.create_index("ix_vehicles_fuel_type", "vehicles", ["fuel_type"])

    # ── driving_profiles: V4 driving history + geo-risk + derived features ──
    op.add_column(
        "driving_profiles",
        sa.Column("months_since_last_claim", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("policy_tenure_years", sa.Numeric(5, 2), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("no_claim_bonus_pct", sa.Numeric(4, 1), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("region_risk_category", sa.String(10), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("flood_risk_index", sa.Numeric(4, 2), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("theft_risk_index", sa.Numeric(4, 2), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("monsoon_exposure_index", sa.Numeric(4, 2), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("pincode_risk_score", sa.Numeric(6, 4), nullable=True),
    )
    op.add_column(
        "driving_profiles",
        sa.Column("policy_inception_month", sa.SmallInteger(), nullable=True),
    )

    # ── predictions: point-in-time snapshot of all 31 inference features ─────
    op.add_column(
        "predictions",
        sa.Column(
            "inference_features_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # predictions
    op.drop_column("predictions", "inference_features_json")

    # driving_profiles
    op.drop_column("driving_profiles", "policy_inception_month")
    op.drop_column("driving_profiles", "pincode_risk_score")
    op.drop_column("driving_profiles", "monsoon_exposure_index")
    op.drop_column("driving_profiles", "theft_risk_index")
    op.drop_column("driving_profiles", "flood_risk_index")
    op.drop_column("driving_profiles", "region_risk_category")
    op.drop_column("driving_profiles", "no_claim_bonus_pct")
    op.drop_column("driving_profiles", "policy_tenure_years")
    op.drop_column("driving_profiles", "months_since_last_claim")

    # vehicles
    op.drop_index("ix_vehicles_fuel_type", table_name="vehicles")
    op.drop_column("vehicles", "repair_cost_band")
    op.drop_column("vehicles", "vehicle_body_style")
    op.drop_column("vehicles", "airbags_count")
    op.drop_column("vehicles", "vehicle_safety_rating")
    op.drop_column("vehicles", "parking_type")
    op.drop_column("vehicles", "vehicle_usage_type")
    op.drop_column("vehicles", "fuel_type")

    # customers
    op.drop_index("ix_customers_occupation", table_name="customers")
    op.drop_column("customers", "annual_income_band")
    op.drop_column("customers", "marital_status")
    op.drop_column("customers", "occupation")
