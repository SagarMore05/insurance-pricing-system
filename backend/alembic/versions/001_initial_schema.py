"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("gender", sa.String(20), nullable=False),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("age >= 18 AND age <= 85", name="ck_customers_age"),
        sa.PrimaryKeyConstraint("customer_id"),
    )
    op.create_index("ix_customers_city", "customers", ["city"])
    op.create_index("ix_customers_created_at", "customers", ["created_at"])

    op.create_table(
        "vehicles",
        sa.Column("vehicle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("car_brand", sa.String(100), nullable=False),
        sa.Column("car_model", sa.String(100), nullable=False),
        sa.Column("engine_cc", sa.Integer(), nullable=False),
        sa.Column("vehicle_age_years", sa.Integer(), nullable=False),
        sa.Column("vehicle_value_inr", sa.Numeric(14, 2), nullable=False),
        sa.CheckConstraint("engine_cc > 0", name="ck_vehicles_engine_cc"),
        sa.CheckConstraint("vehicle_age_years >= 0", name="ck_vehicles_age"),
        sa.CheckConstraint("vehicle_value_inr > 0", name="ck_vehicles_value"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"]),
        sa.PrimaryKeyConstraint("vehicle_id"),
    )
    op.create_index("ix_vehicles_customer_id", "vehicles", ["customer_id"])

    op.create_table(
        "driving_profiles",
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("driving_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("annual_mileage_km", sa.Integer(), nullable=False),
        sa.Column("previous_claims_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("years_licensed", sa.Integer(), nullable=False),
        sa.CheckConstraint("driving_score >= 0 AND driving_score <= 100", name="ck_profile_driving_score"),
        sa.CheckConstraint("annual_mileage_km >= 0", name="ck_profile_mileage"),
        sa.CheckConstraint("previous_claims_count >= 0", name="ck_profile_claims"),
        sa.CheckConstraint("years_licensed >= 0", name="ck_profile_years_licensed"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"]),
        sa.PrimaryKeyConstraint("profile_id"),
    )
    op.create_index("ix_driving_profiles_customer_id", "driving_profiles", ["customer_id"])

    op.create_table(
        "policies",
        sa.Column("policy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vehicle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("premium_amount_inr", sa.Numeric(14, 2), nullable=False),
        sa.Column("risk_level", sa.String(10), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="ck_policies_risk_level"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"]),
        sa.ForeignKeyConstraint(["vehicle_id"], ["vehicles.vehicle_id"]),
        sa.PrimaryKeyConstraint("policy_id"),
    )
    op.create_index("ix_policies_customer_id", "policies", ["customer_id"])
    op.create_index("ix_policies_vehicle_id", "policies", ["vehicle_id"])
    op.create_index("ix_policies_created_at", "policies", ["created_at"])
    op.create_index("ix_policies_risk_level", "policies", ["risk_level"])

    op.create_table(
        "predictions",
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("policy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("expected_claim_amount_inr", sa.Numeric(14, 2), nullable=False),
        sa.Column("final_premium_inr", sa.Numeric(14, 2), nullable=False),
        sa.Column("explanation_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("claim_probability >= 0 AND claim_probability <= 1", name="ck_predictions_probability"),
        sa.ForeignKeyConstraint(["policy_id"], ["policies.policy_id"]),
        sa.PrimaryKeyConstraint("prediction_id"),
    )
    op.create_index("ix_predictions_policy_id", "predictions", ["policy_id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])

    op.create_table(
        "claims",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("policy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claimed_amount_inr", sa.Numeric(14, 2), nullable=False),
        sa.Column("approved_amount_inr", sa.Numeric(14, 2), nullable=True),
        sa.Column("claim_date", sa.Date(), nullable=False),
        sa.Column("claim_status", sa.String(20), nullable=False, server_default="'pending'"),
        sa.CheckConstraint("claim_status IN ('pending', 'approved', 'rejected')", name="ck_claims_status"),
        sa.CheckConstraint("claimed_amount_inr > 0", name="ck_claims_amount"),
        sa.ForeignKeyConstraint(["policy_id"], ["policies.policy_id"]),
        sa.PrimaryKeyConstraint("claim_id"),
    )
    op.create_index("ix_claims_policy_id", "claims", ["policy_id"])
    op.create_index("ix_claims_status", "claims", ["claim_status"])
    op.create_index("ix_claims_date", "claims", ["claim_date"])

    op.create_table(
        "model_feedback",
        sa.Column("feedback_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actual_claim_occurred", sa.Boolean(), nullable=False),
        sa.Column("actual_claim_amount_inr", sa.Numeric(14, 2), nullable=True),
        sa.Column("feedback_date", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["predictions.prediction_id"]),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index("ix_model_feedback_prediction_id", "model_feedback", ["prediction_id"])
    op.create_index("ix_model_feedback_date", "model_feedback", ["feedback_date"])

    op.create_table(
        "model_versions",
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_tag", sa.String(20), nullable=False),
        sa.Column("model_type", sa.String(20), nullable=False),
        sa.Column("trained_on", sa.Date(), nullable=False),
        sa.Column("accuracy_metrics", postgresql.JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.CheckConstraint("model_type IN ('frequency', 'severity')", name="ck_model_versions_type"),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index("ix_model_versions_type", "model_versions", ["model_type"])
    op.create_index("ix_model_versions_active", "model_versions", ["is_active"])


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("model_feedback")
    op.drop_table("claims")
    op.drop_table("predictions")
    op.drop_table("policies")
    op.drop_table("driving_profiles")
    op.drop_table("vehicles")
    op.drop_table("customers")
