import uuid
from datetime import datetime, date
from sqlalchemy import (
    Column, String, Integer, SmallInteger, Numeric, Boolean, DateTime, Date,
    ForeignKey, CheckConstraint, Index, text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    city = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # V4 personal attributes (added migration 006)
    occupation = Column(String(30), nullable=True)
    marital_status = Column(String(20), nullable=True)
    annual_income_band = Column(String(20), nullable=True)

    __table_args__ = (
        CheckConstraint("age >= 18 AND age <= 85", name="ck_customers_age"),
        Index("ix_customers_city", "city"),
        Index("ix_customers_created_at", "created_at"),
        Index("ix_customers_occupation", "occupation"),
    )

    vehicles = relationship("Vehicle", back_populates="customer")
    driving_profiles = relationship("DrivingProfile", back_populates="customer")
    policies = relationship("Policy", back_populates="customer")


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False)
    car_brand = Column(String(100), nullable=False)
    car_model = Column(String(100), nullable=False)
    engine_cc = Column(Integer, nullable=False)
    vehicle_age_years = Column(Integer, nullable=False)
    vehicle_value_inr = Column(Numeric(14, 2), nullable=False)

    # V4 vehicle attributes + lookup-derived specs (added migration 006)
    fuel_type = Column(String(20), nullable=True)
    vehicle_usage_type = Column(String(20), nullable=True)
    parking_type = Column(String(20), nullable=True)
    vehicle_safety_rating = Column(SmallInteger, nullable=True)
    airbags_count = Column(SmallInteger, nullable=True)
    vehicle_body_style = Column(String(20), nullable=True)
    repair_cost_band = Column(String(20), nullable=True)

    __table_args__ = (
        CheckConstraint("engine_cc >= 0", name="ck_vehicles_engine_cc"),
        CheckConstraint("vehicle_age_years >= 0", name="ck_vehicles_age"),
        CheckConstraint("vehicle_value_inr > 0", name="ck_vehicles_value"),
        Index("ix_vehicles_customer_id", "customer_id"),
        Index("ix_vehicles_fuel_type", "fuel_type"),
    )

    customer = relationship("Customer", back_populates="vehicles")
    policies = relationship("Policy", back_populates="vehicle")


class DrivingProfile(Base):
    __tablename__ = "driving_profiles"

    profile_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False)
    driving_score = Column(Numeric(5, 2), nullable=False)
    annual_mileage_km = Column(Integer, nullable=False)
    previous_claims_count = Column(Integer, nullable=False, default=0)
    years_licensed = Column(Integer, nullable=False)

    # V4 driving history + geo-risk + system-derived (added migration 006)
    months_since_last_claim = Column(SmallInteger, nullable=True)
    policy_tenure_years = Column(Numeric(5, 2), nullable=True)
    no_claim_bonus_pct = Column(Numeric(4, 1), nullable=True)
    region_risk_category = Column(String(10), nullable=True)
    flood_risk_index = Column(Numeric(4, 2), nullable=True)
    theft_risk_index = Column(Numeric(4, 2), nullable=True)
    monsoon_exposure_index = Column(Numeric(4, 2), nullable=True)
    pincode_risk_score = Column(Numeric(6, 4), nullable=True)
    policy_inception_month = Column(SmallInteger, nullable=True)

    __table_args__ = (
        CheckConstraint("driving_score >= 0 AND driving_score <= 100", name="ck_profile_driving_score"),
        CheckConstraint("annual_mileage_km >= 0", name="ck_profile_mileage"),
        CheckConstraint("previous_claims_count >= 0", name="ck_profile_claims"),
        CheckConstraint("years_licensed >= 0", name="ck_profile_years_licensed"),
        Index("ix_driving_profiles_customer_id", "customer_id"),
    )

    customer = relationship("Customer", back_populates="driving_profiles")


class Policy(Base):
    __tablename__ = "policies"

    policy_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False)
    vehicle_id = Column(UUID(as_uuid=True), ForeignKey("vehicles.vehicle_id"), nullable=False)
    premium_amount_inr = Column(Numeric(14, 2), nullable=False)
    risk_level = Column(String(10), nullable=False)
    model_version = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="ck_policies_risk_level"),
        Index("ix_policies_customer_id", "customer_id"),
        Index("ix_policies_vehicle_id", "vehicle_id"),
        Index("ix_policies_created_at", "created_at"),
        Index("ix_policies_risk_level", "risk_level"),
    )

    customer = relationship("Customer", back_populates="policies")
    vehicle = relationship("Vehicle", back_populates="policies")
    predictions = relationship("Prediction", back_populates="policy")
    claims = relationship("Claim", back_populates="policy")


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_id = Column(UUID(as_uuid=True), ForeignKey("policies.policy_id"), nullable=False)
    claim_probability = Column(Numeric(6, 5), nullable=False)
    expected_claim_amount_inr = Column(Numeric(14, 2), nullable=False)
    final_premium_inr = Column(Numeric(14, 2), nullable=False)
    explanation_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Point-in-time snapshot of all 31 raw features passed to the predictor (added migration 006)
    inference_features_json = Column(JSONB, nullable=True)

    __table_args__ = (
        CheckConstraint("claim_probability >= 0 AND claim_probability <= 1", name="ck_predictions_probability"),
        Index("ix_predictions_policy_id", "policy_id"),
        Index("ix_predictions_created_at", "created_at"),
    )

    policy = relationship("Policy", back_populates="predictions")
    feedbacks = relationship("ModelFeedback", back_populates="prediction")


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_id = Column(UUID(as_uuid=True), ForeignKey("policies.policy_id"), nullable=False)
    claimed_amount_inr = Column(Numeric(14, 2), nullable=False)
    approved_amount_inr = Column(Numeric(14, 2), nullable=True)
    claim_date = Column(Date, nullable=False, default=date.today)
    claim_status = Column(String(20), nullable=False, default="pending")

    __table_args__ = (
        CheckConstraint("claim_status IN ('pending', 'approved', 'rejected')", name="ck_claims_status"),
        CheckConstraint("claimed_amount_inr > 0", name="ck_claims_amount"),
        Index("ix_claims_policy_id", "policy_id"),
        Index("ix_claims_status", "claim_status"),
        Index("ix_claims_date", "claim_date"),
    )

    policy = relationship("Policy", back_populates="claims")


class ModelFeedback(Base):
    __tablename__ = "model_feedback"

    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.prediction_id"), nullable=False)
    actual_claim_occurred = Column(Boolean, nullable=False)
    actual_claim_amount_inr = Column(Numeric(14, 2), nullable=True)
    feedback_date = Column(Date, nullable=False, default=date.today)

    # HITL enrichment columns (added in migration 004)
    agent_run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.run_id"), nullable=True)
    human_decision = Column(String(20), nullable=True)   # 'approved' | 'rejected'
    agent_decision = Column(String(20), nullable=True)   # 'approve' | 'flag' | 'refer'
    fraud_signal_count = Column(Integer, nullable=True)
    high_severity_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_model_feedback_prediction_id", "prediction_id"),
        Index("ix_model_feedback_date", "feedback_date"),
        Index("ix_model_feedback_agent_run_id", "agent_run_id"),
        Index("ix_model_feedback_human_decision", "human_decision"),
    )

    prediction = relationship("Prediction", back_populates="feedbacks")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    intent = Column(String(50), nullable=True)
    input_text = Column(String, nullable=False)
    output_text = Column(String, nullable=True)
    status = Column(String(20), nullable=False, default="running")
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # HITL review columns (added in migration 003)
    review_outcome = Column(String(20), nullable=True)   # 'approved' | 'rejected'
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_note = Column(String, nullable=True)
    result_json = Column(JSONB, nullable=True)           # snapshot stored when paused

    # Retraining feedback link (added in migration 004)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.prediction_id"), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'awaiting_review', 'completed', 'failed')",
            name="ck_agent_runs_status",
        ),
        CheckConstraint("agent_type IN ('analytics', 'underwriting')", name="ck_agent_runs_type"),
        Index("ix_agent_runs_session_id", "session_id"),
        Index("ix_agent_runs_agent_type", "agent_type"),
        Index("ix_agent_runs_started_at", "started_at"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_prediction_id", "prediction_id"),
    )

    tool_calls = relationship("AgentToolCall", back_populates="run")


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"

    call_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.run_id"), nullable=False)
    tool_name = Column(String(100), nullable=False)
    tool_input = Column(JSONB, nullable=True)
    tool_output = Column(String, nullable=True)
    called_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    duration_ms = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_agent_tool_calls_run_id", "run_id"),
        Index("ix_agent_tool_calls_tool_name", "tool_name"),
    )

    run = relationship("AgentRun", back_populates="tool_calls")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_tag = Column(String(20), nullable=False)
    model_type = Column(String(20), nullable=False)
    trained_on = Column(Date, nullable=False)
    accuracy_metrics = Column(JSONB, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        CheckConstraint("model_type IN ('frequency', 'severity')", name="ck_model_versions_type"),
        Index("ix_model_versions_type", "model_type"),
        Index("ix_model_versions_active", "is_active"),
    )


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    request_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String(20), nullable=False)
    model_version = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    submitted_by = Column(String(100), nullable=False, default="admin")
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_note = Column(String, nullable=True)
    model_card = Column(JSONB, nullable=True)
    recommendation = Column(String(50), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "model_type IN ('frequency', 'severity')",
            name="ck_approval_model_type",
        ),
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_approval_status",
        ),
        Index("ix_approval_requests_status", "status"),
        Index("ix_approval_requests_model_type", "model_type"),
        Index("ix_approval_requests_submitted_at", "submitted_at"),
    )

    audit_logs = relationship(
        "ApprovalAuditLog",
        back_populates="approval_request",
        cascade="all, delete-orphan",
        order_by="ApprovalAuditLog.created_at",
    )


class ApprovalAuditLog(Base):
    __tablename__ = "approval_audit_logs"

    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(
        UUID(as_uuid=True),
        ForeignKey("approval_requests.request_id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type = Column(String(50), nullable=False)
    actor = Column(String(100), nullable=True)
    note = Column(String, nullable=True)
    event_data = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_approval_audit_logs_request_id", "request_id"),
        Index("ix_approval_audit_logs_created_at", "created_at"),
    )

    approval_request = relationship("ApprovalRequest", back_populates="audit_logs")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_tag = Column(String(100), nullable=False, unique=True)
    triggered_by = Column(String(100), nullable=False, default="admin")
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default="RUNNING")
    dataset_version = Column(String(50), nullable=True)
    dataset_rows = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    maturity_days = Column(Integer, nullable=False, default=90)
    train_rows = Column(Integer, nullable=True)
    val_rows = Column(Integer, nullable=True)
    test_rows = Column(Integer, nullable=True)
    error_message = Column(String, nullable=True)
    summary = Column(JSONB, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'VALIDATION_FAILED')",
            name="ck_training_runs_status",
        ),
        Index("ix_training_runs_triggered_at", "triggered_at"),
        Index("ix_training_runs_status", "status"),
        Index("ix_training_runs_run_tag", "run_tag"),
    )

    candidates = relationship(
        "CandidateModel",
        back_populates="training_run",
        cascade="all, delete-orphan",
        order_by="CandidateModel.started_at",
    )


class CandidateModel(Base):
    __tablename__ = "candidate_models"

    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    algorithm = Column(String(50), nullable=False)
    model_type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="TRAINING")
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Numeric(10, 2), nullable=True)
    train_rows = Column(Integer, nullable=True)
    val_rows = Column(Integer, nullable=True)
    test_rows = Column(Integer, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)
    frequency_metrics = Column(JSONB, nullable=True)
    severity_metrics = Column(JSONB, nullable=True)
    feature_importance = Column(JSONB, nullable=True)
    artifact_path = Column(String(500), nullable=True)
    error_message = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "algorithm IN ('xgboost', 'catboost', 'lightgbm')",
            name="ck_candidate_algorithm",
        ),
        CheckConstraint(
            "model_type IN ('frequency', 'severity')",
            name="ck_candidate_model_type",
        ),
        CheckConstraint(
            "status IN ('TRAINING', 'COMPLETED', 'FAILED')",
            name="ck_candidate_status",
        ),
        Index("ix_candidate_models_run_id", "run_id"),
        Index("ix_candidate_models_algorithm", "algorithm"),
        Index("ix_candidate_models_status", "status"),
        Index("ix_candidate_models_model_type", "model_type"),
    )

    training_run = relationship("TrainingRun", back_populates="candidates")


class ModelPromotion(Base):
    __tablename__ = "model_promotions"

    promotion_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    frequency_candidate_id = Column(UUID(as_uuid=True), nullable=True)
    severity_candidate_id = Column(UUID(as_uuid=True), nullable=True)
    frequency_approval_id = Column(UUID(as_uuid=True), nullable=True)
    severity_approval_id = Column(UUID(as_uuid=True), nullable=True)

    old_frequency_champion = Column(JSONB, nullable=True)
    old_severity_champion = Column(JSONB, nullable=True)
    new_frequency_champion = Column(JSONB, nullable=True)
    new_severity_champion = Column(JSONB, nullable=True)

    evaluation_report = Column(JSONB, nullable=True)
    gates_passed = Column(JSONB, nullable=True)
    gates_failed = Column(JSONB, nullable=True)

    status = Column(String(30), nullable=False, default="PENDING")
    promoted_by = Column(String(200), nullable=True)
    promoted_at = Column(DateTime, nullable=True)
    promotion_duration_seconds = Column(Numeric(10, 2), nullable=True)
    backup_path = Column(String(500), nullable=True)
    error_message = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING','EVALUATING','APPROVED','REJECTED','PROMOTING','ACTIVE','ROLLED_BACK','FAILED')",
            name="ck_model_promotions_status",
        ),
        Index("ix_model_promotions_run_id", "run_id"),
        Index("ix_model_promotions_status", "status"),
        Index("ix_model_promotions_created_at", "created_at"),
    )

    rollback_records = relationship(
        "RollbackHistory",
        back_populates="promotion",
        cascade="all, delete-orphan",
        order_by="RollbackHistory.rolled_back_at",
    )


class RollbackHistory(Base):
    __tablename__ = "rollback_history"

    rollback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    promotion_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_promotions.promotion_id", ondelete="CASCADE"),
        nullable=False,
    )

    restored_frequency_champion = Column(JSONB, nullable=True)
    restored_severity_champion = Column(JSONB, nullable=True)

    rollback_reason = Column(String(1000), nullable=True)
    rollback_trigger = Column(String(50), nullable=False, default="manual")
    rollback_status = Column(String(20), nullable=False, default="SUCCESS")
    rollback_duration_seconds = Column(Numeric(10, 2), nullable=True)
    rolled_back_by = Column(String(200), nullable=True)
    rolled_back_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    error_message = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "rollback_trigger IN ('health_check','load_failure','prediction_failure','registry_failure','manual','auto')",
            name="ck_rollback_trigger",
        ),
        CheckConstraint(
            "rollback_status IN ('SUCCESS','FAILED','PARTIAL')",
            name="ck_rollback_status",
        ),
        Index("ix_rollback_history_promotion_id", "promotion_id"),
        Index("ix_rollback_history_rolled_back_at", "rolled_back_at"),
    )

    promotion = relationship("ModelPromotion", back_populates="rollback_records")


class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(100), nullable=False, unique=True)
    quote_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Champion identification
    champion_model_name = Column(String(100), nullable=False)
    champion_version = Column(String(20), nullable=False)

    # Challenger identification (null until a challenger is registered)
    challenger_model_name = Column(String(100), nullable=True)
    challenger_version = Column(String(20), nullable=True)

    # Prediction payloads
    champion_prediction_json = Column(JSONB, nullable=True)
    challenger_prediction_json = Column(JSONB, nullable=True)

    # Comparison metrics (null until challenger runs)
    premium_difference = Column(Numeric(14, 2), nullable=True)
    risk_difference = Column(String(20), nullable=True)
    prediction_latency_ms = Column(Integer, nullable=True)

    # Status and result
    status = Column(String(30), nullable=False, default="WAITING_FOR_CHALLENGER")
    comparison_result = Column(String(30), nullable=True)
    notes = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('WAITING_FOR_CHALLENGER', 'PENDING', 'COMPLETED', 'FAILED')",
            name="ck_shadow_status",
        ),
        CheckConstraint(
            "comparison_result IS NULL OR comparison_result IN "
            "('MATCH', 'DIVERGENT', 'CHALLENGER_HIGHER', 'CHALLENGER_LOWER')",
            name="ck_shadow_comparison_result",
        ),
        Index("ix_shadow_predictions_created_at", "created_at"),
        Index("ix_shadow_predictions_status", "status"),
        Index("ix_shadow_predictions_quote_id", "quote_id"),
        Index("ix_shadow_predictions_request_id", "request_id"),
    )
