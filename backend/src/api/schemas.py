from pydantic import BaseModel, Field, field_validator, UUID4
from typing import Optional, List, Any, Dict
from datetime import datetime, date
from enum import Enum

# ─── V2 Enum Constants ────────────────────────────────────────────────────────

class GenderV2(str, Enum):
    male   = "Male"
    female = "Female"
    other  = "Other"

class FuelType(str, Enum):
    petrol = "Petrol"
    diesel = "Diesel"
    ev     = "EV"
    cng    = "CNG"

class VehicleUsageType(str, Enum):
    personal   = "Personal"
    commercial = "Commercial"
    rideshare  = "Rideshare"

class Occupation(str, Enum):
    salaried      = "Salaried"
    self_employed = "Self-Employed"
    professional  = "Professional"
    retired       = "Retired"
    student       = "Student"
    manual        = "Manual"

class MaritalStatus(str, Enum):
    single   = "Single"
    married  = "Married"
    divorced = "Divorced"
    widowed  = "Widowed"

class AnnualIncomeBand(str, Enum):
    below_3l  = "<3 LPA"
    band_3_7  = "3-7 LPA"
    band_7_15 = "7-15 LPA"
    band_15_30 = "15-30 LPA"
    above_30l = ">30 LPA"

class ParkingType(str, Enum):
    garage        = "Garage"
    street        = "Street"
    open_compound = "Open_Compound"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"



class ClaimStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


# ─── Request Models ───────────────────────────────────────────────────────────

class CustomerQuoteRequest(BaseModel):
    # Personal
    age: int = Field(..., ge=18, le=70, description="Customer age (18–70)")
    gender: str = Field(..., min_length=1, max_length=20)
    city: str = Field(..., min_length=1, max_length=100)
    annual_income_inr: Optional[float] = Field(None, ge=0, le=50_000_000, description="Annual income in INR")

    # Vehicle — uses master dataset column names
    vehicle_brand: str = Field(..., min_length=1, max_length=100, description="Vehicle brand (e.g. Maruti, Hyundai, Tata)")
    vehicle_segment: str = Field(..., description="Hatchback / Sedan / SUV / Luxury")
    fuel_type: str = Field(..., description="Petrol / Diesel / CNG / EV")
    vehicle_age_years: int = Field(..., ge=0, le=30)
    vehicle_value_inr: float = Field(..., gt=0, le=50_000_000)
    # Legacy DB fields — optional; not used by ML but stored for policy records
    car_model: Optional[str] = Field(None, max_length=100, description="Vehicle model name (e.g. Swift, Creta)")
    engine_cc: Optional[int] = Field(None, ge=0, le=10000, description="Engine displacement in cc")

    # Driving profile
    driving_score: Optional[float] = Field(None, ge=0, le=100, description="Telematics driving score (0–100); auto-computed when absent")
    annual_mileage_km: int = Field(..., ge=0, le=200_000)
    previous_claims: int = Field(..., ge=0, le=20, description="Number of prior insurance claims (0–20)")
    years_licensed: int = Field(..., ge=0, le=67)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"male", "female", "other"}
        if v.lower() not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v.lower()

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v):
        return v.strip().lower()

    @field_validator("vehicle_brand")
    @classmethod
    def normalize_brand(cls, v):
        return v.strip().lower()

    @field_validator("vehicle_segment")
    @classmethod
    def normalize_vehicle_segment(cls, v):
        _map = {"hatchback": "Hatchback", "sedan": "Sedan", "suv": "SUV",
                "luxury": "Luxury", "muv": "SUV", "mpv": "SUV"}
        return _map.get(v.strip().lower(), v.strip().title())

    @field_validator("fuel_type")
    @classmethod
    def normalize_fuel_type(cls, v):
        _map = {"petrol": "Petrol", "diesel": "Diesel", "cng": "CNG",
                "ev": "EV", "electric": "EV", "hybrid": "Petrol"}
        return _map.get(v.strip().lower(), v.strip())

    model_config = {"str_strip_whitespace": True}


class PolicyCreateRequest(BaseModel):
    prediction_id: UUID4
    customer_data: CustomerQuoteRequest


class ClaimRequest(BaseModel):
    policy_id: UUID4
    claimed_amount_inr: float = Field(..., gt=0)
    claim_date: Optional[date] = None


class FeedbackRequest(BaseModel):
    prediction_id: UUID4
    actual_claim_occurred: bool
    actual_claim_amount_inr: Optional[float] = Field(None, ge=0)


class AssistantQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)


class ClaimUpdateRequest(BaseModel):
    claim_status: ClaimStatus
    approved_amount_inr: Optional[float] = Field(None, ge=0)


# ─── Response Models ──────────────────────────────────────────────────────────

class DrivingScoreResult(BaseModel):
    score: int
    factors: List[str]


class AdjustmentDetail(BaseModel):
    reason: str
    factor: float
    impact_inr: float


class ExplanationDetail(BaseModel):
    base_premium: float
    expected_loss: float
    adjustments: List[AdjustmentDetail]
    final_premium: float
    risk_level: RiskLevel
    claim_probability: float
    summary: str


class PremiumQuoteResponse(BaseModel):
    prediction_id: UUID4
    premium_amount_inr: float
    risk_level: RiskLevel
    claim_probability: float
    expected_claim_amount_inr: float
    explanation: ExplanationDetail
    model_version: str
    created_at: datetime
    driving_score_info: Optional[DrivingScoreResult] = None


class PolicyResponse(BaseModel):
    policy_id: UUID4
    customer_id: UUID4
    vehicle_id: UUID4
    premium_amount_inr: float
    risk_level: RiskLevel
    model_version: str
    created_at: datetime
    is_active: bool
    prediction: Optional[PremiumQuoteResponse] = None


class ClaimResponse(BaseModel):
    claim_id: UUID4
    policy_id: UUID4
    claimed_amount_inr: float
    approved_amount_inr: Optional[float]
    claim_date: date
    claim_status: ClaimStatus


class DashboardStats(BaseModel):
    total_policies: int
    active_policies: int
    avg_premium_inr: float
    total_claims: int
    claims_ratio: float
    total_premium_collected: float
    premium_by_risk_level: dict
    policies_by_risk_level: dict


class CustomerListItem(BaseModel):
    customer_id: UUID4
    age: int
    gender: str
    city: str
    created_at: datetime
    policy_count: int
    latest_risk_level: Optional[str]
    latest_premium: Optional[float]


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class PremiumTrendPoint(BaseModel):
    month: str
    risk_level: str
    avg_premium: float
    policy_count: int


class AssistantQueryResponse(BaseModel):
    answer: str
    sql_used: Optional[str]
    data: Optional[List[dict]]
    chart_suggestion: str = "none"


class FeedbackResponse(BaseModel):
    feedback_id: UUID4
    prediction_id: UUID4
    actual_claim_occurred: bool
    actual_claim_amount_inr: Optional[float]
    feedback_date: date


class RetrainResponse(BaseModel):
    message: str
    triggered: bool
    performance_report: Optional[dict] = None
    # Enterprise retraining fields (present when multi-algorithm engine is used)
    postponed: Optional[bool] = None
    labeled_count: Optional[int] = None
    threshold: Optional[int] = None
    frequency_winner: Optional[str] = None
    severity_winner: Optional[str] = None
    training_record: Optional[dict] = None


class ClaimAdminItem(BaseModel):
    claim_id: UUID4
    policy_id: UUID4
    claimed_amount_inr: float
    approved_amount_inr: Optional[float]
    claim_date: date
    claim_status: ClaimStatus
    risk_level: Optional[str] = None


# ─── Underwriting Agent ───────────────────────────────────────────────────────

class UnderwritingSignal(BaseModel):
    signal: str
    severity: str       # "low" | "medium" | "high"
    description: str


class ProfileRiskStats(BaseModel):
    similar_policies_count: int
    avg_premium_in_segment: float
    segment_claim_frequency: float
    total_claims_in_segment: int


class UnderwritingDecision(BaseModel):
    decision: str               # "approve" | "flag" | "refer"
    confidence: float
    risk_adjustment_factor: float
    agent_adjusted_premium: float


class AgentQuoteResponse(BaseModel):
    # Standard quote fields — identical to PremiumQuoteResponse
    prediction_id: UUID4
    premium_amount_inr: float
    risk_level: RiskLevel
    claim_probability: float
    expected_claim_amount_inr: float
    explanation: ExplanationDetail
    model_version: str
    created_at: datetime

    # Underwriting agent fields
    session_id: str
    fraud_signals: List[UnderwritingSignal]
    customer_history: ProfileRiskStats
    underwriting_decision: UnderwritingDecision
    reasoning_trace: str
    node_execution_log: List[dict]
    driving_score_info: Optional[DrivingScoreResult] = None


# ─── Analytics Agent ──────────────────────────────────────────────────────────

class AgentAnalyticsRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    session_id: Optional[str] = Field(default=None, max_length=100)


class AnomalyItem(BaseModel):
    column: str
    value: float
    row_data: dict
    direction: str
    deviation_score: float


class RecommendationItem(BaseModel):
    category: str
    text: str
    priority: str


class AgentAnalyticsResponse(BaseModel):
    answer: str
    sql_used: Optional[str] = None
    data: Optional[List[dict]] = None
    chart_suggestion: str = "none"
    session_id: str
    intent: str
    anomalies: Optional[List[AnomalyItem]] = None
    recommendations: Optional[List[RecommendationItem]] = None


# ─── Agent Observability ──────────────────────────────────────────────────────

class AgentToolCallItem(BaseModel):
    call_id: UUID4
    tool_name: str
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    called_at: datetime
    duration_ms: Optional[int] = None


class AgentRunSummary(BaseModel):
    run_id: UUID4
    session_id: str
    agent_type: str
    intent: Optional[str] = None
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_text: str
    review_outcome: Optional[str] = None   # "approved" | "rejected" — set after HITL review


class AgentRunDetail(AgentRunSummary):
    output_text: Optional[str] = None
    tool_calls: List[AgentToolCallItem] = []


# ─── HITL Review ──────────────────────────────────────────────────────────────

class ReviewDecisionRequest(BaseModel):
    reviewer_note: Optional[str] = Field(None, max_length=500)


class ReviewDecisionResponse(BaseModel):
    run_id: UUID4
    review_outcome: str         # "approved" | "rejected"
    reviewed_at: datetime
    reviewer_note: Optional[str]
    agent_decision: Optional[str]


class ReviewQueueItem(BaseModel):
    run_id: UUID4
    prediction_id: str
    started_at: datetime
    minutes_pending: int
    agent_decision: Optional[str]
    confidence_score: float
    claim_probability: float
    fraud_signal_count: int
    high_severity_count: int
    premium_amount_inr: float
    review_triggers: List[str]
    input_summary: str


class ReviewDetail(ReviewQueueItem):
    fraud_signals: List[UnderwritingSignal]
    customer_history: ProfileRiskStats
    underwriting_decision: UnderwritingDecision
    reasoning_trace: str
    node_execution_log: List[dict]


class AgentQuoteAwaitingResponse(BaseModel):
    run_id: str
    prediction_id: str
    status: str                 # always "awaiting_review"
    premium_amount_inr: float
    risk_level: str
    claim_probability: float
    agent_decision: str         # "approve" | "flag" | "refer"
    confidence: float
    fraud_signal_count: int
    review_triggers: List[str]
    message: str


# ─── V2 API Contract (Phase 2 — V4 Readiness) ───────────────────────────────

class CustomerQuoteRequestV2(BaseModel):
    """
    Expanded quote request schema for V4 model readiness.

    Adds 11 new fields beyond V1 (occupation, marital_status,
    annual_income_band, months_since_last_claim, no_claim_bonus_pct,
    policy_tenure_years, fuel_type, vehicle_usage_type, parking_type,
    and renames car_brand/car_model to vehicle_make/vehicle_model).

    Auto-derived by feature enrichment (NOT supplied by user):
      vehicle_safety_rating, airbags_count, vehicle_body_style,
      repair_cost_band, region_risk_category, flood_risk_index,
      theft_risk_index, monsoon_exposure_index, driving_score,
      pincode_risk_score, policy_inception_month.
    """

    # ── Personal ──────────────────────────────────────────────────────────
    age:               int           = Field(..., ge=18,  le=70,        description="Customer age in years (18-70)")
    gender:            GenderV2      = Field(...,                        description="Male | Female | Other")
    occupation:        Occupation    = Field(...,                        description="Salaried | Self-Employed | Professional | Retired | Student | Manual")
    marital_status:    MaritalStatus = Field(...,                        description="Single | Married | Divorced | Widowed")
    annual_income_band: AnnualIncomeBand = Field(...,                   description="<3 LPA | 3-7 LPA | 7-15 LPA | 15-30 LPA | >30 LPA")

    # ── Location ──────────────────────────────────────────────────────────
    city:              str           = Field(..., min_length=1, max_length=100, description="City name (must match V4 supported cities for lookup enrichment)")

    # ── Driving history ───────────────────────────────────────────────────
    years_licensed:         int   = Field(..., ge=0,   le=80,    description="Years held a driving licence (0-80)")
    previous_claims_count:  int   = Field(..., ge=0,   le=20,    description="Number of prior insurance claims (0-20)")
    months_since_last_claim: int  = Field(..., ge=0,   le=240,   description="Months since most recent claim; 0 = never claimed (0-240)")
    no_claim_bonus_pct:     float = Field(default=0.0, ge=0.0, le=50.0, description="(Auto-derived by enrichment service — do not send from frontend) No-claim bonus %; overridden by derive_ncb_pct()")
    policy_tenure_years:    float = Field(..., ge=0.0, le=30.0,  description="Planned policy duration in years (0-30)")

    # ── Vehicle identification ────────────────────────────────────────────
    vehicle_make:       str   = Field(..., min_length=1, max_length=100, description="Manufacturer name (e.g. Hyundai, Tata, Maruti)")
    vehicle_model:      str   = Field(..., min_length=1, max_length=100, description="Model name (e.g. Creta, Nexon, Swift)")
    vehicle_age_years:  int   = Field(..., ge=0,  le=30,                 description="Age of vehicle in years (0-30)")
    vehicle_value_inr:  float = Field(..., gt=0,  le=50_000_000,         description="Current market value in INR")
    engine_cc:          int   = Field(..., ge=0,  le=10_000,             description="Engine displacement in cc (0 for Electric vehicles)")

    # ── Vehicle usage ─────────────────────────────────────────────────────
    fuel_type:           FuelType         = Field(..., description="Petrol | Diesel | EV | CNG")
    vehicle_usage_type:  VehicleUsageType = Field(..., description="Personal | Commercial | Rideshare")
    annual_mileage_km:   int              = Field(..., ge=0, le=200_000, description="Annual distance driven in km (0-200000)")
    parking_type:        ParkingType      = Field(..., description="Garage | Street | Open_Compound")

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("vehicle_make")
    @classmethod
    def normalize_make(cls, v: str) -> str:
        return v.strip()

    @field_validator("vehicle_model")
    @classmethod
    def normalize_model(cls, v: str) -> str:
        return v.strip()

    model_config = {"str_strip_whitespace": True}


# ─── V2 Enriched Feature Audit ───────────────────────────────────────────────

class VehicleEnrichment(BaseModel):
    vehicle_safety_rating: int    = Field(..., description="NCAP safety rating (1-5 stars)")
    airbags_count:         int    = Field(..., description="Factory-fitted airbag count")
    vehicle_body_style:    str    = Field(..., description="Hatchback | Sedan | SUV | MUV")
    repair_cost_band:      str    = Field(..., description="Budget | Standard | Premium | Luxury")
    lookup_hit:            bool   = Field(..., description="True if exact match found in vehicle_specs.json")

class CityEnrichment(BaseModel):
    region_risk_category:  str   = Field(..., description="High | Medium")
    flood_risk_index:      float = Field(..., description="Flood risk on 1-10 scale")
    theft_risk_index:      float = Field(..., description="Vehicle theft risk on 1-10 scale")
    monsoon_exposure_index: float = Field(..., description="Monsoon exposure on 1-10 scale")
    pincode_risk_score:    float  = Field(..., description="Composite risk score (0.35×flood + 0.40×theft + 0.25×monsoon)")
    lookup_hit:            bool   = Field(..., description="True if exact match found in city_geo_risk.json")

class DrivingScoreEnrichment(BaseModel):
    score:   int        = Field(..., description="Computed driving score (0-100)")
    factors: List[str]  = Field(..., description="Score adjustment factors")

class EnrichedFeaturesV2(BaseModel):
    driving_score:       DrivingScoreEnrichment
    vehicle:             VehicleEnrichment
    city:                CityEnrichment
    policy_inception_month: int   = Field(..., description="Month of policy inception (1-12)")
    no_claim_bonus_pct:     float = Field(..., description="Auto-derived NCB %: 0/20/25/35/45/50 based on claims and tenure")

class ChampionMetadataV2(BaseModel):
    frequency_model:    str = Field(..., description="Name of frequency model used")
    severity_model:     str = Field(..., description="Name of severity model used")
    model_version:      str = Field(..., description="Active model version tag")
    v4_ready:           bool = Field(..., description="True when V4 champions are active")


# ─── V2 Response ─────────────────────────────────────────────────────────────

class QuoteResponseV2(BaseModel):
    """
    Extended quote response for V2 API.

    Includes all V1 PremiumQuoteResponse fields plus:
      - enriched_features: Audit of all auto-derived inputs
      - champion_metadata: Which models produced this prediction
    """

    # V1-compatible core fields
    prediction_id:             UUID4              = Field(...)
    premium_amount_inr:        float              = Field(...)
    risk_level:                RiskLevel          = Field(...)
    claim_probability:         float              = Field(...)
    expected_claim_amount_inr: float              = Field(...)
    explanation:               ExplanationDetail  = Field(...)
    model_version:             str                = Field(...)
    created_at:                datetime           = Field(...)
    driving_score_info:        Optional[DrivingScoreResult] = None

    # V2 additions
    enriched_features:  EnrichedFeaturesV2  = Field(..., description="Auto-derived feature audit")
    champion_metadata:  ChampionMetadataV2  = Field(..., description="Model provenance")
    api_version:        str                 = Field(default="v2", description="API contract version")


class QuotePreviewResponseV2(BaseModel):
    """
    Lightweight preview for V2: returns enriched features without ML prediction.
    Use this endpoint to validate enrichment before triggering full quote.
    """
    enriched_features:   EnrichedFeaturesV2
    champion_metadata:   ChampionMetadataV2
    request_summary:     dict = Field(..., description="Echo of key request fields")
    api_version:         str  = Field(default="v2")
    message:             str  = Field(default="Preview only — no ML prediction performed")


# ─── V4 Production Monitoring Dashboard ──────────────────────────────────────

class MonitoringVolumeStat(BaseModel):
    date: str
    count: int
    v2_count: int
    v1_count: int


class MonitoringPremiumBand(BaseModel):
    risk_level: str
    count: int
    mean_premium: float
    min_premium: float
    max_premium: float
    at_min_cap_count: int
    at_max_cap_count: int


class MonitoringRiskBand(BaseModel):
    risk_level: str
    count: int
    pct: float
    mean_claim_prob: float


class DataQualityCheck(BaseModel):
    check_id: str
    description: str
    table_name: str
    total_rows: int
    passing_rows: int
    fill_rate_pct: float
    target_pct: float
    status: str  # 'ok' | 'warning' | 'critical'


class MonitoringHITLStats(BaseModel):
    queue_depth: int
    reviews_last_30d: int
    override_rate: Optional[float]
    avg_review_hours: Optional[float]


class MonitoringLabelStats(BaseModel):
    total_predictions: int
    labelled_predictions: int
    coverage_pct: float
    matured_policies_90d: int
    matured_labelled_90d: int
    matured_coverage_pct: float


class MonitoringDashboard(BaseModel):
    generated_at: str
    total_quotes: int
    quotes_today: int
    quotes_last_7d: int
    quotes_last_30d: int
    v2_share_pct: float
    overall_mean_premium: float
    premium_by_risk: List[MonitoringPremiumBand]
    risk_distribution: List[MonitoringRiskBand]
    data_quality: List[DataQualityCheck]
    inference_json_fill_pct: float
    hitl_stats: MonitoringHITLStats
    label_stats: MonitoringLabelStats


class MonitoringVolumeTrend(BaseModel):
    trend: List[MonitoringVolumeStat]


# ─── V5 Groundwork — Dataset Builder / Export ─────────────────────────────────

class V5DatasetStatus(BaseModel):
    generated_at: str
    total_predictions: int
    labelled_predictions: int
    matured_predictions: int
    matured_labelled: int
    rows_valid: int
    rows_dropped_missing_json: int
    rows_dropped_validation: int
    feature_fill_rates: Dict[str, float]
    missing_features: List[str]
    is_ready_for_retraining: bool
    readiness_reason: str
    label_maturity_days: int = 90
    min_samples_required: int = 500


class V5ExportResponse(BaseModel):
    status: str          # 'ok' | 'empty'
    rows_exported: int
    filename: str
    generated_at: str
    is_ready_for_retraining: bool
    readiness_reason: str


# ─── Phase 4A — Drift Detection ───────────────────────────────────────────────

class FeatureDriftResult(BaseModel):
    feature: str
    feature_type: str                   # "numeric" | "categorical"
    psi: float
    severity: str                       # "low" | "medium" | "high" | "insufficient_data"
    ref_n: int
    prod_n: int
    ref_bins: List[float]
    prod_bins: List[float]
    bin_labels: List[str]
    alert: Optional[str] = None


class DriftSnapshot(BaseModel):
    computed_at: str
    window_days: int
    prod_sample_n: int
    psi_low_threshold: float
    psi_high_threshold: float
    feature_results: List[FeatureDriftResult]
    overall_psi: float
    overall_severity: str               # "low" | "medium" | "high" | "no_data"
    recommendation: str
    alerts: List[str]
    high_drift_features: List[str]
    medium_drift_features: List[str]
    features_computed: int
    features_skipped: int


class DriftHistoryResponse(BaseModel):
    snapshots: List[Dict[str, Any]]
    count: int


# ─── Phase 4B — Human Approval Workflow ──────────────────────────────────────

class ModelCardFrequencyMetrics(BaseModel):
    roc_auc: float
    pr_auc: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None


class ModelCardSeverityMetrics(BaseModel):
    r2: float
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape_pct: Optional[float] = None
    median_abs_error: Optional[float] = None


class ModelCardDriftSummary(BaseModel):
    computed_at: Optional[str] = None
    overall_psi: Optional[float] = None
    overall_severity: Optional[str] = None
    high_drift_features: List[str] = []
    medium_drift_features: List[str] = []
    features_computed: Optional[int] = None
    recommendation: Optional[str] = None


class ModelCard(BaseModel):
    model_type: str
    model_version: str
    algorithm: str
    training_date: str
    dataset_size: int
    preprocessor_version: str
    champion_since: str
    frequency_metrics: Optional[ModelCardFrequencyMetrics] = None
    severity_metrics: Optional[ModelCardSeverityMetrics] = None
    feature_importance_summary: Optional[Dict[str, float]] = None
    drift_summary: Optional[ModelCardDriftSummary] = None
    recommendation: str


class ApprovalRequestCreate(BaseModel):
    model_type: str = Field(..., description="'frequency' or 'severity'")
    notes: Optional[str] = Field(None, max_length=1000)


class ApprovalReviewAction(BaseModel):
    reviewer_note: Optional[str] = Field(None, max_length=2000)


class ApprovalAuditLogEntry(BaseModel):
    log_id: str
    request_id: str
    event_type: str
    actor: Optional[str] = None
    note: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    created_at: str


class ApprovalRequestResponse(BaseModel):
    request_id: str
    model_type: str
    model_version: str
    status: str
    submitted_by: str
    submitted_at: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewer_note: Optional[str] = None
    model_card: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None


class ApprovalRequestDetailResponse(ApprovalRequestResponse):
    audit_logs: List[ApprovalAuditLogEntry] = []


class ApprovalRequestListResponse(BaseModel):
    items: List[ApprovalRequestResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─── Phase 4C — Shadow Deployment Framework ───────────────────────────────────

class ShadowPredictionRecord(BaseModel):
    id: str
    request_id: str
    quote_id: Optional[str] = None
    created_at: str
    champion_model_name: str
    champion_version: str
    challenger_model_name: Optional[str] = None
    challenger_version: Optional[str] = None
    champion_prediction_json: Optional[Dict[str, Any]] = None
    challenger_prediction_json: Optional[Dict[str, Any]] = None
    premium_difference: Optional[float] = None
    risk_difference: Optional[str] = None
    prediction_latency_ms: Optional[int] = None
    status: str
    comparison_result: Optional[str] = None
    notes: Optional[str] = None


class ShadowStatusResponse(BaseModel):
    framework_status: str
    challenger_available: bool
    challenger_status: str
    total_shadow_predictions: int
    waiting_for_challenger: int
    completed_comparisons: int
    failed_comparisons: int
    champion_model: str
    champion_version: str
    last_recorded_at: Optional[str] = None
    message: str


class ShadowStatisticsResponse(BaseModel):
    window_days: int
    total_shadow_predictions: int
    completed_comparisons: int
    waiting_for_challenger: int
    completion_rate_pct: float
    avg_premium_difference_inr: Optional[float] = None
    avg_prediction_latency_ms: Optional[float] = None
    challenger_available: bool


class ShadowHistoryResponse(BaseModel):
    items: List[ShadowPredictionRecord]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─── Phase 5A — V5 Training Engine ───────────────────────────────────────────

class TrainingValidationReport(BaseModel):
    passed: bool
    dataset_rows: int
    matured_labelled: int
    valid_rows: int
    missing_features: List[str]
    reasons: List[str]
    maturity_days: int
    min_samples: int


class CandidateModelResponse(BaseModel):
    model_id: str
    run_id: str
    algorithm: str
    model_type: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    test_rows: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    frequency_metrics: Optional[Dict[str, Any]] = None
    severity_metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None


class TrainingRunResponse(BaseModel):
    run_id: str
    run_tag: str
    triggered_by: str
    triggered_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str
    dataset_rows: Optional[int] = None
    feature_count: Optional[int] = None
    maturity_days: int = 90
    train_rows: Optional[int] = None
    val_rows: Optional[int] = None
    test_rows: Optional[int] = None
    error_message: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    candidates: List[CandidateModelResponse] = []


class TrainingRunListResponse(BaseModel):
    items: List[TrainingRunResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class TrainingTriggerResponse(BaseModel):
    run_id: str
    run_tag: str
    status: str
    message: str
    triggered_by: str
    triggered_at: str



# ─── Phase 5B — Champion Promotion Engine ─────────────────────────────────────

class PromotionEvaluationRequest(BaseModel):
    run_id: str
    frequency_candidate_id: str
    severity_candidate_id: str
    approval_id: Optional[str] = None
    bypass_shadow: bool = False
    bypass_drift: bool = False
    notes: Optional[str] = None


class GateResultSchema(BaseModel):
    gate: str
    passed: bool
    reason: str
    details: Optional[Dict[str, Any]] = None


class PromotionEvaluationResponse(BaseModel):
    promotion_id: str
    run_id: str
    status: str
    all_gates_passed: bool
    gates: Dict[str, GateResultSchema]
    freq_comparison: Optional[Dict[str, Any]] = None
    sev_comparison: Optional[Dict[str, Any]] = None
    can_promote: bool
    message: str
    created_at: str


class PromotionExecuteRequest(BaseModel):
    promoted_by: str = "admin"
    notes: Optional[str] = None


class RollbackRequest(BaseModel):
    reason: str
    rolled_back_by: str = "admin"


class RollbackHistoryResponse(BaseModel):
    rollback_id: str
    promotion_id: str
    rollback_reason: Optional[str] = None
    rollback_trigger: str
    rollback_status: str
    rollback_duration_seconds: Optional[float] = None
    rolled_back_by: Optional[str] = None
    rolled_back_at: str
    error_message: Optional[str] = None
    restored_frequency_champion: Optional[Dict[str, Any]] = None
    restored_severity_champion: Optional[Dict[str, Any]] = None


class PromotionDetailResponse(BaseModel):
    promotion_id: str
    run_id: str
    frequency_candidate_id: Optional[str] = None
    severity_candidate_id: Optional[str] = None
    status: str
    all_gates_passed: Optional[bool] = None
    evaluation_report: Optional[Dict[str, Any]] = None
    gates_passed: Optional[Dict[str, Any]] = None
    gates_failed: Optional[Dict[str, Any]] = None
    old_frequency_champion: Optional[Dict[str, Any]] = None
    old_severity_champion: Optional[Dict[str, Any]] = None
    new_frequency_champion: Optional[Dict[str, Any]] = None
    new_severity_champion: Optional[Dict[str, Any]] = None
    promoted_by: Optional[str] = None
    promoted_at: Optional[str] = None
    promotion_duration_seconds: Optional[float] = None
    backup_path: Optional[str] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    rollback_records: List[RollbackHistoryResponse] = []


class PromotionListResponse(BaseModel):
    items: List[PromotionDetailResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
