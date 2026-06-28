"""
quote_v2.py
===========
V2 quote endpoints — parallel to V1; backward-compatible; V4-ready.

Routes:
  POST /quote        → full ML prediction  (QuoteResponseV2)
  POST /quote/preview → enrichment only, no ML  (QuotePreviewResponseV2)

Design constraints:
  - V1 endpoints NOT touched
  - V4 champions NOT activated
  - Champion registry NOT modified
  - Pricing logic NOT modified
  - React frontend NOT modified

Under the hood the V2 endpoints still call the V1 CombinedPredictor
(because V4 champions are "validated", not "active").  The inference_payload
built by feature_enrichment_service already aliases vehicle_make/model → car_brand/car_model
so the V1 preprocessor receives exactly the field names it expects.
"""

import time
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import get_db
from src.database.models import Customer, Vehicle, DrivingProfile, Policy, Prediction
from src.api.schemas import (
    ExplanationDetail,
    AdjustmentDetail,
    DrivingScoreResult,
    CustomerQuoteRequestV2,
    QuoteResponseV2,
    QuotePreviewResponseV2,
    EnrichedFeaturesV2,
    VehicleEnrichment,
    CityEnrichment,
    DrivingScoreEnrichment,
    ChampionMetadataV2,
)
from src.models.combined_predictor import get_predictor
from src.models.model_registry import get_active_model_version
from src.engine.pricing_engine import get_pricing_engine
from src.services.feature_enrichment_service import enrich_v2_request
from src.services.shadow_deployment import ShadowDeploymentService

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_enriched_features(meta: dict) -> EnrichedFeaturesV2:
    v = meta["vehicle"]
    c = meta["city"]
    d = meta["driving_score"]
    return EnrichedFeaturesV2(
        driving_score=DrivingScoreEnrichment(score=d["score"], factors=d["factors"]),
        vehicle=VehicleEnrichment(
            vehicle_safety_rating=v["vehicle_safety_rating"],
            airbags_count=v["airbags_count"],
            vehicle_body_style=v["vehicle_body_style"],
            repair_cost_band=v["repair_cost_band"],
            lookup_hit=v["lookup_hit"],
        ),
        city=CityEnrichment(
            region_risk_category=c["region_risk_category"],
            flood_risk_index=c["flood_risk_index"],
            theft_risk_index=c["theft_risk_index"],
            monsoon_exposure_index=c["monsoon_exposure_index"],
            pincode_risk_score=c["pincode_risk_score"],
            lookup_hit=c["lookup_hit"],
        ),
        policy_inception_month=meta["policy_inception_month"],
        no_claim_bonus_pct=meta["no_claim_bonus_pct"],
    )


def _build_champion_metadata() -> ChampionMetadataV2:
    predictor = get_predictor()
    v4_ready = predictor._loaded_from_registry

    if v4_ready:
        freq_name = getattr(predictor.frequency_model, "__class__", type(predictor.frequency_model)).__name__
        sev_name  = getattr(predictor.severity_model,  "__class__", type(predictor.severity_model)).__name__
        version   = getattr(predictor, "version", get_active_model_version())
    else:
        freq_name = "XGBClassifier-V1"
        sev_name  = "XGBRegressor-V1"
        version   = get_active_model_version()

    return ChampionMetadataV2(
        frequency_model=freq_name,
        severity_model=sev_name,
        model_version=version,
        v4_ready=v4_ready,
    )


# ─── POST /api/v2/quote ───────────────────────────────────────────────────────

@router.post(
    "/quote",
    response_model=QuoteResponseV2,
    status_code=status.HTTP_201_CREATED,
    summary="V2 Full Quote",
    description=(
        "Accepts the expanded V2 request (20 fields). "
        "Auto-enriches 11 features from lookup tables and driving-score algorithm. "
        "Runs ML prediction via V1 champion (V4 pending Phase 3 activation). "
        "Returns V1-compatible premium fields plus enriched_features audit and champion_metadata."
    ),
)
async def get_quote_v2(
    request: CustomerQuoteRequestV2,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    t_start = time.monotonic()

    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Run training pipeline first.",
        )

    engine = get_pricing_engine()

    # ── Feature enrichment ────────────────────────────────────────────────────
    enrichment = enrich_v2_request(request)
    meta     = enrichment["enrichment_meta"]
    payload  = enrichment["inference_payload"]

    effective_driving_score = meta["driving_score"]["score"]

    # ── ML prediction (V1 predictor, backward-compatible payload) ─────────────
    ml_result = predictor.predict(payload)

    # ── Pricing engine ────────────────────────────────────────────────────────
    pricing_result = engine.calculate(
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        age=request.age,
        driving_score=effective_driving_score,
        previous_claims_count=request.previous_claims_count,
        annual_mileage_km=request.annual_mileage_km,
        vehicle_value_inr=request.vehicle_value_inr,
        no_claim_bonus_pct=meta["no_claim_bonus_pct"],
    )

    # ── DB persistence ────────────────────────────────────────────────────────
    v_meta = meta["vehicle"]
    c_meta = meta["city"]

    customer = Customer(
        customer_id=uuid.uuid4(),
        age=request.age,
        gender=request.gender.value,
        city=request.city,
        occupation=request.occupation.value,
        marital_status=request.marital_status.value,
        annual_income_band=request.annual_income_band.value,
    )
    db.add(customer)
    await db.flush()

    vehicle = Vehicle(
        vehicle_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        car_brand=request.vehicle_make,     # V1 DB column — store make here
        car_model=request.vehicle_model,    # V1 DB column
        engine_cc=request.engine_cc,
        vehicle_age_years=request.vehicle_age_years,
        vehicle_value_inr=request.vehicle_value_inr,
        fuel_type=request.fuel_type.value,
        vehicle_usage_type=request.vehicle_usage_type.value,
        parking_type=request.parking_type.value,
        vehicle_safety_rating=v_meta["vehicle_safety_rating"],
        airbags_count=v_meta["airbags_count"],
        vehicle_body_style=v_meta["vehicle_body_style"],
        repair_cost_band=v_meta["repair_cost_band"],
    )
    db.add(vehicle)

    profile = DrivingProfile(
        profile_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        driving_score=effective_driving_score,
        annual_mileage_km=request.annual_mileage_km,
        previous_claims_count=request.previous_claims_count,
        years_licensed=request.years_licensed,
        months_since_last_claim=request.months_since_last_claim,
        policy_tenure_years=request.policy_tenure_years,
        no_claim_bonus_pct=meta["no_claim_bonus_pct"],
        region_risk_category=c_meta["region_risk_category"],
        flood_risk_index=c_meta["flood_risk_index"],
        theft_risk_index=c_meta["theft_risk_index"],
        monsoon_exposure_index=c_meta["monsoon_exposure_index"],
        pincode_risk_score=c_meta["pincode_risk_score"],
        policy_inception_month=meta["policy_inception_month"],
    )
    db.add(profile)

    policy = Policy(
        policy_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        vehicle_id=vehicle.vehicle_id,
        premium_amount_inr=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        model_version=get_active_model_version(),
        is_active=False,
    )
    db.add(policy)
    await db.flush()

    prediction_row = Prediction(
        prediction_id=uuid.uuid4(),
        policy_id=policy.policy_id,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        final_premium_inr=pricing_result.final_premium,
        explanation_json=pricing_result.to_dict(),
        inference_features_json=payload,
    )
    db.add(prediction_row)
    await db.flush()

    # ── Build response ────────────────────────────────────────────────────────
    explanation = ExplanationDetail(
        base_premium=pricing_result.base_premium,
        expected_loss=pricing_result.expected_loss,
        adjustments=[
            AdjustmentDetail(reason=a.reason, factor=a.factor, impact_inr=a.impact_inr)
            for a in pricing_result.adjustments
        ],
        final_premium=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        claim_probability=ml_result["claim_probability"],
        summary=pricing_result.summary,
    )

    # ── Shadow deployment recording (non-blocking) ────────────────────────────
    # Records champion prediction for future challenger comparison.
    # Customer response is never delayed by this task.
    champion_version = get_active_model_version()
    latency_ms = int((time.monotonic() - t_start) * 1000)
    shadow_payload = {
        "claim_probability": ml_result["claim_probability"],
        "expected_claim_amount_inr": ml_result["expected_claim_amount_inr"],
        "final_premium_inr": pricing_result.final_premium,
        "risk_level": pricing_result.risk_level,
    }
    background_tasks.add_task(
        ShadowDeploymentService.record_background,
        quote_id=str(prediction_row.prediction_id),
        champion_prediction=shadow_payload,
        champion_model_name="XGBoost_V4+CatBoost_V4",
        champion_version=champion_version,
        latency_ms=latency_ms,
    )

    return QuoteResponseV2(
        prediction_id=prediction_row.prediction_id,
        premium_amount_inr=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        explanation=explanation,
        model_version=champion_version,
        created_at=prediction_row.created_at or datetime.utcnow(),
        driving_score_info=DrivingScoreResult(
            score=meta["driving_score"]["score"],
            factors=meta["driving_score"]["factors"],
        ),
        enriched_features=_build_enriched_features(meta),
        champion_metadata=_build_champion_metadata(),
    )


# ─── POST /api/v2/quote/preview ───────────────────────────────────────────────

@router.post(
    "/quote/preview",
    response_model=QuotePreviewResponseV2,
    status_code=status.HTTP_200_OK,
    summary="V2 Enrichment Preview",
    description=(
        "Validates V2 request, runs feature enrichment, and returns enriched features "
        "WITHOUT executing ML prediction or persisting to DB. "
        "Use this to verify lookup hits and enriched inputs before calling /quote."
    ),
)
async def preview_quote_v2(request: CustomerQuoteRequestV2):
    enrichment = enrich_v2_request(request)
    meta = enrichment["enrichment_meta"]

    request_summary = {
        "age":              request.age,
        "gender":           request.gender.value,
        "city":             request.city,
        "vehicle_make":     request.vehicle_make,
        "vehicle_model":    request.vehicle_model,
        "vehicle_age_years": request.vehicle_age_years,
        "engine_cc":        request.engine_cc,
        "fuel_type":        request.fuel_type.value,
        "annual_mileage_km": request.annual_mileage_km,
        "previous_claims_count": request.previous_claims_count,
        "months_since_last_claim": request.months_since_last_claim,
        "no_claim_bonus_pct": meta["no_claim_bonus_pct"],  # derived, not from request
        "policy_tenure_years": request.policy_tenure_years,
    }

    return QuotePreviewResponseV2(
        enriched_features=_build_enriched_features(meta),
        champion_metadata=_build_champion_metadata(),
        request_summary=request_summary,
    )
