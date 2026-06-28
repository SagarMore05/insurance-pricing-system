"""
feature_enrichment_service.py
==============================
Enriches a CustomerQuoteRequestV2 with auto-derived features before
passing the payload to the ML predictor.

Three enrichment sources:
  1. vehicle_specs.json  → vehicle_safety_rating, airbags_count,
                            vehicle_body_style, repair_cost_band
  2. city_geo_risk.json  → region_risk_category, flood_risk_index,
                            theft_risk_index, monsoon_exposure_index,
                            pincode_risk_score
  3. driving_score.py    → driving_score, driving_score_factors

System-derived:
  4. policy_inception_month  → current calendar month
  5. no_claim_bonus_pct      → auto-derived from claims + tenure (NOT user-supplied)

V4 NOT activated. This service prepares enriched payloads for:
  - V2 response audit (always)
  - V4 model inference (when V4 champions become active in Phase 3)

months_since_last_claim mapping:
  API value 0  → V4 internal sentinel 999 (= "never claimed")
  API value 1+ → passed through unchanged

NCB derivation rules (Indian motor insurance standard):
  previous_claims_count > 0  → NCB = 0  (any claim resets bonus)
  claims = 0, tenure = 1 yr  → NCB = 20%
  claims = 0, tenure = 2 yr  → NCB = 25%
  claims = 0, tenure = 3 yr  → NCB = 35%
  claims = 0, tenure = 4 yr  → NCB = 45%
  claims = 0, tenure >= 5 yr → NCB = 50%
"""

import logging
from datetime import datetime
from typing import Any, Dict

from src.services.lookup_loader import (
    get_vehicle_specs,
    get_city_risk,
    CITY_DEFAULTS,
    _load_vehicle_specs,
    _load_city_risk,
)
from src.engine.driving_score import calculate_driving_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pincode risk score formula
# (same formula as used during V4 model training; indices are 1-10 scale)
# ---------------------------------------------------------------------------

def _compute_pincode_risk_score(
    flood: float,
    theft: float,
    monsoon: float,
) -> float:
    """0.35×flood + 0.40×theft + 0.25×monsoon  (result on ~1-10 scale)."""
    return round(0.35 * flood + 0.40 * theft + 0.25 * monsoon, 4)


# ---------------------------------------------------------------------------
# NCB derivation
# Customers do NOT enter NCB; it is derived entirely from claim history
# and policy tenure, matching Indian motor insurance standard slab rates.
# The V4 preprocessor receives No_Claim_Bonus_Pct as a numeric feature.
# ---------------------------------------------------------------------------

def derive_ncb_pct(previous_claims_count: int, policy_tenure_years: float) -> float:
    """
    Derive No-Claim Bonus percentage from claims count and policy tenure.

    Returns one of: 0.0, 20.0, 25.0, 35.0, 45.0, 50.0
    """
    if previous_claims_count > 0:
        return 0.0
    tenure = int(policy_tenure_years)  # floor to completed years
    if tenure <= 0:
        return 0.0
    if tenure == 1:
        return 20.0
    if tenure == 2:
        return 25.0
    if tenure == 3:
        return 35.0
    if tenure == 4:
        return 45.0
    return 50.0  # tenure >= 5


# ---------------------------------------------------------------------------
# Public enrichment function
# ---------------------------------------------------------------------------

def enrich_v2_request(request) -> Dict[str, Any]:
    """
    Accept a CustomerQuoteRequestV2 and return a fully enriched feature dict
    ready for ML inference.

    Parameters
    ----------
    request : CustomerQuoteRequestV2

    Returns
    -------
    dict with keys:
      enrichment_meta     — audit dict (vehicle_hit, city_hit, driving_score_info)
      inference_payload   — complete dict for predictor.predict()
    """
    # ── 1. Vehicle specs enrichment ─────────────────────────────────────────
    v_specs = get_vehicle_specs(request.vehicle_make, request.vehicle_model)

    # Direct table presence check avoids false miss when a real vehicle's specs
    # happen to equal VEHICLE_DEFAULTS (e.g. Hyundai i20 matches all 4 defaults)
    try:
        _vtable = _load_vehicle_specs()
        _make_key = (request.vehicle_make or "").strip()
        _model_key = (request.vehicle_model or "").strip()
        vehicle_hit = bool(
            _make_key
            and _model_key
            and _make_key in _vtable
            and _model_key in _vtable[_make_key]
        )
    except Exception:
        vehicle_hit = False

    logger.debug(
        "Vehicle enrichment: make=%r model=%r hit=%s specs=%s",
        request.vehicle_make, request.vehicle_model, vehicle_hit, v_specs,
    )

    # ── 2. City geo risk enrichment ──────────────────────────────────────────
    c_risk = get_city_risk(request.city)

    try:
        _ctable = _load_city_risk()
        _city_key = (request.city or "").strip()
        city_hit = bool(
            _city_key
            and (
                _city_key in _ctable
                or _city_key.title() in _ctable
            )
        )
    except Exception:
        city_hit = False

    flood   = c_risk["flood_risk_index"]
    theft   = c_risk["theft_risk_index"]
    monsoon = c_risk["monsoon_exposure_index"]
    pincode_risk_score = _compute_pincode_risk_score(flood, theft, monsoon)

    logger.debug(
        "City enrichment: city=%r hit=%s risk=%s pincode_score=%.4f",
        request.city, city_hit, c_risk, pincode_risk_score,
    )

    # ── 3. NCB auto-derivation (computed before driving score so it can be used as input) ──
    # Customers never enter NCB directly. System derives it from claim history
    # and policy tenure using standard Indian motor insurance slab rates.
    derived_ncb = derive_ncb_pct(
        previous_claims_count=request.previous_claims_count,
        policy_tenure_years=request.policy_tenure_years,
    )

    # ── 4. Driving score (enriched with full V2 profile) ─────────────────────
    ds_result = calculate_driving_score(
        annual_mileage=request.annual_mileage_km,
        prior_claims=request.previous_claims_count,
        years_licensed=request.years_licensed,
        age=request.age,
        occupation=request.occupation.value,
        marital_status=request.marital_status.value,
        vehicle_age=request.vehicle_age_years,
        vehicle_value=request.vehicle_value_inr,
        fuel_type=request.fuel_type.value,
        vehicle_segment=v_specs.get("vehicle_body_style"),
        vehicle_usage_type=request.vehicle_usage_type.value,
        parking_type=request.parking_type.value,
        ncb_pct=derived_ncb,
        city_risk_category=c_risk["region_risk_category"],
    )
    driving_score = ds_result["score"]

    # ── 5. months_since_last_claim sentinel mapping ──────────────────────────
    # API: 0 = never claimed  →  V4 preprocessor sentinel: 999
    months_internal = 999 if request.months_since_last_claim == 0 else request.months_since_last_claim

    # ── 6. Policy inception month ────────────────────────────────────────────
    policy_inception_month = datetime.utcnow().month

    logger.debug(
        "NCB derivation: claims=%d tenure=%.1f → ncb=%.1f%%",
        request.previous_claims_count, request.policy_tenure_years, derived_ncb,
    )

    # ── Build enrichment metadata (for QuoteResponseV2 audit) ────────────────
    enrichment_meta = {
        "vehicle": {
            "vehicle_safety_rating": v_specs["vehicle_safety_rating"],
            "airbags_count":         v_specs["airbags_count"],
            "vehicle_body_style":    v_specs["vehicle_body_style"],
            "repair_cost_band":      v_specs["repair_cost_band"],
            "lookup_hit":            vehicle_hit,
        },
        "city": {
            "region_risk_category":   c_risk["region_risk_category"],
            "flood_risk_index":        flood,
            "theft_risk_index":        theft,
            "monsoon_exposure_index":  monsoon,
            "pincode_risk_score":      pincode_risk_score,
            "lookup_hit":              city_hit,
        },
        "driving_score": {
            "score":   driving_score,
            "factors": ds_result["factors"],
        },
        "policy_inception_month": policy_inception_month,
        "no_claim_bonus_pct":     derived_ncb,
    }

    # ── Build complete inference payload ─────────────────────────────────────
    # Maps V2 fields → V1 predictor field names (car_brand, car_model)
    # and V4 field names for when V4 is activated.
    inference_payload = {
        # V1 predictor fields (backward compat)
        "age":                    request.age,
        "gender":                 request.gender.value,
        "city":                   request.city,
        "car_brand":              request.vehicle_make,     # V1 alias
        "car_model":              request.vehicle_model,    # V1 alias
        "engine_cc":              request.engine_cc,
        "vehicle_age_years":      request.vehicle_age_years,
        "vehicle_value_inr":      request.vehicle_value_inr,
        "annual_mileage_km":      request.annual_mileage_km,
        "previous_claims_count":  request.previous_claims_count,
        "years_licensed":         request.years_licensed,
        "driving_score":          driving_score,

        # V4-specific fields (new in V2)
        "vehicle_make":           request.vehicle_make,
        "vehicle_model":          request.vehicle_model,
        "occupation":             request.occupation.value,
        "marital_status":         request.marital_status.value,
        "annual_income_band":     request.annual_income_band.value,
        "fuel_type":              request.fuel_type.value,
        "vehicle_usage_type":     request.vehicle_usage_type.value,
        "parking_type":           request.parking_type.value,
        "months_since_last_claim": months_internal,
        "no_claim_bonus_pct":     derived_ncb,   # system-derived, NOT from request
        "policy_tenure_years":    request.policy_tenure_years,

        # Lookup-derived (V4 STD_COLS / OHE_COLS / PASS_COLS)
        "vehicle_safety_rating":  v_specs["vehicle_safety_rating"],
        "airbags_count":          v_specs["airbags_count"],
        "vehicle_body_style":     v_specs["vehicle_body_style"],
        "repair_cost_band":       v_specs["repair_cost_band"],
        "region_risk_category":   c_risk["region_risk_category"],
        "flood_risk_index":        flood,
        "theft_risk_index":        theft,
        "monsoon_exposure_index":  monsoon,
        "pincode_risk_score":      pincode_risk_score,

        # System-derived
        "policy_inception_month": policy_inception_month,
    }

    return {
        "enrichment_meta":    enrichment_meta,
        "inference_payload":  inference_payload,
    }
