"""
api_compatibility_tests.py
==========================
Phase 2 API compatibility tests.

Sections:
  1. V2 schema validation (CustomerQuoteRequestV2)
  2. Feature enrichment service unit tests
  3. V2/V1 schema compatibility (QuoteResponseV2 superset of PremiumQuoteResponse)
  4. Enum coverage tests
  5. Preview endpoint contract tests
  6. Months-since-last-claim sentinel mapping
  7. Lookup hit / miss propagation through enrichment

Run:
    cd backend
    python -m tests.api_compatibility_tests

All sections print PASS/FAIL per assertion. Final summary shows total counts.
"""

import sys
import traceback
from datetime import datetime

# ─── Test helpers ─────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
ERRORS = []


def check(label: str, condition: bool):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS  {label}")
    else:
        FAIL_COUNT += 1
        ERRORS.append(label)
        print(f"  FAIL  {label}")


def section(title: str):
    print()
    print(f"{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


# ─── Minimal valid V2 request payload ─────────────────────────────────────────

VALID_PAYLOAD = {
    "age": 35,
    "gender": "Male",
    "occupation": "Salaried",
    "marital_status": "Married",
    "annual_income_band": "7-15 LPA",
    "city": "Mumbai",
    "years_licensed": 10,
    "previous_claims_count": 1,
    "months_since_last_claim": 24,
    "no_claim_bonus_pct": 10.0,
    "policy_tenure_years": 2.0,
    "vehicle_make": "Hyundai",
    "vehicle_model": "Creta",
    "vehicle_age_years": 3,
    "vehicle_value_inr": 1_200_000.0,
    "engine_cc": 1500,
    "fuel_type": "Petrol",
    "vehicle_usage_type": "Personal",
    "annual_mileage_km": 15000,
    "parking_type": "Garage",
}


# ─── Section 1: V2 Schema Validation ──────────────────────────────────────────

def test_v2_schema_validation():
    section("SECTION 1 — V2 Schema Validation")

    from src.api.schemas import CustomerQuoteRequestV2

    # Valid payload
    try:
        r = CustomerQuoteRequestV2(**VALID_PAYLOAD)
        check("Valid full payload is accepted", True)
        check("city is title-cased: 'Mumbai'", r.city == "Mumbai")
        check("vehicle_make preserved: 'Hyundai'", r.vehicle_make == "Hyundai")
        check("vehicle_model preserved: 'Creta'", r.vehicle_model == "Creta")
        check("gender is GenderV2 enum", r.gender.value == "Male")
        check("fuel_type is FuelType enum", r.fuel_type.value == "Petrol")
        check("parking_type is ParkingType enum", r.parking_type.value == "Garage")
    except Exception as e:
        check("Valid full payload is accepted", False)
        print(f"    Exception: {e}")

    # Missing required field
    import pydantic
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "occupation"}
    try:
        CustomerQuoteRequestV2(**incomplete)
        check("Missing occupation raises ValidationError", False)
    except pydantic.ValidationError:
        check("Missing occupation raises ValidationError", True)

    # Age out of range
    bad_age = dict(VALID_PAYLOAD, age=17)
    try:
        CustomerQuoteRequestV2(**bad_age)
        check("age=17 raises ValidationError (ge=18)", False)
    except pydantic.ValidationError:
        check("age=17 raises ValidationError (ge=18)", True)

    # months_since_last_claim=0 accepted (never claimed)
    zero_months = dict(VALID_PAYLOAD, months_since_last_claim=0)
    try:
        r2 = CustomerQuoteRequestV2(**zero_months)
        check("months_since_last_claim=0 accepted (never claimed)", r2.months_since_last_claim == 0)
    except Exception as e:
        check("months_since_last_claim=0 accepted (never claimed)", False)
        print(f"    Exception: {e}")

    # city normalisation — lowercase input → title-cased
    lower_city = dict(VALID_PAYLOAD, city="  mumbai  ")
    try:
        r3 = CustomerQuoteRequestV2(**lower_city)
        check("city='  mumbai  ' normalised to 'Mumbai'", r3.city == "Mumbai")
    except Exception as e:
        check("city='  mumbai  ' normalised to 'Mumbai'", False)
        print(f"    Exception: {e}")

    # vehicle_make strip
    padded_make = dict(VALID_PAYLOAD, vehicle_make="  Hyundai  ")
    try:
        r4 = CustomerQuoteRequestV2(**padded_make)
        check("vehicle_make stripped of whitespace", r4.vehicle_make == "Hyundai")
    except Exception as e:
        check("vehicle_make stripped of whitespace", False)
        print(f"    Exception: {e}")

    # Invalid enum value
    bad_fuel = dict(VALID_PAYLOAD, fuel_type="LPG")
    try:
        CustomerQuoteRequestV2(**bad_fuel)
        check("Invalid fuel_type='LPG' raises ValidationError", False)
    except pydantic.ValidationError:
        check("Invalid fuel_type='LPG' raises ValidationError", True)

    # vehicle_value_inr > 0 constraint
    zero_value = dict(VALID_PAYLOAD, vehicle_value_inr=0)
    try:
        CustomerQuoteRequestV2(**zero_value)
        check("vehicle_value_inr=0 raises ValidationError (gt=0)", False)
    except pydantic.ValidationError:
        check("vehicle_value_inr=0 raises ValidationError (gt=0)", True)


# ─── Section 2: Feature Enrichment Service ────────────────────────────────────

def test_feature_enrichment_service():
    section("SECTION 2 — Feature Enrichment Service")

    from src.api.schemas import CustomerQuoteRequestV2
    from src.services.feature_enrichment_service import enrich_v2_request

    r = CustomerQuoteRequestV2(**VALID_PAYLOAD)
    result = enrich_v2_request(r)

    check("enrich_v2_request returns dict with 'enrichment_meta'", "enrichment_meta" in result)
    check("enrich_v2_request returns dict with 'inference_payload'", "inference_payload" in result)

    meta    = result["enrichment_meta"]
    payload = result["inference_payload"]

    # enrichment_meta structure
    check("meta has 'vehicle' key", "vehicle" in meta)
    check("meta has 'city' key", "city" in meta)
    check("meta has 'driving_score' key", "driving_score" in meta)
    check("meta has 'policy_inception_month' key", "policy_inception_month" in meta)

    # vehicle enrichment — Hyundai Creta is a known vehicle
    v = meta["vehicle"]
    check("Hyundai Creta: lookup_hit=True", v["lookup_hit"] is True)
    check("Hyundai Creta: vehicle_safety_rating is int", isinstance(v["vehicle_safety_rating"], int))
    check("Hyundai Creta: airbags_count >= 2", v["airbags_count"] >= 2)
    check("Hyundai Creta: vehicle_body_style='SUV'", v["vehicle_body_style"] == "SUV")
    check("Hyundai Creta: repair_cost_band='Standard'", v["repair_cost_band"] == "Standard")

    # city enrichment — Mumbai is known
    c = meta["city"]
    check("Mumbai: lookup_hit=True", c["lookup_hit"] is True)
    check("Mumbai: region_risk_category='High'", c["region_risk_category"] == "High")
    check("Mumbai: flood_risk_index in [1,10]", 1.0 <= c["flood_risk_index"] <= 10.0)
    check("Mumbai: theft_risk_index in [1,10]", 1.0 <= c["theft_risk_index"] <= 10.0)
    check("Mumbai: monsoon_exposure_index in [1,10]", 1.0 <= c["monsoon_exposure_index"] <= 10.0)
    check("Mumbai: pincode_risk_score is float", isinstance(c["pincode_risk_score"], float))
    check("Mumbai: pincode_risk_score in [1,10]", 1.0 <= c["pincode_risk_score"] <= 10.0)

    # driving score
    d = meta["driving_score"]
    check("driving_score.score is int", isinstance(d["score"], int))
    check("driving_score.score in [0,100]", 0 <= d["score"] <= 100)
    check("driving_score.factors is list", isinstance(d["factors"], list))

    # policy inception month
    pim = meta["policy_inception_month"]
    check("policy_inception_month is current month", pim == datetime.utcnow().month)

    # inference_payload field presence
    required_v1_fields = ["age", "gender", "city", "car_brand", "car_model", "engine_cc",
                          "vehicle_age_years", "vehicle_value_inr", "annual_mileage_km",
                          "previous_claims_count", "years_licensed", "driving_score"]
    for f in required_v1_fields:
        check(f"inference_payload has V1 field '{f}'", f in payload)

    required_v4_fields = ["vehicle_make", "vehicle_model", "occupation", "marital_status",
                          "annual_income_band", "fuel_type", "vehicle_usage_type", "parking_type",
                          "months_since_last_claim", "no_claim_bonus_pct", "policy_tenure_years",
                          "vehicle_safety_rating", "airbags_count", "vehicle_body_style",
                          "repair_cost_band", "region_risk_category", "flood_risk_index",
                          "theft_risk_index", "monsoon_exposure_index", "pincode_risk_score",
                          "policy_inception_month"]
    for f in required_v4_fields:
        check(f"inference_payload has V4 field '{f}'", f in payload)

    # V1 alias check
    check("car_brand == vehicle_make in payload", payload["car_brand"] == r.vehicle_make)
    check("car_model == vehicle_model in payload", payload["car_model"] == r.vehicle_model)

    # driving_score in payload matches meta
    check("payload driving_score matches meta", payload["driving_score"] == meta["driving_score"]["score"])


# ─── Section 3: months_since_last_claim sentinel mapping ──────────────────────

def test_months_sentinel_mapping():
    section("SECTION 3 — months_since_last_claim Sentinel Mapping")

    from src.api.schemas import CustomerQuoteRequestV2
    from src.services.feature_enrichment_service import enrich_v2_request

    # 0 → 999 (never claimed)
    r_never = CustomerQuoteRequestV2(**dict(VALID_PAYLOAD, months_since_last_claim=0))
    result = enrich_v2_request(r_never)
    check("months_since_last_claim=0 maps to 999 in inference_payload",
          result["inference_payload"]["months_since_last_claim"] == 999)

    # Non-zero → passed through
    r_12 = CustomerQuoteRequestV2(**dict(VALID_PAYLOAD, months_since_last_claim=12))
    result2 = enrich_v2_request(r_12)
    check("months_since_last_claim=12 passes through as 12",
          result2["inference_payload"]["months_since_last_claim"] == 12)

    r_240 = CustomerQuoteRequestV2(**dict(VALID_PAYLOAD, months_since_last_claim=240))
    result3 = enrich_v2_request(r_240)
    check("months_since_last_claim=240 (max) passes through as 240",
          result3["inference_payload"]["months_since_last_claim"] == 240)


# ─── Section 4: Lookup hit/miss propagation ────────────────────────────────────

def test_lookup_hit_miss():
    section("SECTION 4 — Lookup Hit / Miss Propagation")

    from src.api.schemas import CustomerQuoteRequestV2
    from src.services.feature_enrichment_service import enrich_v2_request
    from src.services.lookup_loader import VEHICLE_DEFAULTS, CITY_DEFAULTS

    # Import enrich early for the i20 test in section 1 (defined at end of section 1)
    # — already imported here for use below

    # Unknown vehicle → defaults
    unknown_v = dict(VALID_PAYLOAD, vehicle_make="Unknown", vehicle_model="Phantom")
    r = CustomerQuoteRequestV2(**unknown_v)
    result = enrich_v2_request(r)
    v = result["enrichment_meta"]["vehicle"]
    check("Unknown vehicle: vehicle_safety_rating == VEHICLE_DEFAULTS",
          v["vehicle_safety_rating"] == VEHICLE_DEFAULTS["vehicle_safety_rating"])
    check("Unknown vehicle: repair_cost_band == VEHICLE_DEFAULTS",
          v["repair_cost_band"] == VEHICLE_DEFAULTS["repair_cost_band"])

    # Unknown city → defaults
    unknown_c = dict(VALID_PAYLOAD, city="Atlantis")
    r2 = CustomerQuoteRequestV2(**unknown_c)
    result2 = enrich_v2_request(r2)
    c = result2["enrichment_meta"]["city"]
    check("Unknown city: flood_risk_index == CITY_DEFAULTS",
          c["flood_risk_index"] == CITY_DEFAULTS["flood_risk_index"])
    check("Unknown city: region_risk_category == CITY_DEFAULTS",
          c["region_risk_category"] == CITY_DEFAULTS["region_risk_category"])

    # None/empty vehicle_make → defaults
    empty_make = dict(VALID_PAYLOAD, vehicle_make="", vehicle_model="Creta")
    try:
        import pydantic
        CustomerQuoteRequestV2(**empty_make)
        check("Empty vehicle_make rejected by schema (min_length=1)", False)
    except pydantic.ValidationError:
        check("Empty vehicle_make rejected by schema (min_length=1)", True)

    # Hyundai i20 — specs exactly match VEHICLE_DEFAULTS; must still report lookup_hit=True
    from src.services.feature_enrichment_service import enrich_v2_request as _enrich
    r_i20 = CustomerQuoteRequestV2(**dict(VALID_PAYLOAD, vehicle_make="Hyundai", vehicle_model="i20"))
    result_i20 = _enrich(r_i20)
    check("Hyundai i20 (defaults-matching specs): lookup_hit=True",
          result_i20["enrichment_meta"]["vehicle"]["lookup_hit"] is True)


# ─── Section 5: V2/V1 schema compatibility ────────────────────────────────────

def test_schema_compatibility():
    section("SECTION 5 — V2/V1 Schema Compatibility")

    from src.api.schemas import PremiumQuoteResponse, QuoteResponseV2

    v1_fields = set(PremiumQuoteResponse.model_fields.keys())
    v2_fields = set(QuoteResponseV2.model_fields.keys())

    # Every V1 field must exist in V2
    for f in v1_fields:
        check(f"V2 response includes V1 field '{f}'", f in v2_fields)

    # V2-only additions
    v2_only = v2_fields - v1_fields
    check("V2 adds 'enriched_features'", "enriched_features" in v2_only)
    check("V2 adds 'champion_metadata'", "champion_metadata" in v2_only)
    check("V2 adds 'api_version'", "api_version" in v2_only)

    # api_version default
    from src.api.schemas import (
        EnrichedFeaturesV2, VehicleEnrichment, CityEnrichment,
        DrivingScoreEnrichment, ChampionMetadataV2, ExplanationDetail,
        AdjustmentDetail, RiskLevel,
    )
    import uuid as _uuid
    ef = EnrichedFeaturesV2(
        driving_score=DrivingScoreEnrichment(score=75, factors=["good"]),
        vehicle=VehicleEnrichment(
            vehicle_safety_rating=5, airbags_count=6,
            vehicle_body_style="SUV", repair_cost_band="Standard", lookup_hit=True,
        ),
        city=CityEnrichment(
            region_risk_category="High", flood_risk_index=8.5, theft_risk_index=6.0,
            monsoon_exposure_index=9.0, pincode_risk_score=7.38, lookup_hit=True,
        ),
        policy_inception_month=6,
    )
    cm = ChampionMetadataV2(
        frequency_model="XGBClassifier-V1", severity_model="XGBRegressor-V1",
        model_version="v1.0.0", v4_ready=False,
    )
    exp = ExplanationDetail(
        base_premium=10000.0, expected_loss=5000.0,
        adjustments=[AdjustmentDetail(reason="test", factor=1.0, impact_inr=0.0)],
        final_premium=10000.0, risk_level=RiskLevel.medium,
        claim_probability=0.05, summary="test summary",
    )
    resp = QuoteResponseV2(
        prediction_id=_uuid.uuid4(),
        premium_amount_inr=10000.0, risk_level=RiskLevel.medium,
        claim_probability=0.05, expected_claim_amount_inr=5000.0,
        explanation=exp, model_version="v1.0.0",
        created_at=datetime.utcnow(),
        enriched_features=ef, champion_metadata=cm,
    )
    check("QuoteResponseV2 api_version defaults to 'v2'", resp.api_version == "v2")
    check("QuoteResponseV2 instantiation succeeds with all fields", resp is not None)


# ─── Section 6: Enum coverage ─────────────────────────────────────────────────

def test_enum_coverage():
    section("SECTION 6 — Enum Coverage")

    from src.api.schemas import (
        GenderV2, FuelType, VehicleUsageType,
        Occupation, MaritalStatus, AnnualIncomeBand, ParkingType,
    )

    check("GenderV2 has Male", GenderV2("Male") == GenderV2.male)
    check("GenderV2 has Female", GenderV2("Female") == GenderV2.female)
    check("GenderV2 has Other", GenderV2("Other") == GenderV2.other)

    check("FuelType has Petrol", FuelType("Petrol") == FuelType.petrol)
    check("FuelType has Diesel", FuelType("Diesel") == FuelType.diesel)
    check("FuelType has Electric", FuelType("Electric") == FuelType.electric)
    check("FuelType has Hybrid", FuelType("Hybrid") == FuelType.hybrid)

    check("VehicleUsageType has Personal", VehicleUsageType("Personal") == VehicleUsageType.personal)
    check("VehicleUsageType has Commercial", VehicleUsageType("Commercial") == VehicleUsageType.commercial)
    check("VehicleUsageType has Rideshare", VehicleUsageType("Rideshare") == VehicleUsageType.rideshare)

    check("Occupation has Salaried", Occupation("Salaried") == Occupation.salaried)
    check("Occupation has Self-Employed", Occupation("Self-Employed") == Occupation.self_employed)
    check("Occupation has Professional", Occupation("Professional") == Occupation.professional)
    check("Occupation has Retired", Occupation("Retired") == Occupation.retired)
    check("Occupation has Student", Occupation("Student") == Occupation.student)
    check("Occupation has Manual", Occupation("Manual") == Occupation.manual)

    check("MaritalStatus has Single", MaritalStatus("Single") == MaritalStatus.single)
    check("MaritalStatus has Married", MaritalStatus("Married") == MaritalStatus.married)
    check("MaritalStatus has Divorced", MaritalStatus("Divorced") == MaritalStatus.divorced)
    check("MaritalStatus has Widowed", MaritalStatus("Widowed") == MaritalStatus.widowed)

    check("AnnualIncomeBand has <3 LPA", AnnualIncomeBand("<3 LPA") == AnnualIncomeBand.below_3l)
    check("AnnualIncomeBand has 3-7 LPA", AnnualIncomeBand("3-7 LPA") == AnnualIncomeBand.band_3_7)
    check("AnnualIncomeBand has 7-15 LPA", AnnualIncomeBand("7-15 LPA") == AnnualIncomeBand.band_7_15)
    check("AnnualIncomeBand has 15-30 LPA", AnnualIncomeBand("15-30 LPA") == AnnualIncomeBand.band_15_30)
    check("AnnualIncomeBand has >30 LPA", AnnualIncomeBand(">30 LPA") == AnnualIncomeBand.above_30l)

    check("ParkingType has Garage", ParkingType("Garage") == ParkingType.garage)
    check("ParkingType has Street", ParkingType("Street") == ParkingType.street)
    check("ParkingType has Open_Compound", ParkingType("Open_Compound") == ParkingType.open_compound)


# ─── Section 7: V2 request field count ────────────────────────────────────────

def test_v2_request_field_count():
    section("SECTION 7 — V2 Request Field Count")

    from src.api.schemas import CustomerQuoteRequestV2, CustomerQuoteRequest

    v1_field_count = len(CustomerQuoteRequest.model_fields)
    v2_field_count = len(CustomerQuoteRequestV2.model_fields)

    check("CustomerQuoteRequestV2 has exactly 20 fields", v2_field_count == 20)
    check("V2 has more fields than V1", v2_field_count > v1_field_count)
    print(f"    V1 fields: {v1_field_count}  |  V2 fields: {v2_field_count}")

    # Verify all VALID_PAYLOAD keys are model fields
    for key in VALID_PAYLOAD:
        check(f"VALID_PAYLOAD key '{key}' is a V2 model field",
              key in CustomerQuoteRequestV2.model_fields)


# ─── Section 8: pincode_risk_score formula ────────────────────────────────────

def test_pincode_risk_formula():
    section("SECTION 8 — pincode_risk_score Formula Verification")

    from src.services.feature_enrichment_service import _compute_pincode_risk_score

    # Mumbai: flood=8.5, theft=6.0, monsoon=9.0
    # Expected: 0.35*8.5 + 0.40*6.0 + 0.25*9.0 = 2.975 + 2.4 + 2.25 = 7.625
    score = _compute_pincode_risk_score(8.5, 6.0, 9.0)
    check("Mumbai pincode_risk_score == 7.625", abs(score - 7.625) < 0.001)

    # Min boundary: all indices = 1.0 → score = 1.0
    min_score = _compute_pincode_risk_score(1.0, 1.0, 1.0)
    check("Min all-1 pincode_risk_score == 1.0", abs(min_score - 1.0) < 0.001)

    # Max boundary: all indices = 10.0 → score = 10.0
    max_score = _compute_pincode_risk_score(10.0, 10.0, 10.0)
    check("Max all-10 pincode_risk_score == 10.0", abs(max_score - 10.0) < 0.001)

    # Default city: flood=5.0, theft=5.0, monsoon=5.5
    # Expected: 0.35*5 + 0.40*5 + 0.25*5.5 = 1.75 + 2.0 + 1.375 = 5.125
    def_score = _compute_pincode_risk_score(5.0, 5.0, 5.5)
    check("Default city pincode_risk_score == 5.125", abs(def_score - 5.125) < 0.001)


# ─── Main runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    print()
    print("=" * 68)
    print("  API COMPATIBILITY TESTS — Phase 2 V4 Activation")
    print("  Generated: 2026-06-22")
    print("=" * 68)

    sections = [
        test_v2_schema_validation,
        test_feature_enrichment_service,
        test_months_sentinel_mapping,
        test_lookup_hit_miss,
        test_schema_compatibility,
        test_enum_coverage,
        test_v2_request_field_count,
        test_pincode_risk_formula,
    ]

    for fn in sections:
        try:
            fn()
        except Exception as exc:
            print(f"\n  ERROR in {fn.__name__}: {exc}")
            traceback.print_exc()
            FAIL_COUNT += 1
            ERRORS.append(fn.__name__ + " (uncaught exception)")

    print()
    print("=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"  Total assertions : {PASS_COUNT + FAIL_COUNT}")
    print(f"  Passed           : {PASS_COUNT}")
    print(f"  Failed           : {FAIL_COUNT}")
    if ERRORS:
        print()
        print("  FAILED CHECKS:")
        for e in ERRORS:
            print(f"    - {e}")
    print()
    if FAIL_COUNT == 0:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")
        sys.exit(1)
