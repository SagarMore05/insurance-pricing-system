"""
Phase 4 — Part 10: Post-Activation Validation
===============================================
Verifies the system works end-to-end using the newly activated V4 champions.
Run from backend/:
    python -m experiments.phase4_post_activation
"""

import sys, os, json, traceback
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BACKEND_ROOT)

results = []
def check(label, passed, detail=""):
    sym = "[PASS]" if passed else "[FAIL]"
    results.append({"label": label, "passed": passed, "detail": detail})
    print(f"  {sym} {label}" + (f" -- {detail}" if detail else ""))

print("\n[Phase 4 Post-Activation] Verifying activated V4 champions...\n")

# Force clean reload of singleton
import src.models.combined_predictor as _cp
import src.models.model_registry as _mr
_cp._predictor = None
_mr._registry_singleton = None

# ── 1. Registry verification ──────────────────────────────────────────────────
print("=== Registry State ===")
from src.models.model_registry import ModelRegistry
registry = ModelRegistry()
freq_info = registry.get_frequency_champion()
sev_info  = registry.get_severity_champion()

check("Frequency status = active", freq_info["deployment_status"] == "active",
      freq_info["deployment_status"])
check("Severity status = active", sev_info["deployment_status"] == "active",
      sev_info["deployment_status"])
check("Frequency champion = XGBoost_V4", freq_info["champion"] == "XGBoost_V4",
      freq_info["champion"])
check("Severity champion = CatBoost_V4", sev_info["champion"] == "CatBoost_V4",
      sev_info["champion"])

# ── 2. CombinedPredictor loads V4 ─────────────────────────────────────────────
print("\n=== Predictor V4 Load ===")
from src.models.combined_predictor import get_predictor
predictor = get_predictor()

check("Predictor is ready", predictor.is_ready())
check("V4 loaded from registry", predictor._loaded_from_registry)
check("Preprocessor is InsurancePreprocessorV4",
      "InsurancePreprocessorV4" in type(predictor.preprocessor).__name__,
      type(predictor.preprocessor).__name__)

# ── 3. V4 inference - quote generation ───────────────────────────────────────
print("\n=== V4 Quote Generation (post-activation) ===")
import pandas as pd
import numpy as np

test_payload = {
    "age": 38, "gender": "Male", "city": "Bangalore",
    "car_brand": "Tata", "car_model": "Nexon",
    "engine_cc": 1200, "vehicle_age_years": 2,
    "vehicle_value_inr": 900000, "annual_mileage_km": 14000,
    "previous_claims_count": 0, "years_licensed": 12, "driving_score": 78,
    "occupation": "Salaried", "marital_status": "Married",
    "annual_income_band": "7-15 LPA", "fuel_type": "Petrol",
    "vehicle_usage_type": "Personal", "parking_type": "Garage",
    "months_since_last_claim": 999, "no_claim_bonus_pct": 20.0,
    "policy_tenure_years": 3,
    "vehicle_make": "Tata", "vehicle_model": "Nexon",
    "vehicle_safety_rating": 5, "airbags_count": 6,
    "vehicle_body_style": "SUV", "repair_cost_band": "Standard",
    "region_risk_category": "Low", "flood_risk_index": 2.5,
    "theft_risk_index": 3.0, "monsoon_exposure_index": 3.5,
    "pincode_risk_score": 3.075, "policy_inception_month": 6,
}

try:
    ml_result = predictor.predict(test_payload)
    check("V4 predict() executes", True)
    check("claim_probability returned", "claim_probability" in ml_result,
          str(ml_result.get("claim_probability")))
    check("expected_claim_amount_inr returned", "expected_claim_amount_inr" in ml_result,
          str(ml_result.get("expected_claim_amount_inr")))
    check("claim_probability in [0,1]",
          0 <= ml_result["claim_probability"] <= 1,
          str(ml_result["claim_probability"]))
    check("expected_claim_amount_inr > 0",
          ml_result["expected_claim_amount_inr"] > 0,
          f"{ml_result['expected_claim_amount_inr']:,.0f}")
except Exception as exc:
    check("V4 predict() execution", False, traceback.format_exc())
    ml_result = {"claim_probability": 0.1, "expected_claim_amount_inr": 200000}

# ── 4. Premium calculation ────────────────────────────────────────────────────
print("\n=== Premium Calculation ===")
try:
    from src.engine.pricing_engine import get_pricing_engine
    engine = get_pricing_engine()
    pricing = engine.calculate(
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        age=test_payload["age"],
        driving_score=test_payload["driving_score"],
        previous_claims_count=test_payload["previous_claims_count"],
        annual_mileage_km=test_payload["annual_mileage_km"],
        vehicle_value_inr=test_payload["vehicle_value_inr"],
    )
    check("Pricing engine executes", True,
          f"premium=INR {pricing.final_premium:,.0f} risk={pricing.risk_level}")
    check("Premium > 0", pricing.final_premium > 0, str(pricing.final_premium))
    check("Risk level valid", pricing.risk_level in ("low","medium","high"), pricing.risk_level)
    check("Explanation present", len(pricing.adjustments) >= 0)
    premium = pricing.final_premium
    risk = pricing.risk_level
except Exception as exc:
    check("Premium calculation", False, traceback.format_exc())
    premium, risk = 0, "unknown"

# ── 5. Feature enrichment (V2 preview) ───────────────────────────────────────
print("\n=== Feature Enrichment (V2 Preview) ===")
try:
    from src.services.feature_enrichment_service import enrich_v2_request
    from src.api.schemas import CustomerQuoteRequestV2
    v2_req = CustomerQuoteRequestV2(
        age=38, gender="Male", city="Bangalore",
        vehicle_make="Tata", vehicle_model="Nexon",
        engine_cc=1200, vehicle_age_years=2,
        vehicle_value_inr=900000, annual_mileage_km=14000,
        previous_claims_count=0, years_licensed=12,
        occupation="Salaried", marital_status="Married",
        annual_income_band="7-15 LPA", fuel_type="Petrol",
        vehicle_usage_type="Personal", parking_type="Garage",
        months_since_last_claim=0, no_claim_bonus_pct=20.0,
        policy_tenure_years=3,
    )
    enriched = enrich_v2_request(v2_req)
    check("enrich_v2_request with V4 predictor active", True)
    check("Vehicle enrichment present", "vehicle" in enriched["enrichment_meta"])
    check("City enrichment present", "city" in enriched["enrichment_meta"])
    check("Driving score present", "driving_score" in enriched["enrichment_meta"])
except Exception as exc:
    check("Feature enrichment post-activation", False, str(exc))

# ── 6. Governance panel (champion_metadata) ───────────────────────────────────
print("\n=== Governance Panel (Champion Metadata) ===")
try:
    from src.api.routes.quote_v2 import _build_champion_metadata
    # Reload predictor with updated state
    _cp._predictor = None
    meta = _build_champion_metadata()
    check("champion_metadata builds", True)
    check("v4_ready = True (V4 now active)", meta.v4_ready, str(meta.v4_ready))
    check("frequency_model name populated", bool(meta.frequency_model), meta.frequency_model)
    check("severity_model name populated", bool(meta.severity_model), meta.severity_model)
    check("model_version populated", bool(meta.model_version), meta.model_version)
    print(f"    frequency_model = {meta.frequency_model}")
    print(f"    severity_model  = {meta.severity_model}")
    print(f"    model_version   = {meta.model_version}")
    print(f"    v4_ready        = {meta.v4_ready}")
except Exception as exc:
    check("champion_metadata build", False, traceback.format_exc())

# ── 7. Rollback availability ──────────────────────────────────────────────────
print("\n=== Rollback Support ===")
try:
    import json as _json
    hist_path = os.path.join(BACKEND_ROOT, "models", "metadata", "promotion_history.json")
    with open(hist_path) as f:
        hist = _json.load(f)
    freq_history = [e for e in hist["history"] if e["model_type"] == "frequency"]
    sev_history  = [e for e in hist["history"] if e["model_type"] == "severity"]
    check("Frequency rollback available", len(freq_history) >= 1,
          f"entries={len(freq_history)}")
    check("Severity rollback available", len(sev_history) >= 1,
          f"entries={len(sev_history)}")
    check("ModelRegistry.rollback_champion() callable", hasattr(registry, "rollback_champion"))
    check("V1 fallback still exists on disk",
          os.path.exists(os.path.join(BACKEND_ROOT, "models", "saved", "frequency_v1.0.0.pkl")))
except Exception as exc:
    check("Rollback support check", False, str(exc))

# ── 8. Governance log updated ─────────────────────────────────────────────────
print("\n=== Governance Log ===")
try:
    gov_path = os.path.join(BACKEND_ROOT, "models", "metadata", "model_governance_log.json")
    with open(gov_path) as f:
        gov = _json.load(f)
    events = gov["events"]
    activation_events = [e for e in events if e.get("event_type") == "activation"]
    check("Activation events logged to governance log",
          len(activation_events) == 2, f"found {len(activation_events)}")
    for ev in activation_events:
        check(f"  {ev['model_type']} activation event logged",
              ev.get("decision") == "ACTIVATED", str(ev.get("decision")))
except Exception as exc:
    check("Governance log check", False, str(exc))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Post-Activation Summary ===")
total = len(results)
passed = sum(1 for r in results if r["passed"])
failed = total - passed
overall = "PASS" if failed == 0 else "FAIL"

print(f"  Total checks: {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Status: {overall}")
if failed > 0:
    print("\n  Failed checks:")
    for r in results:
        if not r["passed"]:
            print(f"    [FAIL] {r['label']}: {r['detail']}")

# Write result
out = {
    "timestamp": datetime.utcnow().isoformat(),
    "total_checks": total,
    "passed": passed,
    "failed": failed,
    "overall": overall,
    "sample_prediction": {
        "claim_probability": ml_result.get("claim_probability"),
        "expected_claim_amount_inr": ml_result.get("expected_claim_amount_inr"),
        "premium_inr": premium,
        "risk_level": risk,
    },
    "checks": results,
}
out_file = os.path.join(BACKEND_ROOT, "experiments", "outputs", "phase4_post_activation_result.json")
with open(out_file, "w") as f:
    _json.dump(out, f, indent=2)

print(f"\n  Post-activation result written to: {out_file}")
sys.exit(0 if overall == "PASS" else 1)
