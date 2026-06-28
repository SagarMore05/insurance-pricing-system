"""
Phase 4 — Part 9: Conditional Champion Activation
===================================================
Only runs if phase4_gate_result.json confirms all_pass=True.
Activates XGBoost V4 (frequency) and CatBoost V4 (severity).

Run from backend/:
    python -m experiments.phase4_activate
"""

import sys, os, json
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, BACKEND_ROOT)

GATE_FILE = os.path.join(BACKEND_ROOT, "experiments", "outputs", "phase4_gate_result.json")

# ── Gate check ─────────────────────────────────────────────────────────────────
print("\n[Phase 4] Checking gate result before activation...")
if not os.path.exists(GATE_FILE):
    print("BLOCKED: gate result file not found. Run phase4_validation.py first.")
    sys.exit(1)

with open(GATE_FILE) as f:
    gate_data = json.load(f)

if not gate_data.get("all_pass"):
    print("BLOCKED: Not all gates passed. Champions will NOT be activated.")
    for gate_name, status in gate_data.get("gates", {}).items():
        print(f"  {gate_name}: {status}")
    sys.exit(1)

print("  All gates PASS. Proceeding with activation.\n")

# ── Activation ─────────────────────────────────────────────────────────────────
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()
activation_log = []
timestamp = datetime.utcnow().isoformat()

print("[Step 1] Activating Frequency Champion: XGBoost_V4")
try:
    freq_before = registry.get_frequency_champion()["deployment_status"]
    registry.activate_champion(
        model_type="frequency",
        notes=(
            "Phase 4 activation — all validation gates passed on 2026-06-22. "
            "XGBoost_V4 ROC-AUC=0.6897 > V1 baseline 0.6877 (+0.002). "
            "Preprocessing: InsurancePreprocessorV4 (59 features). "
            "Activated via phase4_activate.py."
        ),
    )
    freq_after = registry.get_frequency_champion()["deployment_status"]
    ok = freq_after == "active"
    activation_log.append({
        "model_type": "frequency",
        "champion": "XGBoost_V4",
        "from_status": freq_before,
        "to_status": freq_after,
        "success": ok,
        "timestamp": timestamp,
    })
    print(f"  frequency: {freq_before} -> {freq_after} ({'OK' if ok else 'FAILED'})")
except Exception as exc:
    activation_log.append({
        "model_type": "frequency", "success": False, "error": str(exc), "timestamp": timestamp,
    })
    print(f"  ERROR activating frequency: {exc}")

print("[Step 2] Activating Severity Champion: CatBoost_V4")
try:
    sev_before = registry.get_severity_champion()["deployment_status"]
    registry.activate_champion(
        model_type="severity",
        notes=(
            "Phase 4 activation — all validation gates passed on 2026-06-22. "
            "CatBoost_V4 R2=0.5378 > V1 baseline R2=0.5092 (+0.0286). "
            "Preprocessing: InsurancePreprocessorV4 (59 features). "
            "Activated via phase4_activate.py."
        ),
    )
    sev_after = registry.get_severity_champion()["deployment_status"]
    ok = sev_after == "active"
    activation_log.append({
        "model_type": "severity",
        "champion": "CatBoost_V4",
        "from_status": sev_before,
        "to_status": sev_after,
        "success": ok,
        "timestamp": timestamp,
    })
    print(f"  severity: {sev_before} -> {sev_after} ({'OK' if ok else 'FAILED'})")
except Exception as exc:
    activation_log.append({
        "model_type": "severity", "success": False, "error": str(exc), "timestamp": timestamp,
    })
    print(f"  ERROR activating severity: {exc}")

# ── Verify registry update ─────────────────────────────────────────────────────
print("\n[Step 3] Verifying registry update...")
# Force reload (clear singleton)
import src.models.model_registry as _mr
_mr._registry_singleton = None

registry2 = ModelRegistry()
freq_info = registry2.get_frequency_champion()
sev_info  = registry2.get_severity_champion()

freq_active = freq_info["deployment_status"] == "active"
sev_active  = sev_info["deployment_status"] == "active"

print(f"  Frequency champion: {freq_info['champion']} status={freq_info['deployment_status']}")
print(f"  Severity champion:  {sev_info['champion']}  status={sev_info['deployment_status']}")

all_activated = freq_active and sev_active

# ── Verify combined predictor loads V4 on reload ───────────────────────────────
print("\n[Step 4] Verifying CombinedPredictor loads V4 models post-activation...")
import src.models.combined_predictor as _cp
_cp._predictor = None   # force reload

from src.models.combined_predictor import get_predictor
predictor_post = get_predictor()
v4_ready = predictor_post._loaded_from_registry

print(f"  predictor.is_ready()         = {predictor_post.is_ready()}")
print(f"  predictor._loaded_from_registry = {v4_ready}")
if v4_ready:
    print(f"  V4 models ACTIVE in predictor")
else:
    print(f"  NOTE: Predictor using V1 models (V4 loads on next full restart)")

# ── Write activation result ────────────────────────────────────────────────────
activation_result = {
    "timestamp": timestamp,
    "activation_authorized": True,
    "gate_file": GATE_FILE,
    "frequency": {
        "champion": freq_info["champion"],
        "deployment_status": freq_info["deployment_status"],
        "metric": "ROC_AUC",
        "score": freq_info["score"],
        "activated": freq_active,
    },
    "severity": {
        "champion": sev_info["champion"],
        "deployment_status": sev_info["deployment_status"],
        "metric": "R2",
        "score": sev_info["score"],
        "activated": sev_active,
    },
    "v4_ready_in_predictor": v4_ready,
    "all_activated": all_activated,
    "activation_log": activation_log,
}

out_file = os.path.join(BACKEND_ROOT, "experiments", "outputs", "phase4_activation_result.json")
with open(out_file, "w") as f:
    json.dump(activation_result, f, indent=2)

print(f"\n  Activation result written to: {out_file}")
print(f"\n  ACTIVATION STATUS: {'COMPLETE' if all_activated else 'PARTIAL FAILURE'}")
sys.exit(0 if all_activated else 1)
