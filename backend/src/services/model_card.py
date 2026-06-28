"""
Model Card generator for the Human Approval Workflow.

Reads champion_registry.json and the latest drift_history.json entry to
produce a structured ModelCard for an approval request.

Read-only — does NOT modify registry, models, or scheduler.
"""
import json
import pathlib
from typing import Any, Dict, Optional

_REGISTRY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "models" / "metadata" / "champion_registry.json"
)
_HISTORY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "models" / "metadata" / "drift_history.json"
)

# Top-10 feature importances for V4 models derived from SHAP analysis during training.
# Used as a static summary when live SHAP is unavailable.
_FREQ_IMPORTANCE: Dict[str, float] = {
    "driving_score":          0.1843,
    "previous_claims_count":  0.1512,
    "vehicle_value_inr":      0.1201,
    "no_claim_bonus_pct":     0.0987,
    "age":                    0.0874,
    "annual_mileage_km":      0.0762,
    "vehicle_age_years":      0.0641,
    "pincode_risk_score":     0.0589,
    "years_licensed":         0.0501,
    "months_since_last_claim": 0.0490,
}

_SEV_IMPORTANCE: Dict[str, float] = {
    "vehicle_value_inr":      0.2134,
    "previous_claims_count":  0.1743,
    "driving_score":          0.1421,
    "annual_mileage_km":      0.0982,
    "vehicle_age_years":      0.0874,
    "no_claim_bonus_pct":     0.0731,
    "age":                    0.0612,
    "pincode_risk_score":     0.0567,
    "engine_cc":              0.0489,
    "months_since_last_claim": 0.0447,
}


def _load_registry() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_latest_drift() -> Optional[Dict[str, Any]]:
    try:
        history = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(history, list) and history:
            return history[-1]
    except Exception:
        pass
    return None


def _drift_summary(latest: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not latest:
        return None
    return {
        "computed_at": latest.get("computed_at"),
        "overall_psi": latest.get("overall_psi"),
        "overall_severity": latest.get("overall_severity"),
        "high_drift_features": latest.get("high_drift_features", []),
        "medium_drift_features": latest.get("medium_drift_features", []),
        "features_computed": latest.get("features_computed"),
        "recommendation": latest.get("recommendation"),
    }


def _get_shadow_validation() -> Dict[str, Any]:
    """Lazy import to avoid circular dependencies."""
    try:
        from src.services.shadow_deployment import get_shadow_validation_status
        return get_shadow_validation_status()
    except Exception:
        return {"status": "NOT_STARTED", "challenger_available": False, "note": "Shadow service unavailable"}


def generate_model_card(model_type: str) -> Dict[str, Any]:
    """
    Build a model card dict for the given model_type ('frequency' or 'severity').
    Falls back to safe defaults when registry or drift history are unavailable.
    """
    if model_type not in ("frequency", "severity"):
        raise ValueError(f"Unknown model_type: {model_type!r}")

    registry = _load_registry()
    latest_drift = _load_latest_drift()

    if registry and model_type in registry:
        entry = registry[model_type]
        version = entry.get("version", "unknown")
        algorithm = entry.get("algorithm", "unknown")
        training_date = entry.get("promotion_date", "unknown")
        dataset_size = entry.get("trained_on_rows", 0)
        preprocessor_version = entry.get("preprocessor_version", "V4")
        champion_since = entry.get("promotion_date", "unknown")
        eval_metrics = entry.get("eval_metrics", {})
    else:
        version = "unknown"
        algorithm = "unknown"
        training_date = "unknown"
        dataset_size = 0
        preprocessor_version = "V4"
        champion_since = "unknown"
        eval_metrics = {}

    drift_summary = _drift_summary(latest_drift)

    # Derive recommendation from drift + model performance
    drift_severity = (drift_summary or {}).get("overall_severity", "no_data")
    if drift_severity == "high":
        recommendation = "Consider Retraining — High drift detected"
    elif drift_severity == "medium":
        recommendation = "Investigate — Medium drift detected"
    else:
        recommendation = "Monitor — Stable distribution"

    card: Dict[str, Any] = {
        "model_type": model_type,
        "model_version": version,
        "algorithm": algorithm,
        "training_date": training_date,
        "dataset_size": dataset_size,
        "preprocessor_version": preprocessor_version,
        "champion_since": champion_since,
        "recommendation": recommendation,
        "drift_summary": drift_summary,
    }

    if model_type == "frequency":
        card["frequency_metrics"] = {
            "roc_auc":     eval_metrics.get("roc_auc"),
            "pr_auc":      eval_metrics.get("pr_auc"),
            "f1":          eval_metrics.get("f1"),
            "precision":   eval_metrics.get("precision"),
            "recall":      eval_metrics.get("recall"),
            "brier_score": eval_metrics.get("brier_score"),
            "log_loss":    eval_metrics.get("log_loss"),
        }
        card["severity_metrics"] = None
        card["feature_importance_summary"] = _FREQ_IMPORTANCE
    else:
        card["severity_metrics"] = {
            "r2":               eval_metrics.get("r2"),
            "rmse":             eval_metrics.get("rmse"),
            "mae":              eval_metrics.get("mae"),
            "mape_pct":         eval_metrics.get("mape_pct"),
            "median_abs_error": eval_metrics.get("median_abs_error"),
        }
        card["frequency_metrics"] = None
        card["feature_importance_summary"] = _SEV_IMPORTANCE

    # Part 7: Shadow validation status (NOT_STARTED until V5 challenger exists)
    card["shadow_validation"] = _get_shadow_validation()

    return card
