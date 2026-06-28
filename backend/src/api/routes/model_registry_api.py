"""
Model Registry API (v2) — serves canonical model_registry.json.

Endpoints:
  GET /admin/model-registry            — full registry (production + candidate + archived)
  GET /admin/model-registry/production — currently serving models only
  GET /admin/model-registry/candidate  — candidate models (if any)
  GET /admin/model-registry/comparison — side-by-side metric comparison
"""
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import verify_admin_key

router = APIRouter(dependencies=[Depends(verify_admin_key)])

_BACKEND_DIR = Path(__file__).parent.parent.parent.parent
_REGISTRY_PATH = _BACKEND_DIR / "models" / "registry" / "model_registry.json"


def _load_registry() -> dict:
    if not _REGISTRY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="model_registry.json not found at models/registry/",
        )
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


@router.get("/admin/model-registry")
async def get_model_registry():
    """Full model registry: production, candidate, and archived for all model types."""
    return _load_registry()


@router.get("/admin/model-registry/production")
async def get_production_models():
    """Currently serving production models for frequency and severity."""
    data = _load_registry()
    return {
        "frequency": data["frequency"]["production"],
        "severity": data["severity"]["production"],
        "last_updated": data["last_updated"],
    }


@router.get("/admin/model-registry/candidate")
async def get_candidate_models():
    """Latest candidate models (or null if no candidate exists)."""
    data = _load_registry()
    return {
        "frequency": data["frequency"].get("candidate"),
        "severity": data["severity"].get("candidate"),
        "last_updated": data["last_updated"],
    }


@router.get("/admin/model-registry/comparison")
async def get_model_comparison():
    """
    Side-by-side comparison: production vs candidate.
    Returns computed deltas and improvement/decline verdicts.
    """
    data = _load_registry()

    freq_prod = data["frequency"]["production"]["metrics"]
    freq_cand = data["frequency"].get("candidate")
    sev_prod = data["severity"]["production"]["metrics"]
    sev_cand = data["severity"].get("candidate")

    def _delta(prod_val, cand_val):
        if prod_val is None or cand_val is None:
            return None
        return round(cand_val - prod_val, 6)

    def _verdict(d, higher_is_better: bool = True) -> str:
        if d is None:
            return "unavailable"
        threshold = 0.001
        if higher_is_better:
            if d > threshold:
                return "improved"
            if d < -threshold:
                return "declined"
        else:
            if d < -threshold:
                return "improved"
            if d > threshold:
                return "declined"
        return "unchanged"

    freq_comparison = None
    if freq_cand:
        cm = freq_cand["metrics"]
        auc_d   = _delta(freq_prod.get("auc_roc"),    cm.get("auc_roc"))
        f1_d    = _delta(freq_prod.get("f1"),          cm.get("f1"))
        prec_d  = _delta(freq_prod.get("precision"),   cm.get("precision"))
        rec_d   = _delta(freq_prod.get("recall"),      cm.get("recall"))
        brier_d = _delta(freq_prod.get("brier_score"), cm.get("brier_score"))
        freq_comparison = {
            "auc_roc":     {"production": freq_prod.get("auc_roc"),    "candidate": cm.get("auc_roc"),    "delta": auc_d,   "verdict": _verdict(auc_d,   True)},
            "f1":          {"production": freq_prod.get("f1"),         "candidate": cm.get("f1"),         "delta": f1_d,    "verdict": _verdict(f1_d,    True)},
            "precision":   {"production": freq_prod.get("precision"),  "candidate": cm.get("precision"),  "delta": prec_d,  "verdict": _verdict(prec_d,  True)},
            "recall":      {"production": freq_prod.get("recall"),     "candidate": cm.get("recall"),     "delta": rec_d,   "verdict": _verdict(rec_d,   True)},
            "brier_score": {"production": freq_prod.get("brier_score"),"candidate": cm.get("brier_score"),"delta": brier_d, "verdict": _verdict(brier_d, False)},
        }

    sev_comparison = None
    if sev_cand:
        sm = sev_cand["metrics"]
        r2_d   = _delta(sev_prod.get("r2"),   sm.get("r2"))
        rmse_d = _delta(sev_prod.get("rmse"), sm.get("rmse"))
        mae_d  = _delta(sev_prod.get("mae"),  sm.get("mae"))
        sev_comparison = {
            "r2":              {"production": sev_prod.get("r2"),   "candidate": sm.get("r2"),   "delta": r2_d,   "verdict": _verdict(r2_d,   True)},
            "rmse":            {"production": sev_prod.get("rmse"), "candidate": sm.get("rmse"), "delta": rmse_d, "verdict": _verdict(rmse_d, False)},
            "mae":             {"production": sev_prod.get("mae"),  "candidate": sm.get("mae"),  "delta": mae_d,  "verdict": _verdict(mae_d,  False)},
            "comparison_note": sev_cand.get("comparison_note"),
        }

    return {
        "frequency": freq_comparison,
        "severity": sev_comparison,
        "last_updated": data["last_updated"],
    }
