"""
Model Selection API — exposes Multi-Algorithm Engine results.

Endpoints:
  GET  /admin/model-selection/latest      — latest training run results
  GET  /admin/model-selection/history     — all training runs (paginated)
  GET  /admin/model-selection/comparison  — algorithm comparison from latest run
  POST /admin/model-selection/retrain     — trigger new multi-algorithm training run
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies import verify_admin_key

logger   = logging.getLogger("insurance_api.model_selection_api")
router   = APIRouter(dependencies=[Depends(verify_admin_key)])

_BACKEND_DIR  = Path(__file__).parent.parent.parent.parent
_HISTORY_PATH = _BACKEND_DIR / "models" / "registry" / "training_history.json"

_RETRAIN_STATUS: Dict[str, Any] = {"running": False, "last_run_id": None, "error": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_history() -> Dict[str, Any]:
    if not _HISTORY_PATH.exists():
        return {
            "schema_version": "1.0.0",
            "training_runs":  [],
            "latest_run_id":  None,
            "total_runs":     0,
        }
    with open(_HISTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/admin/model-selection/latest")
async def get_latest_run() -> Dict[str, Any]:
    """Return the most recent multi-algorithm training run record."""
    history = _load_history()
    if not history["training_runs"]:
        return {
            "has_runs":    False,
            "message":     "No training runs found. Execute the multi-algorithm training pipeline first.",
            "total_runs":  0,
            "latest_run":  None,
        }
    latest = history["training_runs"][-1]
    return {
        "has_runs":   True,
        "total_runs": history["total_runs"],
        "latest_run": latest,
    }


@router.get("/admin/model-selection/history")
async def get_training_history(
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Return paginated training run history (newest first)."""
    history = _load_history()
    runs = list(reversed(history["training_runs"]))
    total = len(runs)
    start = (page - 1) * page_size
    end   = start + page_size

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, -(-total // page_size)),
        "runs":        runs[start:end],
        "latest_run_id": history.get("latest_run_id"),
    }


@router.get("/admin/model-selection/comparison")
async def get_algorithm_comparison() -> Dict[str, Any]:
    """
    Return side-by-side algorithm comparison table from the latest training run.
    Each row: algorithm, metrics, is_winner, training_time_sec.
    """
    history = _load_history()
    if not history["training_runs"]:
        return {
            "has_runs":            False,
            "message":             "No training runs found.",
            "frequency_comparison": [],
            "severity_comparison":  [],
            "frequency_winner":     None,
            "severity_winner":      None,
        }

    latest = history["training_runs"][-1]
    return {
        "has_runs":             True,
        "run_id":               latest["run_id"],
        "run_number":           latest["run_number"],
        "training_timestamp":   latest["training_timestamp"],
        "dataset":              latest["dataset"],
        "frequency_comparison": latest.get("frequency_results", []),
        "severity_comparison":  latest.get("severity_results", []),
        "frequency_winner":     latest.get("frequency_winner"),
        "severity_winner":      latest.get("severity_winner"),
        "promotion_reason_frequency": latest.get("promotion_reason_frequency"),
        "promotion_reason_severity":  latest.get("promotion_reason_severity"),
    }


@router.get("/admin/model-selection/status")
async def get_retrain_status() -> Dict[str, Any]:
    """Return current background retraining status."""
    return _RETRAIN_STATUS


@router.post("/admin/model-selection/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Trigger a new multi-algorithm training run as a background task.
    Returns immediately; poll /admin/model-selection/status for progress.
    """
    if _RETRAIN_STATUS["running"]:
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress. Wait for it to complete.",
        )

    _RETRAIN_STATUS["running"] = True
    _RETRAIN_STATUS["error"]   = None
    background_tasks.add_task(_run_training_background)

    return {
        "triggered": True,
        "message":   "Multi-algorithm training started in background. "
                     "Poll GET /admin/model-selection/status for progress.",
    }


def _run_training_background() -> None:
    """Background worker for POST /admin/model-selection/retrain."""
    try:
        from src.training.multi_algorithm_engine import MultiAlgorithmEngine
        engine = MultiAlgorithmEngine()
        record = engine.run()
        _RETRAIN_STATUS["last_run_id"] = record["run_id"]
        logger.info("Background training completed: run=%s", record["run_id"])
    except Exception as exc:
        logger.exception("Background training failed: %s", exc)
        _RETRAIN_STATUS["error"] = str(exc)
    finally:
        _RETRAIN_STATUS["running"] = False
