"""Drift detection API endpoints — Phase 4A.

Read-only. Does not modify champions, pipeline, scheduler, or any models.

Endpoints:
  GET  /admin/drift/current     — Compute live drift snapshot (not persisted)
  POST /admin/drift/snapshot    — Compute + persist snapshot to drift_history.json
  GET  /admin/drift/history     — Return recent persisted snapshots
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import DriftHistoryResponse, DriftSnapshot
from src.database.session import get_db
from src.services.drift_detection import (
    DriftDetectionService,
    load_drift_history,
    save_drift_snapshot,
)

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.get("/admin/drift/current", response_model=DriftSnapshot)
async def get_drift_current(
    days: int = Query(default=30, ge=1, le=180, description="Lookback window in days (1-180)"),
    psi_low: float = Query(default=0.10, ge=0.0, le=1.0, description="PSI threshold for medium drift"),
    psi_high: float = Query(default=0.25, ge=0.0, le=1.0, description="PSI threshold for high drift"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute real-time drift snapshot comparing production feature distributions
    (last `days` days) against V4 training reference distributions.

    Response includes per-feature PSI, severity classification, and an
    overall recommendation: Monitor / Investigate / Consider Retraining.

    Does NOT persist the snapshot to history.
    """
    svc = DriftDetectionService(db)
    snapshot = await svc.compute_drift(
        window_days=days,
        psi_low=psi_low,
        psi_high=psi_high,
    )
    return DriftSnapshot(**snapshot.to_dict())


@router.post("/admin/drift/snapshot", response_model=DriftSnapshot)
async def post_drift_snapshot(
    days: int = Query(default=30, ge=1, le=180, description="Lookback window in days (1-180)"),
    psi_low: float = Query(default=0.10, ge=0.0, le=1.0, description="PSI threshold for medium drift"),
    psi_high: float = Query(default=0.25, ge=0.0, le=1.0, description="PSI threshold for high drift"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute drift snapshot and persist it to drift_history.json.

    Use this to build a historical record of drift over time.
    Snapshots are retained up to 90 entries (oldest rotated out).
    """
    svc = DriftDetectionService(db)
    snapshot = await svc.compute_drift(
        window_days=days,
        psi_low=psi_low,
        psi_high=psi_high,
    )
    save_drift_snapshot(snapshot)
    return DriftSnapshot(**snapshot.to_dict())


@router.get("/admin/drift/history", response_model=DriftHistoryResponse)
async def get_drift_history(
    limit: int = Query(default=30, ge=1, le=90, description="Max number of snapshots to return"),
):
    """
    Return the most recent drift snapshots from drift_history.json.

    Returns an empty list if no snapshots have been persisted yet.
    Each snapshot includes overall PSI, severity, recommendation, and
    the list of features that showed high or medium drift at the time.
    """
    history = load_drift_history(limit=limit)
    return DriftHistoryResponse(snapshots=history, count=len(history))
