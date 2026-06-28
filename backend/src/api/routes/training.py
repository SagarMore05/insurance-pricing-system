"""
Phase 5A — V5 Training Engine admin endpoints.

All endpoints require JWT admin authentication.
No champion modification, no promotion, no registry update.
Challenger models only.

Routes:
  POST /admin/training/run       — trigger a new training run (returns immediately)
  GET  /admin/training/runs      — list training runs (paginated)
  GET  /admin/training/runs/{id} — run detail with all candidate models
  GET  /admin/training/candidates/{id} — single candidate model detail
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    CandidateModelResponse,
    TrainingRunListResponse,
    TrainingRunResponse,
    TrainingTriggerResponse,
)
from src.database.models import CandidateModel, TrainingRun
from src.database.session import get_db
from src.training.engine import (
    V5TrainingEngine,
    _candidate_to_dict,
    _run_to_dict,
    get_training_run_by_id,
    list_training_runs,
)

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.post(
    "/admin/training/run",
    response_model=TrainingTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger V5 challenger training run",
)
async def trigger_training_run(
    background_tasks: BackgroundTasks,
    triggered_by: str = Query("admin", description="Actor label for audit"),
    db: AsyncSession = Depends(get_db),
) -> TrainingTriggerResponse:
    """
    Trigger a V5 challenger training run.

    The run executes asynchronously (BackgroundTasks).
    Returns the run_id immediately so the caller can poll /runs/{id} for status.

    SAFETY: Does not modify champion registry, production models, or pricing.
    """
    now = datetime.utcnow()
    run_id = str(uuid.uuid4())
    run_tag = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Pre-create the DB record so callers can poll immediately
    db_run = TrainingRun(
        run_id=uuid.UUID(run_id),
        run_tag=run_tag,
        triggered_by=triggered_by,
        triggered_at=now,
        status="RUNNING",
    )
    db.add(db_run)
    await db.commit()

    # Launch training in background (non-blocking)
    background_tasks.add_task(
        V5TrainingEngine.run_background,
        run_id_placeholder=run_id,
        triggered_by=triggered_by,
        preexisting_run_tag=run_tag,
    )

    return TrainingTriggerResponse(
        run_id=run_id,
        run_tag=run_tag,
        status="RUNNING",
        message=(
            "Training run started. Poll GET /admin/training/runs/{run_id} for status. "
            "No champion modification will occur — challenger models only."
        ),
        triggered_by=triggered_by,
        triggered_at=now.isoformat(),
    )


@router.get(
    "/admin/training/runs",
    response_model=TrainingRunListResponse,
    summary="List training runs",
)
async def list_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> TrainingRunListResponse:
    """Return paginated list of training runs, newest first."""
    data = await list_training_runs(db, page=page, page_size=page_size)
    items = [TrainingRunResponse(**item) for item in data["items"]]
    return TrainingRunListResponse(
        items=items,
        total=data["total"],
        page=data["page"],
        page_size=data["page_size"],
        total_pages=data["total_pages"],
    )


@router.get(
    "/admin/training/runs/{run_id}",
    response_model=TrainingRunResponse,
    summary="Training run detail",
)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> TrainingRunResponse:
    """Return full detail for a training run including all candidate models."""
    run = await get_training_run_by_id(db, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run '{run_id}' not found.",
        )
    return TrainingRunResponse(**_run_to_dict(run))


@router.get(
    "/admin/training/candidates/{model_id}",
    response_model=CandidateModelResponse,
    summary="Candidate model detail",
)
async def get_candidate(
    model_id: str,
    db: AsyncSession = Depends(get_db),
) -> CandidateModelResponse:
    """Return detail for a single candidate model."""
    try:
        uid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate model '{model_id}' not found.",
        )
    result = await db.execute(
        select(CandidateModel).where(CandidateModel.model_id == uid)
    )
    cand = result.scalar_one_or_none()
    if cand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate model '{model_id}' not found.",
        )
    return CandidateModelResponse(**_candidate_to_dict(cand))
