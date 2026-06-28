"""
Shadow Deployment Framework — read-only admin endpoints (Phase 4C).

All endpoints are READ-ONLY.
No write endpoints exist — shadow predictions are created automatically
by the V2 quote route via BackgroundTasks.

SAFETY: These endpoints do not modify champions, registry, pricing, or scheduler.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    ShadowHistoryResponse,
    ShadowPredictionRecord,
    ShadowStatisticsResponse,
    ShadowStatusResponse,
)
from src.database.session import get_db
from src.services.shadow_deployment import ShadowDeploymentService

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.get(
    "/admin/shadow/status",
    response_model=ShadowStatusResponse,
    summary="Shadow framework status",
)
async def get_shadow_status(
    db: AsyncSession = Depends(get_db),
) -> ShadowStatusResponse:
    """
    Return the overall shadow deployment framework status.
    Shows challenger availability, total predictions captured, and framework health.
    """
    svc = ShadowDeploymentService(db)
    data = await svc.get_status()
    return ShadowStatusResponse(**data)


@router.get(
    "/admin/shadow/statistics",
    response_model=ShadowStatisticsResponse,
    summary="Shadow comparison statistics",
)
async def get_shadow_statistics(
    window_days: int = Query(30, ge=1, le=180),
    db: AsyncSession = Depends(get_db),
) -> ShadowStatisticsResponse:
    """
    Return aggregate statistics for shadow predictions within the given window.
    All comparison metrics are null until a challenger model is registered.
    """
    svc = ShadowDeploymentService(db)
    data = await svc.get_statistics(window_days=window_days)
    return ShadowStatisticsResponse(**data)


@router.get(
    "/admin/shadow/history",
    response_model=ShadowHistoryResponse,
    summary="Shadow prediction history (paginated)",
)
async def get_shadow_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: str = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> ShadowHistoryResponse:
    """
    Return paginated list of shadow predictions, newest first.
    Optional status filter: WAITING_FOR_CHALLENGER | PENDING | COMPLETED | FAILED
    """
    svc = ShadowDeploymentService(db)
    data = await svc.get_history(
        page=page,
        page_size=page_size,
        status_filter=status_filter,
    )
    items = [ShadowPredictionRecord(**item) for item in data["items"]]
    return ShadowHistoryResponse(
        items=items,
        total=data["total"],
        page=data["page"],
        page_size=data["page_size"],
        total_pages=data["total_pages"],
    )


@router.get(
    "/admin/shadow/{shadow_id}",
    response_model=ShadowPredictionRecord,
    summary="Shadow prediction detail",
)
async def get_shadow_record(
    shadow_id: str,
    db: AsyncSession = Depends(get_db),
) -> ShadowPredictionRecord:
    """Return full detail for a specific shadow prediction record."""
    svc = ShadowDeploymentService(db)
    row = await svc.get_by_id(shadow_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shadow prediction '{shadow_id}' not found.",
        )
    from src.services.shadow_deployment import _row_to_dict
    return ShadowPredictionRecord(**_row_to_dict(row))
