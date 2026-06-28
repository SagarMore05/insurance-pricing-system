"""
Phase 5B — Champion Promotion Engine admin endpoints.

All endpoints require JWT admin authentication.

Routes:
  POST /admin/promotions/evaluate         — evaluate a training run's candidates
  POST /admin/promotions/{id}/promote     — execute an APPROVED promotion
  POST /admin/promotions/{id}/rollback    — manually rollback an ACTIVE promotion
  GET  /admin/promotions                  — list all promotions (paginated)
  GET  /admin/promotions/{id}             — promotion detail with rollback history
  GET  /admin/promotions/rollbacks        — rollback history (all promotions)

Safety:
  - evaluate() NEVER modifies champions
  - promote() requires status=APPROVED (all gates passed)
  - All operations are DB-recorded with full audit trail
  - Automatic rollback on any promotion step failure
"""
import math
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    GateResultSchema,
    PromotionDetailResponse,
    PromotionEvaluationRequest,
    PromotionEvaluationResponse,
    PromotionExecuteRequest,
    PromotionListResponse,
    RollbackHistoryResponse,
    RollbackRequest,
)
from src.database.models import ModelPromotion, RollbackHistory
from src.database.session import get_db
from src.promotions.engine import Phase5BPromotionEngine

router = APIRouter(dependencies=[Depends(verify_admin_key)])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _rb_to_schema(rb: RollbackHistory) -> RollbackHistoryResponse:
    return RollbackHistoryResponse(
        rollback_id=str(rb.rollback_id),
        promotion_id=str(rb.promotion_id),
        rollback_reason=rb.rollback_reason,
        rollback_trigger=rb.rollback_trigger,
        rollback_status=rb.rollback_status,
        rollback_duration_seconds=(
            float(rb.rollback_duration_seconds)
            if rb.rollback_duration_seconds is not None else None
        ),
        rolled_back_by=rb.rolled_back_by,
        rolled_back_at=_fmt(rb.rolled_back_at) or "",
        error_message=rb.error_message,
        restored_frequency_champion=rb.restored_frequency_champion,
        restored_severity_champion=rb.restored_severity_champion,
    )


def _promo_to_schema(p: ModelPromotion) -> PromotionDetailResponse:
    gates_passed = p.gates_passed or {}
    gates_failed = p.gates_failed or {}
    all_passed   = bool(gates_passed) and not gates_failed if (gates_passed or gates_failed) else None
    return PromotionDetailResponse(
        promotion_id=str(p.promotion_id),
        run_id=str(p.run_id),
        frequency_candidate_id=str(p.frequency_candidate_id) if p.frequency_candidate_id else None,
        severity_candidate_id=str(p.severity_candidate_id) if p.severity_candidate_id else None,
        status=p.status,
        all_gates_passed=all_passed,
        evaluation_report=p.evaluation_report,
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        old_frequency_champion=p.old_frequency_champion,
        old_severity_champion=p.old_severity_champion,
        new_frequency_champion=p.new_frequency_champion,
        new_severity_champion=p.new_severity_champion,
        promoted_by=p.promoted_by,
        promoted_at=_fmt(p.promoted_at),
        promotion_duration_seconds=(
            float(p.promotion_duration_seconds)
            if p.promotion_duration_seconds is not None else None
        ),
        backup_path=p.backup_path,
        error_message=p.error_message,
        notes=p.notes,
        created_at=_fmt(p.created_at) or "",
        updated_at=_fmt(p.updated_at),
        rollback_records=[_rb_to_schema(rb) for rb in (p.rollback_records or [])],
    )


async def _get_promotion_or_404(promotion_id: str, db: AsyncSession) -> ModelPromotion:
    try:
        uid = uuid.UUID(promotion_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    result = await db.execute(
        select(ModelPromotion)
        .options(selectinload(ModelPromotion.rollback_records))
        .where(ModelPromotion.promotion_id == uid)
    )
    p = result.scalar_one_or_none()
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Promotion '{promotion_id}' not found.",
        )
    return p


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/admin/promotions/evaluate",
    response_model=PromotionEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Evaluate training run for champion promotion",
)
async def evaluate_promotion(
    body: PromotionEvaluationRequest,
    db: AsyncSession = Depends(get_db),
) -> PromotionEvaluationResponse:
    """
    Run all seven governance gates for a training run's candidate models.

    Creates a ModelPromotion record with status APPROVED or REJECTED.
    Does NOT modify champions, registry, or in-memory predictor.

    Gates:
      1. human_approval      — valid ApprovalRequest with status="approved"
      2. shadow_completed    — completed shadow_predictions exist
      3. shadow_count        — ≥ MIN_SHADOW_OBSERVATIONS completed
      4. no_critical_drift   — no high-severity drift in past 30 days
      5. performance         — challenger outperforms champion
      6. artifacts_exist     — model files present on disk
      7. registry_valid      — champion registry paths valid
    """
    engine = Phase5BPromotionEngine(db)
    try:
        result = await engine.evaluate(
            run_id=body.run_id,
            freq_candidate_id=body.frequency_candidate_id,
            sev_candidate_id=body.severity_candidate_id,
            approval_id=body.approval_id,
            bypass_shadow=body.bypass_shadow,
            bypass_drift=body.bypass_drift,
            notes=body.notes,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Evaluation failed: {exc}",
        )

    gates_schema = {
        k: GateResultSchema(
            gate=v["gate"],
            passed=v["passed"],
            reason=v["reason"],
            details=v.get("details"),
        )
        for k, v in result.gates_report.get("gates", {}).items()
    }

    return PromotionEvaluationResponse(
        promotion_id=result.promotion_id,
        run_id=result.run_id,
        status=result.status,
        all_gates_passed=result.all_gates_passed,
        gates=gates_schema,
        freq_comparison=result.freq_comparison,
        sev_comparison=result.sev_comparison,
        can_promote=result.all_gates_passed,
        message=result.message,
        created_at=result.created_at,
    )


@router.post(
    "/admin/promotions/{promotion_id}/promote",
    response_model=PromotionDetailResponse,
    summary="Execute champion promotion",
)
async def execute_promotion(
    promotion_id: str,
    body: PromotionExecuteRequest,
    db: AsyncSession = Depends(get_db),
) -> PromotionDetailResponse:
    """
    Atomically promote challenger models to champion status.

    Requires the promotion to have status=APPROVED (all governance gates passed).

    Steps executed:
      1. Backup current champion artifacts
      2. Install challenger model files into champion directory
      3. Update champion registry (champion_registry.json)
      4. Hot-reload in-memory predictor
      5. Health check with probe data

    On any failure: automatic rollback to pre-promotion state.
    On success: status → ACTIVE, registry updated, predictor live.
    """
    # Verify promotion exists and is APPROVED
    p = await _get_promotion_or_404(promotion_id, db)
    if p.status != "APPROVED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Promotion {promotion_id} has status='{p.status}'. "
                "Only APPROVED promotions can be executed."
            ),
        )

    engine = Phase5BPromotionEngine(db)
    try:
        result = await engine.execute(
            promotion_id=promotion_id,
            promoted_by=body.promoted_by,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Promotion execution failed: {exc}",
        )

    p = await _get_promotion_or_404(promotion_id, db)
    return _promo_to_schema(p)


@router.post(
    "/admin/promotions/{promotion_id}/rollback",
    response_model=PromotionDetailResponse,
    summary="Manually rollback an active champion promotion",
)
async def rollback_promotion(
    promotion_id: str,
    body: RollbackRequest,
    db: AsyncSession = Depends(get_db),
) -> PromotionDetailResponse:
    """
    Roll back an ACTIVE promotion to the pre-promotion champion state.

    Restores:
      - Champion model files (from backup created at promotion time)
      - champion_registry.json
      - In-memory predictor (hot-reload)

    Status → ROLLED_BACK on success.
    """
    p = await _get_promotion_or_404(promotion_id, db)
    if p.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Promotion {promotion_id} has status='{p.status}'. "
                "Only ACTIVE promotions can be rolled back."
            ),
        )

    engine = Phase5BPromotionEngine(db)
    try:
        await engine.rollback(
            promotion_id=promotion_id,
            reason=body.reason,
            rolled_back_by=body.rolled_back_by,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback failed: {exc}",
        )

    p = await _get_promotion_or_404(promotion_id, db)
    return _promo_to_schema(p)


@router.get(
    "/admin/promotions",
    response_model=PromotionListResponse,
    summary="List all promotion evaluations",
)
async def list_promotions(
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PromotionListResponse:
    """Return paginated list of promotion evaluations, newest first."""
    base_q = select(ModelPromotion)
    count_q = select(func.count()).select_from(ModelPromotion)

    if status_filter:
        base_q  = base_q.where(ModelPromotion.status == status_filter)
        count_q = count_q.where(ModelPromotion.status == status_filter)

    total = (await db.execute(count_q)).scalar() or 0

    rows = (
        await db.execute(
            base_q
            .options(selectinload(ModelPromotion.rollback_records))
            .order_by(desc(ModelPromotion.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    return PromotionListResponse(
        items=[_promo_to_schema(p) for p in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, math.ceil(total / page_size)),
    )


@router.get(
    "/admin/promotions/rollbacks",
    response_model=List[RollbackHistoryResponse],
    summary="List all rollback events",
)
async def list_rollbacks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[RollbackHistoryResponse]:
    """Return all rollback events across all promotions, newest first."""
    rows = (
        await db.execute(
            select(RollbackHistory)
            .order_by(desc(RollbackHistory.rolled_back_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()
    return [_rb_to_schema(rb) for rb in rows]


@router.get(
    "/admin/promotions/{promotion_id}",
    response_model=PromotionDetailResponse,
    summary="Get promotion detail",
)
async def get_promotion(
    promotion_id: str,
    db: AsyncSession = Depends(get_db),
) -> PromotionDetailResponse:
    """Return full detail for a promotion including rollback history."""
    p = await _get_promotion_or_404(promotion_id, db)
    return _promo_to_schema(p)
