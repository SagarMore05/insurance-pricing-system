"""
Shadow Deployment Framework — Phase 4C.

Records every V2 champion prediction to the shadow_predictions table.
Compares champion vs. challenger when a challenger model is available.

Current state: No V5 challenger exists → status = WAITING_FOR_CHALLENGER.
The framework is fully operational and accumulating champion baselines.

Safety constraints (enforced here):
  - Does NOT modify champion_registry.json
  - Does NOT reload or replace champion models
  - Does NOT trigger promotion or rollback
  - Does NOT influence pricing
  - Is completely passive (read-record-compare only)

Architecture for zero-change future activation:
  - get_challenger() returns None → swap to return ChallengerWrapper when V5 is ready
  - _run_challenger_inference() is stubbed → fill in when challenger model is registered
  - All comparison columns in shadow_predictions are already defined
"""

import asyncio
import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ShadowPrediction
from src.database.session import AsyncSessionLocal
from src.models.model_registry import get_active_model_version

logger = logging.getLogger("insurance_api.shadow")

# ── Status constants ────────────────────────────────────────────────────────────

STATUS_WAITING   = "WAITING_FOR_CHALLENGER"
STATUS_PENDING   = "PENDING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED    = "FAILED"

RESULT_MATCH      = "MATCH"
RESULT_DIVERGENT  = "DIVERGENT"
RESULT_HIGHER     = "CHALLENGER_HIGHER"
RESULT_LOWER      = "CHALLENGER_LOWER"

# Threshold: challenger premium differs by >5% from champion → DIVERGENT
_PREMIUM_DIVERGENCE_THRESHOLD_PCT = 5.0


# ── Challenger detection ────────────────────────────────────────────────────────

def get_challenger() -> Optional[Dict[str, Any]]:
    """
    Return challenger model metadata if one is registered, else None.

    Currently returns None — no V5 challenger exists.
    Replace the body of this function with registry lookup when V5 trains.

    Future implementation:
        registry = load_challenger_registry()
        if registry and registry.get("challenger_available"):
            return {
                "model_name": registry["challenger"]["model_name"],
                "version": registry["challenger"]["version"],
                "model_path": registry["challenger"]["model_path"],
            }
    """
    return None  # No challenger yet — framework waits


def _challenger_status_str() -> str:
    return "AVAILABLE" if get_challenger() is not None else "UNAVAILABLE"


# ── Core shadow service ─────────────────────────────────────────────────────────

class ShadowDeploymentService:
    """
    Service for recording and comparing shadow predictions.

    Usage (from quote_v2.py via FastAPI BackgroundTasks):
        background_tasks.add_task(
            ShadowDeploymentService.record_background,
            quote_id=str(prediction_id),
            champion_prediction={...},
            latency_ms=elapsed_ms,
            champion_model_name="XGBoost_V4+CatBoost_V4",
            champion_version="v4.0.0",
        )
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Public: record a shadow prediction ────────────────────────────────────

    async def record_prediction(
        self,
        quote_id: str,
        champion_prediction: Dict[str, Any],
        champion_model_name: str,
        champion_version: str,
        latency_ms: Optional[int] = None,
    ) -> ShadowPrediction:
        """
        Record a champion prediction and attempt challenger comparison.
        If no challenger exists → status = WAITING_FOR_CHALLENGER.
        If challenger exists → run inference, store comparison.
        """
        request_id = str(uuid.uuid4())
        challenger = get_challenger()

        challenger_name = None
        challenger_ver = None
        challenger_json = None
        premium_diff = None
        risk_diff = None
        status = STATUS_WAITING
        comparison_result = None
        notes = "Framework operational — awaiting V5 challenger registration."

        if challenger is not None:
            try:
                challenger_name = challenger["model_name"]
                challenger_ver = challenger["version"]
                challenger_json, challenger_latency = await _run_challenger_inference(
                    challenger, champion_prediction.get("_request_payload", {})
                )
                premium_diff, risk_diff, comparison_result = _compare_predictions(
                    champion_prediction, challenger_json
                )
                if latency_ms is not None and challenger_latency is not None:
                    latency_ms = challenger_latency  # store total shadow latency
                status = STATUS_COMPLETED
                notes = f"Shadow comparison completed. Result: {comparison_result}"
            except Exception as exc:
                status = STATUS_FAILED
                notes = f"Challenger inference failed: {type(exc).__name__}: {exc}"
                logger.warning("Shadow inference failed for quote %s: %s", quote_id, exc)

        try:
            quote_uuid = uuid.UUID(quote_id)
        except (ValueError, AttributeError):
            quote_uuid = None

        row = ShadowPrediction(
            id=uuid.uuid4(),
            request_id=request_id,
            quote_id=quote_uuid,
            created_at=datetime.utcnow(),
            champion_model_name=champion_model_name,
            champion_version=champion_version,
            challenger_model_name=challenger_name,
            challenger_version=challenger_ver,
            champion_prediction_json=champion_prediction,
            challenger_prediction_json=challenger_json,
            premium_difference=premium_diff,
            risk_difference=risk_diff,
            prediction_latency_ms=latency_ms,
            status=status,
            comparison_result=comparison_result,
            notes=notes,
        )
        self._db.add(row)
        await self._db.commit()
        return row

    # ── Public: query methods ─────────────────────────────────────────────────

    async def get_status(self) -> Dict[str, Any]:
        """Return overall shadow framework status."""
        total = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
        )).scalar() or 0

        waiting = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.status == STATUS_WAITING)
        )).scalar() or 0

        completed = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.status == STATUS_COMPLETED)
        )).scalar() or 0

        failed = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.status == STATUS_FAILED)
        )).scalar() or 0

        last_row = (await self._db.execute(
            select(ShadowPrediction.created_at)
            .order_by(ShadowPrediction.created_at.desc())
            .limit(1)
        )).scalar()

        return {
            "framework_status": "OPERATIONAL",
            "challenger_available": get_challenger() is not None,
            "challenger_status": _challenger_status_str(),
            "total_shadow_predictions": total,
            "waiting_for_challenger": waiting,
            "completed_comparisons": completed,
            "failed_comparisons": failed,
            "champion_model": "XGBoost_V4 + CatBoost_V4",
            "champion_version": get_active_model_version(),
            "last_recorded_at": last_row.isoformat() if last_row else None,
            "message": (
                "Shadow framework is operational. Recording all V2 predictions. "
                "Awaiting V5 challenger model registration."
                if not get_challenger() else
                "Shadow comparison active — challenger model available."
            ),
        }

    async def get_statistics(self, window_days: int = 30) -> Dict[str, Any]:
        """Return aggregate statistics for the shadow comparison window."""
        since = datetime.utcnow() - timedelta(days=window_days)

        total = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.created_at >= since)
        )).scalar() or 0

        completed = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(and_(
                ShadowPrediction.status == STATUS_COMPLETED,
                ShadowPrediction.created_at >= since,
            ))
        )).scalar() or 0

        waiting = (await self._db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(and_(
                ShadowPrediction.status == STATUS_WAITING,
                ShadowPrediction.created_at >= since,
            ))
        )).scalar() or 0

        avg_premium_diff: Optional[float] = None
        avg_latency: Optional[float] = None

        if completed > 0:
            avg_premium_diff_raw = (await self._db.execute(
                select(func.avg(ShadowPrediction.premium_difference))
                .where(and_(
                    ShadowPrediction.status == STATUS_COMPLETED,
                    ShadowPrediction.created_at >= since,
                ))
            )).scalar()
            avg_premium_diff = float(avg_premium_diff_raw) if avg_premium_diff_raw is not None else None

            avg_latency_raw = (await self._db.execute(
                select(func.avg(ShadowPrediction.prediction_latency_ms))
                .where(and_(
                    ShadowPrediction.status == STATUS_COMPLETED,
                    ShadowPrediction.created_at >= since,
                    ShadowPrediction.prediction_latency_ms.isnot(None),
                ))
            )).scalar()
            avg_latency = float(avg_latency_raw) if avg_latency_raw is not None else None

        completion_rate = round(completed / total * 100, 1) if total > 0 else 0.0

        return {
            "window_days": window_days,
            "total_shadow_predictions": total,
            "completed_comparisons": completed,
            "waiting_for_challenger": waiting,
            "completion_rate_pct": completion_rate,
            "avg_premium_difference_inr": avg_premium_diff,
            "avg_prediction_latency_ms": avg_latency,
            "challenger_available": get_challenger() is not None,
        }

    async def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return paginated shadow prediction history."""
        filters = []
        if status_filter:
            filters.append(ShadowPrediction.status == status_filter)

        count_q = select(func.count()).select_from(ShadowPrediction)
        if filters:
            count_q = count_q.where(and_(*filters))
        total = (await self._db.execute(count_q)).scalar() or 0

        rows_q = (
            select(ShadowPrediction)
            .order_by(ShadowPrediction.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        if filters:
            rows_q = rows_q.where(and_(*filters))

        rows = (await self._db.execute(rows_q)).scalars().all()
        total_pages = max(1, math.ceil(total / page_size))

        return {
            "items": [_row_to_dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        }

    async def get_by_id(self, shadow_id: str) -> Optional[ShadowPrediction]:
        """Return a specific shadow prediction by UUID string."""
        try:
            uid = uuid.UUID(shadow_id)
        except ValueError:
            return None
        result = await self._db.execute(
            select(ShadowPrediction).where(ShadowPrediction.id == uid)
        )
        return result.scalar_one_or_none()

    # ── Class method: background recording (independent session) ──────────────

    @classmethod
    async def record_background(
        cls,
        quote_id: str,
        champion_prediction: Dict[str, Any],
        champion_model_name: str,
        champion_version: str,
        latency_ms: Optional[int] = None,
    ) -> None:
        """
        Entry point for FastAPI BackgroundTasks.
        Creates its own DB session so it runs after the request session closes.
        Failures are logged but never bubble to the customer.
        """
        try:
            async with AsyncSessionLocal() as session:
                svc = cls(session)
                await svc.record_prediction(
                    quote_id=quote_id,
                    champion_prediction=champion_prediction,
                    champion_model_name=champion_model_name,
                    champion_version=champion_version,
                    latency_ms=latency_ms,
                )
        except Exception as exc:
            logger.warning(
                "Shadow recording failed for quote %s (non-blocking): %s",
                quote_id, exc,
            )


# ── Internal helpers ────────────────────────────────────────────────────────────

async def _run_challenger_inference(
    challenger: Dict[str, Any],
    request_payload: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[int]]:
    """
    Run inference against the challenger model.
    Currently stubbed — fill in when V5 challenger is registered.

    Future implementation:
        t0 = time.monotonic()
        result = challenger_model.predict(request_payload)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {"claim_probability": ..., "premium": ..., "risk_level": ...}, latency_ms
    """
    raise NotImplementedError(
        "No challenger model registered. This path should never be reached "
        "while get_challenger() returns None."
    )


def _compare_predictions(
    champion: Dict[str, Any],
    challenger: Dict[str, Any],
) -> tuple[Optional[float], Optional[str], str]:
    """
    Compare champion vs challenger predictions.
    Returns (premium_difference, risk_difference, comparison_result).
    """
    champ_premium = champion.get("final_premium_inr") or champion.get("premium_amount_inr")
    chal_premium  = challenger.get("final_premium_inr") or challenger.get("premium_amount_inr")

    premium_diff: Optional[float] = None
    if champ_premium and chal_premium:
        premium_diff = round(float(chal_premium) - float(champ_premium), 2)
        pct_diff = abs(premium_diff) / float(champ_premium) * 100

        if pct_diff <= _PREMIUM_DIVERGENCE_THRESHOLD_PCT:
            result = RESULT_MATCH
        elif premium_diff > 0:
            result = RESULT_HIGHER
        else:
            result = RESULT_LOWER
    else:
        result = RESULT_DIVERGENT

    champ_risk = champion.get("risk_level")
    chal_risk  = challenger.get("risk_level")
    risk_diff: Optional[str] = None
    if champ_risk and chal_risk and champ_risk != chal_risk:
        risk_diff = f"{champ_risk}→{chal_risk}"

    return premium_diff, risk_diff, result


def _row_to_dict(row: ShadowPrediction) -> Dict[str, Any]:
    return {
        "id": str(row.id),
        "request_id": row.request_id,
        "quote_id": str(row.quote_id) if row.quote_id else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "champion_model_name": row.champion_model_name,
        "champion_version": row.champion_version,
        "challenger_model_name": row.challenger_model_name,
        "challenger_version": row.challenger_version,
        "champion_prediction_json": row.champion_prediction_json,
        "challenger_prediction_json": row.challenger_prediction_json,
        "premium_difference": float(row.premium_difference) if row.premium_difference is not None else None,
        "risk_difference": row.risk_difference,
        "prediction_latency_ms": row.prediction_latency_ms,
        "status": row.status,
        "comparison_result": row.comparison_result,
        "notes": row.notes,
    }


def get_shadow_validation_status() -> Dict[str, Any]:
    """
    Returns the shadow validation status for model card generation.
    Currently NOT_STARTED (no challenger exists).
    """
    challenger = get_challenger()
    return {
        "status": "NOT_STARTED" if challenger is None else "IN_PROGRESS",
        "challenger_available": challenger is not None,
        "note": (
            "No challenger model registered. Shadow framework is operational "
            "and recording champion baselines."
            if challenger is None else
            f"Shadow comparison active against {challenger['model_name']} {challenger['version']}"
        ),
    }
