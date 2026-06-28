import logging
import uuid
from datetime import date as date_cls
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models import Claim, ModelFeedback, Policy, Prediction
from src.api.schemas import ClaimRequest, ClaimResponse, ClaimUpdateRequest
from src.api.dependencies import verify_admin_key

logger = logging.getLogger("claims")

router = APIRouter()


@router.get("/claims", response_model=List[ClaimResponse])
async def get_claims_by_policy(
    policy_id: Optional[uuid.UUID] = Query(None, description="Return all claims for this policy"),
    db: AsyncSession = Depends(get_db),
):
    if policy_id is None:
        return []
    result = await db.execute(
        select(Claim)
        .where(Claim.policy_id == policy_id)
        .order_by(Claim.claim_date.desc())
    )
    claims = result.scalars().all()
    return [
        ClaimResponse(
            claim_id=c.claim_id,
            policy_id=c.policy_id,
            claimed_amount_inr=float(c.claimed_amount_inr),
            approved_amount_inr=float(c.approved_amount_inr) if c.approved_amount_inr else None,
            claim_date=c.claim_date,
            claim_status=c.claim_status,
        )
        for c in claims
    ]


@router.post("/claims", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def file_claim(request: ClaimRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Policy).where(Policy.policy_id == request.policy_id)
    )
    policy = result.scalar_one_or_none()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    if not policy.is_active:
        raise HTTPException(status_code=400, detail="Cannot file claim on inactive policy")

    claim = Claim(
        claim_id=uuid.uuid4(),
        policy_id=request.policy_id,
        claimed_amount_inr=request.claimed_amount_inr,
        claim_date=request.claim_date or date_cls.today(),
        claim_status="pending",
    )
    db.add(claim)
    await db.flush()

    return ClaimResponse(
        claim_id=claim.claim_id,
        policy_id=claim.policy_id,
        claimed_amount_inr=float(claim.claimed_amount_inr),
        approved_amount_inr=None,
        claim_date=claim.claim_date,
        claim_status=claim.claim_status,
    )


@router.patch(
    "/claims/{claim_id}",
    response_model=ClaimResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def update_claim(
    claim_id: uuid.UUID,
    request: ClaimUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Claim).where(Claim.claim_id == claim_id))
    claim = result.scalar_one_or_none()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim.claim_status = request.claim_status.value
    if request.approved_amount_inr is not None:
        claim.approved_amount_inr = request.approved_amount_inr
    await db.flush()

    # Auto-create ModelFeedback when a claim is approved so the retraining
    # pipeline can use real claim outcomes without requiring manual feedback.
    if request.claim_status.value == "approved":
        await _auto_create_feedback(db, claim, request)

    return ClaimResponse(
        claim_id=claim.claim_id,
        policy_id=claim.policy_id,
        claimed_amount_inr=float(claim.claimed_amount_inr),
        approved_amount_inr=float(claim.approved_amount_inr) if claim.approved_amount_inr else None,
        claim_date=claim.claim_date,
        claim_status=claim.claim_status,
    )


async def _auto_create_feedback(
    db: AsyncSession,
    claim: Claim,
    request: ClaimUpdateRequest,
) -> None:
    """
    Insert a ModelFeedback row when a claim is approved.

    Uses the most recent Prediction for the claim's policy as the anchor.
    Skips silently if a feedback record already exists for that prediction,
    so manual feedback submitted via POST /feedback is never overwritten.
    """
    try:
        pred_result = await db.execute(
            select(Prediction)
            .where(Prediction.policy_id == claim.policy_id)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        prediction = pred_result.scalar_one_or_none()
        if prediction is None:
            return

        existing = await db.execute(
            select(ModelFeedback)
            .where(ModelFeedback.prediction_id == prediction.prediction_id)
        )
        if existing.scalar_one_or_none() is not None:
            return

        approved_amount = (
            float(request.approved_amount_inr)
            if request.approved_amount_inr is not None
            else float(claim.claimed_amount_inr)
        )
        db.add(ModelFeedback(
            prediction_id=prediction.prediction_id,
            actual_claim_occurred=True,
            actual_claim_amount_inr=approved_amount,
            feedback_date=date_cls.today(),
        ))
        logger.info(
            "Auto-created ModelFeedback for prediction %s from approved claim %s",
            prediction.prediction_id, claim.claim_id,
        )
    except Exception as exc:
        # Non-fatal: log and continue — the claim approval itself succeeded
        logger.warning(
            "Auto-feedback creation failed for claim %s: %s", claim.claim_id, exc
        )
