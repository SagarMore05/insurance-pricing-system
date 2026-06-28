import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from src.database.session import get_db
from src.database.models import Policy, Prediction, Customer, Vehicle
from src.api.schemas import PolicyCreateRequest, PolicyResponse, PremiumQuoteResponse, ExplanationDetail, AdjustmentDetail

router = APIRouter()


@router.get("/policies", response_model=List[PolicyResponse])
async def list_policies(
    customer_id: Optional[uuid.UUID] = Query(None, description="Filter by customer ID"),
    db: AsyncSession = Depends(get_db),
):
    q = select(Policy)
    if customer_id is not None:
        q = q.where(Policy.customer_id == customer_id)
    result = await db.execute(q.order_by(Policy.created_at.desc()))
    policies = result.scalars().all()
    return [
        PolicyResponse(
            policy_id=p.policy_id,
            customer_id=p.customer_id,
            vehicle_id=p.vehicle_id,
            premium_amount_inr=float(p.premium_amount_inr),
            risk_level=p.risk_level,
            model_version=p.model_version,
            created_at=p.created_at,
            is_active=p.is_active,
        )
        for p in policies
    ]


@router.post("/policies", response_model=PolicyResponse, status_code=status.HTTP_201_CREATED)
async def create_policy(request: PolicyCreateRequest, db: AsyncSession = Depends(get_db)):
    # Find the prediction
    result = await db.execute(
        select(Prediction).where(Prediction.prediction_id == request.prediction_id)
    )
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Activate the policy
    policy_result = await db.execute(
        select(Policy).where(Policy.policy_id == prediction.policy_id)
    )
    policy = policy_result.scalar_one_or_none()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    policy.is_active = True
    await db.flush()

    return PolicyResponse(
        policy_id=policy.policy_id,
        customer_id=policy.customer_id,
        vehicle_id=policy.vehicle_id,
        premium_amount_inr=float(policy.premium_amount_inr),
        risk_level=policy.risk_level,
        model_version=policy.model_version,
        created_at=policy.created_at,
        is_active=policy.is_active,
    )


@router.get("/policies/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Policy).where(Policy.policy_id == policy_id)
    )
    policy = result.scalar_one_or_none()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    pred_result = await db.execute(
        select(Prediction).where(Prediction.policy_id == policy_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    prediction = pred_result.scalar_one_or_none()

    pred_response = None
    if prediction and prediction.explanation_json:
        exp = prediction.explanation_json
        pred_response = PremiumQuoteResponse(
            prediction_id=prediction.prediction_id,
            premium_amount_inr=float(prediction.final_premium_inr),
            risk_level=policy.risk_level,
            claim_probability=float(prediction.claim_probability),
            expected_claim_amount_inr=float(prediction.expected_claim_amount_inr),
            explanation=ExplanationDetail(
                base_premium=exp.get("base_premium", 0),
                expected_loss=exp.get("expected_loss", 0),
                adjustments=[
                    AdjustmentDetail(**a) for a in exp.get("adjustments", [])
                ],
                final_premium=exp.get("final_premium", 0),
                risk_level=policy.risk_level,
                claim_probability=float(prediction.claim_probability),
                summary=exp.get("summary", ""),
            ),
            model_version=policy.model_version,
            created_at=prediction.created_at,
        )

    return PolicyResponse(
        policy_id=policy.policy_id,
        customer_id=policy.customer_id,
        vehicle_id=policy.vehicle_id,
        premium_amount_inr=float(policy.premium_amount_inr),
        risk_level=policy.risk_level,
        model_version=policy.model_version,
        created_at=policy.created_at,
        is_active=policy.is_active,
        prediction=pred_response,
    )
