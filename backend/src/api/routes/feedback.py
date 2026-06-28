import uuid
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models import ModelFeedback, Prediction
from src.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    pred_result = await db.execute(
        select(Prediction).where(Prediction.prediction_id == request.prediction_id)
    )
    if not pred_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Prediction not found")

    feedback = ModelFeedback(
        feedback_id=uuid.uuid4(),
        prediction_id=request.prediction_id,
        actual_claim_occurred=request.actual_claim_occurred,
        actual_claim_amount_inr=request.actual_claim_amount_inr,
        feedback_date=date.today(),
    )
    db.add(feedback)
    await db.flush()

    return FeedbackResponse(
        feedback_id=feedback.feedback_id,
        prediction_id=feedback.prediction_id,
        actual_claim_occurred=feedback.actual_claim_occurred,
        actual_claim_amount_inr=float(feedback.actual_claim_amount_inr) if feedback.actual_claim_amount_inr else None,
        feedback_date=feedback.feedback_date,
    )
