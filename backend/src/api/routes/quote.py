import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models import Customer, Vehicle, DrivingProfile, Policy, Prediction
from src.api.schemas import CustomerQuoteRequest, PremiumQuoteResponse, ExplanationDetail, AdjustmentDetail, DrivingScoreResult
from src.models.combined_predictor import get_predictor
from src.models.model_registry import get_active_model_version
from src.engine.pricing_engine import get_pricing_engine
from src.engine.driving_score import calculate_driving_score

router = APIRouter()


@router.post("/quote", response_model=PremiumQuoteResponse, status_code=status.HTTP_201_CREATED)
async def get_quote(request: CustomerQuoteRequest, db: AsyncSession = Depends(get_db)):
    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Run training pipeline first.",
        )

    engine = get_pricing_engine()

    input_data = request.model_dump()

    # Compute driving score when not supplied; use provided value for backward compat
    if request.driving_score is None:
        ds = calculate_driving_score(
            annual_mileage=request.annual_mileage_km,
            prior_claims=request.previous_claims,
            years_licensed=request.years_licensed,
            age=request.age,
        )
        effective_driving_score = ds["score"]
        driving_score_info = DrivingScoreResult(score=ds["score"], factors=ds["factors"])
    else:
        effective_driving_score = int(request.driving_score)
        driving_score_info = None

    input_data["driving_score"] = effective_driving_score

    ml_result = predictor.predict(input_data)

    pricing_result = engine.calculate(
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        age=request.age,
        driving_score=effective_driving_score,
        previous_claims_count=request.previous_claims,
        annual_mileage_km=request.annual_mileage_km,
        vehicle_value_inr=request.vehicle_value_inr,
    )

    # Persist customer/vehicle/profile
    customer = Customer(
        customer_id=uuid.uuid4(),
        age=request.age,
        gender=request.gender,
        city=request.city,
    )
    db.add(customer)
    await db.flush()

    vehicle = Vehicle(
        vehicle_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        car_brand=request.vehicle_brand,           # master-dataset name → DB column
        car_model=request.car_model or "N/A",      # optional in form; DB requires non-null
        engine_cc=request.engine_cc or 0,          # optional in form; DB requires non-null
        vehicle_age_years=request.vehicle_age_years,
        vehicle_value_inr=request.vehicle_value_inr,
        fuel_type=request.fuel_type,
    )
    db.add(vehicle)

    profile = DrivingProfile(
        profile_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        driving_score=effective_driving_score,
        annual_mileage_km=request.annual_mileage_km,
        previous_claims_count=request.previous_claims,  # master name → DB column
        years_licensed=request.years_licensed,
    )
    db.add(profile)

    policy = Policy(
        policy_id=uuid.uuid4(),
        customer_id=customer.customer_id,
        vehicle_id=vehicle.vehicle_id,
        premium_amount_inr=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        model_version=get_active_model_version(),
        is_active=False,  # becomes active only after purchase
    )
    db.add(policy)
    await db.flush()

    explanation_dict = pricing_result.to_dict()
    prediction = Prediction(
        prediction_id=uuid.uuid4(),
        policy_id=policy.policy_id,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        final_premium_inr=pricing_result.final_premium,
        explanation_json=explanation_dict,
    )
    db.add(prediction)
    await db.flush()

    explanation = ExplanationDetail(
        base_premium=pricing_result.base_premium,
        expected_loss=pricing_result.expected_loss,
        adjustments=[
            AdjustmentDetail(reason=a.reason, factor=a.factor, impact_inr=a.impact_inr)
            for a in pricing_result.adjustments
        ],
        final_premium=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        claim_probability=ml_result["claim_probability"],
        summary=pricing_result.summary,
    )

    return PremiumQuoteResponse(
        prediction_id=prediction.prediction_id,
        premium_amount_inr=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        explanation=explanation,
        model_version=get_active_model_version(),
        created_at=prediction.created_at or datetime.utcnow(),
        driving_score_info=driving_score_info,
    )
