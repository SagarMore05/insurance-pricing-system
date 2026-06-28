"""
POST /api/v1/agent/quote

Underwriting Agent endpoint — Phase 3B (HITL).

Returns 201 when the graph completes without triggering human review.
Returns 202 when the graph pauses at interrupt_before=["human_review"] and
the run is stored as status='awaiting_review' pending a reviewer decision.

Execution order inside this handler:
  1.  Validate ML readiness
  2.  Run CombinedPredictor (ML)
  3.  Run PricingEngine (deterministic rules)
  4.  Fetch segment risk stats from DB
  5.  Persist Customer / Vehicle / DrivingProfile / Policy / Prediction
  6.  Create AgentRun record (status = "running")
  7.  Build initial LangGraph state and invoke the graph with a thread config
  8.  Check aget_state() to detect interrupt (awaiting_review)
  9a. [If paused]   Store result_json, return 202 AgentQuoteAwaitingResponse
  9b. [If complete] Write tool calls, update AgentRun, return 201 AgentQuoteResponse
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    AdjustmentDetail,
    AgentQuoteAwaitingResponse,
    AgentQuoteResponse,
    CustomerQuoteRequest,
    DrivingScoreResult,
    ExplanationDetail,
    ProfileRiskStats,
    UnderwritingDecision,
    UnderwritingSignal,
)
from src.models.model_registry import get_active_model_version
from src.database.models import (
    AgentRun,
    AgentToolCall,
    Customer,
    DrivingProfile,
    Policy,
    Prediction,
    Vehicle,
)
from src.database.session import get_db
from src.engine.driving_score import calculate_driving_score
from src.engine.pricing_engine import get_pricing_engine
from src.models.combined_predictor import get_predictor

from src.agents.underwriting.tools import async_fetch_profile_stats
from src.agents.underwriting.graph import get_underwriting_graph
from src.training.dataset_schema import NUMERIC_RANGES

logger = logging.getLogger("underwriting.routes")

router = APIRouter(dependencies=[Depends(verify_admin_key)])


def _compute_review_triggers(final_state: dict, ml_result: dict) -> list[str]:
    """Mirrors needs_human_review() — identifies which triggers fired."""
    triggers = []
    if final_state.get("confidence_score", 1.0) < 0.85:
        triggers.append("low_confidence")
    if final_state.get("fraud_signals"):
        triggers.append("fraud_signals_present")
    if float(ml_result.get("claim_probability", 0)) > 0.40:
        triggers.append("high_claim_probability")
    return triggers


@router.post(
    "/agent/quote",
    responses={
        201: {"model": AgentQuoteResponse,          "description": "Assessment complete — no review required"},
        202: {"model": AgentQuoteAwaitingResponse,  "description": "Awaiting human review"},
    },
    summary="Agentic underwriting quote",
    description=(
        "Runs the full ML + pricing pipeline and layers an AI underwriting "
        "assessment on top: fraud signal detection, segment comparison, "
        "LLM-generated reasoning, and a final approve / flag / refer decision. "
        "Returns 201 when fully resolved, 202 when the application requires "
        "human review before the decision is finalised."
    ),
)
async def agent_quote(
    request: CustomerQuoteRequest,
    db: AsyncSession = Depends(get_db),
):
    # ── 1. ML readiness check ──────────────────────────────────────────────────
    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Run training pipeline first.",
        )

    # ── 2. ML prediction ──────────────────────────────────────────────────────
    input_data = request.model_dump()

    # OOD check: log a warning when any numeric input falls outside the training range
    _ood = [
        f"{field}={input_data.get(field)} (training range {lo}–{hi})"
        for field, (lo, hi) in NUMERIC_RANGES.items()
        if input_data.get(field) is not None and not (lo <= input_data[field] <= hi)
    ]
    if _ood:
        logger.warning("[underwriting] out-of-distribution inputs: %s", "; ".join(_ood))

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

    ml_result  = predictor.predict(input_data)

    # ── 3. Pricing engine ─────────────────────────────────────────────────────
    engine = get_pricing_engine()
    pricing_result = engine.calculate(
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        age=request.age,
        driving_score=effective_driving_score,
        previous_claims_count=request.previous_claims,
        annual_mileage_km=request.annual_mileage_km,
        vehicle_value_inr=request.vehicle_value_inr,
    )

    # ── 4. Segment risk stats ─────────────────────────────────────────────────
    profile_stats = await async_fetch_profile_stats(db, input_data)

    # ── 5. Persist DB records ─────────────────────────────────────────────────
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
        is_active=False,
    )
    db.add(policy)
    await db.flush()

    prediction = Prediction(
        prediction_id=uuid.uuid4(),
        policy_id=policy.policy_id,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        final_premium_inr=pricing_result.final_premium,
        explanation_json=pricing_result.to_dict(),
    )
    db.add(prediction)
    await db.flush()

    # ── 6. Create AgentRun record ─────────────────────────────────────────────
    run_id = uuid.uuid4()
    agent_run = AgentRun(
        run_id=run_id,
        session_id=str(prediction.prediction_id),
        agent_type="underwriting",
        prediction_id=prediction.prediction_id,
        input_text=(
            f"age={request.age} gender={request.gender} city={request.city} "
            f"vehicle={request.vehicle_brand} segment={request.vehicle_segment} "
            f"fuel={request.fuel_type} value=₹{request.vehicle_value_inr:,.0f} "
            f"score={effective_driving_score} prev_claims={request.previous_claims}"
        ),
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(agent_run)
    await db.flush()

    # ── 7. Build initial state and invoke graph ────────────────────────────────
    initial_state = {
        "request_data":          input_data,
        "session_id":            str(prediction.prediction_id),
        "ml_result":             ml_result,
        "pricing_result":        pricing_result.to_dict(),
        "prediction_id":         str(prediction.prediction_id),
        "customer_history":      profile_stats,
        "fraud_signals":         [],
        "reasoning_trace":       "",
        "underwriting_decision": "",
        "confidence_score":      0.0,
        "risk_adjustment_factor": 1.0,
        "node_log":              [],
        "human_decision":        None,
        "review_note":           None,
    }

    # thread_id links this run to its LangGraph checkpoint
    config = {"configurable": {"thread_id": str(run_id)}}
    graph  = get_underwriting_graph()

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        agent_run.status       = "failed"
        agent_run.output_text  = str(exc)[:500]
        agent_run.completed_at = datetime.utcnow()
        logger.error(
            "Underwriting graph failed for prediction %s: %s",
            prediction.prediction_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Underwriting agent failed. Check server logs.",
        )

    # ── 8. Detect interrupt (graph paused at human_review) ────────────────────
    is_paused = False
    try:
        graph_state = await graph.aget_state(config)
        is_paused = bool(graph_state.next)
    except Exception:
        # Graph compiled without checkpointer — treat as complete
        pass

    # ── 9a. AWAITING REVIEW path (202) ────────────────────────────────────────
    if is_paused:
        review_triggers = _compute_review_triggers(final_state, ml_result)
        signals = final_state.get("fraud_signals", [])

        # Write partial tool calls from nodes that ran
        for entry in final_state.get("node_log", []):
            db.add(AgentToolCall(
                call_id=uuid.uuid4(),
                run_id=run_id,
                tool_name=entry.get("node", "unknown"),
                tool_input={"summary": entry.get("input_summary", "")},
                tool_output=entry.get("output_summary", ""),
                called_at=datetime.utcnow(),
                duration_ms=entry.get("duration_ms"),
            ))

        # Snapshot all state needed for the review UI and for resumption context
        agent_run.status      = "awaiting_review"
        agent_run.intent      = final_state.get("underwriting_decision", "")
        agent_run.result_json = {
            "fraud_signals":         signals,
            "customer_history":      profile_stats,
            "underwriting_decision": final_state.get("underwriting_decision", ""),
            "confidence_score":      final_state.get("confidence_score", 0.0),
            "risk_adjustment_factor": final_state.get("risk_adjustment_factor", 1.0),
            "reasoning_trace":       final_state.get("reasoning_trace", ""),
            "node_execution_log":    final_state.get("node_log", []),
            "claim_probability":     float(ml_result["claim_probability"]),
            "expected_claim_amount_inr": float(ml_result["expected_claim_amount_inr"]),
            "premium_amount_inr":    float(pricing_result.final_premium),
            "risk_level":            pricing_result.risk_level,
            "review_triggers":       review_triggers,
            "request_data":          input_data,
        }

        body = AgentQuoteAwaitingResponse(
            run_id=str(run_id),
            prediction_id=str(prediction.prediction_id),
            status="awaiting_review",
            premium_amount_inr=float(pricing_result.final_premium),
            risk_level=pricing_result.risk_level,
            claim_probability=float(ml_result["claim_probability"]),
            agent_decision=final_state.get("underwriting_decision", ""),
            confidence=final_state.get("confidence_score", 0.0),
            fraud_signal_count=len(signals),
            review_triggers=review_triggers,
            message=(
                "This application requires human review before a decision can be finalised. "
                "Visit the Review Queue tab to action this request."
            ),
        ).model_dump(mode="json")

        logger.info(
            "Underwriting run %s paused for human review (triggers=%s)",
            run_id, review_triggers,
        )
        return JSONResponse(content=body, status_code=202)

    # ── 9b. COMPLETE path (201) ────────────────────────────────────────────────
    for entry in final_state.get("node_log", []):
        db.add(AgentToolCall(
            call_id=uuid.uuid4(),
            run_id=run_id,
            tool_name=entry.get("node", "unknown"),
            tool_input={"summary": entry.get("input_summary", "")},
            tool_output=entry.get("output_summary", ""),
            called_at=datetime.utcnow(),
            duration_ms=entry.get("duration_ms"),
        ))

    agent_run.status       = "completed"
    agent_run.intent       = final_state.get("underwriting_decision", "")
    agent_run.output_text  = final_state.get("reasoning_trace", "")[:2000]
    agent_run.completed_at = datetime.utcnow()

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

    adj_factor = final_state.get("risk_adjustment_factor", 1.0)

    body = AgentQuoteResponse(
        prediction_id=prediction.prediction_id,
        premium_amount_inr=pricing_result.final_premium,
        risk_level=pricing_result.risk_level,
        claim_probability=ml_result["claim_probability"],
        expected_claim_amount_inr=ml_result["expected_claim_amount_inr"],
        explanation=explanation,
        model_version=get_active_model_version(),
        created_at=prediction.created_at or datetime.utcnow(),
        session_id=str(prediction.prediction_id),
        fraud_signals=[
            UnderwritingSignal(**s)
            for s in final_state.get("fraud_signals", [])
        ],
        customer_history=ProfileRiskStats(**profile_stats),
        underwriting_decision=UnderwritingDecision(
            decision=final_state.get("underwriting_decision", "approve"),
            confidence=final_state.get("confidence_score", 1.0),
            risk_adjustment_factor=adj_factor,
            agent_adjusted_premium=round(pricing_result.final_premium * adj_factor, 2),
        ),
        reasoning_trace=final_state.get("reasoning_trace", ""),
        node_execution_log=final_state.get("node_log", []),
        driving_score_info=driving_score_info,
    ).model_dump(mode="json")

    return JSONResponse(content=body, status_code=201)
