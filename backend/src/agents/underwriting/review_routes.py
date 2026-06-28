"""
Human-in-the-Loop review endpoints — Phase 3B.

  GET  /admin/review-queue          → list runs awaiting human review
  GET  /admin/review/{run_id}       → full detail for a single queued run
  POST /admin/review/{run_id}/approve  → resume graph, mark approved
  POST /admin/review/{run_id}/reject   → skip resume, mark rejected
"""
from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    PaginatedResponse,
    ProfileRiskStats,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewDetail,
    ReviewQueueItem,
    UnderwritingDecision,
    UnderwritingSignal,
)
from src.database.models import AgentRun, AgentToolCall
from src.database.session import get_db
from src.agents.underwriting.graph import get_underwriting_graph

logger = logging.getLogger("underwriting.review_routes")

router = APIRouter(dependencies=[Depends(verify_admin_key)])


def _queue_item_from_run(run: AgentRun) -> ReviewQueueItem:
    """Build a ReviewQueueItem from an AgentRun ORM object."""
    rj = run.result_json or {}
    now = datetime.utcnow()
    pending_seconds = (now - run.started_at).total_seconds() if run.started_at else 0
    minutes_pending = int(pending_seconds // 60)

    signals = rj.get("fraud_signals", [])
    high_count = sum(1 for s in signals if s.get("severity") == "high")

    return ReviewQueueItem(
        run_id=run.run_id,
        prediction_id=run.session_id,
        started_at=run.started_at,
        minutes_pending=minutes_pending,
        agent_decision=rj.get("underwriting_decision"),
        confidence_score=float(rj.get("confidence_score", 0.0)),
        claim_probability=float(rj.get("claim_probability", 0.0)),
        fraud_signal_count=len(signals),
        high_severity_count=high_count,
        premium_amount_inr=float(rj.get("premium_amount_inr", 0.0)),
        review_triggers=rj.get("review_triggers", []),
        input_summary=run.input_text,
    )


@router.get("/admin/review-queue", response_model=PaginatedResponse)
async def get_review_queue(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all underwriting runs currently awaiting human review, newest first."""
    offset = (page - 1) * page_size

    q = (
        select(AgentRun)
        .where(AgentRun.status == "awaiting_review")
        .where(AgentRun.agent_type == "underwriting")
    )
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows = (
        await db.execute(q.order_by(AgentRun.started_at.desc()).offset(offset).limit(page_size))
    ).scalars().all()

    items = [_queue_item_from_run(r) for r in rows]

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/admin/review/{run_id}", response_model=ReviewDetail)
async def get_review_detail(run_id: str, db: AsyncSession = Depends(get_db)):
    """Full structured detail for a single queued underwriting run."""
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid run_id — must be a UUID")

    run = await db.get(AgentRun, run_uuid)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun not found")
    if run.status != "awaiting_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is not awaiting review (status={run.status})",
        )

    rj = run.result_json or {}
    signals = rj.get("fraud_signals", [])
    high_count = sum(1 for s in signals if s.get("severity") == "high")
    now = datetime.utcnow()
    minutes_pending = int((now - run.started_at).total_seconds() // 60) if run.started_at else 0

    ch = rj.get("customer_history", {})
    ud_decision = rj.get("underwriting_decision", "")
    confidence = float(rj.get("confidence_score", 0.0))
    adj_factor = float(rj.get("risk_adjustment_factor", 1.0))
    premium = float(rj.get("premium_amount_inr", 0.0))

    return ReviewDetail(
        run_id=run.run_id,
        prediction_id=run.session_id,
        started_at=run.started_at,
        minutes_pending=minutes_pending,
        agent_decision=ud_decision,
        confidence_score=confidence,
        claim_probability=float(rj.get("claim_probability", 0.0)),
        fraud_signal_count=len(signals),
        high_severity_count=high_count,
        premium_amount_inr=premium,
        review_triggers=rj.get("review_triggers", []),
        input_summary=run.input_text,
        fraud_signals=[UnderwritingSignal(**s) for s in signals],
        customer_history=ProfileRiskStats(
            similar_policies_count=ch.get("similar_policies_count", 0),
            avg_premium_in_segment=float(ch.get("avg_premium_in_segment", 0.0)),
            segment_claim_frequency=float(ch.get("segment_claim_frequency", 0.0)),
            total_claims_in_segment=ch.get("total_claims_in_segment", 0),
        ),
        underwriting_decision=UnderwritingDecision(
            decision=ud_decision or "approve",
            confidence=confidence,
            risk_adjustment_factor=adj_factor,
            agent_adjusted_premium=round(premium * adj_factor, 2),
        ),
        reasoning_trace=rj.get("reasoning_trace", ""),
        node_execution_log=rj.get("node_execution_log", []),
    )


@router.post("/admin/review/{run_id}/approve", response_model=ReviewDecisionResponse)
async def approve_review(
    run_id: str,
    body: ReviewDecisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Approve a queued underwriting application.

    Injects human_decision='approved' into the LangGraph checkpoint, resumes
    the graph (running human_review_node), then marks the AgentRun completed.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid run_id — must be a UUID")

    run = await db.get(AgentRun, run_uuid)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun not found")
    if run.status != "awaiting_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is not awaiting review (status={run.status})",
        )

    graph  = get_underwriting_graph()
    config = {"configurable": {"thread_id": str(run_uuid)}}

    try:
        await graph.aupdate_state(config, {
            "human_decision": "approved",
            "review_note": body.reviewer_note or "",
        })
        await graph.ainvoke(None, config=config)
    except Exception as exc:
        logger.error("Graph resume failed for run %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph resume failed — check server logs",
        )

    now = datetime.utcnow()
    run.status        = "completed"
    run.review_outcome = "approved"
    run.reviewed_at   = now
    run.reviewer_note = body.reviewer_note
    run.completed_at  = now

    db.add(AgentToolCall(
        call_id=uuid.uuid4(),
        run_id=run_uuid,
        tool_name="human_review",
        tool_input={"decision": "approved"},
        tool_output=(
            f"Approved by reviewer."
            + (f" Note: {body.reviewer_note}" if body.reviewer_note else "")
        ),
        called_at=now,
        duration_ms=0,
    ))

    agent_decision = (run.result_json or {}).get("underwriting_decision")
    logger.info("Run %s approved by reviewer", run_id)

    return ReviewDecisionResponse(
        run_id=run.run_id,
        review_outcome="approved",
        reviewed_at=now,
        reviewer_note=body.reviewer_note,
        agent_decision=agent_decision,
    )


@router.get("/underwriting/review-queue", response_model=PaginatedResponse)
async def get_underwriting_review_queue(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Alias for /admin/review-queue — underwriting applications awaiting human review."""
    offset = (page - 1) * page_size
    q = (
        select(AgentRun)
        .where(AgentRun.status == "awaiting_review")
        .where(AgentRun.agent_type == "underwriting")
    )
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows = (
        await db.execute(q.order_by(AgentRun.started_at.desc()).offset(offset).limit(page_size))
    ).scalars().all()
    items = [_queue_item_from_run(r) for r in rows]
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/underwriting/stats")
async def get_underwriting_stats(db: AsyncSession = Depends(get_db)):
    """Aggregate statistics for the underwriting agent."""
    total = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
    )).scalar()
    awaiting = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
        .where(AgentRun.status == "awaiting_review")
    )).scalar()
    completed = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
        .where(AgentRun.status == "completed")
    )).scalar()
    approved = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
        .where(AgentRun.review_outcome == "approved")
    )).scalar()
    rejected = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
        .where(AgentRun.review_outcome == "rejected")
    )).scalar()
    failed = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.agent_type == "underwriting")
        .where(AgentRun.status == "failed")
    )).scalar()
    return {
        "total_runs": total,
        "awaiting_review": awaiting,
        "completed": completed,
        "approved": approved,
        "rejected": rejected,
        "failed": failed,
    }


@router.post("/admin/review/{run_id}/reject", response_model=ReviewDecisionResponse)
async def reject_review(
    run_id: str,
    body: ReviewDecisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Reject a queued underwriting application.

    Does NOT resume the graph — the run is simply closed as rejected.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid run_id — must be a UUID")

    run = await db.get(AgentRun, run_uuid)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun not found")
    if run.status != "awaiting_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run is not awaiting review (status={run.status})",
        )

    now = datetime.utcnow()
    run.status         = "completed"
    run.review_outcome = "rejected"
    run.reviewed_at    = now
    run.reviewer_note  = body.reviewer_note
    run.completed_at   = now

    agent_decision = (run.result_json or {}).get("underwriting_decision")
    logger.info("Run %s rejected by reviewer", run_id)

    return ReviewDecisionResponse(
        run_id=run.run_id,
        review_outcome="rejected",
        reviewed_at=now,
        reviewer_note=body.reviewer_note,
        agent_decision=agent_decision,
    )
