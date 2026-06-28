import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import verify_admin_key

_BACKEND_DIR = Path(__file__).parent.parent.parent.parent
from src.api.schemas import (
    AgentRunDetail,
    AgentRunSummary,
    AgentToolCallItem,
    ClaimAdminItem,
    CustomerListItem,
    DashboardStats,
    PaginatedResponse,
    PremiumTrendPoint,
)
from src.database.models import AgentRun, AgentToolCall, Claim, Customer, Policy, Prediction
from src.database.session import get_db

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.get("/admin/dashboard", response_model=DashboardStats)
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    total_policies = (await db.execute(select(func.count()).select_from(Policy))).scalar()
    active_policies = (await db.execute(
        select(func.count()).select_from(Policy).where(Policy.is_active == True)
    )).scalar()
    avg_premium = (await db.execute(select(func.avg(Policy.premium_amount_inr)))).scalar() or 0
    total_claims = (await db.execute(select(func.count()).select_from(Claim))).scalar()
    total_premium = (await db.execute(select(func.sum(Policy.premium_amount_inr)).where(Policy.is_active == True))).scalar() or 0

    claims_ratio = (total_claims / total_policies) if total_policies > 0 else 0

    risk_rows = (await db.execute(
        select(Policy.risk_level, func.count().label("cnt"), func.avg(Policy.premium_amount_inr).label("avg_premium"))
        .group_by(Policy.risk_level)
    )).all()

    premium_by_risk = {r.risk_level: float(r.avg_premium or 0) for r in risk_rows}
    policies_by_risk = {r.risk_level: r.cnt for r in risk_rows}

    return DashboardStats(
        total_policies=total_policies,
        active_policies=active_policies,
        avg_premium_inr=float(avg_premium),
        total_claims=total_claims,
        claims_ratio=round(claims_ratio, 4),
        total_premium_collected=float(total_premium),
        premium_by_risk_level=premium_by_risk,
        policies_by_risk_level=policies_by_risk,
    )


@router.get("/admin/customers", response_model=PaginatedResponse)
async def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = None,
    city: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size

    filters = []
    if risk_level:
        filters.append(Policy.risk_level == risk_level)
    if city:
        filters.append(Customer.city.ilike(f"%{city}%"))
    if date_from:
        filters.append(Customer.created_at >= date_from)
    if date_to:
        filters.append(Customer.created_at <= date_to)

    # Subquery for latest policy
    subq = (
        select(
            Policy.customer_id,
            func.max(Policy.created_at).label("latest_created"),
        )
        .group_by(Policy.customer_id)
        .subquery()
    )
    latest_policy = (
        select(Policy)
        .join(subq, and_(Policy.customer_id == subq.c.customer_id,
                          Policy.created_at == subq.c.latest_created))
        .subquery()
    )

    base_query = (
        select(
            Customer,
            func.count(Policy.policy_id).label("policy_count"),
            latest_policy.c.risk_level.label("latest_risk"),
            latest_policy.c.premium_amount_inr.label("latest_premium"),
        )
        .outerjoin(Policy, Policy.customer_id == Customer.customer_id)
        .outerjoin(latest_policy, latest_policy.c.customer_id == Customer.customer_id)
        .group_by(Customer.customer_id, latest_policy.c.risk_level, latest_policy.c.premium_amount_inr)
    )
    if filters:
        base_query = base_query.where(and_(*filters))

    total_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
    total = total_result.scalar()

    rows = (await db.execute(base_query.offset(offset).limit(page_size))).all()

    items = [
        CustomerListItem(
            customer_id=row.Customer.customer_id,
            age=row.Customer.age,
            gender=row.Customer.gender,
            city=row.Customer.city,
            created_at=row.Customer.created_at,
            policy_count=row.policy_count,
            latest_risk_level=row.latest_risk,
            latest_premium=float(row.latest_premium) if row.latest_premium else None,
        )
        for row in rows
    ]

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/admin/analytics/premium-trends", response_model=list)
async def premium_trends(db: AsyncSession = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(days=365)

    rows = (await db.execute(
        select(
            func.to_char(Policy.created_at, "YYYY-MM").label("month"),
            Policy.risk_level,
            func.avg(Policy.premium_amount_inr).label("avg_premium"),
            func.count().label("policy_count"),
        )
        .where(Policy.created_at >= cutoff)
        .group_by("month", Policy.risk_level)
        .order_by("month", Policy.risk_level)
    )).all()

    return [
        PremiumTrendPoint(
            month=row.month,
            risk_level=row.risk_level,
            avg_premium=float(row.avg_premium or 0),
            policy_count=row.policy_count,
        )
        for row in rows
    ]


@router.get("/admin/claims", response_model=PaginatedResponse)
async def list_all_claims(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size

    filters = []
    if status:
        filters.append(Claim.claim_status == status)

    base_q = (
        select(Claim, Policy.risk_level.label("risk_level"))
        .join(Policy, Policy.policy_id == Claim.policy_id)
    )
    if filters:
        base_q = base_q.where(and_(*filters))

    total = (await db.execute(
        select(func.count()).select_from(base_q.subquery())
    )).scalar()

    rows = (await db.execute(
        base_q.order_by(Claim.claim_date.desc()).offset(offset).limit(page_size)
    )).all()

    items = [
        ClaimAdminItem(
            claim_id=row.Claim.claim_id,
            policy_id=row.Claim.policy_id,
            claimed_amount_inr=float(row.Claim.claimed_amount_inr),
            approved_amount_inr=float(row.Claim.approved_amount_inr) if row.Claim.approved_amount_inr else None,
            claim_date=row.Claim.claim_date,
            claim_status=row.Claim.claim_status,
            risk_level=row.risk_level,
        )
        for row in rows
    ]

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/admin/agent-runs", response_model=PaginatedResponse)
async def list_agent_runs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent_type: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size

    q = select(AgentRun)
    if agent_type:
        q = q.where(AgentRun.agent_type == agent_type)
    if status_filter:
        q = q.where(AgentRun.status == status_filter)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar()
    rows = (
        await db.execute(q.order_by(AgentRun.started_at.desc()).offset(offset).limit(page_size))
    ).scalars().all()

    items = []
    for r in rows:
        dur = None
        if r.completed_at and r.started_at:
            dur = int((r.completed_at - r.started_at).total_seconds() * 1000)
        items.append(AgentRunSummary(
            run_id=r.run_id,
            session_id=r.session_id,
            agent_type=r.agent_type,
            intent=r.intent,
            status=r.status,
            started_at=r.started_at,
            completed_at=r.completed_at,
            duration_ms=dur,
            input_text=r.input_text,
            review_outcome=r.review_outcome,
        ))

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/admin/agent-runs/{run_id}", response_model=AgentRunDetail)
async def get_agent_run(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid run_id format — must be a UUID")

    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.run_id == run_uuid)
        .options(selectinload(AgentRun.tool_calls))
    )
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AgentRun not found")

    dur = None
    if run.completed_at and run.started_at:
        dur = int((run.completed_at - run.started_at).total_seconds() * 1000)

    tool_calls = [
        AgentToolCallItem(
            call_id=tc.call_id,
            tool_name=tc.tool_name,
            tool_input=tc.tool_input,
            tool_output=tc.tool_output,
            called_at=tc.called_at,
            duration_ms=tc.duration_ms,
        )
        for tc in sorted(run.tool_calls, key=lambda t: t.called_at)
    ]

    return AgentRunDetail(
        run_id=run.run_id,
        session_id=run.session_id,
        agent_type=run.agent_type,
        intent=run.intent,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=dur,
        input_text=run.input_text,
        output_text=run.output_text,
        tool_calls=tool_calls,
        review_outcome=run.review_outcome,
    )


# ─── Champion Registry & Governance (read-only, no DB) ────────────────────────

@router.get("/admin/registry")
async def get_champion_registry():
    path = _BACKEND_DIR / "models" / "metadata" / "champion_registry.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="champion_registry.json not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/admin/governance")
async def get_governance_log():
    path = _BACKEND_DIR / "models" / "metadata" / "model_governance_log.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_governance_log.json not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/admin/gate-results")
async def get_gate_results():
    path = _BACKEND_DIR / "experiments" / "outputs" / "phase4_gate_result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="phase4_gate_result.json not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
