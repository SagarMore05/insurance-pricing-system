"""V4 Production Monitoring — read-only analytics endpoints.

No model mutations. No retraining triggers. No champion changes.
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import (
    DataQualityCheck,
    MonitoringDashboard,
    MonitoringHITLStats,
    MonitoringLabelStats,
    MonitoringPremiumBand,
    MonitoringRiskBand,
    MonitoringVolumeStat,
    MonitoringVolumeTrend,
)
from src.database.models import (
    AgentRun, Customer, ModelFeedback, Policy, Prediction, Vehicle,
)
from src.database.session import get_db

router = APIRouter(dependencies=[Depends(verify_admin_key)])

_MIN_PREMIUM = 3000.0
_MAX_PREMIUM = 150000.0


def _fill_pct(passing: int, total: int) -> float:
    return round((passing / total * 100) if total > 0 else 0.0, 1)


def _dq_status(rate: float) -> str:
    if rate >= 90:
        return "ok"
    if rate >= 70:
        return "warning"
    return "critical"


@router.get("/admin/monitoring/dashboard", response_model=MonitoringDashboard)
async def get_monitoring_dashboard(db: AsyncSession = Depends(get_db)):
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago  = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)
    ninety_days_ago = now - timedelta(days=90)

    # ── Quote volume ──────────────────────────────────────────────────────────
    total_quotes = (await db.execute(
        select(func.count()).select_from(Prediction)
    )).scalar() or 0

    quotes_today = (await db.execute(
        select(func.count()).select_from(Prediction)
        .where(Prediction.created_at >= today_start)
    )).scalar() or 0

    quotes_7d = (await db.execute(
        select(func.count()).select_from(Prediction)
        .where(Prediction.created_at >= seven_days_ago)
    )).scalar() or 0

    quotes_30d = (await db.execute(
        select(func.count()).select_from(Prediction)
        .where(Prediction.created_at >= thirty_days_ago)
    )).scalar() or 0

    v2_quotes = (await db.execute(
        select(func.count()).select_from(Prediction)
        .where(Prediction.inference_features_json.isnot(None))
    )).scalar() or 0

    v2_share_pct = round((v2_quotes / total_quotes * 100) if total_quotes > 0 else 0.0, 1)

    # ── Premium stats by risk level ───────────────────────────────────────────
    overall_mean = (await db.execute(
        select(func.avg(Prediction.final_premium_inr))
    )).scalar() or 0

    premium_rows = (await db.execute(
        select(
            Policy.risk_level,
            func.count().label("cnt"),
            func.avg(Prediction.final_premium_inr).label("mean_p"),
            func.min(Prediction.final_premium_inr).label("min_p"),
            func.max(Prediction.final_premium_inr).label("max_p"),
            func.count().filter(Prediction.final_premium_inr <= _MIN_PREMIUM + 1).label("at_min"),
            func.count().filter(Prediction.final_premium_inr >= _MAX_PREMIUM - 1).label("at_max"),
        )
        .select_from(Prediction)
        .join(Policy, Policy.policy_id == Prediction.policy_id, isouter=True)
        .group_by(Policy.risk_level)
    )).all()

    premium_by_risk = [
        MonitoringPremiumBand(
            risk_level=r.risk_level or "unknown",
            count=r.cnt,
            mean_premium=round(float(r.mean_p or 0), 2),
            min_premium=round(float(r.min_p or 0), 2),
            max_premium=round(float(r.max_p or 0), 2),
            at_min_cap_count=int(r.at_min or 0),
            at_max_cap_count=int(r.at_max or 0),
        )
        for r in premium_rows
    ]

    # ── Risk distribution ─────────────────────────────────────────────────────
    total_policies = (await db.execute(
        select(func.count()).select_from(Policy)
    )).scalar() or 0

    risk_rows = (await db.execute(
        select(
            Policy.risk_level,
            func.count().label("cnt"),
            func.avg(Prediction.claim_probability).label("mean_cp"),
        )
        .select_from(Policy)
        .join(Prediction, Prediction.policy_id == Policy.policy_id, isouter=True)
        .group_by(Policy.risk_level)
    )).all()

    risk_distribution = [
        MonitoringRiskBand(
            risk_level=r.risk_level or "unknown",
            count=r.cnt,
            pct=round((r.cnt / total_policies * 100) if total_policies > 0 else 0.0, 1),
            mean_claim_prob=round(float(r.mean_cp or 0), 4),
        )
        for r in risk_rows
    ]

    # ── Data quality checks ───────────────────────────────────────────────────
    v2_30d = (await db.execute(
        select(func.count()).select_from(Prediction)
        .where(and_(
            Prediction.created_at >= thirty_days_ago,
            Prediction.inference_features_json.isnot(None),
        ))
    )).scalar() or 0
    inf_json_fill = _fill_pct(v2_30d, quotes_30d)

    cust_30d = (await db.execute(
        select(func.count()).select_from(Customer)
        .where(Customer.created_at >= thirty_days_ago)
    )).scalar() or 0
    occ_30d = (await db.execute(
        select(func.count()).select_from(Customer)
        .where(and_(Customer.created_at >= thirty_days_ago, Customer.occupation.isnot(None)))
    )).scalar() or 0
    marital_30d = (await db.execute(
        select(func.count()).select_from(Customer)
        .where(and_(Customer.created_at >= thirty_days_ago, Customer.marital_status.isnot(None)))
    )).scalar() or 0
    income_30d = (await db.execute(
        select(func.count()).select_from(Customer)
        .where(and_(Customer.created_at >= thirty_days_ago, Customer.annual_income_band.isnot(None)))
    )).scalar() or 0

    # Vehicle has no created_at — proxy via Customer join
    veh_30d = (await db.execute(
        select(func.count()).select_from(Vehicle)
        .join(Customer, Customer.customer_id == Vehicle.customer_id)
        .where(Customer.created_at >= thirty_days_ago)
    )).scalar() or 0
    fuel_30d = (await db.execute(
        select(func.count()).select_from(Vehicle)
        .join(Customer, Customer.customer_id == Vehicle.customer_id)
        .where(and_(Customer.created_at >= thirty_days_ago, Vehicle.fuel_type.isnot(None)))
    )).scalar() or 0
    safety_30d = (await db.execute(
        select(func.count()).select_from(Vehicle)
        .join(Customer, Customer.customer_id == Vehicle.customer_id)
        .where(and_(Customer.created_at >= thirty_days_ago, Vehicle.vehicle_safety_rating.isnot(None)))
    )).scalar() or 0

    data_quality = [
        DataQualityCheck(
            check_id="DQ-001",
            description="V4 Feature JSON stored (inference_features_json) — last 30d",
            table_name="predictions",
            total_rows=quotes_30d,
            passing_rows=v2_30d,
            fill_rate_pct=inf_json_fill,
            target_pct=100.0,
            status=_dq_status(inf_json_fill),
        ),
        DataQualityCheck(
            check_id="DQ-002",
            description="Customer Occupation field — last 30d",
            table_name="customers",
            total_rows=cust_30d,
            passing_rows=occ_30d,
            fill_rate_pct=_fill_pct(occ_30d, cust_30d),
            target_pct=100.0,
            status=_dq_status(_fill_pct(occ_30d, cust_30d)),
        ),
        DataQualityCheck(
            check_id="DQ-003",
            description="Customer Marital Status field — last 30d",
            table_name="customers",
            total_rows=cust_30d,
            passing_rows=marital_30d,
            fill_rate_pct=_fill_pct(marital_30d, cust_30d),
            target_pct=100.0,
            status=_dq_status(_fill_pct(marital_30d, cust_30d)),
        ),
        DataQualityCheck(
            check_id="DQ-004",
            description="Customer Annual Income Band field — last 30d",
            table_name="customers",
            total_rows=cust_30d,
            passing_rows=income_30d,
            fill_rate_pct=_fill_pct(income_30d, cust_30d),
            target_pct=100.0,
            status=_dq_status(_fill_pct(income_30d, cust_30d)),
        ),
        DataQualityCheck(
            check_id="DQ-005",
            description="Vehicle Fuel Type field — last 30d",
            table_name="vehicles",
            total_rows=veh_30d,
            passing_rows=fuel_30d,
            fill_rate_pct=_fill_pct(fuel_30d, veh_30d),
            target_pct=100.0,
            status=_dq_status(_fill_pct(fuel_30d, veh_30d)),
        ),
        DataQualityCheck(
            check_id="DQ-006",
            description="Vehicle Safety Rating (lookup-derived) — last 30d",
            table_name="vehicles",
            total_rows=veh_30d,
            passing_rows=safety_30d,
            fill_rate_pct=_fill_pct(safety_30d, veh_30d),
            target_pct=100.0,
            status=_dq_status(_fill_pct(safety_30d, veh_30d)),
        ),
    ]

    # ── HITL stats ─────────────────────────────────────────────────────────────
    queue_depth = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(AgentRun.status == "awaiting_review")
    )).scalar() or 0

    reviews_30d = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(and_(AgentRun.status == "completed", AgentRun.completed_at >= thirty_days_ago))
    )).scalar() or 0

    rejected_30d = (await db.execute(
        select(func.count()).select_from(AgentRun)
        .where(and_(AgentRun.review_outcome == "rejected", AgentRun.completed_at >= thirty_days_ago))
    )).scalar() or 0

    override_rate = round((rejected_30d / reviews_30d * 100), 1) if reviews_30d > 0 else None

    avg_secs = (await db.execute(
        select(func.avg(
            func.extract("epoch", AgentRun.completed_at) - func.extract("epoch", AgentRun.started_at)
        ))
        .select_from(AgentRun)
        .where(and_(
            AgentRun.status == "completed",
            AgentRun.completed_at.isnot(None),
            AgentRun.completed_at >= thirty_days_ago,
        ))
    )).scalar()
    avg_review_hours = round(float(avg_secs) / 3600, 2) if avg_secs is not None else None

    hitl_stats = MonitoringHITLStats(
        queue_depth=queue_depth,
        reviews_last_30d=reviews_30d,
        override_rate=override_rate,
        avg_review_hours=avg_review_hours,
    )

    # ── Label coverage ─────────────────────────────────────────────────────────
    labelled = (await db.execute(
        select(func.count()).select_from(ModelFeedback)
    )).scalar() or 0

    matured_90d = (await db.execute(
        select(func.count()).select_from(Policy)
        .where(Policy.created_at <= ninety_days_ago)
    )).scalar() or 0

    matured_labelled = (await db.execute(
        select(func.count()).select_from(ModelFeedback)
        .join(Prediction, Prediction.prediction_id == ModelFeedback.prediction_id)
        .join(Policy, Policy.policy_id == Prediction.policy_id)
        .where(Policy.created_at <= ninety_days_ago)
    )).scalar() or 0

    label_stats = MonitoringLabelStats(
        total_predictions=total_quotes,
        labelled_predictions=labelled,
        coverage_pct=_fill_pct(labelled, total_quotes),
        matured_policies_90d=matured_90d,
        matured_labelled_90d=matured_labelled,
        matured_coverage_pct=_fill_pct(matured_labelled, matured_90d),
    )

    return MonitoringDashboard(
        generated_at=now.isoformat(),
        total_quotes=total_quotes,
        quotes_today=quotes_today,
        quotes_last_7d=quotes_7d,
        quotes_last_30d=quotes_30d,
        v2_share_pct=v2_share_pct,
        overall_mean_premium=round(float(overall_mean), 2),
        premium_by_risk=premium_by_risk,
        risk_distribution=risk_distribution,
        data_quality=data_quality,
        inference_json_fill_pct=inf_json_fill,
        hitl_stats=hitl_stats,
        label_stats=label_stats,
    )


@router.get("/admin/monitoring/volume-trend", response_model=MonitoringVolumeTrend)
async def get_volume_trend(db: AsyncSession = Depends(get_db)):
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    rows = (await db.execute(
        select(
            func.date(Prediction.created_at).label("day"),
            func.count().label("count"),
            func.count().filter(Prediction.inference_features_json.isnot(None)).label("v2_count"),
            func.count().filter(Prediction.inference_features_json.is_(None)).label("v1_count"),
        )
        .where(Prediction.created_at >= thirty_days_ago)
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at))
    )).all()

    return MonitoringVolumeTrend(
        trend=[
            MonitoringVolumeStat(
                date=str(r.day),
                count=r.count,
                v2_count=int(r.v2_count or 0),
                v1_count=int(r.v1_count or 0),
            )
            for r in rows
        ]
    )
