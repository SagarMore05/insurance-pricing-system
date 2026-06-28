"""
Underwriting Agent tools.

Two responsibilities:
  1. async_fetch_profile_stats — DB aggregate query for the applicant's city segment
  2. evaluate_fraud_signals    — rule-based red-flag analysis, no LLM required
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Claim, Customer, Policy

if TYPE_CHECKING:
    pass

logger = logging.getLogger("underwriting.tools")

# ─── Tool 1: Customer / segment history ───────────────────────────────────────

async def async_fetch_profile_stats(db: AsyncSession, request_data: dict) -> dict:
    """
    Query aggregate risk stats for the applicant's city segment.

    Because no Customer record exists yet at quote time, we use city + gender
    as the segment proxy — the closest comparable population on the books.
    """
    city: str = request_data.get("city", "")
    gender: str = request_data.get("gender", "")

    try:
        # Policy count + avg premium in segment
        stmt_policies = (
            select(
                func.count(Policy.policy_id.distinct()).label("cnt"),
                func.avg(Policy.premium_amount_inr).label("avg_prem"),
            )
            .join(Customer, Policy.customer_id == Customer.customer_id)
            .where(Customer.city == city)
        )
        row_p = (await db.execute(stmt_policies)).one_or_none()
        similar_count = int(row_p.cnt or 0) if row_p else 0
        avg_premium = float(row_p.avg_prem or 0.0) if row_p else 0.0

        # Claim frequency in segment
        stmt_claims = (
            select(
                func.count(Claim.claim_id.distinct()).label("claims"),
                func.count(Policy.policy_id.distinct()).label("policies"),
            )
            .join(Policy, Claim.policy_id == Policy.policy_id)
            .join(Customer, Policy.customer_id == Customer.customer_id)
            .where(Customer.city == city)
        )
        row_c = (await db.execute(stmt_claims)).one_or_none()
        total_claims = int(row_c.claims or 0) if row_c else 0
        total_policies = max(int(row_c.policies or 1) if row_c else 1, 1)

        return {
            "similar_policies_count": similar_count,
            "avg_premium_in_segment": round(avg_premium, 2),
            "segment_claim_frequency": round(total_claims / total_policies, 4),
            "total_claims_in_segment": total_claims,
        }

    except Exception as exc:
        logger.warning("Profile stats query failed: %s — returning zeros", exc)
        return {
            "similar_policies_count": 0,
            "avg_premium_in_segment": 0.0,
            "segment_claim_frequency": 0.0,
            "total_claims_in_segment": 0,
        }


# ─── Tool 2: Fraud signal detection ───────────────────────────────────────────

_HIGH = "high"
_MED  = "medium"
_LOW  = "low"


def evaluate_fraud_signals(request_data: dict, profile_stats: dict) -> list[dict]:
    """
    Rule-based fraud / anomaly signals.  No LLM — fully deterministic.

    Returns a list of signal dicts: {signal, severity, description}
    Severity: "high" triggers refer, "medium" triggers flag, "low" is advisory.
    """
    signals: list[dict] = []

    age              = int(request_data.get("age") or 30)
    vehicle_value    = float(request_data.get("vehicle_value_inr") or 0)
    vehicle_age      = int(request_data.get("vehicle_age_years") or 0)
    engine_cc        = int(request_data.get("engine_cc") or 0)
    driving_score    = float(request_data.get("driving_score") or 50)
    mileage          = int(request_data.get("annual_mileage_km") or 0)
    prev_claims      = int(request_data.get("previous_claims") or request_data.get("previous_claims_count") or 0)
    segment_freq     = float(profile_stats.get("segment_claim_frequency", 0))

    # 1. Excessive prior claims — strongest single predictor of future claims
    if prev_claims >= 4:
        signals.append({
            "signal": "excessive_prior_claims",
            "severity": _HIGH,
            "description": (
                f"{prev_claims} prior claims is well above the segment average. "
                "Refer for manual review before binding."
            ),
        })
    elif prev_claims == 3:
        signals.append({
            "signal": "elevated_prior_claims",
            "severity": _MED,
            "description": f"{prev_claims} prior claims is above average for this profile.",
        })

    # 2. Young driver + high-performance vehicle
    if age < 25 and engine_cc > 2500:
        signals.append({
            "signal": "young_high_performance",
            "severity": _MED,
            "description": (
                f"Driver aged {age} with {engine_cc}cc engine. "
                "Young drivers in high-performance vehicles carry significantly elevated accident risk."
            ),
        })

    # 3. Young driver + high-value vehicle (potential over-insurance)
    if age < 25 and vehicle_value > 1_500_000:
        signals.append({
            "signal": "young_high_value_vehicle",
            "severity": _MED,
            "description": (
                f"Vehicle value ₹{vehicle_value:,.0f} is unusually high for a driver aged {age}. "
                "Verify ownership and valuation documentation."
            ),
        })

    # 4. Low driving score + high mileage (high exposure, poor driving)
    if driving_score < 35 and mileage > 40_000:
        signals.append({
            "signal": "low_score_high_exposure",
            "severity": _HIGH,
            "description": (
                f"Driving score {driving_score}/100 combined with {mileage:,} km/year "
                "indicates high-risk, high-exposure profile."
            ),
        })

    # 5. Claim history contradicts driving score (score inflation suspected)
    if prev_claims >= 2 and driving_score > 80:
        signals.append({
            "signal": "score_claims_mismatch",
            "severity": _MED,
            "description": (
                f"Driving score {driving_score}/100 is inconsistent with "
                f"{prev_claims} prior claims. Score may not reflect true driving behaviour."
            ),
        })

    # 6. Very high mileage — potential commercial use on personal policy
    if mileage > 80_000:
        signals.append({
            "signal": "extreme_annual_mileage",
            "severity": _MED,
            "description": (
                f"{mileage:,} km/year exceeds typical personal-use thresholds. "
                "Confirm vehicle is not used for commercial or hire purposes."
            ),
        })

    # 7. Old high-value vehicle (valuation risk)
    if vehicle_age > 12 and vehicle_value > 1_800_000:
        signals.append({
            "signal": "aged_high_value_discrepancy",
            "severity": _LOW,
            "description": (
                f"A {vehicle_age}-year-old vehicle declared at ₹{vehicle_value:,.0f} "
                "may be over-valued relative to market depreciation."
            ),
        })

    return signals
