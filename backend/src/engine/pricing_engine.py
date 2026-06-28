from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import math


MIN_PREMIUM_INR = 3_000.0
MAX_PREMIUM_INR = 150_000.0
LOADING_FACTOR = 1.35


@dataclass
class Adjustment:
    reason: str
    factor: float
    impact_inr: float


@dataclass
class PricingResult:
    base_premium: float
    expected_loss: float
    adjustments: List[Adjustment]
    final_premium: float
    risk_level: str
    claim_probability: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_premium": round(self.base_premium, 2),
            "expected_loss": round(self.expected_loss, 2),
            "adjustments": [
                {
                    "reason": a.reason,
                    "factor": a.factor,
                    "impact_inr": round(a.impact_inr, 2),
                }
                for a in self.adjustments
            ],
            "final_premium": round(self.final_premium, 2),
            "risk_level": self.risk_level,
            "claim_probability": round(self.claim_probability, 5),
            "summary": self.summary,
        }


class PricingEngine:
    """
    Deterministic rules-based pricing engine.
    Takes ML model outputs and applies business rules to compute final premium.
    No LLM calls — fully deterministic.
    """

    def __init__(
        self,
        min_premium: float = MIN_PREMIUM_INR,
        max_premium: float = MAX_PREMIUM_INR,
        loading_factor: float = LOADING_FACTOR,
    ):
        self.min_premium = min_premium
        self.max_premium = max_premium
        self.loading_factor = loading_factor

    def _classify_risk(self, claim_probability: float) -> str:
        # Thresholds calibrated to the XGBoost model's output distribution:
        # p50 ≈ 0.226, p75 ≈ 0.405 → meaningful three-band segmentation.
        if claim_probability < 0.20:
            return "low"
        elif claim_probability < 0.40:
            return "medium"
        return "high"

    def _apply_business_rules(
        self, age: int, driving_score: float, previous_claims: int,
        annual_mileage_km: int, vehicle_value_inr: float,
        no_claim_bonus_pct: float = 0.0,
    ) -> List[Tuple[str, float]]:
        multipliers: List[Tuple[str, float]] = []

        # Age adjustments
        if age < 25:
            multipliers.append(("young_driver_surcharge", 1.30))
        elif age > 70:
            multipliers.append(("senior_driver_surcharge", 1.15))

        # Driving score
        if driving_score >= 85:
            multipliers.append(("safe_driver_discount", 0.85))
        elif driving_score < 50:
            multipliers.append(("poor_driver_surcharge", 1.25))

        # Claims history — IRDAI NCB schedule (0/20/25/35/45/50%)
        if no_claim_bonus_pct > 0:
            factor = round(1.0 - no_claim_bonus_pct / 100.0, 4)
            multipliers.append(("no_claims_bonus", factor))
        elif previous_claims >= 3:
            multipliers.append(("high_claims_surcharge", 1.40))

        # Mileage
        if annual_mileage_km > 30_000:
            multipliers.append(("high_mileage_surcharge", 1.20))
        elif annual_mileage_km < 5_000:
            multipliers.append(("low_mileage_discount", 0.92))

        # Vehicle value
        if vehicle_value_inr > 2_000_000:
            multipliers.append(("luxury_vehicle_surcharge", 1.15))

        return multipliers

    def _build_summary(
        self, final_premium: float, risk_level: str,
        adjustments: List[Adjustment], base_premium: float
    ) -> str:
        discount_items = [(a.reason, a.impact_inr) for a in adjustments if a.factor < 1]
        surcharge_items = [(a.reason, a.impact_inr) for a in adjustments if a.factor > 1]

        risk_label = risk_level.capitalize()
        premium_str = f"₹{final_premium:,.0f}"
        parts = [f"Your premium is {premium_str} ({risk_label} Risk)."]

        readable = {
            "safe_driver_discount": "strong driving score",
            "no_claims_bonus": "clean claims history",
            "low_mileage_discount": "low annual mileage",
            "young_driver_surcharge": "young driver age",
            "senior_driver_surcharge": "senior driver age",
            "poor_driver_surcharge": "driving record",
            "high_claims_surcharge": "previous claims history",
            "high_mileage_surcharge": "high annual mileage",
            "luxury_vehicle_surcharge": "high-value vehicle",
        }

        if discount_items:
            name, amt = max(discount_items, key=lambda x: abs(x[1]))
            label = readable.get(name, name.replace("_", " "))
            parts.append(f"Your {label} saved you ₹{abs(amt):,.0f}.")

        if surcharge_items:
            name, amt = max(surcharge_items, key=lambda x: abs(x[1]))
            label = readable.get(name, name.replace("_", " "))
            parts.append(f"Your {label} added ₹{abs(amt):,.0f}.")

        return " ".join(parts)

    def calculate(
        self,
        claim_probability: float,
        expected_claim_amount_inr: float,
        age: int,
        driving_score: float,
        previous_claims_count: int,
        annual_mileage_km: int,
        vehicle_value_inr: float,
        no_claim_bonus_pct: float = 0.0,
    ) -> PricingResult:
        # Step 1 — Base premium
        expected_loss = claim_probability * expected_claim_amount_inr
        base_premium = expected_loss * self.loading_factor

        # Step 2 — Business rules
        rule_multipliers = self._apply_business_rules(
            age=age,
            driving_score=driving_score,
            previous_claims=previous_claims_count,
            annual_mileage_km=annual_mileage_km,
            vehicle_value_inr=vehicle_value_inr,
            no_claim_bonus_pct=no_claim_bonus_pct,
        )

        # Step 3 — Compute adjustments with INR impact
        running_premium = base_premium
        adjustments: List[Adjustment] = []
        for reason, factor in rule_multipliers:
            new_premium = running_premium * factor
            impact = new_premium - running_premium
            adjustments.append(Adjustment(reason=reason, factor=factor, impact_inr=impact))
            running_premium = new_premium

        # Step 3 — Apply floor/cap
        final_premium = max(self.min_premium, min(self.max_premium, running_premium))

        # Step 4 — Risk classification
        risk_level = self._classify_risk(claim_probability)

        # Step 5 — Summary
        summary = self._build_summary(final_premium, risk_level, adjustments, base_premium)

        return PricingResult(
            base_premium=base_premium,
            expected_loss=expected_loss,
            adjustments=adjustments,
            final_premium=final_premium,
            risk_level=risk_level,
            claim_probability=claim_probability,
            summary=summary,
        )


# Module-level singleton
_engine = PricingEngine()


def get_pricing_engine() -> PricingEngine:
    return _engine
