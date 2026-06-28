"""
V4DatasetBuilder
================
Reads persisted V4 inference records from the predictions table via the
inference_features_json JSONB column and joins model_feedback labels to
produce a training-ready pandas DataFrame for future V5 retraining.

Design decisions:
- Reads from inference_features_json JSONB — NOT from normalized columns.
  (pipeline.py collect_features_with_labels() joins only 12 V1 columns; this
   class supersedes it for V4 records.)
- Maturity filter: policies must be >= LABEL_MATURITY_DAYS old so that
  claim outcomes are observable (first labels expected ~late September 2026).
- Returns all 31 raw V4 features + claim_occurred + actual_claim_amount_inr.
- Reports per-feature fill rates and rows dropped at each validation stage.
- Does NOT train models. Does NOT modify champions. Read-only DB access.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ModelFeedback, Policy, Prediction

logger = logging.getLogger("dataset_builder_v4")

# ── 31 raw V4 feature keys expected in inference_features_json JSONB ─────────
V4_RAW_FEATURES: List[str] = [
    # Personal
    "age", "gender", "occupation", "marital_status", "annual_income_band",
    # Location
    "city",
    # Driving history
    "years_licensed", "previous_claims_count", "months_since_last_claim",
    "no_claim_bonus_pct", "policy_tenure_years",
    # Vehicle identification
    "vehicle_make", "vehicle_model",
    # Vehicle usage
    "fuel_type", "vehicle_usage_type", "annual_mileage_km", "parking_type",
    # Telematics / risk
    "driving_score", "pincode_risk_score", "region_risk_category",
    "flood_risk_index", "theft_risk_index", "monsoon_exposure_index",
    # Vehicle specs
    "vehicle_value_inr", "vehicle_age_years", "engine_cc",
    "vehicle_safety_rating", "airbags_count",
    # Categorical specs
    "repair_cost_band", "vehicle_body_style",
    # System-derived
    "policy_inception_month",
]

# Fields that must be non-null in a valid training row
_REQUIRED_COLS: List[str] = [
    "age", "gender", "city", "years_licensed", "previous_claims_count",
    "annual_mileage_km", "driving_score", "vehicle_make", "vehicle_model",
    "vehicle_value_inr", "vehicle_age_years", "engine_cc",
    "fuel_type", "vehicle_usage_type", "occupation", "marital_status",
]

LABEL_MATURITY_DAYS: int = 90
RETRAIN_MIN_SAMPLES: int = 500


@dataclass
class DatasetBuildResult:
    """Summary of a V4 dataset extraction run."""
    extraction_ts: str
    total_predictions: int            # V2 predictions (inference_features_json IS NOT NULL)
    labelled_predictions: int         # have a model_feedback row
    matured_predictions: int          # policy_created_at <= now - 90d
    matured_labelled: int             # matured AND labelled — eligible for training
    rows_valid: int                   # passed required-field validation
    rows_dropped_missing_json: int    # inference_features_json IS NULL or unparsable
    rows_dropped_validation: int      # failed required-field checks
    feature_fill_rates: Dict[str, float] = field(default_factory=dict)
    missing_features: List[str] = field(default_factory=list)   # fill < 80 %
    is_ready_for_retraining: bool = False
    readiness_reason: str = ""
    dataset: Optional[pd.DataFrame] = None   # populated when include_dataset=True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "dataset"}


class V4DatasetBuilder:
    """
    Build a training-ready dataset from persisted V4 production records.

    Reads inference_features_json JSONB + model_feedback labels.
    Does not touch the V1 normalized columns (age, car_brand, etc.).

    Usage (async context):
        builder = V4DatasetBuilder(db)
        result = await builder.build(include_dataset=True)
        df = result.dataset   # pandas DataFrame, or None if no valid rows
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def build(
        self,
        days_back: Optional[int] = None,
        maturity_days: int = LABEL_MATURITY_DAYS,
        include_dataset: bool = False,
    ) -> DatasetBuildResult:
        """
        Extract, validate, and optionally return the training DataFrame.

        Parameters
        ----------
        days_back : int | None
            Limit extraction to predictions created in the last N days.
            None = all time.
        maturity_days : int
            Minimum policy age in days for label credibility (default 90).
        include_dataset : bool
            If True, attach the full pandas DataFrame to result.dataset.
        """
        now = datetime.utcnow()
        maturity_cutoff = now - timedelta(days=maturity_days)

        # ── Query all V2 predictions (inference_features_json NOT NULL) ───────
        # LEFT JOIN model_feedback so unlabelled rows are also counted.
        stmt = (
            select(
                Prediction.prediction_id,
                Prediction.inference_features_json,
                Prediction.claim_probability,
                Prediction.expected_claim_amount_inr,
                Prediction.created_at.label("prediction_ts"),
                Policy.created_at.label("policy_ts"),
                ModelFeedback.actual_claim_occurred,
                ModelFeedback.actual_claim_amount_inr.label("label_amount"),
            )
            .join(Policy, Policy.policy_id == Prediction.policy_id)
            .outerjoin(
                ModelFeedback,
                ModelFeedback.prediction_id == Prediction.prediction_id,
            )
            .where(Prediction.inference_features_json.isnot(None))
        )

        if days_back is not None:
            cutoff = now - timedelta(days=days_back)
            stmt = stmt.where(Prediction.created_at >= cutoff)

        all_rows = (await self.db.execute(stmt)).all()

        total_predictions = len(all_rows)
        labelled_count = sum(1 for r in all_rows if r.actual_claim_occurred is not None)
        matured_count = sum(1 for r in all_rows if r.policy_ts <= maturity_cutoff)
        matured_labelled = sum(
            1 for r in all_rows
            if r.policy_ts <= maturity_cutoff and r.actual_claim_occurred is not None
        )

        # ── Parse JSONB and build valid training rows ─────────────────────────
        records: List[Dict[str, Any]] = []
        dropped_missing_json: int = 0
        dropped_validation: int = 0

        for r in all_rows:
            # Training rows must be matured AND labelled
            if r.actual_claim_occurred is None:
                continue
            if r.policy_ts > maturity_cutoff:
                continue

            # Parse JSONB (SQLAlchemy asyncpg may return dict or str)
            feat_json = r.inference_features_json
            if not isinstance(feat_json, dict):
                try:
                    feat_json = json.loads(feat_json) if feat_json else {}
                except (json.JSONDecodeError, TypeError):
                    dropped_missing_json += 1
                    continue

            if not feat_json:
                dropped_missing_json += 1
                continue

            # Extract 31 raw features
            row: Dict[str, Any] = {feat: feat_json.get(feat) for feat in V4_RAW_FEATURES}

            # Validate required fields
            missing_required = [col for col in _REQUIRED_COLS if row.get(col) is None]
            if missing_required:
                logger.debug(
                    "Row %s dropped — missing required: %s",
                    r.prediction_id, missing_required,
                )
                dropped_validation += 1
                continue

            # Attach labels and model outputs
            row["claim_occurred"]              = int(r.actual_claim_occurred)
            row["actual_claim_amount_inr"]     = float(r.label_amount or 0.0)
            row["claim_probability_pred"]      = float(r.claim_probability)
            row["expected_claim_amount_pred"]  = float(r.expected_claim_amount_inr)
            row["prediction_id"]               = str(r.prediction_id)
            row["prediction_ts"]               = (
                r.prediction_ts.isoformat() if r.prediction_ts else None
            )
            records.append(row)

        rows_valid = len(records)

        # ── Per-feature fill rates (across all valid rows) ────────────────────
        fill_rates: Dict[str, float] = {}
        if records:
            temp_df = pd.DataFrame(records)
            for feat in V4_RAW_FEATURES:
                if feat in temp_df.columns:
                    fill_rates[feat] = round(
                        float(temp_df[feat].notna().sum()) / len(temp_df) * 100, 1
                    )
                else:
                    fill_rates[feat] = 0.0
        else:
            fill_rates = {f: 0.0 for f in V4_RAW_FEATURES}

        missing_features = [f for f, rate in fill_rates.items() if rate < 80.0]

        # ── Retraining readiness check ────────────────────────────────────────
        if rows_valid >= RETRAIN_MIN_SAMPLES:
            is_ready = True
            readiness_reason = (
                f"Ready: {rows_valid} valid rows >= {RETRAIN_MIN_SAMPLES} minimum."
            )
        elif matured_labelled == 0:
            is_ready = False
            readiness_reason = (
                "No matured labelled records. Labels accumulate as policies mature "
                f"(>= {maturity_days} days). First labels expected approx. late September 2026."
            )
        else:
            is_ready = False
            readiness_reason = (
                f"Insufficient data: {rows_valid} valid rows < {RETRAIN_MIN_SAMPLES} minimum. "
                f"Matured labelled: {matured_labelled}."
            )

        dataset_df: Optional[pd.DataFrame] = (
            pd.DataFrame(records) if include_dataset and records else None
        )

        return DatasetBuildResult(
            extraction_ts=now.isoformat(),
            total_predictions=total_predictions,
            labelled_predictions=labelled_count,
            matured_predictions=matured_count,
            matured_labelled=matured_labelled,
            rows_valid=rows_valid,
            rows_dropped_missing_json=dropped_missing_json,
            rows_dropped_validation=dropped_validation,
            feature_fill_rates=fill_rates,
            missing_features=missing_features,
            is_ready_for_retraining=is_ready,
            readiness_reason=readiness_reason,
            dataset=dataset_df,
        )
