"""
Enterprise Retraining Endpoint
================================
POST /admin/retrain

Workflow
--------
1. Count new labelled production records in PostgreSQL (ModelFeedback table).
2. If count < RETRAIN_MIN_SAMPLES (default 500):
       Return graceful "Retraining postponed" response — no error raised.
3. If count >= RETRAIN_MIN_SAMPLES:
   a. Load master dataset (motor_insurance_master_dataset_50000.csv).
   b. Load production records from PostgreSQL (joins ModelFeedback → Prediction
      → Policy → Customer → Vehicle → DrivingProfile).
   c. Normalise production column names to master-dataset schema.
   d. Merge master + production records.
   e. Deduplicate.
   f. Run MultiAlgorithmEngine on merged DataFrame.
   g. Return training results (frequency winner, severity winner, metrics).

Safety guarantees
-----------------
- Master dataset is NEVER discarded — it is always the historical foundation.
- Registry / production models are only updated when training succeeds.
- A failed run leaves existing production models untouched.
- No errors are raised when the threshold is not reached.
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import RetrainResponse
from src.config import settings
from src.database.models import (
    Customer, DrivingProfile, ModelFeedback, Policy, Prediction, Vehicle,
)
from src.database.session import get_db

logger = logging.getLogger("insurance_api.retraining")

router = APIRouter(dependencies=[Depends(verify_admin_key)])

_BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
_MASTER_DATASET_PATH = _BACKEND_DIR / "data" / "master" / "motor_insurance_master_dataset_50000.csv"

# Columns from master dataset needed for training (excludes metadata / target leakage columns)
_MASTER_TRAIN_COLS = [
    "age", "gender", "city", "annual_income_inr",
    "vehicle_brand", "vehicle_segment", "fuel_type",
    "vehicle_age_years", "vehicle_value_inr", "years_licensed",
    "annual_mileage_km", "previous_claims", "driving_score",
    "claim_occurred", "actual_claim_amount_inr",
]

# In-process status for background retraining (mirrors model_selection_api pattern)
_RETRAIN_STATUS: Dict[str, Any] = {
    "running": False,
    "last_result": None,
    "error": None,
    "started_at": None,
}


# ── Helper: count labelled production records ──────────────────────────────────

async def _count_labeled_records(db: AsyncSession) -> int:
    """Count ModelFeedback rows that have an actual outcome (labelled records)."""
    result = await db.execute(
        select(func.count())
        .select_from(ModelFeedback)
        .where(ModelFeedback.actual_claim_occurred.isnot(None))
    )
    return result.scalar() or 0


# ── Helper: load production records as DataFrame ───────────────────────────────

async def _load_production_records(db: AsyncSession) -> pd.DataFrame:
    """
    Load labelled production records from PostgreSQL and return a DataFrame
    with column names matching the master-dataset schema:
      vehicle_brand, previous_claims, actual_claim_amount_inr, ...

    Joins: ModelFeedback → Prediction → Policy → Customer → Vehicle → DrivingProfile
    """
    stmt = (
        select(
            # Targets
            ModelFeedback.actual_claim_occurred.label("claim_occurred"),
            ModelFeedback.actual_claim_amount_inr.label("actual_claim_amount_inr"),
            # Customer features
            Customer.age,
            Customer.gender,
            Customer.city,
            # Vehicle features — rename car_brand → vehicle_brand to match master schema
            Vehicle.car_brand.label("vehicle_brand"),
            Vehicle.fuel_type,
            Vehicle.vehicle_age_years,
            Vehicle.vehicle_value_inr,
            # Driving features — rename previous_claims_count → previous_claims
            DrivingProfile.driving_score,
            DrivingProfile.annual_mileage_km,
            DrivingProfile.previous_claims_count.label("previous_claims"),
            DrivingProfile.years_licensed,
        )
        .join(Prediction, Prediction.prediction_id == ModelFeedback.prediction_id)
        .join(Policy, Policy.policy_id == Prediction.policy_id)
        .join(Customer, Customer.customer_id == Policy.customer_id)
        .join(Vehicle, Vehicle.vehicle_id == Policy.vehicle_id)
        .join(DrivingProfile, DrivingProfile.customer_id == Policy.customer_id)
        .where(ModelFeedback.actual_claim_occurred.isnot(None))
    )

    rows = (await db.execute(stmt)).all()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "claim_occurred", "actual_claim_amount_inr",
        "age", "gender", "city",
        "vehicle_brand", "fuel_type", "vehicle_age_years", "vehicle_value_inr",
        "driving_score", "annual_mileage_km", "previous_claims", "years_licensed",
    ])

    # Cast types
    df["claim_occurred"] = df["claim_occurred"].astype(int)
    df["actual_claim_amount_inr"] = df["actual_claim_amount_inr"].fillna(0).astype(float)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(35).astype(int)
    df["vehicle_age_years"] = pd.to_numeric(df["vehicle_age_years"], errors="coerce").fillna(3).astype(int)
    df["vehicle_value_inr"] = pd.to_numeric(df["vehicle_value_inr"], errors="coerce").fillna(800000).astype(float)
    df["driving_score"] = pd.to_numeric(df["driving_score"], errors="coerce").fillna(70.0).astype(float)
    df["annual_mileage_km"] = pd.to_numeric(df["annual_mileage_km"], errors="coerce").fillna(15000).astype(int)
    df["previous_claims"] = pd.to_numeric(df["previous_claims"], errors="coerce").fillna(0).astype(int)
    df["years_licensed"] = pd.to_numeric(df["years_licensed"], errors="coerce").fillna(5).astype(int)

    logger.info("Loaded %d production training records from PostgreSQL.", len(df))
    return df


# ── Helper: build merged training DataFrame ────────────────────────────────────

def _build_merged_dataset(prod_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load master dataset and merge with production records.

    Master dataset columns used: _MASTER_TRAIN_COLS
    Production records already use master-dataset column naming convention.

    Steps:
    1. Load master CSV → select training columns
    2. Concat master + production
    3. Deduplicate on age/city/vehicle columns (keep last → production rows win)
    4. Reset index
    """
    master_df = pd.read_csv(_MASTER_DATASET_PATH)

    # Select only the training-relevant columns that exist
    available_master_cols = [c for c in _MASTER_TRAIN_COLS if c in master_df.columns]
    master_df = master_df[available_master_cols].copy()

    logger.info("Master dataset: %d rows, %d columns.", len(master_df), len(master_df.columns))

    if prod_df.empty:
        return master_df

    # Align production DataFrame columns to master schema (fill missing optional cols)
    for col in _MASTER_TRAIN_COLS:
        if col not in prod_df.columns:
            prod_df[col] = None

    # Keep only columns that exist in master
    prod_df = prod_df[[c for c in _MASTER_TRAIN_COLS if c in prod_df.columns]].copy()

    # Merge
    merged = pd.concat([master_df, prod_df], ignore_index=True)

    # Deduplicate: drop exact duplicate rows (same features + targets)
    dedup_cols = ["age", "city", "vehicle_brand", "vehicle_age_years",
                  "vehicle_value_inr", "annual_mileage_km", "driving_score",
                  "claim_occurred"]
    dedup_cols = [c for c in dedup_cols if c in merged.columns]
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    after = len(merged)
    if before > after:
        logger.info("Deduplication removed %d rows (%d → %d).", before - after, before, after)

    logger.info(
        "Merged dataset: %d total rows (%d master + %d production).",
        len(merged), len(master_df), len(prod_df),
    )
    return merged


# ── Main retrain endpoint ──────────────────────────────────────────────────────

@router.post("/admin/retrain", response_model=RetrainResponse)
async def trigger_retraining(db: AsyncSession = Depends(get_db)):
    """
    Enterprise retraining endpoint.

    Step 1 — Count labelled production records.
    Step 2 — If < threshold: return graceful postponement message (no training).
    Step 3 — Merge master dataset + production records.
    Step 4 — Run MultiAlgorithmEngine (trains XGBoost, LightGBM, CatBoost for
              both frequency and severity; auto-selects winner; updates registry).
    Step 5 — Return training results.
    """
    threshold = settings.RETRAIN_MIN_SAMPLES  # default 500

    # ── Step 1: Count labelled production records ──────────────────────────────
    labeled_count = await _count_labeled_records(db)
    logger.info("Enterprise retrain triggered. Labelled records: %d / %d threshold.", labeled_count, threshold)

    # ── Step 2: Threshold check ────────────────────────────────────────────────
    if labeled_count < threshold:
        msg = (
            f"Retraining postponed. "
            f"New labelled records collected: {labeled_count}. "
            f"Minimum required: {threshold}."
        )
        logger.info(msg)
        return RetrainResponse(
            message=msg,
            triggered=True,
            postponed=True,
            labeled_count=labeled_count,
            threshold=threshold,
            performance_report={
                "n_samples": labeled_count,
                "threshold": threshold,
                "postponed": True,
                "reason": (
                    f"Insufficient labelled records: {labeled_count} < {threshold}. "
                    "Continue collecting production feedback. "
                    "Retraining will proceed automatically once the threshold is reached."
                ),
                "evaluation_date": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )

    # ── Step 3: Load + merge master & production data ──────────────────────────
    try:
        prod_df = await _load_production_records(db)
        merged_df = _build_merged_dataset(prod_df)
        n_master = len(pd.read_csv(_MASTER_DATASET_PATH))
        n_prod = len(prod_df)
        dataset_label = f"{_MASTER_DATASET_PATH.name}+{n_prod}_production"
    except Exception as exc:
        logger.error("Failed to build merged dataset: %s", exc, exc_info=True)
        return RetrainResponse(
            message=f"Dataset preparation failed: {exc}",
            triggered=True,
            postponed=False,
            labeled_count=labeled_count,
            threshold=threshold,
            performance_report={
                "n_samples": labeled_count,
                "error": str(exc),
                "evaluation_date": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )

    # ── Step 4: Run multi-algorithm training ───────────────────────────────────
    try:
        from src.training.multi_algorithm_engine import MultiAlgorithmEngine

        engine = MultiAlgorithmEngine()
        record = engine.run(dataset_df=merged_df, dataset_label=dataset_label)

        freq_winner = record.get("frequency_winner", "unknown")
        sev_winner  = record.get("severity_winner", "unknown")
        elapsed     = record.get("total_elapsed_sec", 0)

        freq_results = {r["algorithm"]: r for r in record.get("frequency_results", [])}
        sev_results  = {r["algorithm"]: r for r in record.get("severity_results", [])}

        freq_metrics = freq_results.get(freq_winner, {}).get("metrics", {})
        sev_metrics  = sev_results.get(sev_winner, {}).get("metrics", {})

        logger.info(
            "Enterprise retraining complete. Freq winner: %s (AUC=%.4f). "
            "Sev winner: %s (R²=%.4f). Elapsed: %.1fs.",
            freq_winner, freq_metrics.get("roc_auc", 0) or 0,
            sev_winner,  sev_metrics.get("r2", 0) or 0,
            elapsed,
        )

        return RetrainResponse(
            message=(
                f"Retraining complete. "
                f"Frequency winner: {freq_winner.upper()} "
                f"(ROC-AUC {freq_metrics.get('roc_auc', 0):.4f}). "
                f"Severity winner: {sev_winner.upper()} "
                f"(R² {sev_metrics.get('r2', 0):.4f}). "
                f"Elapsed: {elapsed:.1f}s."
            ),
            triggered=True,
            postponed=False,
            labeled_count=labeled_count,
            threshold=threshold,
            frequency_winner=freq_winner,
            severity_winner=sev_winner,
            training_record=record,
            performance_report={
                "evaluation_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "n_samples":       len(merged_df),
                "master_rows":     n_master,
                "production_rows": n_prod,
                "labeled_count":   labeled_count,
                "dataset_label":   dataset_label,
                # Frequency metrics (for frontend MetricItem display)
                "frequency_auc":   round(freq_metrics.get("roc_auc", 0) or 0, 4),
                "frequency_f1":    round(freq_metrics.get("f1", 0) or 0, 4),
                "frequency_winner": freq_winner,
                # Severity metrics
                "severity_r2":     round(sev_metrics.get("r2", 0) or 0, 4),
                "severity_rmse":   round(sev_metrics.get("rmse", 0) or 0, 2),
                "severity_winner": sev_winner,
                # All algorithm comparison
                "algorithm_comparison": {
                    "frequency": {
                        algo: r.get("metrics", {})
                        for algo, r in freq_results.items()
                    },
                    "severity": {
                        algo: r.get("metrics", {})
                        for algo, r in sev_results.items()
                    },
                },
                "promotion_reason_frequency": record.get("promotion_reason_frequency"),
                "promotion_reason_severity":  record.get("promotion_reason_severity"),
                "total_elapsed_sec": elapsed,
                "retrain_recommended": True,
            },
        )

    except Exception as exc:
        logger.error("Enterprise retraining failed: %s", exc, exc_info=True)
        return RetrainResponse(
            message=f"Retraining failed: {exc}",
            triggered=True,
            postponed=False,
            labeled_count=labeled_count,
            threshold=threshold,
            performance_report={
                "n_samples": len(merged_df),
                "error": str(exc),
                "evaluation_date": datetime.utcnow().strftime("%Y-%m-%d"),
            },
        )
