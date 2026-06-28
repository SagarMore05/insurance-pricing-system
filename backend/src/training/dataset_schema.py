"""
Canonical master dataset schema — single source of truth for feature definitions.

All training, preprocessing, retraining, and validation modules import from here.
Never define feature lists independently in other modules.
"""
from __future__ import annotations

import logging
import pathlib
from difflib import get_close_matches
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("insurance_api.dataset_schema")

# ── Paths ─────────────────────────────────────────────────────────────────────

_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MASTER_DATASET_PATH = _BACKEND_ROOT / "data" / "master" / "motor_insurance_master_dataset_50000.csv"

# ── Feature / target definitions ──────────────────────────────────────────────

# 13 input features (master dataset column names)
FEATURE_COLUMNS: List[str] = [
    "age",
    "gender",
    "city",
    "annual_income_inr",
    "vehicle_brand",
    "vehicle_segment",
    "fuel_type",
    "vehicle_age_years",
    "vehicle_value_inr",
    "years_licensed",
    "annual_mileage_km",
    "previous_claims",
    "driving_score",
]

FREQUENCY_TARGET = "claim_occurred"
SEVERITY_TARGET  = "actual_claim_amount_inr"

# Columns present in CSV but excluded from training
METADATA_COLS: List[str] = [
    "customer_id", "claim_probability", "expected_loss_inr",
    "risk_level", "premium_inr",
]

# All columns required for a complete training run
TRAINING_COLS: List[str] = FEATURE_COLUMNS + [FREQUENCY_TARGET, SEVERITY_TARGET]

# ── Categorical allowed values ─────────────────────────────────────────────────

CATEGORICAL_VALUES: Dict[str, List[str]] = {
    "vehicle_brand": [
        "Maruti", "Hyundai", "Ford", "Skoda", "Jeep", "Kia", "Volkswagen",
        "Renault", "BMW", "Mercedes", "MG", "Toyota", "Nissan", "Honda",
        "Mahindra", "Audi", "Tata",
    ],
    "vehicle_segment": ["Hatchback", "SUV", "Sedan", "Luxury"],
    "fuel_type":       ["Petrol", "Diesel", "EV", "CNG"],
    "gender":          ["Male", "Female"],
}

# ── Numeric ranges (from master dataset) ──────────────────────────────────────

NUMERIC_RANGES: Dict[str, tuple] = {
    "age":               (18, 70),
    "driving_score":     (20.0, 100.0),
    "previous_claims":   (0, 5),
    "vehicle_value_inr": (300_004, 3_999_906),
    "annual_income_inr": (200_645, 4_999_988),
    "vehicle_age_years": (0, 15),
}

# ── Alias mapping: any reasonable name → canonical master-dataset column name ──

COLUMN_ALIASES: Dict[str, str] = {
    # vehicle_brand aliases
    "car_brand":    "vehicle_brand",
    "brand":        "vehicle_brand",
    "make":         "vehicle_brand",
    "vehicle_make": "vehicle_brand",
    # previous_claims aliases
    "previous_claims_count": "previous_claims",
    "prior_claims":          "previous_claims",
    "num_claims":            "previous_claims",
    # actual_claim_amount_inr aliases
    "claim_amount_inr":    "actual_claim_amount_inr",
    "actual_claim_amount": "actual_claim_amount_inr",
    "claim_amount":        "actual_claim_amount_inr",
    "paid_claim_amount":   "actual_claim_amount_inr",
    # annual_income_inr aliases
    "income":        "annual_income_inr",
    "annual_income": "annual_income_inr",
}

# ── Reverse alias: canonical → all known aliases (for error messages) ──────────

_REVERSE_ALIASES: Dict[str, List[str]] = {}
for _alias, _canon in COLUMN_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_canon, []).append(_alias)


# ── Normalization helpers ──────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame, silent: bool = False) -> pd.DataFrame:
    """
    Rename any known column aliases to their canonical master-dataset names.

    Applied BEFORE validation — never validate before normalizing.
    Logs every mapping that is applied.
    """
    rename_map: Dict[str, str] = {}
    for alias, canon in COLUMN_ALIASES.items():
        if alias in df.columns and canon not in df.columns:
            rename_map[alias] = canon

    if rename_map:
        for alias, canon in rename_map.items():
            if not silent:
                logger.info("[Schema] Column normalized: %s → %s", alias, canon)
        df = df.rename(columns=rename_map)

    return df


def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    context: str = "dataset",
) -> None:
    """
    Validate that all required columns are present in df.

    Raises ValueError with helpful suggestions (closest matches + alias hints)
    rather than a generic missing-column error.
    """
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return

    msgs: List[str] = []
    all_cols = df.columns.tolist()
    for col in missing:
        parts: List[str] = [f"  Missing required column: {col}"]

        # Suggest closest match by string similarity
        closest = get_close_matches(col, all_cols, n=3, cutoff=0.4)
        if closest:
            parts.append(f"    Closest in dataset: {closest}")
            parts.append(f"    Suggested mapping:  {closest[0]} → {col}")

        # Suggest known aliases
        known_aliases = _REVERSE_ALIASES.get(col, [])
        in_df_aliases = [a for a in known_aliases if a in all_cols]
        if in_df_aliases:
            parts.append(f"    Known alias present: {in_df_aliases[0]!r} → call normalize_columns() first")

        msgs.append("\n".join(parts))

    raise ValueError(
        f"[{context}] Dataset missing {len(missing)} required column(s):\n"
        + "\n".join(msgs)
    )


def log_dataset_info(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Print key dataset statistics before training begins."""
    logger.info("=" * 60)
    logger.info("[%s] Rows: %d | Columns: %d", label, len(df), len(df.columns))
    logger.info("[%s] Columns: %s", label, list(df.columns))

    if FREQUENCY_TARGET in df.columns:
        claim_rate = df[FREQUENCY_TARGET].mean()
        logger.info("[%s] Frequency target (%s): claim rate = %.3f", label, FREQUENCY_TARGET, claim_rate)

    if SEVERITY_TARGET in df.columns:
        nonzero = (df[SEVERITY_TARGET] > 0).sum()
        logger.info("[%s] Severity target (%s): %d non-zero rows", label, SEVERITY_TARGET, nonzero)

    features_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    features_missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    logger.info("[%s] Feature columns present: %d / %d", label, len(features_present), len(FEATURE_COLUMNS))
    if features_missing:
        logger.warning("[%s] Feature columns missing: %s", label, features_missing)
    logger.info("=" * 60)
