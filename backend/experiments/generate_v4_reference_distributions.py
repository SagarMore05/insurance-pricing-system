#!/usr/bin/env python3
"""
Generate V4 reference feature distributions from the training CSV.

Produces:  backend/models/metadata/v4_reference_distributions.json

Used by the production monitoring system to compute PSI (Population Stability
Index) for feature drift detection.  Once generated, this file is read by the
MonitoringPanel frontend component (Phase 3 deliverable — replaces the
"Insufficient production data" placeholder shown in Phase 2).

Run from the repository root:
    python backend/experiments/generate_v4_reference_distributions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_PATH   = (
    REPO_ROOT
    / "backend"
    / "experiments"
    / "outputs"
    / "data"
    / "car_insurance_portfolio_v4.csv"
)
OUTPUT_PATH = REPO_ROOT / "backend" / "models" / "metadata" / "v4_reference_distributions.json"

# ── Feature lists matching InsurancePreprocessorV4 input schema ───────────────

V4_NUMERIC_FEATURES = [
    # StandardScaled
    "age", "annual_mileage_km", "vehicle_value_inr", "driving_score",
    "vehicle_safety_rating", "pincode_risk_score",
    "flood_risk_index", "theft_risk_index", "monsoon_exposure_index",
    # MinMaxScaled
    "vehicle_age_years", "years_licensed", "policy_tenure_years",
    # Passthrough (raw)
    "previous_claims_count", "months_since_last_claim",
    "no_claim_bonus_pct", "airbags_count", "policy_inception_month", "engine_cc",
]

V4_CATEGORICAL_FEATURES = [
    # OneHotEncoded (raw before OHE)
    "gender", "occupation", "marital_status", "annual_income_band",
    "city", "fuel_type", "vehicle_usage_type", "parking_type",
    "repair_cost_band", "vehicle_body_style",
    # LabelEncoded
    "vehicle_make",
    # Passthrough
    "region_risk_category",
]


def _numeric_stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {"count": 0}
    return {
        "count": int(len(clean)),
        "mean":  round(float(clean.mean()), 6),
        "std":   round(float(clean.std()),  6),
        "min":   round(float(clean.min()),  6),
        "max":   round(float(clean.max()),  6),
        "p10":   round(float(np.percentile(clean, 10)), 6),
        "p25":   round(float(np.percentile(clean, 25)), 6),
        "p50":   round(float(np.percentile(clean, 50)), 6),
        "p75":   round(float(np.percentile(clean, 75)), 6),
        "p90":   round(float(np.percentile(clean, 90)), 6),
        "p99":   round(float(np.percentile(clean, 99)), 6),
    }


def _categorical_stats(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=True)
    total  = int(len(series.dropna()))
    return {
        "count":        total,
        "value_counts": {str(k): int(v) for k, v in counts.items()},
        "value_pcts":   (
            {str(k): round(int(v) / total * 100, 4) for k, v in counts.items()}
            if total > 0 else {}
        ),
    }


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: Training CSV not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {DATA_PATH.name} ...", flush=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns", flush=True)

    distributions: dict = {
        "meta": {
            "generated_at":  pd.Timestamp.utcnow().isoformat(),
            "source_file":   DATA_PATH.name,
            "total_rows":    int(len(df)),
            "model_version": "v4.0.0",
            "description": (
                "Reference feature distributions from V4 training dataset "
                "(car_insurance_portfolio_v4.csv, 50 000 rows). "
                "Used for PSI-based feature drift detection in production monitoring. "
                "PSI < 0.10 = stable (green), 0.10-0.25 = investigate (amber), "
                "> 0.25 = significant drift (red)."
            ),
        },
        "numeric":     {},
        "categorical": {},
    }

    print("\nComputing numeric distributions ...", flush=True)
    for feat in V4_NUMERIC_FEATURES:
        if feat in df.columns:
            stats = _numeric_stats(df[feat])
            distributions["numeric"][feat] = stats
            print(f"  [numeric]  {feat:35s} mean={stats.get('mean', 'N/A')}", flush=True)
        else:
            print(f"  [WARN]     '{feat}' not found in CSV — skipped", flush=True)

    print("\nComputing categorical distributions ...", flush=True)
    for feat in V4_CATEGORICAL_FEATURES:
        if feat in df.columns:
            stats = _categorical_stats(df[feat])
            distributions["categorical"][feat] = stats
            n_vals = len(stats["value_counts"])
            print(
                f"  [categ]    {feat:35s} {n_vals} unique values, "
                f"n={stats['count']:,}",
                flush=True,
            )
        else:
            print(f"  [WARN]     '{feat}' not found in CSV — skipped", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(distributions, fh, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Output written to: {OUTPUT_PATH}", flush=True)
    print(f"  Numeric features:     {len(distributions['numeric'])}", flush=True)
    print(f"  Categorical features: {len(distributions['categorical'])}", flush=True)
    print(f"  Total rows analysed:  {distributions['meta']['total_rows']:,}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
