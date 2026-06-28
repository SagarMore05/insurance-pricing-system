"""
Regenerate ONLY actual_claim_amount_inr using feature-conditioned actuarial business logic.

SCOPE:
  - Modifies ONLY: actual_claim_amount_inr (for claim_occurred == 1 rows)
  - All other columns remain byte-for-byte identical
  - Non-claimant rows (claim_occurred == 0) keep actual_claim_amount_inr = 0

FORMULA:
  amount = vehicle_value_inr x loss_ratio x segment_mult x age_mult
           x fuel_mult x city_mult x driver_mult x mileage_mult x noise

  Where loss_ratio = Uniform(0.01, 0.07) — the stochastic element representing
  accident severity as a fraction of the vehicle's market value.

Usage (from backend/):
    python scripts/regenerate_severity_target.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKEND     = Path(__file__).resolve().parent.parent
MASTER_PATH = BACKEND / "data" / "master" / "motor_insurance_master_dataset_50000.csv"
BACKUP_PATH = BACKEND / "data" / "master" / "motor_insurance_master_dataset_50000_pre_severity_regen.csv"
REPORT_PATH = BACKEND / "experiments" / "outputs" / "reports" / "severity_regen_report.json"

RANDOM_SEED = 42

# ── City tier (mirrors pipeline.py exactly) ────────────────────────────────────
_CITY_TIER = {
    "mumbai": 1, "delhi": 1, "bangalore": 1, "bengaluru": 1,
    "hyderabad": 1, "chennai": 1, "kolkata": 1, "pune": 1, "ahmedabad": 1,
    "jaipur": 2, "lucknow": 2, "kanpur": 2, "nagpur": 2, "indore": 2,
    "bhopal": 2, "patna": 2, "vadodara": 2, "coimbatore": 2, "agra": 2,
    "visakhapatnam": 2, "surat": 2, "chandigarh": 2,
}

# ── Deterministic multipliers (actuarial business logic) ───────────────────────
_SEGMENT_MULT = {"Luxury": 1.40, "SUV": 1.15, "Sedan": 1.00, "Hatchback": 0.85}
_FUEL_MULT    = {"EV": 1.35, "Diesel": 1.10, "Petrol": 1.00, "CNG": 0.90}
_CITY_MULT    = {1: 1.15, 2: 1.05, 3: 1.00}


# ── Formula ────────────────────────────────────────────────────────────────────

def _city_tier_vec(cities: pd.Series) -> np.ndarray:
    return cities.str.lower().map(_CITY_TIER).fillna(3).astype(int).values


def compute_severity_vectorised(
    claimants: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Return uncalibrated, unclipped claim amounts for all claimant rows."""
    n = len(claimants)

    # Primary driver: vehicle value × random loss ratio
    # Loss ratio = fraction of vehicle value damaged (2–12% realistic for partial claims)
    loss_ratio = rng.uniform(0.01, 0.07, size=n)
    base = claimants["vehicle_value_inr"].values * loss_ratio

    # Vehicle segment: Luxury/SUV parts and specialist labour cost more
    seg_mult = claimants["vehicle_segment"].map(_SEGMENT_MULT).fillna(1.0).values

    # Fuel type: EV battery packs are expensive; diesel turbos add repair cost
    fuel_mult = claimants["fuel_type"].map(_FUEL_MULT).fillna(1.0).values

    # Vehicle age: newer vehicles have dearer OEM parts and fewer aftermarket options
    # max(0.85, 1.20 - age*0.025) -> age 0 -> 1.20, age 5 -> 1.075, age 10 -> 0.95
    age_mult = np.maximum(0.85, 1.20 - claimants["vehicle_age_years"].values * 0.025)

    # City tier: metro workshops have higher labour rates and parts premiums
    tier = _city_tier_vec(claimants["city"])
    city_mult = np.vectorize(_CITY_MULT.get)(tier, 1.00)

    # Driver risk: poor driving score and prior claims signal higher-severity accidents
    # driving_score 20–100; previous_claims 0–5
    driver_mult = (
        1.0
        + (100 - claimants["driving_score"].values) / 500.0
        + claimants["previous_claims"].values * 0.03
    )

    # Annual mileage: high-usage vehicles sustain cumulative wear -> slightly larger claims
    mileage_mult = 1.0 + (claimants["annual_mileage_km"].values / 50_000.0) * 0.05

    # Controlled lognormal noise (~8% CV) for realistic variation between
    # identical-profile vehicles (repair shop pricing, exact damage extent, etc.)
    noise = rng.lognormal(mean=0.0, sigma=0.08, size=n)

    return base * seg_mult * age_mult * fuel_mult * city_mult * driver_mult * mileage_mult * noise


# ── Main ───────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    print("=" * 65)
    print("  Severity Target Regeneration")
    print("=" * 65)

    # 1. Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading {MASTER_PATH.name} ...")
    df = pd.read_csv(MASTER_PATH)
    original_cols   = list(df.columns)
    original_dtypes = df.dtypes.to_dict()
    print(f"      Shape: {df.shape}, columns: {original_cols}")

    # 2. Snapshot before stats ─────────────────────────────────────────────────
    claimant_mask = df["claim_occurred"] == 1
    n_claimants   = int(claimant_mask.sum())
    n_non         = int((~claimant_mask).sum())
    y_before = df.loc[claimant_mask, "actual_claim_amount_inr"].values.copy()

    before_stats = {
        "n_claimants":  n_claimants,
        "n_non_claimants": n_non,
        "mean":   float(np.mean(y_before)),
        "median": float(np.median(y_before)),
        "std":    float(np.std(y_before)),
        "min":    float(np.min(y_before)),
        "max":    float(np.max(y_before)),
        "p25":    float(np.percentile(y_before, 25)),
        "p75":    float(np.percentile(y_before, 75)),
        "p99":    float(np.percentile(y_before, 99)),
    }
    print(f"\n[2/7] Before distribution (claimants only):")
    print(f"      mean=Rs.{before_stats['mean']:,.0f}  median=Rs.{before_stats['median']:,.0f}"
          f"  std=Rs.{before_stats['std']:,.0f}")
    print(f"      min=Rs.{before_stats['min']:,.0f}  max=Rs.{before_stats['max']:,.0f}")
    print(f"      p25=Rs.{before_stats['p25']:,.0f}  p75=Rs.{before_stats['p75']:,.0f}")
    print(f"      Pearson r with vehicle_value: "
          f"{np.corrcoef(y_before, df.loc[claimant_mask, 'vehicle_value_inr'].values)[0,1]:.4f}")

    # 3. Back up ───────────────────────────────────────────────────────────────
    print(f"\n[3/7] Creating backup -> {BACKUP_PATH.name} ...")
    if not dry_run:
        shutil.copy(MASTER_PATH, BACKUP_PATH)
        print("      Backup created.")
    else:
        print("      [DRY RUN] Skipped.")

    # 4. Apply formula ─────────────────────────────────────────────────────────
    print(f"\n[4/7] Computing feature-conditioned severity for {n_claimants:,} claimants ...")
    rng = np.random.default_rng(RANDOM_SEED)
    claimants_df = df[claimant_mask].copy()
    raw_amounts  = compute_severity_vectorised(claimants_df, rng)

    raw_mean = float(np.mean(raw_amounts))
    print(f"      Raw (uncalibrated) mean = Rs.{raw_mean:,.0f}")

    # 5. Calibrate to preserve distributional scale ────────────────────────────
    # Scale so the new mean matches the original mean.  This preserves the relative
    # structure (which drives R²) while keeping the absolute scale consistent with
    # expected_loss_inr and premium_inr columns that we are not touching.
    target_mean        = before_stats["mean"]
    calibration_factor = target_mean / raw_mean
    calibrated         = raw_amounts * calibration_factor
    print(f"      Calibration factor: {calibration_factor:.4f}  "
          f"(raw Rs.{raw_mean:,.0f} -> target Rs.{target_mean:,.0f})")

    # Clip to actuarially plausible bounds
    clipped = np.clip(calibrated, 5_000.0, 1_800_000.0)
    n_clipped_low  = int(np.sum(clipped < 5_001))
    n_clipped_high = int(np.sum(calibrated > 1_800_000))
    print(f"      Clipping: {n_clipped_low} below Rs.5,000 | {n_clipped_high} above Rs.18,00,000")

    # Round to nearest rupee (integer, stored as float64 for pandas consistency)
    new_amounts = np.round(clipped)

    # 6. After stats ───────────────────────────────────────────────────────────
    after_stats = {
        "mean":   float(np.mean(new_amounts)),
        "median": float(np.median(new_amounts)),
        "std":    float(np.std(new_amounts)),
        "min":    float(np.min(new_amounts)),
        "max":    float(np.max(new_amounts)),
        "p25":    float(np.percentile(new_amounts, 25)),
        "p75":    float(np.percentile(new_amounts, 75)),
        "p99":    float(np.percentile(new_amounts, 99)),
    }
    print(f"\n[5/7] After distribution (claimants only):")
    print(f"      mean=Rs.{after_stats['mean']:,.0f}  median=Rs.{after_stats['median']:,.0f}"
          f"  std=Rs.{after_stats['std']:,.0f}")
    print(f"      min=Rs.{after_stats['min']:,.0f}  max=Rs.{after_stats['max']:,.0f}")
    print(f"      p25=Rs.{after_stats['p25']:,.0f}  p75=Rs.{after_stats['p75']:,.0f}")

    # New correlations with features
    vv_corr  = float(np.corrcoef(new_amounts, claimants_df["vehicle_value_inr"].values)[0, 1])
    age_corr = float(np.corrcoef(new_amounts, claimants_df["vehicle_age_years"].values)[0, 1])
    ds_corr  = float(np.corrcoef(new_amounts, claimants_df["driving_score"].values)[0, 1])
    pc_corr  = float(np.corrcoef(new_amounts, claimants_df["previous_claims"].values)[0, 1])
    print(f"\n      Feature correlations (Pearson r) after regeneration:")
    print(f"        vehicle_value_inr : {vv_corr:+.4f}")
    print(f"        vehicle_age_years : {age_corr:+.4f}")
    print(f"        driving_score     : {ds_corr:+.4f}")
    print(f"        previous_claims   : {pc_corr:+.4f}")

    # 7. Write back & verify ───────────────────────────────────────────────────
    print(f"\n[6/7] Applying to dataset and verifying schema ...")
    df.loc[claimant_mask, "actual_claim_amount_inr"] = new_amounts

    # Schema checks
    assert list(df.columns) == original_cols, "FATAL: column list changed!"
    assert df.shape == (50_000, 20),          "FATAL: row/column count changed!"
    assert (df.loc[~claimant_mask, "actual_claim_amount_inr"] == 0).all(), \
        "FATAL: non-claimant rows changed!"
    assert df["claim_occurred"].equals(
        pd.read_csv(BACKUP_PATH)["claim_occurred"] if not dry_run else df["claim_occurred"]
    ) or dry_run, "FATAL: claim_occurred changed!"

    # Verify no other column changed
    if not dry_run:
        original_df = pd.read_csv(BACKUP_PATH)
        for col in original_cols:
            if col == "actual_claim_amount_inr":
                continue
            if not df[col].equals(original_df[col]):
                raise AssertionError(f"FATAL: column '{col}' was unexpectedly modified!")
    print("      All schema checks passed. No other columns modified.")

    if not dry_run:
        df.to_csv(MASTER_PATH, index=False)
        print(f"      Saved -> {MASTER_PATH.name}")
    else:
        print("      [DRY RUN] Save skipped.")

    # 8. Report ────────────────────────────────────────────────────────────────
    report = {
        "timestamp":          datetime.utcnow().isoformat(),
        "dataset":            str(MASTER_PATH.name),
        "random_seed":        RANDOM_SEED,
        "calibration_factor": round(calibration_factor, 6),
        "formula": {
            "base":           "vehicle_value_inr * Uniform(0.01, 0.07)",
            "multipliers":    {
                "segment": str(_SEGMENT_MULT),
                "fuel":    str(_FUEL_MULT),
                "age":     "max(0.85, 1.20 - vehicle_age_years * 0.025)",
                "city":    str(_CITY_MULT),
                "driver":  "1.0 + (100 - driving_score)/500 + previous_claims*0.03",
                "mileage": "1.0 + (annual_mileage_km/50000)*0.05",
            },
            "noise":          "lognormal(0, 0.08)",
            "clip":           "[5000, 1800000]",
        },
        "before": before_stats,
        "after":  after_stats,
        "feature_correlations_after": {
            "vehicle_value_inr": round(vv_corr, 4),
            "vehicle_age_years": round(age_corr, 4),
            "driving_score":     round(ds_corr, 4),
            "previous_claims":   round(pc_corr, 4),
        },
        "schema_unchanged": True,
        "non_claimants_unchanged": True,
        "claim_occurred_unchanged": True,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[7/7] Report -> {REPORT_PATH.name}")

    print("\n" + "=" * 65)
    print("  Regeneration complete.")
    print(f"  Before correlation (vehicle_value vs claim): "
          f"{np.corrcoef(y_before, df.loc[claimant_mask, 'vehicle_value_inr'].values)[0,1]:.4f}")
    print(f"  After  correlation (vehicle_value vs claim): {vv_corr:+.4f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview formula output without saving")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
