"""
Generate dissertation figure: Claim Amount Statistics and Log Transformation
of the Severity Model.

Run from the backend/ directory:
    python scripts/generate_log_transform_figure.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH   = Path(__file__).parent.parent / "data" / "synthetic_insurance_data.csv"
OUTPUT_DIR  = Path(__file__).parent.parent / "models" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "claim_amount_log_transformation.png"


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Run generate_synthetic_data.py first.")
        sys.exit(1)

    # ── Load exactly the same data the severity model uses ───────────────────
    df = pd.read_csv(DATA_PATH)
    claimants = df[df["claim_occurred"] == 1]["actual_claim_amount_inr"].values

    print(f"Loaded {len(df):,} total rows from {DATA_PATH.name}")
    print(f"Claimant rows used by severity model: {len(claimants):,}")
    print()
    print("Before log1p transformation:")
    print(f"  Min:    ₹{claimants.min():>12,.0f}")
    print(f"  Max:    ₹{claimants.max():>12,.0f}")
    print(f"  Mean:   ₹{claimants.mean():>12,.0f}")
    print(f"  Median: ₹{np.median(claimants):>12,.0f}")
    print(f"  Std:    ₹{claimants.std():>12,.0f}")

    log_transformed = np.log1p(claimants)
    print()
    print("After log1p transformation:")
    print(f"  Min:    {log_transformed.min():.4f}")
    print(f"  Max:    {log_transformed.max():.4f}")
    print(f"  Mean:   {log_transformed.mean():.4f}")
    print(f"  Median: {np.median(log_transformed):.4f}")
    print(f"  Std:    {log_transformed.std():.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Claim Amount Statistics and Log Transformation of the Severity Model",
        fontsize=14, fontweight="bold", y=1.02
    )

    # Left: raw INR amounts
    ax0 = axes[0]
    ax0.hist(claimants, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
    ax0.axvline(claimants.mean(),   color="red",    linestyle="--", linewidth=1.5,
                label=f"Mean  ₹{claimants.mean():,.0f}")
    ax0.axvline(np.median(claimants), color="orange", linestyle="-",  linewidth=1.5,
                label=f"Median ₹{np.median(claimants):,.0f}")
    ax0.set_title("Before Log Transformation", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Claim Amount (INR)", fontsize=11)
    ax0.set_ylabel("Frequency", fontsize=11)
    ax0.legend(fontsize=9)
    ax0.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{int(x/1_00_000)}L" if x >= 1_00_000 else f"₹{int(x/1000)}k")
    )

    # Right: log1p-transformed values (this is exactly what the model trains on)
    ax1 = axes[1]
    ax1.hist(log_transformed, bins=60, color="darkorange", edgecolor="white", linewidth=0.4)
    ax1.axvline(log_transformed.mean(),     color="red",    linestyle="--", linewidth=1.5,
                label=f"Mean  {log_transformed.mean():.2f}")
    ax1.axvline(np.median(log_transformed), color="steelblue", linestyle="-",  linewidth=1.5,
                label=f"Median {np.median(log_transformed):.2f}")
    ax1.set_title("After Log Transformation  [log₁p(amount)]", fontsize=12, fontweight="bold")
    ax1.set_xlabel("log₁p(Claim Amount)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_FILE), dpi=150, bbox_inches="tight")
    plt.close()

    print()
    print(f"Figure saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
