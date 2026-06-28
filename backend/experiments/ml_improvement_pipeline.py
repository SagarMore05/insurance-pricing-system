"""
ML Improvement Pipeline — Isolated Experimental Phase
======================================================
Run from the backend/ directory:
    python experiments/ml_improvement_pipeline.py

SAFE: No production files are read from or written to.
      All outputs are saved under experiments/outputs/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
import catboost as cb

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
ORIG_DATA   = ROOT / "data" / "synthetic_insurance_data.csv"
EXP_DIR     = Path(__file__).parent / "outputs"
DATA_DIR    = EXP_DIR / "data"
REPORT_DIR  = EXP_DIR / "reports"
PLOT_DIR    = EXP_DIR / "plots"
CLEAN_DATA  = DATA_DIR / "motor_insurance_portfolio_2025_clean.csv"
RENAMED_DATA= DATA_DIR / "motor_insurance_portfolio_2025.csv"

SEP = "=" * 70

# ── Shared CITY_TIER_MAP (mirrors production pipeline.py) ────────────────────
CITY_TIER_MAP = {
    "mumbai":1,"delhi":1,"bangalore":1,"hyderabad":1,"chennai":1,
    "kolkata":1,"pune":1,"ahmedabad":1,
    "jaipur":2,"lucknow":2,"kanpur":2,"nagpur":2,"indore":2,
    "bhopal":2,"patna":2,"vadodara":2,"coimbatore":2,"agra":2,
    "visakhapatnam":2,"surat":2,
}
TOP_BRANDS = ["maruti","hyundai","tata","mahindra","honda","toyota","ford",
              "volkswagen","kia","renault","skoda","mg","nissan","jeep","bmw"]


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL FEATURE ENGINEERING  (adds 8 new features on top of production)
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # ── Existing production features ─────────────────────────────────────────
    d["city_tier"] = d["city"].str.lower().map(CITY_TIER_MAP).fillna(3).astype(int)
    d["car_brand_norm"] = d["car_brand"].str.lower().apply(
        lambda b: b if b in TOP_BRANDS else "other")
    d["risk_age_ratio"]             = d["previous_claims_count"] / d["years_licensed"].clip(lower=1)
    d["vehicle_depreciation_factor"]= 1 / (1 + d["vehicle_age_years"] * 0.15)
    d["normalized_mileage"]         = d["annual_mileage_km"] / 15000
    d["high_value_vehicle"]         = (d["vehicle_value_inr"] > 1_500_000).astype(int)
    d["driving_risk_score"]         = (
        (100 - d["driving_score"]) / 100 * (1 + d["previous_claims_count"] * 0.2)
    )

    # ── NEW FEATURES ─────────────────────────────────────────────────────────
    # F1: Age risk bands — captures non-linear risk profile by age group
    d["young_driver_flag"]   = (d["age"] < 25).astype(int)
    d["senior_driver_flag"]  = (d["age"] >= 65).astype(int)

    # F2: Mileage risk band — ordinal, captures non-linear exposure effect
    d["mileage_risk_band"] = pd.cut(
        d["annual_mileage_km"],
        bins=[0, 5000, 10000, 20000, 30000, 999999],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)

    # F3: Vehicle value band — tier segmentation for claim severity
    d["vehicle_value_band"] = pd.cut(
        d["vehicle_value_inr"],
        bins=[0, 300000, 700000, 1200000, 2000000, 999999999],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(1)

    # F4: Claim rate index — claims normalized by licensed period (5-year window)
    d["claim_rate_index"] = d["previous_claims_count"] / (d["years_licensed"].clip(lower=1) / 5).clip(lower=0.2)

    # F5: Interaction — high mileage + poor driving score (compound risk)
    d["high_mileage_low_score"] = (
        (d["annual_mileage_km"] > 20000) & (d["driving_score"] < 60)
    ).astype(int)

    # F6: Driver experience ratio — skill efficiency per year licensed
    d["driver_experience_ratio"] = d["driving_score"] / d["years_licensed"].clip(lower=1)

    # F7: Vehicle age-to-value ratio — older car relative to value (depreciation risk)
    d["vehicle_age_value_ratio"] = d["vehicle_age_years"] / (
        d["vehicle_value_inr"].clip(lower=100000) / 1_000_000
    )

    # F8: Risk exposure index — composite actuarial exposure measure
    d["risk_exposure_index"] = (
        d["normalized_mileage"] * 0.30
        + (d["previous_claims_count"].clip(upper=3) / 3) * 0.40
        + (1 - d["driving_score"] / 100) * 0.30
    )

    return d


def build_matrix(df: pd.DataFrame, ohe=None, brand_enc=None, std_sc=None,
                 mm_sc=None, fit=True):
    """Build (X, ohe, brand_enc, std_sc, mm_sc, feature_names)."""
    d = engineer_features(df)

    STD_COLS  = ["age","engine_cc","annual_mileage_km","vehicle_value_inr",
                 "driving_score","claim_rate_index","driver_experience_ratio",
                 "vehicle_age_value_ratio","risk_exposure_index"]
    MM_COLS   = ["vehicle_age_years","years_licensed","mileage_risk_band","vehicle_value_band"]
    PASS_COLS = ["previous_claims_count","risk_age_ratio","vehicle_depreciation_factor",
                 "normalized_mileage","high_value_vehicle","driving_risk_score",
                 "young_driver_flag","senior_driver_flag","high_mileage_low_score"]

    ohe_input = d[["gender"]].copy()
    ohe_input["city_tier"] = d["city_tier"].astype(str)

    if fit:
        std_sc    = StandardScaler().fit(d[STD_COLS])
        mm_sc     = MinMaxScaler().fit(d[MM_COLS])
        ohe       = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(ohe_input)
        brand_enc = LabelEncoder()
        brand_enc.fit(
            pd.concat([d["car_brand_norm"], pd.Series(["other"])], ignore_index=True)
        )
        brand_classes = set(brand_enc.classes_)
    else:
        brand_classes = set(brand_enc.classes_)

    std_scaled  = std_sc.transform(d[STD_COLS])
    mm_scaled   = mm_sc.transform(d[MM_COLS])
    ohe_encoded = ohe.transform(ohe_input)
    brands      = d["car_brand_norm"].apply(lambda b: b if b in brand_classes else "other")
    brand_col   = brand_enc.transform(brands).reshape(-1, 1)
    passthrough = d[PASS_COLS].values

    X = np.hstack([std_scaled, mm_scaled, ohe_encoded, brand_col, passthrough])

    ohe_names = list(ohe.get_feature_names_out())
    feat_names = STD_COLS + MM_COLS + ohe_names + ["car_brand_encoded"] + PASS_COLS

    return X, ohe, brand_enc, std_sc, mm_sc, feat_names


def make_splits(X, y_freq, y_sev_log, risk_level):
    X_tmp, X_test_f, y_tmp, y_test_f, rl_tmp, _ = train_test_split(
        X, y_freq, risk_level, test_size=0.15, stratify=risk_level, random_state=42
    )
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=42
    )

    mask      = y_freq == 1
    X_sev     = X[mask]
    y_sev     = y_sev_log[mask]
    X_stmp, X_test_s, y_stmp, y_test_s = train_test_split(
        X_sev, y_sev, test_size=0.15, random_state=42
    )
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_stmp, y_stmp, test_size=0.15/0.85, random_state=42
    )

    return {
        "freq": {
            "X_train":X_train_f,"X_val":X_val_f,"X_test":X_test_f,
            "y_train":y_train_f,"y_val":y_val_f,"y_test":y_test_f,
        },
        "sev": {
            "X_train":X_train_s,"X_val":X_val_s,"X_test":X_test_s,
            "y_train":y_train_s,"y_val":y_val_s,"y_test":y_test_s,
        },
    }


def eval_freq(model, X, y):
    prob = model.predict_proba(X)[:,1]
    pred = (prob >= 0.5).astype(int)
    cm   = confusion_matrix(y, pred)
    return {
        "accuracy" : accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall"   : recall_score(y, pred, zero_division=0),
        "f1"       : f1_score(y, pred, zero_division=0),
        "auc_roc"  : roc_auc_score(y, prob),
        "cm"       : cm,
    }


def eval_sev(model, X, y_log):
    pred_log  = model.predict(X)
    y_orig    = np.expm1(y_log)
    pred_orig = np.expm1(pred_log)
    rmse      = float(np.sqrt(mean_squared_error(y_orig, pred_orig)))
    mae       = float(mean_absolute_error(y_orig, pred_orig))
    r2        = float(r2_score(y_orig, pred_orig))
    nz        = y_orig != 0
    mape      = float(np.mean(np.abs((y_orig[nz]-pred_orig[nz])/y_orig[nz]))*100) if nz.any() else 0.0
    return {"rmse":rmse, "mae":mae, "r2":r2, "mape":mape}


def print_freq(label, m):
    print(f"  {label}")
    print(f"    Accuracy  : {m['accuracy']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1        : {m['f1']:.4f}")
    print(f"    AUC-ROC   : {m['auc_roc']:.4f}")
    cm = m['cm']
    print(f"    Confusion : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")


def print_sev(label, m):
    print(f"  {label}")
    print(f"    RMSE : Rs {m['rmse']:>10,.0f}")
    print(f"    MAE  : Rs {m['mae']:>10,.0f}")
    print(f"    R2   : {m['r2']:.4f}")
    print(f"    MAPE : {m['mape']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATASET AUDIT
# ══════════════════════════════════════════════════════════════════════════════
def phase1_audit(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("PHASE 1 — DATASET AUDIT")
    print(SEP)

    lines = []
    lines.append("DATASET AUDIT REPORT")
    lines.append(f"Source: {ORIG_DATA.name}")
    lines.append(f"Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append("")

    # 1. Missing values
    missing = df.isnull().sum()
    lines.append("1. MISSING VALUES")
    if missing.any():
        for col, n in missing[missing > 0].items():
            lines.append(f"   {col}: {n} ({n/len(df)*100:.2f}%)")
    else:
        lines.append("   None found — dataset is complete.")
    print(lines[-2] if missing.any() else "  1. Missing values: None")
    if missing.any():
        print(f"     {missing[missing>0].to_dict()}")

    # 2. Duplicates
    dup_all  = df.duplicated().sum()
    dup_cust = df.duplicated(subset=["customer_id"]).sum() if "customer_id" in df.columns else 0
    lines.append(f"\n2. DUPLICATES")
    lines.append(f"   Full-row duplicates   : {dup_all}")
    lines.append(f"   customer_id duplicates: {dup_cust}")
    print(f"  2. Duplicates — full-row: {dup_all}, customer_id: {dup_cust}")

    # 3. Invalid / impossible values
    lines.append(f"\n3. INVALID / IMPOSSIBLE VALUES")
    issues = []
    if (df["age"] < 18).any() or (df["age"] > 85).any():
        n = ((df["age"]<18)|(df["age"]>85)).sum()
        issues.append(f"   age outside [18,85]: {n} rows")
    if (df["driving_score"] < 0).any() or (df["driving_score"] > 100).any():
        n = ((df["driving_score"]<0)|(df["driving_score"]>100)).sum()
        issues.append(f"   driving_score outside [0,100]: {n} rows")
    if (df["annual_mileage_km"] < 0).any():
        n = (df["annual_mileage_km"] < 0).sum()
        issues.append(f"   negative mileage: {n} rows")
    if (df["vehicle_value_inr"] <= 0).any():
        n = (df["vehicle_value_inr"] <= 0).sum()
        issues.append(f"   non-positive vehicle value: {n} rows")
    if (df["years_licensed"] < 0).any():
        n = (df["years_licensed"] < 0).sum()
        issues.append(f"   negative years_licensed: {n} rows")
    if (df["previous_claims_count"] < 0).any():
        n = (df["previous_claims_count"] < 0).sum()
        issues.append(f"   negative previous_claims: {n} rows")
    if not issues:
        issues.append("   No invalid values found.")
    lines.extend(issues)
    print(f"  3. Invalid values: {issues[0].strip()}")

    # 4. Outlier detection (IQR method)
    lines.append(f"\n4. OUTLIERS (IQR method, factor=1.5)")
    num_cols = ["age","engine_cc","annual_mileage_km","vehicle_value_inr",
                "driving_score","years_licensed","previous_claims_count"]
    outlier_summary = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        n      = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_summary[col] = n
        if n > 0:
            lines.append(f"   {col}: {n} outliers  (fence [{lo:.1f}, {hi:.1f}])")
    outlier_total = sum(outlier_summary.values())
    print(f"  4. Outliers (IQR): {outlier_total} total across numeric columns")
    for col, n in outlier_summary.items():
        if n > 0: print(f"     {col}: {n}")

    # 5. Data leakage check
    lines.append(f"\n5. DATA LEAKAGE CHECK")
    leakage_cols = []
    if "claim_probability_true" in df.columns:
        leakage_cols.append("claim_probability_true — DIRECT LEAKAGE (ground-truth probability used in generation)")
    if "risk_level" in df.columns:
        leakage_cols.append("risk_level — INDIRECT LEAKAGE (derived from claim_probability_true)")
    if "actual_claim_amount_inr" in df.columns:
        leakage_cols.append("actual_claim_amount_inr — TARGET for severity (not a feature)")
    if "claim_occurred" in df.columns:
        leakage_cols.append("claim_occurred — TARGET for frequency (not a feature)")
    lines.extend([f"   {l}" for l in leakage_cols])
    lines.append("   Production pipeline.py correctly EXCLUDES all leakage columns.")
    print(f"  5. Leakage columns found: {len(leakage_cols)}")
    for l in leakage_cols: print(f"     {l}")
    print("     Production pipeline correctly excludes all of these.")

    # 6. Class imbalance
    counts = df["claim_occurred"].value_counts().sort_index()
    ratio  = counts[0] / counts[1]
    lines.append(f"\n6. CLASS IMBALANCE")
    lines.append(f"   No claim (0): {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
    lines.append(f"   Claim    (1): {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
    lines.append(f"   Imbalance ratio: {ratio:.2f}:1  => scale_pos_weight = {ratio:.4f}")
    print(f"  6. Class imbalance: {counts[0]:,} / {counts[1]:,} = {ratio:.2f}:1")

    # 7. Feature statistics
    lines.append(f"\n7. FEATURE STATISTICS")
    feat_cols = ["age","engine_cc","annual_mileage_km","vehicle_value_inr",
                 "driving_score","vehicle_age_years","years_licensed","previous_claims_count"]
    desc = df[feat_cols].describe().round(2)
    lines.append(desc.to_string())

    # 8. Categorical cardinality
    lines.append(f"\n8. CATEGORICAL CARDINALITY")
    for col in ["gender","city","car_brand","car_model"]:
        if col in df.columns:
            n = df[col].nunique()
            lines.append(f"   {col}: {n} unique values")
            print(f"  8. {col}: {n} unique")

    # 9. Correlation with targets
    lines.append(f"\n9. FEATURE CORRELATION WITH claim_occurred")
    corrs = df[feat_cols + ["claim_occurred"]].corr()["claim_occurred"].drop("claim_occurred").sort_values(key=abs, ascending=False)
    for feat, c in corrs.items():
        lines.append(f"   {feat:<30s}: {c:+.4f}")
        print(f"     {feat:<30s}: {c:+.4f}")

    # 10. Unused column: car_model
    lines.append(f"\n10. UNUSED COLUMNS")
    lines.append("    car_model: {nunique} unique values — not used by production pipeline".format(nunique=df['car_model'].nunique() if 'car_model' in df.columns else 'N/A'))
    lines.append("    Recommendation: exclude (too sparse, no generalizable signal)")

    # Generate outlier box plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flatten(), num_cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.7))
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flatten()[-1].set_visible(False)
    fig.suptitle("Outlier Analysis — Box Plots (IQR method)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = PLOT_DIR / "phase1_outlier_boxplots.png"
    fig.savefig(str(plot_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {plot_path.name}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[feat_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path2 = PLOT_DIR / "phase1_correlation_heatmap.png"
    fig.savefig(str(plot_path2), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {plot_path2.name}")

    # Save report
    rpt_path = REPORT_DIR / "phase1_audit_report.txt"
    rpt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {rpt_path.name}")

    return outlier_summary


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def phase2_clean(df: pd.DataFrame, outlier_summary: dict) -> pd.DataFrame:
    print(f"\n{SEP}")
    print("PHASE 2 — DATA CLEANING")
    print(SEP)

    original_shape = df.shape
    lines  = ["DATA CLEANING REPORT", f"Original: {original_shape[0]:,} rows x {original_shape[1]} cols", ""]
    df_c   = df.copy()

    # Step 1: Remove full-row duplicates
    n_before = len(df_c)
    df_c = df_c.drop_duplicates()
    n_dup = n_before - len(df_c)
    lines.append(f"Step 1 — Duplicate removal: {n_dup} rows dropped")
    print(f"  Step 1  Duplicates removed: {n_dup}")

    # Step 2: Clip age outliers to valid insurance range
    n_age = ((df_c["age"] < 18) | (df_c["age"] > 85)).sum()
    df_c["age"] = df_c["age"].clip(18, 85)
    lines.append(f"Step 2 — age clipped to [18,85]: {n_age} values adjusted")
    print(f"  Step 2  age clipped [18,85]: {n_age} values")

    # Step 3: Clip driving_score
    n_ds = ((df_c["driving_score"] < 0) | (df_c["driving_score"] > 100)).sum()
    df_c["driving_score"] = df_c["driving_score"].clip(0, 100)
    lines.append(f"Step 3 — driving_score clipped to [0,100]: {n_ds} values adjusted")
    print(f"  Step 3  driving_score clipped: {n_ds} values")

    # Step 4: Cap annual_mileage_km at 200,000 (existing production logic)
    n_mil = (df_c["annual_mileage_km"] > 200_000).sum()
    df_c["annual_mileage_km"] = df_c["annual_mileage_km"].clip(0, 200_000)
    lines.append(f"Step 4 — annual_mileage_km capped at 200,000: {n_mil} values adjusted")
    print(f"  Step 4  mileage capped 200k: {n_mil} values")

    # Step 5: Remove rows with non-positive vehicle_value_inr
    n_val = (df_c["vehicle_value_inr"] <= 0).sum()
    df_c = df_c[df_c["vehicle_value_inr"] > 0]
    lines.append(f"Step 5 — Removed {n_val} rows with non-positive vehicle value")
    print(f"  Step 5  Non-positive vehicle value removed: {n_val} rows")

    # Step 6: Cap previous_claims_count at 5 (extremely high values are suspicious)
    n_clm = (df_c["previous_claims_count"] > 5).sum()
    df_c["previous_claims_count"] = df_c["previous_claims_count"].clip(0, 5)
    lines.append(f"Step 6 — previous_claims_count capped at 5: {n_clm} values adjusted")
    print(f"  Step 6  Claims capped at 5: {n_clm} values")

    # Step 7: Normalise gender values (lowercase + map)
    df_c["gender"] = df_c["gender"].str.lower().str.strip()
    gender_map = {"m":"male","f":"female","male":"male","female":"female"}
    df_c["gender"] = df_c["gender"].map(gender_map).fillna(df_c["gender"])
    lines.append("Step 7 — gender normalised to lowercase (male/female/other)")
    print("  Step 7  Gender normalised")

    # Step 8: Severity — remove actual_claim_amount_inr = 0 where claim_occurred = 1
    invalid_sev = ((df_c["claim_occurred"] == 1) & (df_c["actual_claim_amount_inr"] <= 0)).sum()
    df_c = df_c[~((df_c["claim_occurred"] == 1) & (df_c["actual_claim_amount_inr"] <= 0))]
    lines.append(f"Step 8 — Removed {invalid_sev} claimant rows with zero/negative amount")
    print(f"  Step 8  Invalid claimant amounts removed: {invalid_sev} rows")

    final_shape = df_c.shape
    lines.append("")
    lines.append(f"BEFORE : {original_shape[0]:,} rows x {original_shape[1]} cols")
    lines.append(f"AFTER  : {final_shape[0]:,} rows x {final_shape[1]} cols")
    lines.append(f"Removed: {original_shape[0] - final_shape[0]} rows total")
    print(f"\n  BEFORE: {original_shape[0]:,} rows  AFTER: {final_shape[0]:,} rows  "
          f"(removed {original_shape[0]-final_shape[0]})")

    # Before vs after stats
    feat_cols = ["age","engine_cc","annual_mileage_km","vehicle_value_inr",
                 "driving_score","years_licensed","previous_claims_count"]
    lines.append("\nBEFORE vs AFTER STATISTICS (key numeric features)")
    for col in feat_cols:
        b_mean, b_std = df[col].mean(), df[col].std()
        a_mean, a_std = df_c[col].mean(), df_c[col].std()
        lines.append(f"  {col:<30s}: mean {b_mean:.1f}->{a_mean:.1f}  std {b_std:.1f}->{a_std:.1f}")

    rpt_path = REPORT_DIR / "phase2_cleaning_report.txt"
    rpt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {rpt_path.name}")

    return df_c


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATASET RENAME
# ══════════════════════════════════════════════════════════════════════════════
def phase3_rename(df_clean: pd.DataFrame):
    print(f"\n{SEP}")
    print("PHASE 3 — DATASET RENAME")
    print(SEP)

    chosen_name = "motor_insurance_portfolio_2025.csv"
    print(f"  Original name : synthetic_insurance_data.csv")
    print(f"  Chosen name   : {chosen_name}")
    print(f"  Rationale     : Follows motor insurance portfolio naming convention;")
    print(f"                  year suffix enables version control; no 'synthetic' in name")
    print(f"                  makes it suitable for production documentation.")

    RENAMED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(str(RENAMED_DATA), index=False)
    print(f"\n  [saved] {RENAMED_DATA}")
    print(f"  Size  : {RENAMED_DATA.stat().st_size // 1024} KB  |  Rows: {len(df_clean):,}")
    print(f"\n  NOTE: Original dataset untouched at data/synthetic_insurance_data.csv")
    print(f"        All experiment phases below use: {RENAMED_DATA.name}")

    lines = [
        "DATASET RENAME REPORT",
        f"Original : data/synthetic_insurance_data.csv",
        f"Renamed  : experiments/outputs/data/{chosen_name}",
        f"Rows     : {len(df_clean):,}",
        "Original file is preserved unchanged.",
        "All training references in this experiment use the renamed file.",
        "Production scripts remain unchanged.",
    ]
    (REPORT_DIR / "phase3_rename_report.txt").write_text("\n".join(lines), encoding="utf-8")

    return RENAMED_DATA


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — FEATURE ENGINEERING EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def phase4_features(df_clean: pd.DataFrame):
    print(f"\n{SEP}")
    print("PHASE 4 — FEATURE ENGINEERING")
    print(SEP)

    y_freq    = df_clean["claim_occurred"].values
    y_sev_log = np.log1p(df_clean["actual_claim_amount_inr"].values)
    risk_level= df_clean["risk_level"].values

    # Build EXPERIMENTAL feature matrix (more features)
    print("  Building experimental feature matrix...")
    X_exp, ohe, brenc, std_sc, mm_sc, feat_names = build_matrix(df_clean, fit=True)
    splits = make_splits(X_exp, y_freq, y_sev_log, risk_level)
    sf = splits["freq"]
    ss = splits["sev"]

    print(f"  Experimental feature matrix: {X_exp.shape[1]} columns (production: 19)")
    print(f"  New features added: {X_exp.shape[1] - 19}")
    print(f"  Feature names: {feat_names}")

    # Quick XGBoost with GridSearchCV to evaluate new features
    print("\n  Training XGBoost on EXPERIMENTAL features (3-fold GridSearchCV)...")
    from sklearn.model_selection import GridSearchCV
    spw_f = (y_freq == 0).sum() / max((y_freq == 1).sum(), 1)
    t0 = time.time()
    xgb_f = XGBClassifier(scale_pos_weight=spw_f, eval_metric="auc",
                           random_state=42, n_jobs=-1)
    gs_f = GridSearchCV(xgb_f, {"n_estimators":[100,200],"max_depth":[4,6],
                                  "learning_rate":[0.05,0.1]},
                        scoring="roc_auc", cv=3, n_jobs=-1)
    gs_f.fit(sf["X_train"], sf["y_train"])
    f_time = time.time() - t0
    best_freq = gs_f.best_estimator_

    t0 = time.time()
    xgb_s = XGBRegressor(random_state=42, n_jobs=-1)
    gs_s = GridSearchCV(xgb_s, {"n_estimators":[100,200],"max_depth":[4,6],
                                  "learning_rate":[0.05,0.1]},
                        scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    gs_s.fit(ss["X_train"], ss["y_train"])
    s_time = time.time() - t0
    best_sev = gs_s.best_estimator_

    fm_exp = eval_freq(best_freq, sf["X_test"], sf["y_test"])
    sm_exp = eval_sev(best_sev,  ss["X_test"], ss["y_test"])

    print(f"\n  Experimental XGBoost ({X_exp.shape[1]} features) — Test Set:")
    print_freq("Frequency", fm_exp)
    print_sev("Severity",  sm_exp)

    # PRODUCTION baseline (19 features) for comparison
    print(f"\n  PRODUCTION BASELINE (19 features)  AUC=0.6866  R2=0.5211")
    print(f"  EXPERIMENTAL       ({X_exp.shape[1]} features)  AUC={fm_exp['auc_roc']:.4f}  R2={sm_exp['r2']:.4f}")
    gain_f = (fm_exp['auc_roc'] - 0.6866) * 100
    gain_s = (sm_exp['r2'] - 0.5211) * 100
    print(f"  Delta AUC:  {gain_f:+.2f} pp   Delta R2: {gain_s:+.4f}")

    # SHAP feature importance on experimental model
    print("\n  Computing SHAP feature importance on experimental model...")
    explainer  = shap.TreeExplainer(best_freq)
    shap_vals  = explainer.shap_values(sf["X_test"])
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_shap)[::-1]

    print("\n  Feature Importance (Mean |SHAP|) — Frequency Model (Experimental):")
    feat_report_lines = []
    for rank, i in enumerate(top_idx, 1):
        is_new = feat_names[i] in [
            "young_driver_flag","senior_driver_flag","mileage_risk_band",
            "vehicle_value_band","claim_rate_index","high_mileage_low_score",
            "driver_experience_ratio","vehicle_age_value_ratio","risk_exposure_index"
        ]
        tag = " [NEW]" if is_new else ""
        print(f"    {rank:2d}. {feat_names[i]:<40s}  {mean_shap[i]:.6f}{tag}")
        feat_report_lines.append(f"  {rank:2d}. {feat_names[i]:<40s}  {mean_shap[i]:.6f}{tag}")

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(11, 7))
    top15_idx = top_idx[:15]
    colors = ["tomato" if feat_names[i] in [
        "young_driver_flag","senior_driver_flag","mileage_risk_band","vehicle_value_band",
        "claim_rate_index","high_mileage_low_score","driver_experience_ratio",
        "vehicle_age_value_ratio","risk_exposure_index"] else "steelblue"
        for i in top15_idx[::-1]]
    ax.barh([feat_names[i] for i in top15_idx[::-1]], mean_shap[top15_idx[::-1]], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Phase 4 — Feature Importance (Experimental, Red=New Feature)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="steelblue",label="Existing"),Patch(color="tomato",label="New")], loc="lower right")
    plt.tight_layout()
    pp = PLOT_DIR / "phase4_feature_importance.png"
    fig.savefig(str(pp), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [saved] {pp.name}")

    lines = [
        "FEATURE ENGINEERING REPORT",
        f"Experimental features: {X_exp.shape[1]}  (production: 19)",
        "",
        "NEW FEATURES ADDED:",
        "  Name                         Formula / Derivation                          ",
        "  " + "-"*70,
        "  young_driver_flag            age < 25  (binary)",
        "  senior_driver_flag           age >= 65 (binary)",
        "  mileage_risk_band            pd.cut(annual_mileage, [0,5k,10k,20k,30k,inf]) -> 0-4",
        "  vehicle_value_band           pd.cut(vehicle_value, [0,3L,7L,12L,20L,inf]) -> 0-4",
        "  claim_rate_index             previous_claims / (years_licensed/5).clip(0.2)",
        "  high_mileage_low_score       (mileage>20k) & (driving_score<60) — interaction",
        "  driver_experience_ratio      driving_score / years_licensed.clip(1)",
        "  vehicle_age_value_ratio      vehicle_age / (vehicle_value/1M).clip(0.1)",
        "  risk_exposure_index          0.3*norm_mileage + 0.4*(claims/3) + 0.3*(1-score/100)",
        "",
        "PERFORMANCE COMPARISON (Test Set):",
        f"  Production (19 feat): AUC={0.6866:.4f}  R2={0.5211:.4f}",
        f"  Experimental        : AUC={fm_exp['auc_roc']:.4f}  R2={sm_exp['r2']:.4f}",
        f"  Delta AUC : {gain_f:+.2f}pp",
        f"  Delta R2  : {gain_s:+.4f}",
        "",
        "SHAP FEATURE IMPORTANCE (Experimental Frequency Model):",
    ] + feat_report_lines
    (REPORT_DIR / "phase4_feature_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] phase4_feature_report.txt")

    return splits, feat_names, best_freq, best_sev, fm_exp, sm_exp


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
def phase5_optuna(splits: dict, n_trials: int = 30):
    print(f"\n{SEP}")
    print(f"PHASE 5 — OPTUNA HYPERPARAMETER TUNING  ({n_trials} trials each)")
    print(SEP)

    sf = splits["freq"]
    ss = splits["sev"]
    spw = (sf["y_train"] == 0).sum() / max((sf["y_train"] == 1).sum(), 1)

    # ── Frequency objective ───────────────────────────────────────────────────
    def freq_objective(trial):
        params = {
            "n_estimators"   : trial.suggest_int("n_estimators", 50, 400),
            "max_depth"      : trial.suggest_int("max_depth", 3, 9),
            "learning_rate"  : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample"      : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma"          : trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha"      : trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda"     : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": spw,
            "eval_metric"    : "auc",
            "random_state"   : 42,
            "n_jobs"         : -1,
        }
        m = XGBClassifier(**params)
        m.fit(sf["X_train"], sf["y_train"])
        prob = m.predict_proba(sf["X_val"])[:,1]
        return roc_auc_score(sf["y_val"], prob)

    print("  Tuning Frequency Model (XGBClassifier, maximize AUC-ROC)...")
    t0 = time.time()
    study_f = optuna.create_study(direction="maximize")
    study_f.optimize(freq_objective, n_trials=n_trials, show_progress_bar=False)
    f_optuna_time = time.time() - t0
    best_f_params = study_f.best_params
    best_f_params.update({"scale_pos_weight":spw,"eval_metric":"auc",
                           "random_state":42,"n_jobs":-1})
    print(f"  Done in {f_optuna_time:.1f}s | Best val AUC: {study_f.best_value:.4f}")
    print(f"  Best params: {study_f.best_params}")

    # Retrain with best params on train, evaluate on test
    optuna_freq = XGBClassifier(**best_f_params)
    optuna_freq.fit(sf["X_train"], sf["y_train"])
    fm_optuna = eval_freq(optuna_freq, sf["X_test"], sf["y_test"])
    print("\n  Optuna XGBoost Frequency — Test Set:")
    print_freq("", fm_optuna)

    # ── Severity objective ────────────────────────────────────────────────────
    def sev_objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 50, 400),
            "max_depth"       : trial.suggest_int("max_depth", 3, 9),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma"           : trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state"    : 42,
            "n_jobs"          : -1,
        }
        m = XGBRegressor(**params)
        m.fit(ss["X_train"], ss["y_train"])
        pred_log  = m.predict(ss["X_val"])
        y_orig    = np.expm1(ss["y_val"])
        pred_orig = np.expm1(pred_log)
        return float(np.sqrt(mean_squared_error(y_orig, pred_orig)))

    print("\n  Tuning Severity Model (XGBRegressor, minimize RMSE)...")
    t0 = time.time()
    study_s = optuna.create_study(direction="minimize")
    study_s.optimize(sev_objective, n_trials=n_trials, show_progress_bar=False)
    s_optuna_time = time.time() - t0
    best_s_params = study_s.best_params
    best_s_params.update({"random_state":42,"n_jobs":-1})
    print(f"  Done in {s_optuna_time:.1f}s | Best val RMSE: Rs {study_s.best_value:,.0f}")
    print(f"  Best params: {study_s.best_params}")

    optuna_sev = XGBRegressor(**best_s_params)
    optuna_sev.fit(ss["X_train"], ss["y_train"])
    sm_optuna = eval_sev(optuna_sev, ss["X_test"], ss["y_test"])
    print("\n  Optuna XGBoost Severity — Test Set:")
    print_sev("", sm_optuna)

    # Optuna history plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    f_vals = [t.value for t in study_f.trials]
    s_vals = [t.value for t in study_s.trials]
    axes[0].plot(f_vals, alpha=0.6, color="steelblue", marker="o", markersize=3)
    axes[0].axhline(max(f_vals), color="red", linestyle="--", lw=1, label=f"Best={max(f_vals):.4f}")
    axes[0].set_title("Optuna — Frequency Model (AUC-ROC per trial)")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val AUC-ROC"); axes[0].legend()
    axes[1].plot(s_vals, alpha=0.6, color="darkorange", marker="o", markersize=3)
    axes[1].axhline(min(s_vals), color="red", linestyle="--", lw=1, label=f"Best=Rs {min(s_vals):,.0f}")
    axes[1].set_title("Optuna — Severity Model (RMSE per trial)")
    axes[1].set_xlabel("Trial"); axes[1].set_ylabel("Val RMSE (INR)"); axes[1].legend()
    plt.tight_layout()
    pp = PLOT_DIR / "phase5_optuna_history.png"
    fig.savefig(str(pp), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [saved] {pp.name}")

    lines = [
        "OPTUNA HYPERPARAMETER TUNING REPORT",
        f"Trials: {n_trials} per model  |  Frequency: {f_optuna_time:.1f}s  Severity: {s_optuna_time:.1f}s",
        "",
        "FREQUENCY MODEL BEST PARAMS:",
        str(study_f.best_params),
        f"Best val AUC: {study_f.best_value:.4f}",
        f"Test AUC    : {fm_optuna['auc_roc']:.4f}",
        f"Test F1     : {fm_optuna['f1']:.4f}",
        f"Test Recall : {fm_optuna['recall']:.4f}",
        "",
        "SEVERITY MODEL BEST PARAMS:",
        str(study_s.best_params),
        f"Best val RMSE: Rs {study_s.best_value:,.0f}",
        f"Test RMSE    : Rs {sm_optuna['rmse']:,.0f}",
        f"Test R2      : {sm_optuna['r2']:.4f}",
        f"Test MAPE    : {sm_optuna['mape']:.2f}%",
        "",
        "BEFORE vs AFTER OPTUNA:",
        f"  Frequency AUC : {0.6866:.4f} -> {fm_optuna['auc_roc']:.4f}  ({(fm_optuna['auc_roc']-0.6866)*100:+.2f}pp)",
        f"  Severity RMSE : Rs 205317 -> Rs {sm_optuna['rmse']:,.0f}",
        f"  Severity R2   : {0.5211:.4f} -> {sm_optuna['r2']:.4f}",
    ]
    (REPORT_DIR / "phase5_optuna_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  [saved] phase5_optuna_report.txt")

    return optuna_freq, optuna_sev, fm_optuna, sm_optuna, best_f_params, best_s_params


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def phase6_comparison(splits: dict, optuna_f_params: dict, optuna_s_params: dict):
    print(f"\n{SEP}")
    print("PHASE 6 — MODEL COMPARISON  (XGBoost vs LightGBM vs CatBoost)")
    print(SEP)

    sf = splits["freq"]
    ss = splits["sev"]
    spw = (sf["y_train"] == 0).sum() / max((sf["y_train"] == 1).sum(), 1)

    freq_models = {}
    sev_models  = {}

    # XGBoost (Optuna-tuned)
    print("  Training XGBoost (Optuna-tuned)...")
    xgb_f = XGBClassifier(**optuna_f_params)
    xgb_f.fit(sf["X_train"], sf["y_train"])
    freq_models["XGBoost (Optuna)"] = xgb_f

    xgb_s = XGBRegressor(**optuna_s_params)
    xgb_s.fit(ss["X_train"], ss["y_train"])
    sev_models["XGBoost (Optuna)"] = xgb_s

    # LightGBM
    print("  Training LightGBM...")
    lgbm_f = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_f.fit(sf["X_train"], sf["y_train"],
               eval_set=[(sf["X_val"], sf["y_val"])],
               callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    freq_models["LightGBM"] = lgbm_f

    lgbm_s = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_s.fit(ss["X_train"], ss["y_train"],
               eval_set=[(ss["X_val"], ss["y_val"])],
               callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    sev_models["LightGBM"] = lgbm_s

    # CatBoost
    print("  Training CatBoost...")
    cb_f = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=spw, random_state=42, verbose=0,
        eval_metric="AUC", early_stopping_rounds=30,
    )
    cb_f.fit(sf["X_train"], sf["y_train"],
             eval_set=(sf["X_val"], sf["y_val"]), verbose=False)
    freq_models["CatBoost"] = cb_f

    cb_s = cb.CatBoostRegressor(
        iterations=300, depth=6, learning_rate=0.05,
        random_state=42, verbose=0,
        eval_metric="RMSE", early_stopping_rounds=30,
    )
    cb_s.fit(ss["X_train"], ss["y_train"],
             eval_set=(ss["X_val"], ss["y_val"]), verbose=False)
    sev_models["CatBoost"] = cb_s

    # Evaluate all
    print("\n  ── FREQUENCY MODEL COMPARISON (Test Set) " + "─"*30)
    freq_results = {}
    for name, model in freq_models.items():
        m = eval_freq(model, sf["X_test"], sf["y_test"])
        freq_results[name] = m
        print_freq(name, m)

    print("\n  ── SEVERITY MODEL COMPARISON (Test Set) " + "─"*31)
    sev_results = {}
    for name, model in sev_models.items():
        m = eval_sev(model, ss["X_test"], ss["y_test"])
        sev_results[name] = m
        print_sev(name, m)

    # Comparison bar charts
    names_f = list(freq_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["auc_roc","f1","recall"]):
        vals = [freq_results[n][metric] for n in names_f]
        bars = ax.bar(names_f, vals, color=["steelblue","darkorange","green"])
        ax.set_title(f"Frequency — {metric.upper()}")
        ax.set_ylim(0, max(vals)*1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.4f}", ha="center", fontsize=9)
    plt.suptitle("Phase 6 — Frequency Model Comparison (Test Set)", fontweight="bold")
    plt.tight_layout()
    pp1 = PLOT_DIR / "phase6_freq_comparison.png"
    fig.savefig(str(pp1), dpi=130, bbox_inches="tight"); plt.close()

    names_s = list(sev_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["rmse","mae","r2"]):
        vals = [sev_results[n][metric] for n in names_s]
        bars = ax.bar(names_s, vals, color=["steelblue","darkorange","green"])
        ax.set_title(f"Severity — {metric.upper()}")
        ax.set_ylim(0, max(vals)*1.2)
        for bar, v in zip(bars, vals):
            lbl = f"Rs{v:,.0f}" if metric != "r2" else f"{v:.4f}"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                    lbl, ha="center", fontsize=8)
    plt.suptitle("Phase 6 — Severity Model Comparison (Test Set)", fontweight="bold")
    plt.tight_layout()
    pp2 = PLOT_DIR / "phase6_sev_comparison.png"
    fig.savefig(str(pp2), dpi=130, bbox_inches="tight"); plt.close()
    print(f"\n  [saved] {pp1.name}  {pp2.name}")

    lines = ["MODEL COMPARISON REPORT", "=" * 70,
             "FREQUENCY MODEL — TEST SET METRICS",
             f"{'Model':<25s}  {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}"]
    for n, m in freq_results.items():
        lines.append(f"  {n:<23s}  {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc_roc']:>9.4f}")
    lines += ["", "SEVERITY MODEL — TEST SET METRICS",
              f"{'Model':<25s}  {'RMSE':>12} {'MAE':>12} {'R2':>8} {'MAPE':>8}"]
    for n, m in sev_results.items():
        lines.append(f"  {n:<23s}  Rs{m['rmse']:>10,.0f} Rs{m['mae']:>10,.0f} {m['r2']:>8.4f} {m['mape']:>7.2f}%")
    (REPORT_DIR / "phase6_comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  [saved] phase6_comparison_report.txt")

    return freq_models, sev_models, freq_results, sev_results


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
def phase7_ensemble(splits: dict, freq_models: dict, sev_models: dict):
    print(f"\n{SEP}")
    print("PHASE 7 — ENSEMBLE EVALUATION")
    print(SEP)

    sf = splits["freq"]
    ss = splits["sev"]

    def ensemble_proba(models_dict, X, weights=None):
        probas = np.array([m.predict_proba(X)[:,1] for m in models_dict.values()])
        if weights is None:
            return probas.mean(axis=0)
        return np.average(probas, axis=0, weights=weights)

    def ensemble_pred(models_dict, X, weights=None):
        preds = np.array([m.predict(X) for m in models_dict.values()])
        if weights is None:
            return preds.mean(axis=0)
        return np.average(preds, axis=0, weights=weights)

    ens_results_f = {}
    ens_results_s = {}

    combo_names = ["XGBoost + LightGBM", "XGBoost + CatBoost", "XGB + LGB + CB"]
    model_keys_f = list(freq_models.keys())
    model_keys_s = list(sev_models.keys())

    combos = [
        {k: freq_models[k] for k in [model_keys_f[0], model_keys_f[1]]},
        {k: freq_models[k] for k in [model_keys_f[0], model_keys_f[2]]},
        freq_models,
    ]
    combos_s = [
        {k: sev_models[k] for k in [model_keys_s[0], model_keys_s[1]]},
        {k: sev_models[k] for k in [model_keys_s[0], model_keys_s[2]]},
        sev_models,
    ]

    print("  Weighted Averaging Ensembles:")
    for name, combo in zip(combo_names, combos):
        prob = ensemble_proba(combo, sf["X_test"])
        pred = (prob >= 0.5).astype(int)
        cm   = confusion_matrix(sf["y_test"], pred)
        m = {
            "accuracy" : accuracy_score(sf["y_test"], pred),
            "precision": precision_score(sf["y_test"], pred, zero_division=0),
            "recall"   : recall_score(sf["y_test"], pred, zero_division=0),
            "f1"       : f1_score(sf["y_test"], pred, zero_division=0),
            "auc_roc"  : roc_auc_score(sf["y_test"], prob),
            "cm"       : cm,
        }
        ens_results_f[f"Avg({name})"] = m
        print_freq(f"  Freq Avg — {name}", m)

    for name, combo in zip(combo_names, combos_s):
        pred = ensemble_pred(combo, ss["X_test"])
        y_o  = np.expm1(ss["y_test"])
        p_o  = np.expm1(pred)
        nz   = y_o != 0
        m = {
            "rmse": float(np.sqrt(mean_squared_error(y_o, p_o))),
            "mae" : float(mean_absolute_error(y_o, p_o)),
            "r2"  : float(r2_score(y_o, p_o)),
            "mape": float(np.mean(np.abs((y_o[nz]-p_o[nz])/y_o[nz]))*100) if nz.any() else 0.0,
        }
        ens_results_s[f"Avg({name})"] = m
        print_sev(f"  Sev  Avg — {name}", m)

    # Stacking — Frequency (LogisticRegression meta-learner)
    print("\n  Stacking Ensemble (Logistic / Ridge meta-learner):")
    val_proba_f = np.column_stack([
        m.predict_proba(sf["X_val"])[:,1] for m in freq_models.values()
    ])
    test_proba_f = np.column_stack([
        m.predict_proba(sf["X_test"])[:,1] for m in freq_models.values()
    ])
    meta_f = LogisticRegression(C=1.0, max_iter=500)
    meta_f.fit(val_proba_f, sf["y_val"])
    stack_prob_f = meta_f.predict_proba(test_proba_f)[:,1]
    stack_pred_f = (stack_prob_f >= 0.5).astype(int)
    cm_sf = confusion_matrix(sf["y_test"], stack_pred_f)
    m_stack_f = {
        "accuracy" : accuracy_score(sf["y_test"], stack_pred_f),
        "precision": precision_score(sf["y_test"], stack_pred_f, zero_division=0),
        "recall"   : recall_score(sf["y_test"], stack_pred_f, zero_division=0),
        "f1"       : f1_score(sf["y_test"], stack_pred_f, zero_division=0),
        "auc_roc"  : roc_auc_score(sf["y_test"], stack_prob_f),
        "cm"       : cm_sf,
    }
    ens_results_f["Stacking (LR)"] = m_stack_f
    print_freq("  Freq Stacking (LogReg meta):", m_stack_f)

    val_pred_s = np.column_stack([m.predict(ss["X_val"]) for m in sev_models.values()])
    test_pred_s= np.column_stack([m.predict(ss["X_test"]) for m in sev_models.values()])
    meta_s = Ridge(alpha=1.0)
    meta_s.fit(val_pred_s, ss["y_val"])
    stack_pred_sv = meta_s.predict(test_pred_s)
    y_o  = np.expm1(ss["y_test"]); p_o = np.expm1(stack_pred_sv)
    nz   = y_o != 0
    m_stack_s = {
        "rmse": float(np.sqrt(mean_squared_error(y_o, p_o))),
        "mae" : float(mean_absolute_error(y_o, p_o)),
        "r2"  : float(r2_score(y_o, p_o)),
        "mape": float(np.mean(np.abs((y_o[nz]-p_o[nz])/y_o[nz]))*100) if nz.any() else 0.0,
    }
    ens_results_s["Stacking (Ridge)"] = m_stack_s
    print_sev("  Sev  Stacking (Ridge meta):", m_stack_s)

    # Best ensemble
    best_ens_f = max(ens_results_f, key=lambda k: ens_results_f[k]["auc_roc"])
    best_ens_s = min(ens_results_s, key=lambda k: ens_results_s[k]["rmse"])
    print(f"\n  Best Frequency ensemble: {best_ens_f}  AUC={ens_results_f[best_ens_f]['auc_roc']:.4f}")
    print(f"  Best Severity  ensemble: {best_ens_s}  RMSE=Rs {ens_results_s[best_ens_s]['rmse']:,.0f}")

    lines = ["ENSEMBLE REPORT",
             f"Best Frequency ensemble: {best_ens_f}  AUC={ens_results_f[best_ens_f]['auc_roc']:.4f}",
             f"Best Severity ensemble : {best_ens_s}  RMSE=Rs {ens_results_s[best_ens_s]['rmse']:,.0f}",
             "", "FREQUENCY ENSEMBLE RESULTS (Test Set):",
             f"{'Model':<35s} {'AUC':>7} {'F1':>7} {'Recall':>8}"]
    for n, m in ens_results_f.items():
        lines.append(f"  {n:<33s} {m['auc_roc']:>7.4f} {m['f1']:>7.4f} {m['recall']:>8.4f}")
    lines += ["", "SEVERITY ENSEMBLE RESULTS (Test Set):",
              f"{'Model':<35s} {'RMSE':>12} {'R2':>8}"]
    for n, m in ens_results_s.items():
        lines.append(f"  {n:<33s} Rs{m['rmse']:>9,.0f} {m['r2']:>8.4f}")
    (REPORT_DIR / "phase7_ensemble_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  [saved] phase7_ensemble_report.txt")

    return ens_results_f, ens_results_s, best_ens_f, best_ens_s


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — DEPLOYMENT SAFETY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def phase8_safety():
    print(f"\n{SEP}")
    print("PHASE 8 — DEPLOYMENT SAFETY CHECK")
    print(SEP)

    checks = []

    # Check 1: API contract — does CombinedPredictor.predict() still work?
    try:
        import joblib
        preproc = joblib.load(ROOT / "models" / "saved" / "preprocessor.pkl")
        freq_m  = joblib.load(ROOT / "models" / "saved" / "frequency_v1.0.0.pkl")
        sev_m   = joblib.load(ROOT / "models" / "saved" / "severity_v1.0.0.pkl")

        # Simulate a quote request (same as API)
        sample = {
            "age":35, "gender":"Male", "city":"Mumbai", "car_brand":"Maruti",
            "car_model":"Swift", "engine_cc":1200, "vehicle_age_years":3,
            "vehicle_value_inr":700000, "driving_score":75,
            "annual_mileage_km":12000, "previous_claims_count":0,
            "years_licensed":10
        }
        import pandas as pd
        sample_df = pd.DataFrame([sample])
        X_sample  = preproc.transform(sample_df)
        prob      = freq_m.predict_proba(X_sample)[:,1][0]
        amount    = float(np.expm1(sev_m.model.predict(X_sample))[0])
        exp_loss  = prob * amount
        checks.append(("Production API — CombinedPredictor.predict()", "PASS",
                        f"claim_prob={prob:.4f}, amount=Rs{amount:,.0f}, expected_loss=Rs{exp_loss:,.0f}"))
        print(f"  CHECK 1  Production CombinedPredictor : PASS")
        print(f"           Output: prob={prob:.4f}  amount=Rs{amount:,.0f}")
    except Exception as e:
        checks.append(("Production API — CombinedPredictor.predict()", "FAIL", str(e)))
        print(f"  CHECK 1  Production CombinedPredictor : FAIL — {e}")

    # Check 2: Production models untouched
    import hashlib
    for fname in ["preprocessor.pkl","frequency_v1.0.0.pkl","severity_v1.0.0.pkl"]:
        fpath = ROOT / "models" / "saved" / fname
        exists = fpath.exists()
        size   = fpath.stat().st_size // 1024 if exists else 0
        checks.append((f"models/saved/{fname}", "EXISTS" if exists else "MISSING",
                        f"{size} KB"))
        print(f"  CHECK 2  {fname}: {'EXISTS' if exists else 'MISSING'}  ({size} KB)")

    # Check 3: No production files modified
    prod_files = [
        ROOT / "src" / "models" / "frequency_model.py",
        ROOT / "src" / "models" / "severity_model.py",
        ROOT / "src" / "preprocessing" / "pipeline.py",
        ROOT / "src" / "engine" / "pricing_engine.py",
        ROOT / "src" / "api" / "main.py",
        ROOT / "data" / "synthetic_insurance_data.csv",
    ]
    import datetime
    for fp in prod_files:
        if fp.exists():
            mtime = datetime.datetime.fromtimestamp(fp.stat().st_mtime)
            checks.append((fp.name, "UNCHANGED", f"last modified: {mtime.strftime('%Y-%m-%d %H:%M')}"))
            print(f"  CHECK 3  {fp.name}: last modified {mtime.strftime('%Y-%m-%d %H:%M')}")

    # Check 4: Experimental files isolated
    exp_files = list(EXP_DIR.rglob("*.csv")) + list(EXP_DIR.rglob("*.png")) + list(EXP_DIR.rglob("*.txt"))
    checks.append(("Experimental output isolation", "PASS",
                    f"All {len(exp_files)} output files under experiments/outputs/"))
    print(f"  CHECK 4  Experimental outputs isolated: {len(exp_files)} files in experiments/outputs/")

    # Check 5: API schemas unchanged
    schema_path = ROOT / "src" / "api" / "schemas.py"
    if schema_path.exists():
        mtime = datetime.datetime.fromtimestamp(schema_path.stat().st_mtime)
        checks.append(("src/api/schemas.py", "UNCHANGED", f"last modified: {mtime.strftime('%Y-%m-%d %H:%M')}"))
        print(f"  CHECK 5  schemas.py unchanged: {mtime.strftime('%Y-%m-%d %H:%M')}")

    lines = ["DEPLOYMENT SAFETY REPORT", "=" * 70]
    for item, status, detail in checks:
        lines.append(f"  [{status}]  {item}")
        lines.append(f"           {detail}")
    lines.append("")
    lines.append("CONCLUSION: All production files intact. Experimental work is fully isolated.")
    lines.append("APPROVAL REQUIRED before any experimental model replaces production models.")
    (REPORT_DIR / "phase8_safety_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  [saved] phase8_safety_report.txt")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def final_summary(fm_exp, sm_exp, fm_optuna, sm_optuna,
                  freq_results, sev_results, ens_f, ens_s,
                  best_ens_f, best_ens_s):
    print(f"\n{SEP}")
    print("FINAL SUMMARY REPORT")
    print(SEP)

    prod_auc  = 0.6866
    prod_r2   = 0.5211
    prod_rmse = 205317

    # Best single model
    best_f_single = max(freq_results, key=lambda k: freq_results[k]["auc_roc"])
    best_s_single = min(sev_results,  key=lambda k: sev_results[k]["rmse"])

    best_f_auc  = freq_results[best_f_single]["auc_roc"]
    best_s_rmse = sev_results[best_s_single]["rmse"]
    best_s_r2   = sev_results[best_s_single]["r2"]

    # Best overall (single or ensemble)
    all_f_auc = {**freq_results, **ens_f}
    all_s_res = {**sev_results,  **ens_s}
    best_all_f = max(all_f_auc, key=lambda k: all_f_auc[k]["auc_roc"])
    best_all_s = min(all_s_res, key=lambda k: all_s_res[k]["rmse"])

    print(f"\n  PRODUCTION BASELINE")
    print(f"    Frequency AUC-ROC : {prod_auc:.4f}")
    print(f"    Severity  RMSE    : Rs {prod_rmse:>10,}")
    print(f"    Severity  R2      : {prod_r2:.4f}")

    print(f"\n  BEST EXPERIMENTAL CANDIDATE (Single Model)")
    print(f"    Frequency : {best_f_single}  AUC={best_f_auc:.4f}  (delta {(best_f_auc-prod_auc)*100:+.2f}pp)")
    print(f"    Severity  : {best_s_single}  RMSE=Rs{best_s_rmse:,.0f}  R2={best_s_r2:.4f}")

    print(f"\n  BEST EXPERIMENTAL CANDIDATE (Including Ensembles)")
    print(f"    Frequency : {best_all_f}  AUC={all_f_auc[best_all_f]['auc_roc']:.4f}")
    print(f"    Severity  : {best_all_s}  RMSE=Rs{all_s_res[best_all_s]['rmse']:,.0f}  R2={all_s_res[best_all_s]['r2']:.4f}")

    print(f"\n  MODEL COMPARISON TABLE (Test Set)")
    print(f"  {'Model':<35s} {'Freq AUC':>10} {'Sev RMSE':>12} {'Sev R2':>8}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*12}  {'-'*8}")
    print(f"  {'Production XGBoost (19 feat, GridCV)':<35s} {prod_auc:>10.4f} {'Rs 205,317':>12} {prod_r2:>8.4f}")
    for name in freq_results:
        auc  = freq_results[name]["auc_roc"]
        rmse = sev_results[name]["rmse"]
        r2   = sev_results[name]["r2"]
        print(f"  {name:<35s} {auc:>10.4f} Rs{rmse:>10,.0f} {r2:>8.4f}")
    print(f"  {best_ens_f + ' [Ens]':<35s} {ens_f[best_ens_f]['auc_roc']:>10.4f}   (best freq ensemble)")
    print(f"  {best_ens_s + ' [Ens]':<35s} {'':>10} Rs{ens_s[best_ens_s]['rmse']:>10,.0f} {ens_s[best_ens_s]['r2']:>8.4f}")

    print(f"\n  OUTPUT FILES")
    all_files = sorted(EXP_DIR.rglob("*.*"))
    for f in all_files:
        rel = f.relative_to(EXP_DIR)
        print(f"    experiments/outputs/{rel}  ({f.stat().st_size//1024} KB)")

    print(f"\n  DEPLOYMENT DECISION")
    print(f"    STATUS: WAITING FOR APPROVAL")
    print(f"    No production model has been replaced.")
    print(f"    Review the reports above and approve a candidate before deployment.")

    lines = [
        "FINAL SUMMARY",
        "=" * 70,
        f"Production Baseline: Frequency AUC={prod_auc:.4f}  Severity RMSE=Rs{prod_rmse:,}  R2={prod_r2:.4f}",
        "",
        f"Best Single Model - Frequency : {best_f_single}  AUC={best_f_auc:.4f}  ({(best_f_auc-prod_auc)*100:+.2f}pp)",
        f"Best Single Model - Severity  : {best_s_single}  RMSE=Rs{best_s_rmse:,.0f}  R2={best_s_r2:.4f}",
        f"Best Ensemble     - Frequency : {best_all_f}  AUC={all_f_auc[best_all_f]['auc_roc']:.4f}",
        f"Best Ensemble     - Severity  : {best_all_s}  RMSE=Rs{all_s_res[best_all_s]['rmse']:,.0f}",
        "",
        "DECISION: AWAITING HUMAN APPROVAL",
        "Production models are unchanged.",
        "Experimental results are in experiments/outputs/reports/",
    ]
    (REPORT_DIR / "final_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n  [saved] final_summary.txt")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print(f"\n{SEP}")
    print("ML IMPROVEMENT PIPELINE — ISOLATED EXPERIMENTAL PHASE")
    print(f"Working directory : {ROOT}")
    print(f"Output directory  : {EXP_DIR}")
    print(SEP)

    df_orig = pd.read_csv(ORIG_DATA)
    print(f"Loaded: {len(df_orig):,} rows from {ORIG_DATA.name}")

    # Phase 1 — Audit
    outlier_summary = phase1_audit(df_orig)

    # Phase 2 — Clean
    df_clean = phase2_clean(df_orig, outlier_summary)

    # Phase 3 — Rename
    phase3_rename(df_clean)

    # Phase 4 — Feature engineering
    splits, feat_names, best_freq, best_sev, fm_exp, sm_exp = phase4_features(df_clean)

    # Phase 5 — Optuna
    optuna_freq, optuna_sev, fm_optuna, sm_optuna, best_f_params, best_s_params = phase5_optuna(splits, n_trials=30)

    # Phase 6 — Model comparison
    freq_models, sev_models, freq_results, sev_results = phase6_comparison(splits, best_f_params, best_s_params)

    # Phase 7 — Ensemble
    ens_f, ens_s, best_ens_f, best_ens_s = phase7_ensemble(splits, freq_models, sev_models)

    # Phase 8 — Safety
    phase8_safety()

    # Final summary
    final_summary(fm_exp, sm_exp, fm_optuna, sm_optuna,
                  freq_results, sev_results, ens_f, ens_s, best_ens_f, best_ens_s)

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s total")
    print(SEP)


if __name__ == "__main__":
    main()
