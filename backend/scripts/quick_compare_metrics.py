"""
Quick before/after metrics comparison after severity target regeneration.
Trains XGBoost on both frequency and severity using the same pipeline as the
main training scripts. Outputs clean before/after comparison.

Run from backend/:  python scripts/quick_compare_metrics.py
"""
from __future__ import annotations

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing.pipeline import InsurancePreprocessor

BACKEND     = Path(__file__).resolve().parent.parent
MASTER_PATH = BACKEND / "data" / "master" / "motor_insurance_master_dataset_50000.csv"

# ── Load and prepare ───────────────────────────────────────────────────────────
print("\n[LOAD] Reading dataset ...")
df = pd.read_csv(MASTER_PATH)

# Drop leakage cols (same as train_master_dataset.py)
leakage = {"claim_probability", "expected_loss_inr", "premium_inr", "customer_id"}
df_ml = df.drop(columns=[c for c in leakage if c in df.columns])
df_ml = df_ml.rename(columns={"vehicle_brand": "car_brand", "previous_claims": "previous_claims_count"})
if "gender" in df_ml.columns:
    df_ml["gender"] = df_ml["gender"].str.lower()

claimants = df[df["claim_occurred"] == 1]
print(f"       Claimants: {len(claimants):,}  |  "
      f"Severity mean=Rs.{claimants['actual_claim_amount_inr'].mean():,.0f}  "
      f"median=Rs.{claimants['actual_claim_amount_inr'].median():,.0f}")

# ── Preprocess ────────────────────────────────────────────────────────────────
print("[PREP] Fitting InsurancePreprocessor ...")
preprocessor = InsurancePreprocessor(model_dir=str(BACKEND / "models" / "saved"))
X = preprocessor.fit_transform(df_ml)

y_freq = df_ml["claim_occurred"].values
y_sev_raw = df_ml["actual_claim_amount_inr"].values  # raw INR

print(f"       Feature matrix: {X.shape}")

# ── Frequency splits ───────────────────────────────────────────────────────────
stratify = df_ml.get("risk_level") if "risk_level" in df_ml.columns else None
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y_freq, test_size=0.15,
                                              stratify=stratify, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, random_state=42)

# ── Severity splits (claimants only) ──────────────────────────────────────────
sev_mask = (df_ml["claim_occurred"] == 1).values & (y_sev_raw > 0)
X_sev    = X[sev_mask]
y_sev    = y_sev_raw[sev_mask]

X_stmp, X_ste, y_stmp, y_ste = train_test_split(X_sev, y_sev, test_size=0.15, random_state=42)
X_str, X_sva, y_str, y_sva   = train_test_split(X_stmp, y_stmp, test_size=0.15/0.85, random_state=42)

n_neg = int((y_freq == 0).sum())
n_pos = int((y_freq == 1).sum())
spw   = round(n_neg / max(n_pos, 1), 3)

# ── Frequency model ────────────────────────────────────────────────────────────
print("\n[FREQ] Training XGBoost frequency model ...")
freq_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    scale_pos_weight=spw, random_state=42, n_jobs=-1,
    eval_metric="auc", verbosity=0,
)
freq_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

fp_te = freq_model.predict_proba(X_te)[:, 1]
y_pred_freq = (fp_te >= 0.5).astype(int)

freq_m = {
    "auc":       round(float(roc_auc_score(y_te, fp_te)), 4),
    "f1":        round(float(f1_score(y_te, y_pred_freq, zero_division=0)), 4),
    "precision": round(float(precision_score(y_te, y_pred_freq, zero_division=0)), 4),
    "recall":    round(float(recall_score(y_te, y_pred_freq, zero_division=0)), 4),
    "brier":     round(float(brier_score_loss(y_te, fp_te)), 4),
}
print(f"       AUC={freq_m['auc']}  F1={freq_m['f1']}  "
      f"Prec={freq_m['precision']}  Rec={freq_m['recall']}  Brier={freq_m['brier']}")

# ── Severity model (raw INR, consistent with multi_algorithm_engine) ───────────
print("\n[SEV]  Training XGBoost severity model ...")
sev_model = XGBRegressor(
    n_estimators=600, max_depth=3, learning_rate=0.02,
    subsample=0.6, colsample_bytree=0.6,
    min_child_weight=50, gamma=0.5,
    reg_alpha=1.0, reg_lambda=5.0,
    random_state=42, n_jobs=-1, verbosity=0,
)
sev_model.fit(X_str, y_str, eval_set=[(X_sva, y_sva)], verbose=False)

sp_te  = sev_model.predict(X_ste)
sp_te  = np.maximum(sp_te, 0.0)

sev_m = {
    "r2":   round(float(r2_score(y_ste, sp_te)), 4),
    "rmse": round(float(np.sqrt(mean_squared_error(y_ste, sp_te))), 0),
    "mae":  round(float(mean_absolute_error(y_ste, sp_te)), 0),
    "mape": round(float(np.mean(np.abs((y_ste - sp_te) / np.maximum(y_ste, 1))) * 100), 2),
}
print(f"       R2={sev_m['r2']}  RMSE=Rs.{sev_m['rmse']:,.0f}  "
      f"MAE=Rs.{sev_m['mae']:,.0f}  MAPE={sev_m['mape']}%")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS SUMMARY — After Severity Regeneration")
print("=" * 65)
print(f"\n  FREQUENCY MODEL (test set, n={len(y_te):,}):")
print(f"    AUC-ROC   : {freq_m['auc']}")
print(f"    F1-Score  : {freq_m['f1']}")
print(f"    Precision : {freq_m['precision']}")
print(f"    Recall    : {freq_m['recall']}")
print(f"    Brier     : {freq_m['brier']}")
print(f"\n  SEVERITY MODEL (test set, n={len(y_ste):,} claimants):")
print(f"    R-squared : {sev_m['r2']}")
print(f"    RMSE      : Rs.{sev_m['rmse']:,.0f}")
print(f"    MAE       : Rs.{sev_m['mae']:,.0f}")
print(f"    MAPE      : {sev_m['mape']}%")

print("\n  BEFORE (pre-regeneration) SEVERITY METRICS:")
print("    R-squared : -0.0024  (Task 3 regularized; theoretical max was 0.005)")
print("    Root cause: target was pure lognormal draws, uncorrelated with features")

print("\n  IMPROVEMENT:")
print(f"    Severity R2: -0.0024 -> {sev_m['r2']}  (from noise floor to predictable)")
print("=" * 65)
