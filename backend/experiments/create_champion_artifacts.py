"""
Create Champion Model Artifacts
================================
One-time setup script: builds the champion model files that the registry
points to, then copies V1 production models to the challengers/ slot.

Does NOT run any hyperparameter search (no new experiments).

Frequency champion:
  Load v4_freq_model.json (XGBoost, already saved from train_v4_experiment.py)
  Wrap in FrequencyModel → save as models/champion/frequency_best_model.pkl

Severity champion:
  Re-fit CatBoostRegressor with the exact Optuna-found hyperparameters from
  challenger_training_report.txt (random_seed=42 → deterministic, identical
  to the in-memory model that was evaluated during champion_challenger_v4.py)
  → save as models/champion/severity_best_model.cbm  (CatBoost native)
  → save as models/champion/severity_best_model.pkl   (CatBoostSeverityWrapper)

V4 Preprocessor:
  Fit InsurancePreprocessorV4 on the full V4 dataset (40k train rows) and
  save as models/champion/preprocessor_v4.pkl so the registry can load it
  alongside the V4 champion models.

Challengers slot:
  Copy existing V1 pkl files to models/challengers/ (non-destructive).
"""

import os, sys, warnings, shutil
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CHAMPION_DIR = os.path.join(BACKEND_DIR, "models", "champion")
CHALL_DIR    = os.path.join(BACKEND_DIR, "models", "challengers")
SAVED_DIR    = os.path.join(BACKEND_DIR, "models", "saved")
EXP_MODEL    = os.path.join(SCRIPT_DIR, "outputs", "models")
V4_CSV       = os.path.join(SCRIPT_DIR, "outputs", "data", "car_insurance_portfolio_v4.csv")

os.makedirs(CHAMPION_DIR, exist_ok=True)
os.makedirs(CHALL_DIR,    exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import joblib

# InsurancePreprocessorV4 lives in src/preprocessing/pipeline_v4.py so that
# joblib pickles can be unpickled from any context (not __main__).
sys.path.insert(0, BACKEND_DIR)
from src.preprocessing.pipeline_v4 import InsurancePreprocessorV4


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load V4 dataset and fit preprocessor
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Champion Artifact Creation")
print("=" * 60)

print("\n[1/6] Loading V4 dataset ...")
df = pd.read_csv(V4_CSV)
print(f"  Loaded: {len(df):,} rows")

DROP_COLS = [
    "customer_id", "vehicle_model",
    "claim_probability_true", "claim_occurred",
    "actual_claim_amount_inr", "region_risk_category",
]
X_raw  = df[[c for c in df.columns if c not in DROP_COLS]].copy()
y_freq = df["claim_occurred"].values

print("\n[2/6] Fitting InsurancePreprocessorV4 ...")
prep_v4 = InsurancePreprocessorV4()
X_all   = prep_v4.fit_transform(X_raw)
print(f"  Feature matrix: {X_all.shape}")

prep_v4_path = os.path.join(CHAMPION_DIR, "preprocessor_v4.pkl")
joblib.dump(prep_v4, prep_v4_path)
print(f"  Saved: {prep_v4_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train/test split (identical to train_v4_experiment.py)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
    X_all, y_freq, np.arange(len(df)),
    test_size=0.20, stratify=y_freq, random_state=RANDOM_SEED,
)
claim_tr = y_train == 1
claim_te = y_test  == 1

X_sev_tr = X_train[claim_tr]
X_sev_te = X_test[claim_te]
y_sev_tr = df.iloc[idx_tr[claim_tr]]["actual_claim_amount_inr"].values
y_sev_te = df.iloc[idx_te[claim_te]]["actual_claim_amount_inr"].values

print(f"\n  Train: {len(X_train):,}  Test: {len(X_test):,}")
print(f"  Severity train claimants: {len(y_sev_tr):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Frequency champion: wrap v4_freq_model.json → pkl
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Wrapping V4 XGBoost frequency model ...")
import xgboost as xgb

from src.models.frequency_model import FrequencyModel

xgb_src = os.path.join(EXP_MODEL, "v4_freq_model.json")
if not os.path.exists(xgb_src):
    print(f"  ERROR: {xgb_src} not found. Run train_v4_experiment.py first.")
    sys.exit(1)

raw_xgb = xgb.XGBClassifier()
raw_xgb.load_model(xgb_src)

freq_wrapper = FrequencyModel(version="v4.0.0", model_dir=CHAMPION_DIR)
freq_wrapper.model   = raw_xgb
freq_wrapper._fitted = True

freq_dest = os.path.join(CHAMPION_DIR, "frequency_best_model.pkl")
joblib.dump(freq_wrapper, freq_dest)
print(f"  Saved: {freq_dest}")

# Quick sanity check
proba_sample = freq_wrapper.predict_proba(X_test[:5])
print(f"  Sample probabilities: {np.round(proba_sample, 4)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Severity champion: deterministic CatBoost re-fit → cbm + pkl
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Fitting CatBoost severity champion (deterministic, seed=42) ...")

try:
    import catboost as cb
except ImportError:
    print("  ERROR: catboost not installed. Run: pip install catboost")
    sys.exit(1)

# Exact Optuna-found params from challenger_training_report.txt
CATBOOST_PARAMS = {
    "iterations":          300,
    "depth":               4,
    "learning_rate":       0.04525761220189302,
    "l2_leaf_reg":         3.4331074992907338,
    "bagging_temperature": 0.8414531026035467,
    "random_strength":     0.09853817303519963,
    "border_count":        128,
    "random_seed":         RANDOM_SEED,
    "verbose":             0,
}
cat_model = cb.CatBoostRegressor(**CATBOOST_PARAMS)
cat_model.fit(X_sev_tr, y_sev_tr)
print("  CatBoost fitted.")

# Verify R² matches experiment report
from sklearn.metrics import r2_score
y_pred_te = cat_model.predict(X_sev_te)
r2 = r2_score(y_sev_te, y_pred_te)
print(f"  Verified R² on test set: {r2:.4f}  (expected 0.5378)")

# Save .cbm (CatBoost native)
cbm_dest = os.path.join(CHAMPION_DIR, "severity_best_model.cbm")
cat_model.save_model(cbm_dest)
print(f"  Saved: {cbm_dest}")

# Save .pkl (CatBoostSeverityWrapper for registry loading)
from src.models.severity_model import CatBoostSeverityWrapper
sev_wrapper = CatBoostSeverityWrapper(version="v4.0.0")
sev_wrapper.model   = cat_model
sev_wrapper._fitted = True

pkl_dest = os.path.join(CHAMPION_DIR, "severity_best_model.pkl")
joblib.dump(sev_wrapper, pkl_dest)
print(f"  Saved: {pkl_dest}")

# Predict sanity check
pred_sample = sev_wrapper.predict(X_sev_te[:3])
print(f"  Sample predictions (INR): {np.round(pred_sample, 0)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Copy V1 models to challengers/ slot
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Copying V1 models to challengers/ ...")

v1_files = {
    "frequency_v1.0.0.pkl": "xgb_frequency.pkl",
    "severity_v1.0.0.pkl":  "xgb_severity.pkl",
    "preprocessor.pkl":     "preprocessor_v1.pkl",
}
for src_name, dst_name in v1_files.items():
    src = os.path.join(SAVED_DIR, src_name)
    dst = os.path.join(CHALL_DIR, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {src_name} -> challengers/{dst_name}")
    else:
        print(f"  SKIP (not found): {src_name}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Validate champion directory
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Validating champion artifacts ...")
expected_files = [
    "frequency_best_model.pkl",
    "severity_best_model.cbm",
    "severity_best_model.pkl",
    "preprocessor_v4.pkl",
]
all_ok = True
for fname in expected_files:
    path = os.path.join(CHAMPION_DIR, fname)
    size = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    ok   = os.path.exists(path)
    all_ok = all_ok and ok
    status = "OK" if ok else "MISSING"
    print(f"  [{status}] {fname}  ({size} KB)")

print("\n  Challengers slot:")
for fname in os.listdir(CHALL_DIR):
    size = os.path.getsize(os.path.join(CHALL_DIR, fname)) // 1024
    print(f"    {fname}  ({size} KB)")

print("\n" + "=" * 60)
if all_ok:
    print("Champion artifacts created successfully.")
    print("Registry path: models/metadata/champion_registry.json")
    print("\nNext steps:")
    print("  1. When the production API is upgraded to V4 features,")
    print("     call: ModelRegistry().activate_champion('frequency')")
    print("           ModelRegistry().activate_champion('severity')")
    print("  2. Restart the API server — registry-loaded models will be used.")
    print("  3. To rollback: ModelRegistry().rollback_champion('frequency')")
else:
    print("Some artifacts are MISSING. Check errors above.")
print("=" * 60)
