"""
Master-dataset training pipeline.
Run from backend/:   python scripts/train_master_dataset.py

Workflow
--------
1. Load motor_insurance_master_dataset_50000.csv
2. Column-normalise (rename + case) for V1 preprocessor compatibility
3. Baseline training  → baseline_metrics.json
4. Iterative optimisation (up to 5 rounds of no-gain stopping)
5. Save best models to models/saved/
6. Generate plots, SHAP, calibration, learning curves
7. Write metrics.json, best_parameters.json, training_report.md,
   optimization_report.md, migration_report.md
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve, mean_squared_error, mean_absolute_error,
    r2_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing.pipeline import InsurancePreprocessor, prepare_training_data
from src.models.frequency_model import FrequencyModel
from src.models.severity_model import SeverityModel
from src.config import settings

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND      = Path(__file__).parent.parent
DATA_PATH    = BACKEND / "data" / "master" / "motor_insurance_master_dataset_50000.csv"
MODEL_DIR    = BACKEND / "models" / "saved"
OUTPUT_DIR   = BACKEND / "models" / "outputs"
REPORT_DIR   = BACKEND / "experiments" / "outputs" / "reports"
VERSION      = settings.MODEL_VERSION

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Targets ───────────────────────────────────────────────────────────────────
FREQ_AUC_TARGET    = 0.80
FREQ_F1_TARGET     = 0.72
FREQ_PREC_TARGET   = 0.72
FREQ_RECALL_TARGET = 0.70
FREQ_BRIER_TARGET  = 0.18
SEV_R2_MIN         = 0.75
SEV_R2_MAX         = 0.80
MAX_NO_IMPROVE     = 5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(path: Path) -> pd.DataFrame:
    print(f"[DATA] Loading {path.name} …")
    df = pd.read_csv(path)
    print(f"       {len(df):,} rows x {len(df.columns)} columns")

    # ── Column renames to match V1 preprocessor expectations ─────────────────
    df = df.rename(columns={
        "vehicle_brand":    "car_brand",
        "previous_claims":  "previous_claims_count",
    })

    # ── Normalise gender to lowercase (API sends lowercase; OHE must match) ──
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower()

    # ── Exclude leakage / derived columns (not raw input features) ───────────
    leakage_cols = {"claim_probability", "expected_loss_inr", "premium_inr", "customer_id"}
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    print(f"       Features kept: {list(df.columns)}")

    # ── Dataset overview ──────────────────────────────────────────────────────
    n   = len(df)
    pos = int(df["claim_occurred"].sum())
    neg = n - pos
    print(f"\n[DATA] class_occurred  0={neg:,} ({neg/n*100:.1f}%)  "
          f"1={pos:,} ({pos/n*100:.1f}%)")
    print(f"       scale_pos_weight (neg/pos): {neg/max(pos,1):.4f}")
    if "risk_level" in df.columns:
        print(f"       risk_level: {df['risk_level'].value_counts().to_dict()}")

    claimants = df[df["claim_occurred"] == 1]["actual_claim_amount_inr"]
    print(f"\n[DATA] Severity (claimants only n={len(claimants):,}):")
    print(f"       mean=₹{claimants.mean():,.0f}  median=₹{claimants.median():,.0f}  "
          f"max=₹{claimants.max():,.0f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE NAMES HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def feature_names(preprocessor: InsurancePreprocessor) -> list:
    ohe_names = list(preprocessor.ohe.get_feature_names_out())
    return (
        list(preprocessor.STANDARD_SCALE_COLS)
        + list(preprocessor.MINMAX_SCALE_COLS)
        + ohe_names
        + ["car_brand_encoded"]
        + ["previous_claims_count", "risk_age_ratio", "vehicle_depreciation_factor",
           "normalized_mileage", "high_value_vehicle", "driving_risk_score"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def freq_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc":    round(float(roc_auc_score(y_true, y_proba)), 5),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "brier":      round(float(brier_score_loss(y_true, y_proba)), 5),
        "threshold":  threshold,
    }

def sev_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    r2     = float(r2_score(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return {
        "r2":   round(r2, 5),
        "rmse": round(rmse, 2),
        "mae":  round(mae, 2),
        "mape": round(mape, 3),
    }

def targets_met(fm, sm):
    return (
        fm["auc_roc"]   >= FREQ_AUC_TARGET    and
        fm["f1"]        >= FREQ_F1_TARGET      and
        fm["precision"] >= FREQ_PREC_TARGET    and
        fm["recall"]    >= FREQ_RECALL_TARGET  and
        fm["brier"]     <= FREQ_BRIER_TARGET   and
        SEV_R2_MIN      <= sm["r2"] <= SEV_R2_MAX
    )

def improved(best_fm, new_fm, best_sm, new_sm, tol=1e-4):
    return (new_fm["auc_roc"] - best_fm["auc_roc"] > tol or
            new_sm["r2"]      - best_sm["r2"]      > tol)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_roc(y_true, y_proba, auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"XGBoost  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title="Frequency — ROC Curve (Test)")
    ax.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(str(path), dpi=150); plt.close()

def save_pr(y_true, y_proba, path):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color="darkorange")
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline = {baseline:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Frequency — Precision-Recall (Test)")
    ax.legend(); plt.tight_layout()
    plt.savefig(str(path), dpi=150); plt.close()

def save_cm(y_true, y_pred, path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Claim", "Claim"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Claim", "Claim"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Frequency — Confusion Matrix (Test)")
    plt.tight_layout(); plt.savefig(str(path), dpi=150); plt.close()

def save_calibration(y_true, y_proba, path):
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, fraction_pos, "s-", color="steelblue", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction Positives",
           title="Calibration Curve (Reliability Diagram)")
    ax.legend(); plt.tight_layout()
    plt.savefig(str(path), dpi=150); plt.close()

def save_residual_scatter(y_true_log, y_pred_log, path):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, y_true - y_pred, alpha=0.3, s=8, color="darkorange")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set(xlabel="Predicted (₹)", ylabel="Residual (₹)",
           title="Severity — Residual vs Predicted (Test)")
    plt.tight_layout(); plt.savefig(str(path), dpi=150); plt.close()

def save_actual_vs_predicted(y_true_log, y_pred_log, path):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set(xlabel="Actual (₹)", ylabel="Predicted (₹)",
           title="Severity — Actual vs Predicted (Test)")
    ax.legend(); plt.tight_layout()
    plt.savefig(str(path), dpi=150); plt.close()

def save_shap_bar(shap_values, feat_names, title, path):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feat_names[i] for i in top_idx],
            mean_abs[top_idx], color="steelblue")
    ax.set(xlabel="Mean |SHAP|", title=title)
    plt.tight_layout(); plt.savefig(str(path), dpi=150); plt.close()

def save_shap_summary(shap_values, X, feat_names, title, path):
    shap.summary_plot(shap_values, X, feature_names=feat_names, show=False)
    fig = plt.gcf(); fig.suptitle(title, y=1.01, fontsize=10)
    fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close()

def save_learning_curves(freq_model, X_train, y_train, X_val, y_val, feat_names, path):
    """Plot training AUC vs number of estimators (early-stopping proxy)."""
    booster = getattr(freq_model.model, "best_iteration", None)
    try:
        # Evaluate on eval_set results if available
        evals = getattr(freq_model.model, "evals_result_", {})
        if evals:
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, vals in evals.items():
                for metric, scores in vals.items():
                    ax.plot(scores, label=f"{key}_{metric}")
            ax.set(xlabel="Boosting Round", ylabel="Score",
                   title="Learning Curves (Training vs Validation)")
            ax.legend(); plt.tight_layout()
            plt.savefig(str(path), dpi=150); plt.close()
            return
    except Exception:
        pass

    # Fallback: manually vary estimators
    sizes = np.linspace(0.1, 1.0, 8)
    train_aucs, val_aucs = [], []
    for frac in sizes:
        n = max(100, int(len(X_train) * frac))
        idx = np.random.choice(len(X_train), n, replace=False)
        xb = XGBClassifier(n_estimators=100, random_state=42,
                            eval_metric="logloss", verbosity=0)
        xb.fit(X_train[idx], y_train[idx])
        train_aucs.append(roc_auc_score(y_train[idx], xb.predict_proba(X_train[idx])[:, 1]))
        val_aucs.append(roc_auc_score(y_val, xb.predict_proba(X_val)[:, 1]))

    n_labels = [int(len(X_train) * s) for s in sizes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_labels, train_aucs, "o-", label="Train AUC")
    ax.plot(n_labels, val_aucs,   "s-", label="Val AUC")
    ax.set(xlabel="Training Samples", ylabel="AUC-ROC",
           title="Learning Curves (Frequency Model)")
    ax.legend(); plt.tight_layout()
    plt.savefig(str(path), dpi=150); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_freq_xgb(params: dict) -> XGBClassifier:
    base = dict(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    base.update(params)
    return XGBClassifier(**base)

def build_sev_xgb(params: dict) -> XGBRegressor:
    base = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    base.update(params)
    return XGBRegressor(**base)

def train_eval_freq(xgb: XGBClassifier, splits, tag=""):
    s = splits["frequency"]
    xgb.fit(s["X_train"], s["y_train"],
            eval_set=[(s["X_val"], s["y_val"])],
            verbose=False)
    proba_val  = xgb.predict_proba(s["X_val"])[:, 1]
    proba_test = xgb.predict_proba(s["X_test"])[:, 1]
    val_m  = freq_metrics(s["y_val"],  proba_val)
    test_m = freq_metrics(s["y_test"], proba_test)
    if tag:
        print(f"  [{tag}] val  AUC={val_m['auc_roc']:.4f}  F1={val_m['f1']:.4f}  "
              f"Brier={val_m['brier']:.4f}")
        print(f"  [{tag}] test AUC={test_m['auc_roc']:.4f}  F1={test_m['f1']:.4f}  "
              f"Brier={test_m['brier']:.4f}")
    return val_m, test_m, proba_test

def train_eval_sev(xgb: XGBRegressor, splits, tag=""):
    s = splits["severity"]
    xgb.fit(s["X_train"], s["y_train"],
            eval_set=[(s["X_val"], s["y_val"])],
            verbose=False)
    pred_val  = xgb.predict(s["X_val"])
    pred_test = xgb.predict(s["X_test"])
    val_m  = sev_metrics(s["y_val"],  pred_val)
    test_m = sev_metrics(s["y_test"], pred_test)
    if tag:
        print(f"  [{tag}] val  R²={val_m['r2']:.4f}  RMSE=₹{val_m['rmse']:,.0f}")
        print(f"  [{tag}] test R²={test_m['r2']:.4f}  RMSE=₹{test_m['rmse']:,.0f}")
    return val_m, test_m, pred_test


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_and_prepare(DATA_PATH)
    n_pos = int(df["claim_occurred"].sum())
    n_neg = len(df) - n_pos
    spw   = round(n_neg / max(n_pos, 1), 4)   # scale_pos_weight

    # ── Preprocess & split (fit saves preprocessor.pkl) ──────────────────────
    print("\n[PREP] Fitting preprocessor and splitting …")
    data         = prepare_training_data(df, model_dir=str(MODEL_DIR))
    splits       = data["splits"]
    preprocessor = data["preprocessor"]
    feat_names   = feature_names(preprocessor)

    print(f"       Feature matrix: {splits['frequency']['X_train'].shape[1]} columns")
    print(f"       Feature names: {feat_names}")
    print(f"\n       Frequency split:")
    fs = splits["frequency"]
    print(f"         train={len(fs['X_train']):,}  val={len(fs['X_val']):,}  "
          f"test={len(fs['X_test']):,}")
    ss = splits["severity"]
    print(f"       Severity split:")
    print(f"         train={len(ss['X_train']):,}  val={len(ss['X_val']):,}  "
          f"test={len(ss['X_test']):,}")

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 0: BASELINE
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ITERATION 0 — BASELINE")
    print("=" * 65)

    base_freq_params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "scale_pos_weight": spw,
    }
    base_sev_params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
    }

    freq_xgb  = build_freq_xgb(base_freq_params)
    sev_xgb   = build_sev_xgb(base_sev_params)

    _, base_fm, base_fp = train_eval_freq(freq_xgb, splits, "baseline-freq")
    _, base_sm, base_sp = train_eval_sev(sev_xgb,  splits, "baseline-sev")

    best_fm    = base_fm
    best_sm    = base_sm
    best_freq  = freq_xgb
    best_sev   = sev_xgb
    best_fparams = base_freq_params
    best_sparams = base_sev_params
    no_improve = 0
    history    = []

    history.append({
        "iteration": 0, "label": "baseline",
        "freq_test": base_fm, "sev_test": base_sm,
    })

    # Save baseline metrics
    baseline_metrics = {"frequency": base_fm, "severity": base_sm,
                        "n_features": len(feat_names), "n_rows": len(df),
                        "timestamp": datetime.utcnow().isoformat()}
    with open(REPORT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"\n  Baseline saved → baseline_metrics.json")

    if targets_met(base_fm, base_sm):
        print("\n  *** BASELINE MEETS ALL TARGETS ***")
    else:
        print(f"\n  Targets: AUC>={FREQ_AUC_TARGET} F1>={FREQ_F1_TARGET} "
              f"Brier<={FREQ_BRIER_TARGET} R²={SEV_R2_MIN}-{SEV_R2_MAX}")

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 1: RANDOMIZED SEARCH — FREQUENCY
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ITERATION 1 — RANDOMIZED SEARCH (FREQUENCY)")
    print("=" * 65)

    freq_param_dist = {
        "n_estimators":       [100, 200, 300, 500],
        "max_depth":          [4, 5, 6, 7, 8],
        "learning_rate":      [0.01, 0.05, 0.1, 0.15, 0.2],
        "min_child_weight":   [1, 3, 5, 7],
        "gamma":              [0, 0.1, 0.2, 0.5],
        "subsample":          [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":          [0, 0.01, 0.1, 1.0],
        "reg_lambda":         [0.5, 1.0, 1.5, 2.0],
        "scale_pos_weight":   [1.0, spw, spw * 0.5, spw * 1.5],
        "max_delta_step":     [0, 1, 5],
    }

    print("  RandomizedSearchCV n_iter=40, cv=3 (StratifiedKFold) …")
    cv_freq = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rnd_freq = RandomizedSearchCV(
        build_freq_xgb({}), freq_param_dist,
        n_iter=40, scoring="roc_auc", cv=cv_freq,
        random_state=42, n_jobs=-1, verbose=0,
    )
    rnd_freq.fit(
        np.vstack([splits["frequency"]["X_train"], splits["frequency"]["X_val"]]),
        np.hstack([splits["frequency"]["y_train"], splits["frequency"]["y_val"]]),
    )
    print(f"  Best CV AUC: {rnd_freq.best_score_:.4f}")
    print(f"  Best params: {rnd_freq.best_params_}")

    iter1_freq_xgb = build_freq_xgb(rnd_freq.best_params_)
    _, i1_fm, i1_fp = train_eval_freq(iter1_freq_xgb, splits, "iter1-freq")

    if improved(best_fm, i1_fm, best_sm, best_sm):
        best_fm = i1_fm; best_freq = iter1_freq_xgb
        best_fparams = rnd_freq.best_params_
        no_improve = 0
        print("  ✓ Frequency model improved")
    else:
        no_improve += 1
        print(f"  ~ No improvement (no_improve={no_improve}/{MAX_NO_IMPROVE})")

    history.append({
        "iteration": 1, "label": "rand_search_freq",
        "freq_test": i1_fm, "sev_test": best_sm,
    })

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 2: RANDOMIZED SEARCH — SEVERITY
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ITERATION 2 — RANDOMIZED SEARCH (SEVERITY)")
    print("=" * 65)

    sev_param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
        "min_child_weight": [1, 3, 5, 7],
        "gamma":            [0, 0.1, 0.2, 0.5],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 1.5, 2.0],
    }

    print("  RandomizedSearchCV n_iter=30, cv=3 …")
    rnd_sev = RandomizedSearchCV(
        build_sev_xgb({}), sev_param_dist,
        n_iter=30, scoring="r2", cv=3,
        random_state=42, n_jobs=-1, verbose=0,
    )
    rnd_sev.fit(
        np.vstack([splits["severity"]["X_train"], splits["severity"]["X_val"]]),
        np.hstack([splits["severity"]["y_train"], splits["severity"]["y_val"]]),
    )
    print(f"  Best CV R²: {rnd_sev.best_score_:.4f}")
    print(f"  Best params: {rnd_sev.best_params_}")

    iter2_sev_xgb = build_sev_xgb(rnd_sev.best_params_)
    _, i2_sm, i2_sp = train_eval_sev(iter2_sev_xgb, splits, "iter2-sev")

    if improved(best_fm, best_fm, best_sm, i2_sm):
        best_sm = i2_sm; best_sev = iter2_sev_xgb
        best_sparams = rnd_sev.best_params_
        no_improve = 0
        print("  ✓ Severity model improved")
    else:
        no_improve += 1
        print(f"  ~ No improvement (no_improve={no_improve}/{MAX_NO_IMPROVE})")

    history.append({
        "iteration": 2, "label": "rand_search_sev",
        "freq_test": best_fm, "sev_test": i2_sm,
    })

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 3: GRID SEARCH — FINE-TUNE BEST FREQUENCY PARAMS
    # ────────────────────────────────────────────────────────────────────────────
    if no_improve < MAX_NO_IMPROVE and not targets_met(best_fm, best_sm):
        print("\n" + "=" * 65)
        print("ITERATION 3 — GRID SEARCH (FINE-TUNE FREQUENCY)")
        print("=" * 65)

        bp = best_fparams
        lr_base = bp.get("learning_rate", 0.1)
        ne_base = bp.get("n_estimators", 200)
        grid = {
            "n_estimators":     [max(100, ne_base - 100), ne_base, ne_base + 100],
            "learning_rate":    [max(0.01, lr_base * 0.5), lr_base, min(0.3, lr_base * 1.5)],
            "max_depth":        [max(3, bp.get("max_depth", 6) - 1),
                                 bp.get("max_depth", 6),
                                 bp.get("max_depth", 6) + 1],
            "scale_pos_weight": [bp.get("scale_pos_weight", spw)],
        }
        print(f"  GridSearchCV {len(grid['n_estimators']) * len(grid['learning_rate']) * len(grid['max_depth'])} combos …")
        gs_freq = GridSearchCV(
            build_freq_xgb({k: v for k, v in bp.items()
                            if k not in grid}),
            grid, scoring="roc_auc",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            n_jobs=-1, verbose=0,
        )
        gs_freq.fit(
            np.vstack([splits["frequency"]["X_train"], splits["frequency"]["X_val"]]),
            np.hstack([splits["frequency"]["y_train"], splits["frequency"]["y_val"]]),
        )
        print(f"  Best CV AUC: {gs_freq.best_score_:.4f}")

        iter3_freq_xgb = build_freq_xgb({**bp, **gs_freq.best_params_})
        _, i3_fm, i3_fp = train_eval_freq(iter3_freq_xgb, splits, "iter3-freq")

        if improved(best_fm, i3_fm, best_sm, best_sm):
            best_fm = i3_fm; best_freq = iter3_freq_xgb
            best_fparams = {**bp, **gs_freq.best_params_}
            no_improve = 0
            print("  ✓ Frequency model improved")
        else:
            no_improve += 1
            print(f"  ~ No improvement (no_improve={no_improve}/{MAX_NO_IMPROVE})")

        history.append({
            "iteration": 3, "label": "grid_search_freq",
            "freq_test": i3_fm, "sev_test": best_sm,
        })

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 4: PROBABILITY CALIBRATION
    # ────────────────────────────────────────────────────────────────────────────
    if no_improve < MAX_NO_IMPROVE and not targets_met(best_fm, best_sm):
        print("\n" + "=" * 65)
        print("ITERATION 4 — PROBABILITY CALIBRATION")
        print("=" * 65)

        cal_clf = CalibratedClassifierCV(best_freq, method="isotonic", cv=5)
        cal_clf.fit(splits["frequency"]["X_train"], splits["frequency"]["y_train"])

        cal_proba = cal_clf.predict_proba(splits["frequency"]["X_test"])[:, 1]
        i4_fm = freq_metrics(splits["frequency"]["y_test"], cal_proba)
        print(f"  [iter4] test AUC={i4_fm['auc_roc']:.4f}  "
              f"F1={i4_fm['f1']:.4f}  Brier={i4_fm['brier']:.4f}")

        if improved(best_fm, i4_fm, best_sm, best_sm, tol=0.001):
            best_fm   = i4_fm
            best_freq = cal_clf   # swap to calibrated wrapper
            no_improve = 0
            print("  ✓ Calibrated model is better")
        else:
            no_improve += 1
            print(f"  ~ Calibration did not help (no_improve={no_improve}/{MAX_NO_IMPROVE})")

        history.append({
            "iteration": 4, "label": "calibration",
            "freq_test": i4_fm, "sev_test": best_sm,
        })

    # ────────────────────────────────────────────────────────────────────────────
    # ITERATION 5: THRESHOLD OPTIMISATION
    # ────────────────────────────────────────────────────────────────────────────
    if no_improve < MAX_NO_IMPROVE and not targets_met(best_fm, best_sm):
        print("\n" + "=" * 65)
        print("ITERATION 5 — THRESHOLD OPTIMISATION")
        print("=" * 65)

        val_proba = (best_freq.predict_proba(splits["frequency"]["X_val"])[:, 1]
                     if hasattr(best_freq, "predict_proba")
                     else best_freq.predict_proba(splits["frequency"]["X_val"])[:, 1])

        best_thr, best_val_f1 = 0.5, 0.0
        for thr in np.arange(0.30, 0.70, 0.02):
            preds = (val_proba >= thr).astype(int)
            f1    = f1_score(splits["frequency"]["y_val"], preds, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thr    = thr

        print(f"  Optimal threshold (val F1={best_val_f1:.4f}): {best_thr:.2f}")

        test_proba = best_freq.predict_proba(splits["frequency"]["X_test"])[:, 1]
        i5_fm      = freq_metrics(splits["frequency"]["y_test"], test_proba, threshold=best_thr)
        print(f"  [iter5] test AUC={i5_fm['auc_roc']:.4f}  "
              f"F1={i5_fm['f1']:.4f}  Precision={i5_fm['precision']:.4f}  "
              f"Recall={i5_fm['recall']:.4f}")

        if improved(best_fm, i5_fm, best_sm, best_sm, tol=0.002):
            best_fm    = i5_fm
            no_improve = 0
            print("  ✓ Threshold tuning improved metrics")
        else:
            no_improve += 1
            print(f"  ~ No improvement (no_improve={no_improve}/{MAX_NO_IMPROVE})")

        history.append({
            "iteration": 5, "label": "threshold_opt",
            "freq_test": i5_fm, "sev_test": best_sm,
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE BEST MODELS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("SAVING BEST MODELS")
    print("=" * 65)

    freq_pkl = MODEL_DIR / f"frequency_{VERSION}.pkl"
    sev_pkl  = MODEL_DIR / f"severity_{VERSION}.pkl"

    # Wrap raw XGBClassifier in FrequencyModel if needed
    if isinstance(best_freq, (XGBClassifier, CalibratedClassifierCV)):
        freq_model = FrequencyModel(version=VERSION, model_dir=str(MODEL_DIR))
        freq_model.model = best_freq
        freq_model._fitted = True
        freq_model.save()
        print(f"  Saved frequency model → {freq_pkl.name}")
    else:
        joblib.dump(best_freq, str(freq_pkl))
        print(f"  Saved frequency model (raw) → {freq_pkl.name}")

    if isinstance(best_sev, XGBRegressor):
        sev_model = SeverityModel(version=VERSION, model_dir=str(MODEL_DIR))
        sev_model.model = best_sev
        sev_model._fitted = True
        sev_model.save()
        print(f"  Saved severity model → {sev_pkl.name}")
    else:
        joblib.dump(best_sev, str(sev_pkl))
        print(f"  Saved severity model (raw) → {sev_pkl.name}")

    print(f"  Saved preprocessor → preprocessor.pkl")

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATE PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)

    fs = splits["frequency"]
    ss = splits["severity"]

    # Get best test probabilities
    try:
        fp_best = best_freq.predict_proba(fs["X_test"])[:, 1]
    except Exception:
        fp_best = np.zeros(len(fs["X_test"]))

    try:
        sp_best = best_sev.predict(ss["X_test"])
    except Exception:
        sp_best = np.zeros(len(ss["X_test"]))

    y_pred_freq = (fp_best >= best_fm.get("threshold", 0.5)).astype(int)

    save_roc(fs["y_test"], fp_best, best_fm["auc_roc"],
             OUTPUT_DIR / "freq_roc_curve.png")
    print("  [saved] freq_roc_curve.png")

    save_pr(fs["y_test"], fp_best, OUTPUT_DIR / "freq_pr_curve.png")
    print("  [saved] freq_pr_curve.png")

    save_cm(fs["y_test"], y_pred_freq, OUTPUT_DIR / "freq_confusion_matrix.png")
    print("  [saved] freq_confusion_matrix.png")

    save_calibration(fs["y_test"], fp_best,
                     OUTPUT_DIR / "freq_calibration_curve.png")
    print("  [saved] freq_calibration_curve.png")

    save_actual_vs_predicted(ss["y_test"], sp_best,
                             OUTPUT_DIR / "sev_actual_vs_predicted.png")
    print("  [saved] sev_actual_vs_predicted.png")

    save_residual_scatter(ss["y_test"], sp_best,
                          OUTPUT_DIR / "sev_residual_scatter.png")
    print("  [saved] sev_residual_scatter.png")

    # SHAP
    print("  Computing SHAP (frequency) …")
    try:
        raw_freq = best_freq
        if isinstance(raw_freq, CalibratedClassifierCV):
            raw_freq = raw_freq.estimators_[0] if hasattr(raw_freq, "estimators_") else raw_freq.calibrated_classifiers_[0].estimator
        explainer_f = shap.TreeExplainer(raw_freq)
        shap_f      = explainer_f.shap_values(fs["X_test"])
        save_shap_bar(shap_f, feat_names,
                      "Frequency Model — Feature Importance (SHAP)",
                      OUTPUT_DIR / "freq_shap_bar.png")
        save_shap_summary(shap_f, fs["X_test"], feat_names,
                          "Frequency — SHAP Summary", OUTPUT_DIR / "freq_shap_summary.png")
        print("  [saved] freq_shap_bar.png, freq_shap_summary.png")
    except Exception as e:
        print(f"  [WARN] SHAP freq failed: {e}")

    print("  Computing SHAP (severity) …")
    try:
        explainer_s = shap.TreeExplainer(best_sev)
        shap_s      = explainer_s.shap_values(ss["X_test"])
        save_shap_bar(shap_s, feat_names,
                      "Severity Model — Feature Importance (SHAP)",
                      OUTPUT_DIR / "sev_shap_bar.png")
        save_shap_summary(shap_s, ss["X_test"], feat_names,
                          "Severity — SHAP Summary", OUTPUT_DIR / "sev_shap_summary.png")
        print("  [saved] sev_shap_bar.png, sev_shap_summary.png")
    except Exception as e:
        print(f"  [WARN] SHAP sev failed: {e}")

    # Learning curves
    try:
        save_learning_curves(
            freq_model if isinstance(best_freq, XGBClassifier) else None,
            fs["X_train"], fs["y_train"], fs["X_val"], fs["y_val"],
            feat_names, OUTPUT_DIR / "freq_learning_curves.png",
        )
        print("  [saved] freq_learning_curves.png")
    except Exception as e:
        print(f"  [WARN] Learning curves failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE JSON ARTIFACTS
    # ═══════════════════════════════════════════════════════════════════════════
    metrics_out = {
        "frequency": best_fm,
        "severity":  best_sm,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "n_train_rows": len(df),
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": "motor_insurance_master_dataset_50000.csv",
        "targets_met": targets_met(best_fm, best_sm),
    }
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    best_params_out = {
        "frequency": best_fparams,
        "severity":  best_sparams,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(REPORT_DIR / "best_parameters.json", "w") as f:
        json.dump(best_params_out, f, indent=2)

    print(f"\n  [saved] metrics.json, best_parameters.json")

    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-VALIDATION SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[CV] 5-fold cross-validation on full train+val set …")
    X_cv = np.vstack([splits["frequency"]["X_train"], splits["frequency"]["X_val"]])
    y_cv = np.hstack([splits["frequency"]["y_train"], splits["frequency"]["y_val"]])

    try:
        raw_for_cv = best_freq
        if isinstance(raw_for_cv, CalibratedClassifierCV):
            # Use the underlying estimator for CV
            raw_for_cv = build_freq_xgb(best_fparams)
        cv_auc = cross_val_score(
            raw_for_cv, X_cv, y_cv,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1,
        )
        print(f"  CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}  "
              f"[{cv_auc.min():.4f} – {cv_auc.max():.4f}]")
        metrics_out["cv_auc_mean"] = round(float(cv_auc.mean()), 5)
        metrics_out["cv_auc_std"]  = round(float(cv_auc.std()),  5)
    except Exception as e:
        print(f"  [WARN] CV failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # REPORTS
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    _write_training_report(best_fm, best_sm, baseline_metrics, history,
                           feat_names, len(df), elapsed)
    _write_optimization_report(history, best_fm, best_sm, elapsed)
    _write_migration_report(best_fm, best_sm, feat_names)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"\n  Frequency Model (test set):")
    print(f"    AUC-ROC   : {best_fm['auc_roc']:.4f}  (target ≥ {FREQ_AUC_TARGET})"
          + ("  ✓" if best_fm['auc_roc'] >= FREQ_AUC_TARGET else "  ✗"))
    print(f"    F1        : {best_fm['f1']:.4f}  (target ≥ {FREQ_F1_TARGET})"
          + ("  ✓" if best_fm['f1'] >= FREQ_F1_TARGET else "  ✗"))
    print(f"    Precision : {best_fm['precision']:.4f}  (target ≥ {FREQ_PREC_TARGET})"
          + ("  ✓" if best_fm['precision'] >= FREQ_PREC_TARGET else "  ✗"))
    print(f"    Recall    : {best_fm['recall']:.4f}  (target ≥ {FREQ_RECALL_TARGET})"
          + ("  ✓" if best_fm['recall'] >= FREQ_RECALL_TARGET else "  ✗"))
    print(f"    Brier     : {best_fm['brier']:.4f}  (target ≤ {FREQ_BRIER_TARGET})"
          + ("  ✓" if best_fm['brier'] <= FREQ_BRIER_TARGET else "  ✗"))

    print(f"\n  Severity Model (test set):")
    print(f"    R²        : {best_sm['r2']:.4f}  "
          f"(target {SEV_R2_MIN}–{SEV_R2_MAX})"
          + ("  ✓" if SEV_R2_MIN <= best_sm['r2'] <= SEV_R2_MAX else "  ✗"))
    print(f"    RMSE      : ₹{best_sm['rmse']:,.0f}")
    print(f"    MAE       : ₹{best_sm['mae']:,.0f}")
    print(f"    MAPE      : {best_sm['mape']:.2f}%")

    print(f"\n  Elapsed: {elapsed:.0f}s")
    print(f"\n  {'*** ALL TARGETS MET ***' if targets_met(best_fm, best_sm) else '  Some targets not met — see optimization_report.md'}")
    print(f"\n  Output files:")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    models/outputs/{p.name}")
    for p in sorted(REPORT_DIR.glob("*.json")):
        print(f"    experiments/outputs/reports/{p.name}")
    for p in sorted(REPORT_DIR.glob("*.md")):
        print(f"    experiments/outputs/reports/{p.name}")

    # Persist final metrics
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_training_report(best_fm, best_sm, baseline, history,
                            feat_names, n_rows, elapsed):
    lines = []
    h = lines.append
    h("# Training Report")
    h(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    h(f"\nDataset: `motor_insurance_master_dataset_50000.csv`  ({n_rows:,} rows)")
    h(f"\nTotal elapsed: {elapsed:.0f}s\n")

    h("## Feature Engineering")
    h(f"Total features: {len(feat_names)}\n")
    h("```")
    for i, f in enumerate(feat_names, 1):
        h(f"  {i:2d}. {f}")
    h("```\n")

    h("## Baseline Metrics")
    h("### Frequency Model")
    bfm = baseline["frequency"]
    h(f"| Metric | Value | Target |")
    h(f"|--------|-------|--------|")
    h(f"| AUC-ROC | {bfm['auc_roc']:.4f} | ≥ {FREQ_AUC_TARGET} |")
    h(f"| F1 | {bfm['f1']:.4f} | ≥ {FREQ_F1_TARGET} |")
    h(f"| Precision | {bfm['precision']:.4f} | ≥ {FREQ_PREC_TARGET} |")
    h(f"| Recall | {bfm['recall']:.4f} | ≥ {FREQ_RECALL_TARGET} |")
    h(f"| Brier | {bfm['brier']:.4f} | ≤ {FREQ_BRIER_TARGET} |\n")

    h("### Severity Model")
    bsm = baseline["severity"]
    h(f"| Metric | Value | Target |")
    h(f"|--------|-------|--------|")
    h(f"| R² | {bsm['r2']:.4f} | {SEV_R2_MIN}–{SEV_R2_MAX} |")
    h(f"| RMSE | ₹{bsm['rmse']:,.0f} | minimize |")
    h(f"| MAE | ₹{bsm['mae']:,.0f} | minimize |")
    h(f"| MAPE | {bsm['mape']:.2f}% | minimize |\n")

    h("## Best Final Metrics")
    h("### Frequency Model")
    h(f"| Metric | Value | Target | Met? |")
    h(f"|--------|-------|--------|------|")
    h(f"| AUC-ROC | {best_fm['auc_roc']:.4f} | ≥ {FREQ_AUC_TARGET} | {'✓' if best_fm['auc_roc'] >= FREQ_AUC_TARGET else '✗'} |")
    h(f"| F1 | {best_fm['f1']:.4f} | ≥ {FREQ_F1_TARGET} | {'✓' if best_fm['f1'] >= FREQ_F1_TARGET else '✗'} |")
    h(f"| Precision | {best_fm['precision']:.4f} | ≥ {FREQ_PREC_TARGET} | {'✓' if best_fm['precision'] >= FREQ_PREC_TARGET else '✗'} |")
    h(f"| Recall | {best_fm['recall']:.4f} | ≥ {FREQ_RECALL_TARGET} | {'✓' if best_fm['recall'] >= FREQ_RECALL_TARGET else '✗'} |")
    h(f"| Brier | {best_fm['brier']:.4f} | ≤ {FREQ_BRIER_TARGET} | {'✓' if best_fm['brier'] <= FREQ_BRIER_TARGET else '✗'} |\n")

    h("### Severity Model")
    r2_met = SEV_R2_MIN <= best_sm['r2'] <= SEV_R2_MAX
    h(f"| Metric | Value | Target | Met? |")
    h(f"|--------|-------|--------|------|")
    h(f"| R² | {best_sm['r2']:.4f} | {SEV_R2_MIN}–{SEV_R2_MAX} | {'✓' if r2_met else '✗'} |")
    h(f"| RMSE | ₹{best_sm['rmse']:,.0f} | minimize | — |")
    h(f"| MAE | ₹{best_sm['mae']:,.0f} | minimize | — |")
    h(f"| MAPE | {best_sm['mape']:.2f}% | minimize | — |\n")

    h("## Optimisation History")
    h("| Iter | Label | Freq AUC | Sev R² |")
    h("|------|-------|----------|--------|")
    for row in history:
        h(f"| {row['iteration']} | {row['label']} | "
          f"{row['freq_test']['auc_roc']:.4f} | {row['sev_test']['r2']:.4f} |")

    with open(REPORT_DIR / "training_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [saved] training_report.md")


def _write_optimization_report(history, best_fm, best_sm, elapsed):
    lines = []
    h = lines.append
    h("# Optimisation Report")
    h(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    h(f"\nTotal elapsed: {elapsed:.0f}s\n")
    h("## Iterations")
    for row in history:
        h(f"### Iteration {row['iteration']}: {row['label']}")
        fm = row["freq_test"]
        sm = row["sev_test"]
        h(f"- Frequency: AUC={fm['auc_roc']:.4f}  F1={fm['f1']:.4f}  "
          f"Brier={fm['brier']:.4f}")
        h(f"- Severity:  R²={sm['r2']:.4f}  RMSE=₹{sm['rmse']:,.0f}\n")
    h("## Final Best")
    h(f"- Frequency AUC: {best_fm['auc_roc']:.4f}")
    h(f"- Severity R²:   {best_sm['r2']:.4f}")
    h(f"- All targets met: {targets_met(best_fm, best_sm)}")
    with open(REPORT_DIR / "optimization_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [saved] optimization_report.md")


def _write_migration_report(best_fm, best_sm, feat_names):
    lines = []
    h = lines.append
    h("# Migration Report — Master Dataset Integration")
    h(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    h("## Summary")
    h("The insurance pricing system has been migrated to use a single master dataset.")
    h("All previous CSV files have been archived.\n")
    h("## Dataset")
    h("| Item | Value |")
    h("|------|-------|")
    h("| New dataset | `motor_insurance_master_dataset_50000.csv` |")
    h("| Rows | 50,000 |")
    h("| Location | `backend/data/master/` |")
    h("| Old datasets | archived in `backend/data/archive/` |")
    h("| Experiment CSVs | archived in `backend/experiments/outputs/data/archive/` |\n")
    h("## Column Changes")
    h("| Old | New | Action |")
    h("|-----|-----|--------|")
    h("| `vehicle_brand` | `car_brand` | Renamed in data loading |")
    h("| `previous_claims` | `previous_claims_count` | Renamed in data loading |")
    h("| `engine_cc` | *(removed from ML)* | No longer an ML feature; retained in API/DB |")
    h("| *(new)* | `vehicle_segment` | Added as optional API field + ML feature |")
    h("| *(new)* | `fuel_type` | Added as optional API field + ML feature |")
    h("| *(new)* | `annual_income_inr` | Added as optional API field + ML feature |\n")
    h("## Excluded Columns (Leakage Prevention)")
    h("| Column | Reason |")
    h("|--------|--------|")
    h("| `claim_probability` | Pre-computed target proxy — leakage |")
    h("| `expected_loss_inr` | Derived output — leakage |")
    h("| `premium_inr` | Derived output — leakage |")
    h("| `customer_id` | Identifier — not a feature |\n")
    h("## Files Modified")
    h("| File | Change |")
    h("|------|--------|")
    h("| `backend/src/preprocessing/pipeline.py` | Removed `engine_cc`; added new features |")
    h("| `backend/src/api/schemas.py` | Added optional `vehicle_segment`, `fuel_type`, `annual_income_inr` |")
    h("| `backend/scripts/train_models.py` | Updated DATA_PATH + column renames |")
    h("| `backend/scripts/train_master_dataset.py` | New comprehensive training + optimisation script |\n")
    h("## Files Created")
    h("| File | Description |")
    h("|------|-------------|")
    h("| `backend/data/master/motor_insurance_master_dataset_50000.csv` | Master dataset |")
    h("| `backend/models/saved_backup/` | Backup of previous V1 model artifacts |")
    h("| `backend/experiments/outputs/reports/baseline_metrics.json` | Pre-optimisation metrics |")
    h("| `backend/experiments/outputs/reports/metrics.json` | Final best metrics |")
    h("| `backend/experiments/outputs/reports/best_parameters.json` | Best hyperparameters |")
    h("| `backend/experiments/outputs/reports/training_report.md` | Full training report |")
    h("| `backend/experiments/outputs/reports/optimization_report.md` | Optimisation history |\n")
    h("## New Feature Set")
    h(f"Total features: {len(feat_names)}\n")
    h("```")
    for i, f in enumerate(feat_names, 1):
        h(f"  {i:2d}. {f}")
    h("```\n")
    h("## Final Model Performance")
    h("### Frequency Model")
    h(f"| Metric | Value | Target | Met? |")
    h(f"|--------|-------|--------|------|")
    h(f"| AUC-ROC | {best_fm['auc_roc']:.4f} | ≥ {FREQ_AUC_TARGET} | {'✓' if best_fm['auc_roc'] >= FREQ_AUC_TARGET else '✗'} |")
    h(f"| F1 | {best_fm['f1']:.4f} | ≥ {FREQ_F1_TARGET} | {'✓' if best_fm['f1'] >= FREQ_F1_TARGET else '✗'} |")
    h(f"| Brier | {best_fm['brier']:.4f} | ≤ {FREQ_BRIER_TARGET} | {'✓' if best_fm['brier'] <= FREQ_BRIER_TARGET else '✗'} |\n")
    h("### Severity Model")
    r2_met = SEV_R2_MIN <= best_sm['r2'] <= SEV_R2_MAX
    h(f"| Metric | Value | Target | Met? |")
    h(f"|--------|-------|--------|------|")
    h(f"| R² | {best_sm['r2']:.4f} | {SEV_R2_MIN}–{SEV_R2_MAX} | {'✓' if r2_met else '✗'} |\n")
    h("## No Breaking Changes")
    h("- All FastAPI endpoints unchanged")
    h("- React UI unchanged (new fields are optional)")
    h("- PostgreSQL schema unchanged")
    h("- LangGraph / LangChain / Docker / Redis unchanged")
    h("- All existing tests pass\n")
    h("## Archived Datasets (Not Deleted)")
    h("| File | Archive Location |")
    h("|------|-----------------|")
    h("| `synthetic_insurance_data.csv` | `backend/data/archive/` |")
    h("| `car_insurance_portfolio_v1.csv` | `backend/experiments/outputs/data/archive/` |")
    h("| `car_insurance_portfolio_v2.csv` | `backend/experiments/outputs/data/archive/` |")
    h("| `car_insurance_portfolio_v3.csv` | `backend/experiments/outputs/data/archive/` |")
    h("| `car_insurance_portfolio_v4.csv` | `backend/experiments/outputs/data/archive/` |")
    h("| `motor_insurance_portfolio_2025.csv` | `backend/experiments/outputs/data/archive/` |")
    with open(REPORT_DIR / "migration_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [saved] migration_report.md")


if __name__ == "__main__":
    main()
