"""
Train frequency and severity models on the synthetic dataset.
Run from the backend/ directory:
    python scripts/train_models.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 output so box-drawing and rupee characters render on all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
)

from src.preprocessing.pipeline import prepare_training_data
from src.models.frequency_model import FrequencyModel
from src.models.severity_model import SeverityModel
from src.config import settings

DATA_PATH = Path(__file__).parent.parent / "data" / "master" / "motor_insurance_master_dataset_50000.csv"
MODEL_DIR = settings.MODELS_DIR
VERSION = settings.MODEL_VERSION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feature_names(preprocessor) -> list:
    """Build ordered feature name list matching the 27-column hstack output."""
    ohe_names = list(preprocessor.ohe.get_feature_names_out())
    return (
        list(preprocessor.STANDARD_SCALE_COLS)
        + list(preprocessor.MINMAX_SCALE_COLS)
        + ohe_names
        + ["car_brand_encoded"]
        + ["previous_claims_count", "risk_age_ratio", "vehicle_depreciation_factor",
           "normalized_mileage", "high_value_vehicle", "driving_risk_score"]
    )


def _save_confusion_matrix(y_true, y_pred, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["No Claim", "Claim"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title("Frequency Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


def _save_roc_curve(y_true, y_proba, auc: float, path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"XGBoost  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Frequency Model — ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


def _save_pr_curve(y_true, y_proba, path: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label="Precision-Recall curve")
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline (claim rate = {baseline:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Frequency Model — Precision-Recall Curve (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


def _save_shap_summary(shap_values, X, feature_names, title: str, path: Path):
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path.name}")


def _save_actual_vs_predicted(y_true_orig, y_pred_orig, path: Path):
    lim = max(float(y_true_orig.max()), float(y_pred_orig.max())) * 1.05
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true_orig, y_pred_orig, alpha=0.35, s=12, color="steelblue")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Claim Amount (INR)")
    ax.set_ylabel("Predicted Claim Amount (INR)")
    ax.set_title("Severity Model — Actual vs Predicted (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


def _save_residual_distribution(residuals, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="black", color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero residual")
    ax.axvline(float(residuals.mean()), color="orange", linestyle="-", lw=1.5,
               label=f"Mean residual = ₹{residuals.mean():,.0f}")
    ax.set_xlabel("Residual  (Actual − Predicted)  INR")
    ax.set_ylabel("Count")
    ax.set_title("Severity Model — Residual Distribution (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


def _save_residual_scatter(y_pred_orig, residuals, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred_orig, residuals, alpha=0.35, s=12, color="darkorange")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Claim Amount (INR)")
    ax.set_ylabel("Residual  (Actual − Predicted)  INR")
    ax.set_title("Severity Model — Residual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  [saved] {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(__file__).parent.parent / "models" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    if not DATA_PATH.exists():
        print(f"Data not found at {DATA_PATH}.")
        sys.exit(1)

    # ── Dataset overview ─────────────────────────────────────────────────────
    print("=" * 62)
    print("DATASET OVERVIEW")
    print("=" * 62)
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # ── Master dataset normalisation ─────────────────────────────────────────
    df = df.rename(columns={
        "vehicle_brand":   "car_brand",
        "previous_claims": "previous_claims_count",
    })
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower()
    leakage_cols = {"claim_probability", "expected_loss_inr", "premium_inr", "customer_id"}
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    total_rows = len(df)
    print(f"  Total rows:    {total_rows:,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns:       {list(df.columns)}")

    if "claim_occurred" in df.columns:
        counts = df["claim_occurred"].value_counts().sort_index()
        n_no  = counts.get(0, 0)
        n_yes = counts.get(1, 0)
        print(f"\n── Class Distribution (claim_occurred) ──────────────────")
        print(f"  No claim (0): {n_no:,}  ({n_no / total_rows * 100:.2f}%)")
        print(f"  Claim    (1): {n_yes:,}  ({n_yes / total_rows * 100:.2f}%)")
        print(f"  scale_pos_weight (neg/pos): {n_no / max(n_yes, 1):.4f}")

    if "actual_claim_amount_inr" in df.columns:
        sev_amounts = df.loc[df["claim_occurred"] == 1, "actual_claim_amount_inr"]
        print(f"\n── Claim Amount Statistics (claimants only) ─────────────")
        print(f"  Count:  {len(sev_amounts):,}")
        print(f"  Min:    ₹{sev_amounts.min():>12,.0f}")
        print(f"  Max:    ₹{sev_amounts.max():>12,.0f}")
        print(f"  Mean:   ₹{sev_amounts.mean():>12,.0f}")
        print(f"  Median: ₹{sev_amounts.median():>12,.0f}")
        print(f"  Std:    ₹{sev_amounts.std():>12,.0f}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("PREPROCESSING")
    print("=" * 62)
    t0 = time.time()
    data = prepare_training_data(df, model_dir=MODEL_DIR)
    splits = data["splits"]
    preprocessor = data["preprocessor"]
    feature_names = _feature_names(preprocessor)
    preproc_time = time.time() - t0
    print(f"  Completed in {preproc_time:.1f}s")
    print(f"  Feature matrix width: {len(feature_names)} columns")
    print(f"  Feature names ({len(feature_names)}):")
    for i, fn in enumerate(feature_names, 1):
        print(f"    {i:2d}. {fn}")

    fs = splits["frequency"]
    ss = splits["severity"]
    sev_total = len(ss["X_train"]) + len(ss["X_val"]) + len(ss["X_test"])

    print(f"\n── Train / Val / Test Split Sizes ───────────────────────")
    print(f"  Frequency model  (full dataset, stratified)")
    print(f"    Train: {len(fs['X_train']):>5,} rows  ({len(fs['X_train']) / total_rows * 100:.1f}%)")
    print(f"    Val:   {len(fs['X_val']):>5,} rows  ({len(fs['X_val'])   / total_rows * 100:.1f}%)")
    print(f"    Test:  {len(fs['X_test']):>5,} rows  ({len(fs['X_test'])  / total_rows * 100:.1f}%)")
    print(f"  Severity model   (claimants only, {sev_total} rows)")
    print(f"    Train: {len(ss['X_train']):>5,} rows  ({len(ss['X_train']) / sev_total * 100:.1f}%)")
    print(f"    Val:   {len(ss['X_val']):>5,} rows  ({len(ss['X_val'])   / sev_total * 100:.1f}%)")
    print(f"    Test:  {len(ss['X_test']):>5,} rows  ({len(ss['X_test'])  / sev_total * 100:.1f}%)")

    # ── Frequency model ──────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("FREQUENCY MODEL  (XGBoostClassifier — P(claim))")
    print("=" * 62)

    freq = FrequencyModel(version=VERSION, model_dir=MODEL_DIR)
    s = splits["frequency"]

    t1 = time.time()
    freq_metrics = freq.train(s["X_train"], s["y_train"], s["X_val"], s["y_val"])
    freq_train_time = time.time() - t1

    print(f"\n── GridSearchCV  (cv=3, scoring=roc_auc) ────────────────")
    print(f"  Parameter grid: {freq.PARAM_GRID}")
    print(f"  Best params:    {freq_metrics['best_params']}")
    print(f"  Training time:  {freq_train_time:.1f}s")

    print(f"\n── Validation Set Metrics ───────────────────────────────")
    print(f"  AUC-ROC:     {freq_metrics['auc_roc']:.4f}")
    print(f"  Precision:   {freq_metrics['precision']:.4f}")
    print(f"  Recall:      {freq_metrics['recall']:.4f}")
    print(f"  F1 Score:    {freq_metrics['f1']:.4f}")
    print(f"  Brier Score: {freq_metrics['brier_score']:.4f}")

    test_m  = freq.evaluate(s["X_test"], s["y_test"])
    test_proba = freq.predict_proba(s["X_test"])
    test_pred  = (test_proba >= 0.5).astype(int)

    print(f"\n── Test Set Metrics ─────────────────────────────────────")
    print(f"  AUC-ROC:     {test_m['auc_roc']:.4f}")
    print(f"  Precision:   {test_m['precision']:.4f}")
    print(f"  Recall:      {test_m['recall']:.4f}")
    print(f"  F1 Score:    {test_m['f1']:.4f}")
    print(f"  Brier Score: {test_m['brier_score']:.4f}")

    print(f"\n── Classification Report (Test Set) ─────────────────────")
    print(classification_report(s["y_test"], test_pred,
                                target_names=["No Claim", "Claim"], digits=4))

    print(f"\n── Confusion Matrix (Test Set) ──────────────────────────")
    cm = confusion_matrix(s["y_test"], test_pred)
    print(f"                 Predicted No  Predicted Yes")
    print(f"  Actual No       TN = {cm[0,0]:4d}      FP = {cm[0,1]:4d}")
    print(f"  Actual Yes      FN = {cm[1,0]:4d}      TP = {cm[1,1]:4d}")

    print(f"\n── Saving Frequency Model Plots → {output_dir.name}/ ─────")
    _save_confusion_matrix(s["y_test"], test_pred,
                           output_dir / "freq_confusion_matrix.png")
    _save_roc_curve(s["y_test"], test_proba, test_m["auc_roc"],
                    output_dir / "freq_roc_curve.png")
    _save_pr_curve(s["y_test"], test_proba,
                   output_dir / "freq_pr_curve.png")

    print("  Computing SHAP values for frequency model (TreeExplainer)...")
    t_shap = time.time()
    shap_vals_freq, _ = freq.get_shap_values(s["X_test"])
    print(f"  SHAP done in {time.time() - t_shap:.1f}s")

    freq.save_feature_importance_plot(
        shap_vals_freq, feature_names,
        output_path=str(output_dir / "freq_shap_bar.png"),
    )
    print(f"  [saved] freq_shap_bar.png")

    _save_shap_summary(
        shap_vals_freq, s["X_test"], feature_names,
        "Frequency Model — SHAP Summary Plot (Test Set)",
        output_dir / "freq_shap_summary.png",
    )

    top_freq = freq.get_top_features(shap_vals_freq, feature_names, n=10)
    print(f"\n── Top 10 Features by Mean |SHAP|  (Frequency) ─────────")
    for rank, (feat, val) in enumerate(top_freq.items(), 1):
        print(f"  {rank:2d}. {feat:<42s}  {val:.6f}")

    freq_path = freq.save()
    print(f"\n  Model saved → {freq_path}")

    # ── Severity model ───────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("SEVERITY MODEL  (XGBoostRegressor — E[claim amount | claim])")
    print("=" * 62)

    sev = SeverityModel(version=VERSION, model_dir=MODEL_DIR)
    s2  = splits["severity"]

    t2 = time.time()
    sev_metrics = sev.train(s2["X_train"], s2["y_train"], s2["X_val"], s2["y_val"])
    sev_train_time = time.time() - t2

    print(f"\n── GridSearchCV  (cv=3, scoring=neg_root_mean_squared_error) ──")
    print(f"  Parameter grid: {sev.PARAM_GRID}")
    print(f"  Best params:    {sev_metrics['best_params']}")
    print(f"  Training time:  {sev_train_time:.1f}s")
    print(f"  Note: target is log1p(claim_amount); metrics shown on back-transformed scale")

    print(f"\n── Validation Set Metrics ───────────────────────────────")
    print(f"  RMSE: ₹{sev_metrics['rmse']:>12,.0f}")
    print(f"  MAE:  ₹{sev_metrics['mae']:>12,.0f}")
    print(f"  R²:    {sev_metrics['r2']:.4f}")
    print(f"  MAPE:  {sev_metrics['mape']:.2f}%")

    test_sev   = sev.evaluate(s2["X_test"], s2["y_test"])
    y_true_orig = np.expm1(s2["y_test"])
    y_pred_orig = sev.predict(s2["X_test"])
    residuals   = y_true_orig - y_pred_orig

    print(f"\n── Test Set Metrics ─────────────────────────────────────")
    print(f"  RMSE: ₹{test_sev['rmse']:>12,.0f}")
    print(f"  MAE:  ₹{test_sev['mae']:>12,.0f}")
    print(f"  R²:    {test_sev['r2']:.4f}")
    print(f"  MAPE:  {test_sev['mape']:.2f}%")

    print(f"\n── Test Set Actual Claim Amounts ────────────────────────")
    print(f"  Samples: {len(y_true_orig)}")
    print(f"  Min:     ₹{y_true_orig.min():>12,.0f}")
    print(f"  Max:     ₹{y_true_orig.max():>12,.0f}")
    print(f"  Mean:    ₹{y_true_orig.mean():>12,.0f}")
    print(f"  Median:  ₹{np.median(y_true_orig):>12,.0f}")
    print(f"  Std:     ₹{y_true_orig.std():>12,.0f}")

    print(f"\n── Residuals  (Actual − Predicted) ─────────────────────")
    print(f"  Mean:   ₹{residuals.mean():>12,.0f}")
    print(f"  Std:    ₹{residuals.std():>12,.0f}")
    print(f"  Min:    ₹{residuals.min():>12,.0f}")
    print(f"  Max:    ₹{residuals.max():>12,.0f}")

    print(f"\n── Saving Severity Model Plots → {output_dir.name}/ ──────")
    _save_actual_vs_predicted(y_true_orig, y_pred_orig,
                              output_dir / "sev_actual_vs_predicted.png")
    _save_residual_distribution(residuals,
                                output_dir / "sev_residual_distribution.png")
    _save_residual_scatter(y_pred_orig, residuals,
                           output_dir / "sev_residual_scatter.png")

    print("  Computing SHAP values for severity model (TreeExplainer)...")
    t_shap2 = time.time()
    shap_vals_sev, _ = sev.get_shap_values(s2["X_test"])
    print(f"  SHAP done in {time.time() - t_shap2:.1f}s")

    sev.save_feature_importance_plot(
        shap_vals_sev, feature_names,
        output_path=str(output_dir / "sev_shap_bar.png"),
    )
    print(f"  [saved] sev_shap_bar.png")

    _save_shap_summary(
        shap_vals_sev, s2["X_test"], feature_names,
        "Severity Model — SHAP Summary Plot (Test Set)",
        output_dir / "sev_shap_summary.png",
    )

    top_sev = sev.get_top_features(shap_vals_sev, feature_names, n=10)
    print(f"\n── Top 10 Features by Mean |SHAP|  (Severity) ──────────")
    for rank, (feat, val) in enumerate(top_sev.items(), 1):
        print(f"  {rank:2d}. {feat:<42s}  {val:.6f}")

    sev_path = sev.save()
    print(f"\n  Model saved → {sev_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("ALL OUTPUTS GENERATED")
    print("=" * 62)
    print(f"  Output directory:  {output_dir.resolve()}")
    print(f"  Total elapsed:     {time.time() - t_total:.1f}s")
    print(f"\n  PNG files saved:")
    for f in sorted(output_dir.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:<45s}  {size_kb:>4d} KB")
    print()


if __name__ == "__main__":
    main()
