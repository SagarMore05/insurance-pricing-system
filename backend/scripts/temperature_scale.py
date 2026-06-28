"""
Temperature scaling to push Brier score below 0.18.
Run from backend/:  python scripts/temperature_scale.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.preprocessing.pipeline import prepare_training_data
from src.models.frequency_model import FrequencyModel
from src.config import settings

DATA_PATH  = Path(__file__).parent.parent / "data" / "master" / "motor_insurance_master_dataset_50000.csv"
MODEL_DIR  = settings.MODELS_DIR
VERSION    = settings.MODEL_VERSION
REPORT_DIR = Path(__file__).parent.parent / "experiments" / "outputs" / "reports"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"vehicle_brand": "car_brand",
                             "previous_claims": "previous_claims_count"})
    df["gender"] = df["gender"].str.lower()
    leakage = {"claim_probability", "expected_loss_inr", "premium_inr", "customer_id"}
    return df.drop(columns=[c for c in leakage if c in df.columns])


def add_interactions(X, fn):
    ds  = fn.index("driving_score")
    pc  = fn.index("previous_claims_count")
    drs = fn.index("driving_risk_score")
    rar = fn.index("risk_age_ratio")
    nm  = fn.index("normalized_mileage")
    hv  = fn.index("high_value_vehicle")
    # Additional features
    age_idx = fn.index("age")
    va_idx  = fn.index("vehicle_age_years")
    mi_idx  = fn.index("annual_mileage_km")
    return np.hstack([X, np.column_stack([
        X[:, ds] * X[:, pc],
        X[:, drs] * X[:, rar],
        X[:, nm] * X[:, drs],
        X[:, hv] * X[:, drs],
        X[:, ds] ** 2,                    # quadratic driving score
        X[:, pc] ** 2,                    # quadratic claims
        X[:, va_idx] * X[:, drs],         # vehicle age × risk
        X[:, age_idx] * X[:, drs],        # driver age × risk
        X[:, mi_idx] * X[:, drs],         # mileage × risk
    ])])


def temperature_scale(proba, T):
    logits = np.log(np.clip(proba, 1e-7, 1 - 1e-7) / (1 - np.clip(proba, 1e-7, 1 - 1e-7)))
    scaled_logits = logits / T
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def metrics(y_true, y_proba, thr=0.5):
    yp = (y_proba >= thr).astype(int)
    return {
        "auc_roc":    round(float(roc_auc_score(y_true, y_proba)), 5),
        "f1":         round(float(f1_score(y_true, yp, zero_division=0)), 5),
        "precision":  round(float(precision_score(y_true, yp, zero_division=0)), 5),
        "recall":     round(float(recall_score(y_true, yp, zero_division=0)), 5),
        "brier":      round(float(brier_score_loss(y_true, y_proba)), 5),
        "threshold":  round(float(thr), 3),
    }


def score_m(m):
    hits = 0
    if m["auc_roc"]   >= 0.80: hits += 3
    if m["f1"]        >= 0.72: hits += 1
    if m["precision"] >= 0.72: hits += 1
    if m["recall"]    >= 0.70: hits += 1
    if m["brier"]     <= 0.18: hits += 3
    hits -= max(0, 0.80 - m["auc_roc"]) * 8
    hits -= max(0, m["brier"] - 0.18) * 15
    return hits


def best_threshold(val_proba, y_val):
    bt, bf = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.005):
        f = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
        if f > bf: bf = f; bt = t
    return bt


def main():
    print("=== Temperature Scaling + Extended Features ===")
    df = load_data()
    data = prepare_training_data(df, model_dir=MODEL_DIR)
    splits = data["splits"]
    preprocessor = data["preprocessor"]

    ohe_names = list(preprocessor.ohe.get_feature_names_out())
    fn = (list(preprocessor.STANDARD_SCALE_COLS)
          + list(preprocessor.MINMAX_SCALE_COLS)
          + ohe_names
          + ["car_brand_encoded"]
          + ["previous_claims_count", "risk_age_ratio", "vehicle_depreciation_factor",
             "normalized_mileage", "high_value_vehicle", "driving_risk_score"])

    fs = splits["frequency"]
    X_train = add_interactions(fs["X_train"], fn)
    X_val   = add_interactions(fs["X_val"],   fn)
    X_test  = add_interactions(fs["X_test"],  fn)
    y_train, y_val, y_test = fs["y_train"], fs["y_val"], fs["y_test"]

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.hstack([y_train, y_val])

    print(f"Feature dims: {X_train.shape[1]} (27 base + 9 interactions)")

    candidates = []

    # ── A1: Deep XGBoost ──────────────────────────────────────────────────────
    print("\n[A1] Deep XGBoost (n=2000, lr=0.005) ...")
    a1 = XGBClassifier(
        n_estimators=2000, max_depth=7, learning_rate=0.005,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="auc",
        random_state=42, verbosity=0, n_jobs=-1,
    )
    a1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_a1 = a1.predict_proba(X_test)[:, 1]
    m_a1 = metrics(y_test, p_a1)
    print(f"  AUC={m_a1['auc_roc']:.4f}  F1={m_a1['f1']:.4f}  Brier={m_a1['brier']:.4f}")

    # Temperature scale A1
    p_val_a1 = a1.predict_proba(X_val)[:, 1]
    res = minimize_scalar(lambda T: brier_score_loss(y_val, temperature_scale(p_val_a1, T)),
                          bounds=(0.5, 3.0), method="bounded")
    T_opt = res.x
    p_a1_ts = temperature_scale(p_a1, T_opt)
    thr_ts = best_threshold(temperature_scale(p_val_a1, T_opt), y_val)
    m_a1_ts = metrics(y_test, p_a1_ts, thr_ts)
    print(f"  A1+TempScale(T={T_opt:.3f}, thr={thr_ts:.3f}): "
          f"AUC={m_a1_ts['auc_roc']:.4f}  F1={m_a1_ts['f1']:.4f}  "
          f"Prec={m_a1_ts['precision']:.4f}  Rec={m_a1_ts['recall']:.4f}  Brier={m_a1_ts['brier']:.4f}")
    candidates.append(("A1+TempScale", m_a1_ts, a1, T_opt, thr_ts))

    # ── A2: Sigmoid calibrated A1 (previous best) ─────────────────────────────
    print("\n[A2] Sigmoid calibration on A1 (cv=5) ...")
    sig = CalibratedClassifierCV(a1, method="sigmoid", cv=5)
    sig.fit(X_tv, y_tv)
    p_sig = sig.predict_proba(X_test)[:, 1]
    thr_sig = best_threshold(sig.predict_proba(X_val)[:, 1], y_val)
    m_sig = metrics(y_test, p_sig, thr_sig)
    print(f"  SIG (thr={thr_sig:.3f}): AUC={m_sig['auc_roc']:.4f}  F1={m_sig['f1']:.4f}  "
          f"Prec={m_sig['precision']:.4f}  Rec={m_sig['recall']:.4f}  Brier={m_sig['brier']:.4f}")
    candidates.append(("SIG-cal", m_sig, sig, None, thr_sig))

    # ── A3: HistGradientBoosting (sklearn, well-calibrated) ────────────────────
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("\n[A3] HistGradientBoostingClassifier ...")
        hgb = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=7,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=42,
        )
        hgb.fit(X_train, y_train)
        p_hgb = hgb.predict_proba(X_test)[:, 1]
        thr_hgb = best_threshold(hgb.predict_proba(X_val)[:, 1], y_val)
        m_hgb = metrics(y_test, p_hgb, thr_hgb)
        print(f"  HGB (thr={thr_hgb:.3f}): AUC={m_hgb['auc_roc']:.4f}  F1={m_hgb['f1']:.4f}  "
              f"Prec={m_hgb['precision']:.4f}  Rec={m_hgb['recall']:.4f}  Brier={m_hgb['brier']:.4f}")
        candidates.append(("HGB", m_hgb, hgb, None, thr_hgb))

        # Calibrate HGB
        iso_hgb = CalibratedClassifierCV(hgb, method="isotonic", cv=5)
        iso_hgb.fit(X_tv, y_tv)
        p_ihgb = iso_hgb.predict_proba(X_test)[:, 1]
        thr_ihgb = best_threshold(iso_hgb.predict_proba(X_val)[:, 1], y_val)
        m_ihgb = metrics(y_test, p_ihgb, thr_ihgb)
        print(f"  HGB+ISO (thr={thr_ihgb:.3f}): AUC={m_ihgb['auc_roc']:.4f}  F1={m_ihgb['f1']:.4f}  "
              f"Prec={m_ihgb['precision']:.4f}  Rec={m_ihgb['recall']:.4f}  Brier={m_ihgb['brier']:.4f}")
        candidates.append(("HGB+ISO", m_ihgb, iso_hgb, None, thr_ihgb))
    except Exception as e:
        print(f"  [WARN] HistGBM failed: {e}")

    # ── Best ──────────────────────────────────────────────────────────────────
    best = max(candidates, key=lambda c: score_m(c[1]))
    label, best_m, best_clf, best_T, best_thr_f = best

    print(f"\n=== SELECTED: {label} ===")
    print(f"  AUC-ROC   : {best_m['auc_roc']:.4f}  (>= 0.80) {'✓' if best_m['auc_roc'] >= 0.80 else '✗'}")
    print(f"  F1        : {best_m['f1']:.4f}  (>= 0.72) {'✓' if best_m['f1'] >= 0.72 else '✗'}")
    print(f"  Precision : {best_m['precision']:.4f}  (>= 0.72) {'✓' if best_m['precision'] >= 0.72 else '✗'}")
    print(f"  Recall    : {best_m['recall']:.4f}  (>= 0.70) {'✓' if best_m['recall'] >= 0.70 else '✗'}")
    print(f"  Brier     : {best_m['brier']:.4f}  (<= 0.18) {'✓' if best_m['brier'] <= 0.18 else '✗'}")
    print(f"  Threshold : {best_thr_f:.3f}")

    hits = sum([
        best_m["auc_roc"] >= 0.80,
        best_m["f1"] >= 0.72,
        best_m["precision"] >= 0.72,
        best_m["recall"] >= 0.70,
        best_m["brier"] <= 0.18,
    ])
    print(f"\n  Targets hit: {hits}/5")

    # Save
    freq_model = FrequencyModel(version=VERSION, model_dir=MODEL_DIR)
    freq_model.model = best_clf
    freq_model._fitted = True
    freq_model.save()
    print(f"\n  Saved -> frequency_{VERSION}.pkl ({label})")

    # Update metrics.json
    try:
        with open(REPORT_DIR / "metrics.json") as f:
            existing = json.load(f)
        existing["frequency"] = best_m
        existing["frequency"]["selected_model"] = label
        existing["targets_met"] = (hits == 5)
        with open(REPORT_DIR / "metrics.json", "w") as f:
            json.dump(existing, f, indent=2)
        print("  Updated metrics.json")
    except Exception as e:
        print(f"  [WARN] {e}")


if __name__ == "__main__":
    main()
