"""
Champion-Challenger Model Experiment — Dataset V4
=================================================
Evaluates XGBoost (champion) against CatBoost and LightGBM (challengers)
on the V4 insurance dataset. Tests single models and ensembles.

ALL work stays inside experiments/. No production files are modified.

Phases:
  1  Baseline recovery — load saved V4 XGBoost, record metrics
  2  Challenger training — CatBoost + LightGBM with Optuna
  3  Ensemble experiments — 3 averaging combinations
  4  Model comparison table
  5  SHAP analysis on best model
  6  Champion-Challenger decision
  7  Production readiness assessment
  Final  champion_challenger_final_summary.txt
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
    precision_score, recall_score, r2_score, mean_squared_error,
    mean_absolute_error
)
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

RANDOM_SEED = 42
N_OPTUNA_TRIALS = 50
CV_FOLDS = 3
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "outputs", "data")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "outputs", "models")
REPORT_DIR = os.path.join(SCRIPT_DIR, "outputs", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)

V4_CSV = os.path.join(DATA_DIR, "car_insurance_portfolio_v4.csv")

# ─────────────────────────────────────────────────────────────
# PREPROCESSOR  (exact copy from train_v4_experiment.py)
# ─────────────────────────────────────────────────────────────
class InsurancePreprocessorV4:
    STD_COLS = [
        "age", "annual_mileage_km", "vehicle_value_inr", "driving_score",
        "vehicle_safety_rating", "pincode_risk_score",
        "flood_risk_index", "theft_risk_index", "monsoon_exposure_index"
    ]
    MM_COLS  = ["vehicle_age_years", "years_licensed", "policy_tenure_years"]
    OHE_COLS = [
        "gender", "city_tier_str", "fuel_type", "vehicle_usage_type",
        "occupation", "marital_status", "annual_income_band",
        "parking_type", "repair_cost_band", "vehicle_body_style"
    ]
    PASS_COLS = [
        "previous_claims_count", "months_since_last_claim_norm",
        "no_claim_bonus_pct", "airbags_count", "policy_inception_month",
        "risk_age_ratio", "ncb_risk_score", "engine_cc_norm"
    ]
    LE_COL = "vehicle_make"

    def __init__(self):
        self.std_scaler = StandardScaler()
        self.mm_scaler  = MinMaxScaler()
        self.ohe        = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.le         = LabelEncoder()
        self.feature_names_ = None

    def _engineer(self, df):
        out = df.copy()
        CITY_TIER = {
            "Mumbai":1,"Delhi":1,"Bangalore":1,"Hyderabad":1,"Chennai":1,
            "Kolkata":1,"Pune":1,"Ahmedabad":1,
            "Jaipur":2,"Lucknow":2,"Kanpur":2,"Nagpur":2,"Indore":2,
            "Bhopal":2,"Patna":2,"Vadodara":2,"Coimbatore":2,"Surat":2,
            "Agra":2,"Visakhapatnam":2,
        }
        out["city_tier_str"] = out["city"].map(CITY_TIER).fillna(2).astype(int).astype(str)
        out["risk_age_ratio"] = out["previous_claims_count"] / out["years_licensed"].clip(lower=1)
        out["ncb_risk_score"] = (50.0 - out["no_claim_bonus_pct"]) / 50.0
        out["months_since_last_claim_norm"] = np.where(
            out["months_since_last_claim"] == 999, 0.0,
            out["months_since_last_claim"] / 120.0
        )
        out["engine_cc_norm"] = out["engine_cc"] / 3500.0
        return out

    def fit_transform(self, df):
        df2 = self._engineer(df)
        std_arr  = self.std_scaler.fit_transform(df2[self.STD_COLS])
        mm_arr   = self.mm_scaler.fit_transform(df2[self.MM_COLS])
        ohe_arr  = self.ohe.fit_transform(df2[self.OHE_COLS])
        le_vals  = self.le.fit_transform(df2[self.LE_COL])
        pass_arr = df2[self.PASS_COLS].values
        X = np.hstack([std_arr, mm_arr, ohe_arr, le_vals.reshape(-1,1), pass_arr])
        self.feature_names_ = (
            self.STD_COLS + self.MM_COLS +
            list(self.ohe.get_feature_names_out(self.OHE_COLS)) +
            [self.LE_COL + "_encoded"] + self.PASS_COLS
        )
        return X

    def transform(self, df):
        df2 = self._engineer(df)
        std_arr  = self.std_scaler.transform(df2[self.STD_COLS])
        mm_arr   = self.mm_scaler.transform(df2[self.MM_COLS])
        ohe_arr  = self.ohe.transform(df2[self.OHE_COLS])
        le_vals  = self.le.transform(df2[self.LE_COL])
        pass_arr = df2[self.PASS_COLS].values
        return np.hstack([std_arr, mm_arr, ohe_arr, le_vals.reshape(-1,1), pass_arr])


# ─────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────
def freq_metrics(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = (tp + tn) / len(y_true)
    auc  = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                roc_auc=auc, pr_auc=prauc,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def sev_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return dict(r2=r2, rmse=rmse, mae=mae, mape=mape)

def mape_safe(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)


# ─────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FACTORIES
# ─────────────────────────────────────────────────────────────
def xgb_freq_objective(trial, X_tr, y_tr):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth         = trial.suggest_int("max_depth", 2, 5),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        subsample         = trial.suggest_float("subsample", 0.50, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.50, 0.95),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 15),
        gamma             = trial.suggest_float("gamma", 0.0, 0.5),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 15.0),
        scale_pos_weight  = spw,
        use_label_encoder = False, eval_metric="auc",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    m = xgb.XGBClassifier(**p)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_auc(m, X_tr, y_tr, cv))

def xgb_sev_objective(trial, X_tr, y_tr):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth        = trial.suggest_int("max_depth", 2, 5),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        subsample        = trial.suggest_float("subsample", 0.50, 0.95),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 0.95),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0.0, 0.5),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 10.0),
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    m = xgb.XGBRegressor(**p)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_r2(m, X_tr, y_tr, kf))

def cat_freq_objective(trial, X_tr, y_tr):
    p = dict(
        iterations        = trial.suggest_int("iterations", 100, 500, step=100),
        depth             = trial.suggest_int("depth", 3, 7),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength   = trial.suggest_float("random_strength", 0.0, 1.0),
        border_count      = trial.suggest_int("border_count", 32, 128, step=32),
        random_seed=RANDOM_SEED, verbose=0,
        eval_metric="AUC", auto_class_weights="Balanced",
    )
    m = cb.CatBoostClassifier(**p)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_auc(m, X_tr, y_tr, cv))

def cat_sev_objective(trial, X_tr, y_tr):
    p = dict(
        iterations          = trial.suggest_int("iterations", 100, 500, step=100),
        depth               = trial.suggest_int("depth", 3, 7),
        learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength", 0.0, 1.0),
        border_count        = trial.suggest_int("border_count", 32, 128, step=32),
        random_seed=RANDOM_SEED, verbose=0,
    )
    m = cb.CatBoostRegressor(**p)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_r2(m, X_tr, y_tr, kf))

def lgb_freq_objective(trial, X_tr, y_tr):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 100, step=10),
        subsample         = trial.suggest_float("subsample", 0.50, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.50, 0.95),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 10.0),
        scale_pos_weight  = spw,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    m = lgb.LGBMClassifier(**p)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_auc(m, X_tr, y_tr, cv))

def lgb_sev_objective(trial, X_tr, y_tr):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 100, step=10),
        subsample         = trial.suggest_float("subsample", 0.50, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.50, 0.95),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 10.0),
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    m = lgb.LGBMRegressor(**p)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return np.mean(cross_val_r2(m, X_tr, y_tr, kf))


# ─────────────────────────────────────────────────────────────
# CROSS-VAL HELPERS
# ─────────────────────────────────────────────────────────────
def cross_val_auc(model, X, y, cv):
    scores = []
    for tr_idx, val_idx in cv.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], prob))
    return scores

def cross_val_r2(model, X, y, cv):
    scores = []
    for tr_idx, val_idx in cv.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[val_idx])
        scores.append(r2_score(y[val_idx], pred))
    return scores


# ─────────────────────────────────────────────────────────────
# OPTUNA TUNER
# ─────────────────────────────────────────────────────────────
def tune(objective_fn, X_tr, y_tr, direction="maximize", n_trials=N_OPTUNA_TRIALS):
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_tr, y_tr),
        n_trials=n_trials, show_progress_bar=False
    )
    return study.best_params, study.best_value


# ─────────────────────────────────────────────────────────────
# McNemar TEST  (statistical significance of classifier difference)
# ─────────────────────────────────────────────────────────────
def mcnemar_test(y_true, pred_a, pred_b):
    """
    Compare two binary classifiers.
    b = A correct & B wrong
    c = A wrong  & B correct
    McNemar statistic = (|b-c| - 1)^2 / (b+c), chi-sq df=1.
    Returns (statistic, p_value). Small p => significant difference.
    """
    from scipy.stats import chi2
    a_right = (pred_a == y_true)
    b_right = (pred_b == y_true)
    b = int(( a_right & ~b_right).sum())   # A right, B wrong
    c = int((~a_right &  b_right).sum())   # A wrong, B right
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    p    = 1 - chi2.cdf(stat, df=1)
    return stat, p

def bootstrap_r2_diff(y_true, pred_a, pred_b, n_boot=1000, seed=42):
    """
    Bootstrap confidence interval for R²(b) - R²(a).
    Returns (mean_diff, p_value_one_sided).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r2a = r2_score(y_true[idx], pred_a[idx])
        r2b = r2_score(y_true[idx], pred_b[idx])
        diffs.append(r2b - r2a)
    diffs = np.array(diffs)
    mean_diff = float(diffs.mean())
    # p-value: fraction of bootstraps where diff <= 0 (one-sided, H1: B > A)
    p = float((diffs <= 0).mean())
    return mean_diff, p


# ─────────────────────────────────────────────────────────────
# DATA LOADING & SPLIT
# ─────────────────────────────────────────────────────────────
def load_and_split():
    print("Loading V4 dataset ...")
    df = pd.read_csv(V4_CSV)
    print(f"  Shape: {df.shape}  Claim rate: {df['claim_occurred'].mean()*100:.2f}%")

    DROP_COLS = [
        "customer_id", "vehicle_model",
        "claim_probability_true", "claim_occurred",
        "actual_claim_amount_inr", "region_risk_category",
    ]
    X_raw  = df[[c for c in df.columns if c not in DROP_COLS]].copy()
    y_freq = df["claim_occurred"].values

    prep = InsurancePreprocessorV4()
    X_all = prep.fit_transform(X_raw)
    feat_names = prep.feature_names_

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_freq, np.arange(len(df)),
        test_size=0.20, stratify=y_freq, random_state=RANDOM_SEED
    )
    print(f"  Train={len(X_train):,}  Test={len(X_test):,}  "
          f"Test claims={y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    # Severity subsets
    claim_tr = y_train == 1
    claim_te = y_test  == 1
    X_sev_tr = X_train[claim_tr]
    X_sev_te = X_test[claim_te]
    y_sev_tr = df.iloc[idx_train[claim_tr]]["actual_claim_amount_inr"].values
    y_sev_te = df.iloc[idx_test[claim_te]]["actual_claim_amount_inr"].values
    print(f"  Severity train={len(X_sev_tr):,}  test={len(X_sev_te):,}")

    return (X_train, X_test, y_train, y_test,
            X_sev_tr, X_sev_te, y_sev_tr, y_sev_te,
            feat_names, df, idx_test, claim_te)


# ─────────────────────────────────────────────────────────────
# PHASE 1 — BASELINE RECOVERY
# ─────────────────────────────────────────────────────────────
def phase1_baseline(X_train, X_test, y_train, y_test,
                    X_sev_tr, X_sev_te, y_sev_tr, y_sev_te):
    print("\n--- PHASE 1: Baseline Recovery ---")

    # Load saved XGBoost models
    freq_champ = xgb.XGBClassifier()
    freq_champ.load_model(os.path.join(MODEL_DIR, "v4_freq_model.json"))
    sev_champ  = xgb.XGBRegressor()
    sev_champ.load_model(os.path.join(MODEL_DIR, "v4_sev_model.json"))

    y_prob_xgb = freq_champ.predict_proba(X_test)[:, 1]
    y_pred_sev_xgb = sev_champ.predict(X_sev_te)

    fm = freq_metrics(y_test, y_prob_xgb)
    sm = sev_metrics(y_sev_te, y_pred_sev_xgb)

    print(f"  [FREQ] AUC={fm['roc_auc']:.4f}  F1={fm['f1']:.4f}  "
          f"Prec={fm['precision']:.4f}  Rec={fm['recall']:.4f}")
    print(f"  [SEV]  R²={sm['r2']:.4f}  RMSE=INR {sm['rmse']:,.0f}  "
          f"MAE=INR {sm['mae']:,.0f}")

    # Write report
    path = os.path.join(REPORT_DIR, "baseline_v4_metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("BASELINE V4 METRICS — Champion XGBoost\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model:    XGBoostClassifier (v4_freq_model.json)\n")
        f.write("Dataset:  car_insurance_portfolio_v4.csv\n")
        f.write("Split:    80/20 stratified, seed=42\n")
        f.write(f"Test set: {len(y_test):,} rows  "
                f"({y_test.sum():,} claims, {y_test.mean()*100:.2f}%)\n\n")

        f.write("FREQUENCY MODEL (Classification)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:   {fm['accuracy']:.4f}\n")
        f.write(f"  Precision:  {fm['precision']:.4f}\n")
        f.write(f"  Recall:     {fm['recall']:.4f}\n")
        f.write(f"  F1 Score:   {fm['f1']:.4f}\n")
        f.write(f"  ROC-AUC:    {fm['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:     {fm['pr_auc']:.4f}\n")
        f.write(f"  Confusion:  TN={fm['tn']:,}  FP={fm['fp']:,}  "
                f"FN={fm['fn']:,}  TP={fm['tp']:,}\n\n")

        f.write("SEVERITY MODEL (Regression)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  R²:         {sm['r2']:.4f}\n")
        f.write(f"  RMSE:       INR {sm['rmse']:,.0f}\n")
        f.write(f"  MAE:        INR {sm['mae']:,.0f}\n")
        f.write(f"  MAPE:       {sm['mape']:.2f}%\n")

    print(f"  Written: baseline_v4_metrics.txt")
    return fm, sm, freq_champ, sev_champ, y_prob_xgb, y_pred_sev_xgb


# ─────────────────────────────────────────────────────────────
# PHASE 2 — CHALLENGER TRAINING
# ─────────────────────────────────────────────────────────────
def phase2_challengers(X_train, X_test, y_train, y_test,
                       X_sev_tr, X_sev_te, y_sev_tr, y_sev_te,
                       xgb_fm, xgb_sm):
    print("\n--- PHASE 2: Challenger Training ---")
    results = {}

    # ── CatBoost Frequency
    print("  [FREQ] Tuning CatBoost ...")
    t0 = time.time()
    best_p, cv_val = tune(cat_freq_objective, X_train, y_train)
    best_p.update(dict(random_seed=RANDOM_SEED, verbose=0,
                       eval_metric="AUC", auto_class_weights="Balanced"))
    cat_freq = cb.CatBoostClassifier(**best_p)
    cat_freq.fit(X_train, y_train)
    cat_prob = cat_freq.predict_proba(X_test)[:, 1]
    cat_fm   = freq_metrics(y_test, cat_prob)
    elapsed  = time.time() - t0
    print(f"    AUC={cat_fm['roc_auc']:.4f}  F1={cat_fm['f1']:.4f}  ({elapsed:.0f}s)")
    results["catboost_freq"] = dict(metrics=cat_fm, params=best_p, cv_val=cv_val,
                                     model=cat_freq, prob=cat_prob, elapsed=elapsed)

    # ── LightGBM Frequency
    print("  [FREQ] Tuning LightGBM ...")
    t0 = time.time()
    best_p, cv_val = tune(lgb_freq_objective, X_train, y_train)
    best_p.update(dict(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1))
    lgb_freq = lgb.LGBMClassifier(**best_p)
    lgb_freq.fit(X_train, y_train)
    lgb_prob = lgb_freq.predict_proba(X_test)[:, 1]
    lgb_fm   = freq_metrics(y_test, lgb_prob)
    elapsed  = time.time() - t0
    print(f"    AUC={lgb_fm['roc_auc']:.4f}  F1={lgb_fm['f1']:.4f}  ({elapsed:.0f}s)")
    results["lgbm_freq"] = dict(metrics=lgb_fm, params=best_p, cv_val=cv_val,
                                 model=lgb_freq, prob=lgb_prob, elapsed=elapsed)

    # ── CatBoost Severity
    print("  [SEV]  Tuning CatBoost ...")
    t0 = time.time()
    best_p, cv_val = tune(cat_sev_objective, X_sev_tr, y_sev_tr)
    best_p.update(dict(random_seed=RANDOM_SEED, verbose=0))
    cat_sev  = cb.CatBoostRegressor(**best_p)
    cat_sev.fit(X_sev_tr, y_sev_tr)
    cat_pred = cat_sev.predict(X_sev_te)
    cat_sm   = sev_metrics(y_sev_te, cat_pred)
    elapsed  = time.time() - t0
    print(f"    R²={cat_sm['r2']:.4f}  RMSE=INR {cat_sm['rmse']:,.0f}  ({elapsed:.0f}s)")
    results["catboost_sev"] = dict(metrics=cat_sm, params=best_p, cv_val=cv_val,
                                    model=cat_sev, pred=cat_pred, elapsed=elapsed)

    # ── LightGBM Severity
    print("  [SEV]  Tuning LightGBM ...")
    t0 = time.time()
    best_p, cv_val = tune(lgb_sev_objective, X_sev_tr, y_sev_tr)
    best_p.update(dict(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1))
    lgb_sev  = lgb.LGBMRegressor(**best_p)
    lgb_sev.fit(X_sev_tr, y_sev_tr)
    lgb_pred = lgb_sev.predict(X_sev_te)
    lgb_sm   = sev_metrics(y_sev_te, lgb_pred)
    elapsed  = time.time() - t0
    print(f"    R²={lgb_sm['r2']:.4f}  RMSE=INR {lgb_sm['rmse']:,.0f}  ({elapsed:.0f}s)")
    results["lgbm_sev"] = dict(metrics=lgb_sm, params=best_p, cv_val=cv_val,
                                model=lgb_sev, pred=lgb_pred, elapsed=elapsed)

    # Write challenger report
    path = os.path.join(REPORT_DIR, "challenger_training_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("CHALLENGER TRAINING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Training config:\n")
        f.write(f"  Optuna trials per model: {N_OPTUNA_TRIALS}\n")
        f.write(f"  Cross-validation folds:  {CV_FOLDS}\n")
        f.write(f"  Train size: {len(X_train):,}\n")
        f.write(f"  Test size:  {len(X_test):,}\n\n")

        f.write("FREQUENCY CHALLENGERS\n")
        f.write("-" * 40 + "\n\n")

        for algo, label in [("catboost_freq","CatBoost"), ("lgbm_freq","LightGBM")]:
            r = results[algo]
            m = r["metrics"]
            f.write(f"{label} Classifier:\n")
            f.write(f"  CV AUC (best trial):  {r['cv_val']:.4f}\n")
            f.write(f"  Test ROC-AUC:         {m['roc_auc']:.4f}  "
                    f"(XGB champion: {xgb_fm['roc_auc']:.4f}  "
                    f"delta: {m['roc_auc']-xgb_fm['roc_auc']:+.4f})\n")
            f.write(f"  Test PR-AUC:          {m['pr_auc']:.4f}  "
                    f"(XGB: {xgb_fm['pr_auc']:.4f}  delta: {m['pr_auc']-xgb_fm['pr_auc']:+.4f})\n")
            f.write(f"  Test F1:              {m['f1']:.4f}  "
                    f"(XGB: {xgb_fm['f1']:.4f}  delta: {m['f1']-xgb_fm['f1']:+.4f})\n")
            f.write(f"  Precision:            {m['precision']:.4f}\n")
            f.write(f"  Recall:               {m['recall']:.4f}\n")
            f.write(f"  Accuracy:             {m['accuracy']:.4f}\n")
            f.write(f"  Training time:        {r['elapsed']:.0f}s\n")
            f.write(f"  Best params: {r['params']}\n\n")

        f.write("SEVERITY CHALLENGERS\n")
        f.write("-" * 40 + "\n\n")
        for algo, label in [("catboost_sev","CatBoost"), ("lgbm_sev","LightGBM")]:
            r = results[algo]
            m = r["metrics"]
            f.write(f"{label} Regressor:\n")
            f.write(f"  CV R² (best trial):   {r['cv_val']:.4f}\n")
            f.write(f"  Test R²:              {m['r2']:.4f}  "
                    f"(XGB: {xgb_sm['r2']:.4f}  delta: {m['r2']-xgb_sm['r2']:+.4f})\n")
            f.write(f"  Test RMSE:            INR {m['rmse']:,.0f}  "
                    f"(XGB: INR {xgb_sm['rmse']:,.0f})\n")
            f.write(f"  Test MAE:             INR {m['mae']:,.0f}\n")
            f.write(f"  Test MAPE:            {m['mape']:.2f}%\n")
            f.write(f"  Training time:        {r['elapsed']:.0f}s\n")
            f.write(f"  Best params: {r['params']}\n\n")

    print(f"  Written: challenger_training_report.txt")
    return results


# ─────────────────────────────────────────────────────────────
# PHASE 3 — ENSEMBLE EXPERIMENT
# ─────────────────────────────────────────────────────────────
def phase3_ensembles(y_test, y_sev_te,
                     xgb_prob, cat_prob, lgb_prob,
                     xgb_sev_pred, cat_sev_pred, lgb_sev_pred,
                     xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm):
    print("\n--- PHASE 3: Ensemble Experiments ---")

    ens_freq = {}
    ens_sev  = {}

    # Frequency ensembles
    combos_f = [
        ("XGB+CAT",       (xgb_prob + cat_prob) / 2),
        ("XGB+LGB",       (xgb_prob + lgb_prob) / 2),
        ("XGB+CAT+LGB",   (xgb_prob + cat_prob + lgb_prob) / 3),
    ]
    for name, prob in combos_f:
        m = freq_metrics(y_test, prob)
        ens_freq[name] = dict(metrics=m, prob=prob)
        print(f"  [FREQ ENS] {name:<15} AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")

    # Severity ensembles
    combos_s = [
        ("XGB+CAT",       (xgb_sev_pred + cat_sev_pred) / 2),
        ("XGB+LGB",       (xgb_sev_pred + lgb_sev_pred) / 2),
        ("XGB+CAT+LGB",   (xgb_sev_pred + cat_sev_pred + lgb_sev_pred) / 3),
    ]
    for name, pred in combos_s:
        m = sev_metrics(y_sev_te, pred)
        ens_sev[name] = dict(metrics=m, pred=pred)
        print(f"  [SEV  ENS] {name:<15} R²={m['r2']:.4f}  RMSE=INR {m['rmse']:,.0f}")

    # Write ensemble report
    path = os.path.join(REPORT_DIR, "ensemble_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("ENSEMBLE EXPERIMENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: Simple unweighted probability averaging.\n")
        f.write("All constituent models tuned with Optuna (50 trials, 3-fold CV).\n\n")

        f.write("FREQUENCY ENSEMBLES\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Ensemble':<20} {'AUC':>8} {'PR-AUC':>8} {'F1':>8} "
                f"{'Prec':>8} {'Rec':>8}\n")
        f.write("-" * 62 + "\n")

        # Baselines first
        for label, m in [("XGB (champion)", xgb_fm),
                          ("CatBoost", cat_fm),
                          ("LightGBM", lgb_fm)]:
            f.write(f"  {label:<18} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
                    f"{m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}\n")
        f.write("  " + "-" * 60 + "\n")
        for name, res in ens_freq.items():
            m = res["metrics"]
            f.write(f"  Ens {name:<16} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
                    f"{m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}\n")

        f.write("\n\nSEVERITY ENSEMBLES\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Ensemble':<20} {'R2':>8} {'RMSE':>12} {'MAE':>12} {'MAPE%':>8}\n")
        f.write("-" * 62 + "\n")
        for label, m in [("XGB (champion)", xgb_sm),
                          ("CatBoost", cat_sm),
                          ("LightGBM", lgb_sm)]:
            f.write(f"  {label:<18} {m['r2']:>8.4f} {m['rmse']:>12,.0f} "
                    f"{m['mae']:>12,.0f} {m['mape']:>8.2f}\n")
        f.write("  " + "-" * 60 + "\n")
        for name, res in ens_sev.items():
            m = res["metrics"]
            f.write(f"  Ens {name:<16} {m['r2']:>8.4f} {m['rmse']:>12,.0f} "
                    f"{m['mae']:>12,.0f} {m['mape']:>8.2f}\n")

        # Best ensemble
        best_ens_f = max(ens_freq.items(), key=lambda x: x[1]["metrics"]["roc_auc"])
        best_ens_s = max(ens_sev.items(),  key=lambda x: x[1]["metrics"]["r2"])
        f.write(f"\nBest frequency ensemble: {best_ens_f[0]}  "
                f"AUC={best_ens_f[1]['metrics']['roc_auc']:.4f}\n")
        f.write(f"Best severity ensemble:  {best_ens_s[0]}  "
                f"R²={best_ens_s[1]['metrics']['r2']:.4f}\n")

    print(f"  Written: ensemble_report.txt")
    return ens_freq, ens_sev


# ─────────────────────────────────────────────────────────────
# PHASE 4 — MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
def phase4_comparison(y_test, y_sev_te,
                      xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
                      ens_freq, ens_sev,
                      xgb_prob, cat_prob, lgb_prob,
                      xgb_sev_pred, cat_sev_pred, lgb_sev_pred):
    print("\n--- PHASE 4: Model Comparison ---")

    all_freq = [
        ("XGBoost",             xgb_fm, xgb_prob),
        ("CatBoost",            cat_fm, cat_prob),
        ("LightGBM",            lgb_fm, lgb_prob),
        ("Ens XGB+CAT",         ens_freq["XGB+CAT"]["metrics"],     ens_freq["XGB+CAT"]["prob"]),
        ("Ens XGB+LGB",         ens_freq["XGB+LGB"]["metrics"],     ens_freq["XGB+LGB"]["prob"]),
        ("Ens XGB+CAT+LGB",     ens_freq["XGB+CAT+LGB"]["metrics"], ens_freq["XGB+CAT+LGB"]["prob"]),
    ]
    all_sev = [
        ("XGBoost",             xgb_sm, xgb_sev_pred),
        ("CatBoost",            cat_sm, cat_sev_pred),
        ("LightGBM",            lgb_sm, lgb_sev_pred),
        ("Ens XGB+CAT",         ens_sev["XGB+CAT"]["metrics"],     ens_sev["XGB+CAT"]["pred"]),
        ("Ens XGB+LGB",         ens_sev["XGB+LGB"]["metrics"],     ens_sev["XGB+LGB"]["pred"]),
        ("Ens XGB+CAT+LGB",     ens_sev["XGB+CAT+LGB"]["metrics"], ens_sev["XGB+CAT+LGB"]["pred"]),
    ]

    # Sort by AUC / R²
    all_freq_sorted = sorted(all_freq, key=lambda x: x[1]["roc_auc"], reverse=True)
    all_sev_sorted  = sorted(all_sev,  key=lambda x: x[1]["r2"],      reverse=True)

    # Identify overall best
    best_freq_name, best_freq_m, best_freq_prob = all_freq_sorted[0]
    best_sev_name,  best_sev_m,  best_sev_pred  = all_sev_sorted[0]

    path = os.path.join(REPORT_DIR, "model_comparison_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON REPORT — Champion vs Challengers\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 72 + "\n\n")

        f.write("FREQUENCY MODELS — Ranked by ROC-AUC\n")
        f.write("-" * 72 + "\n")
        f.write(f"{'Rank':<5} {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
                f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8}\n")
        f.write("-" * 72 + "\n")
        for rank, (name, m, _) in enumerate(all_freq_sorted, 1):
            tag = " <- CHAMPION" if rank == 1 else ("  [XGB]" if name == "XGBoost" else "")
            f.write(f"  {rank:<3} {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f} "
                    f"{m['pr_auc']:>8.4f}{tag}\n")

        f.write(f"\nBest frequency model: {best_freq_name}  "
                f"AUC={best_freq_m['roc_auc']:.4f}\n")
        delta_f = best_freq_m["roc_auc"] - xgb_fm["roc_auc"]
        f.write(f"Gain vs XGB champion: {delta_f:+.4f}\n\n")

        f.write("SEVERITY MODELS — Ranked by R²\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<5} {'Model':<22} {'R2':>8} {'RMSE':>12} "
                f"{'MAE':>12} {'MAPE%':>8}\n")
        f.write("-" * 60 + "\n")
        for rank, (name, m, _) in enumerate(all_sev_sorted, 1):
            tag = " <- CHAMPION" if rank == 1 else ("  [XGB]" if name == "XGBoost" else "")
            f.write(f"  {rank:<3} {name:<22} {m['r2']:>8.4f} {m['rmse']:>12,.0f} "
                    f"{m['mae']:>12,.0f} {m['mape']:>8.2f}{tag}\n")

        f.write(f"\nBest severity model: {best_sev_name}  "
                f"R²={best_sev_m['r2']:.4f}\n")
        delta_s = best_sev_m["r2"] - xgb_sm["r2"]
        f.write(f"Gain vs XGB champion: {delta_s:+.4f}\n")

    print(f"  Written: model_comparison_report.txt")
    print(f"  Best frequency: {best_freq_name} (AUC={best_freq_m['roc_auc']:.4f})")
    print(f"  Best severity:  {best_sev_name}  (R²={best_sev_m['r2']:.4f})")
    return all_freq_sorted, all_sev_sorted, best_freq_name, best_freq_m, best_freq_prob, \
           best_sev_name, best_sev_m, best_sev_pred


# ─────────────────────────────────────────────────────────────
# PHASE 5 — SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────
def phase5_shap(best_freq_name, best_freq_model_obj,
                best_sev_name,  best_sev_model_obj,
                X_test, X_sev_te, feat_names,
                all_freq_sorted, all_sev_sorted,
                xgb_freq_model, xgb_sev_model,
                cat_freq_model, lgb_freq_model):
    print("\n--- PHASE 5: SHAP Analysis ---")
    path = os.path.join(REPORT_DIR, "best_model_shap_report.txt")

    N_SAMPLE = 2000

    def get_shap_df(model, X_sample, feat_names, is_classifier):
        if not SHAP_OK:
            return None
        idx = np.random.choice(len(X_sample), min(N_SAMPLE, len(X_sample)), replace=False)
        Xs = X_sample[idx]
        try:
            expl = shap.TreeExplainer(model)
            sv   = expl.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1]
            mean_abs = np.abs(sv).mean(axis=0)
            df = pd.DataFrame({"feature": feat_names[:len(mean_abs)],
                                "mean_abs_shap": mean_abs})
            return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        except Exception as e:
            print(f"    SHAP failed: {e}")
            return None

    # SHAP for best frequency model
    print(f"  Computing SHAP for best frequency model: {best_freq_name} ...")
    shap_freq_df = get_shap_df(best_freq_model_obj, X_test, feat_names, True)

    # SHAP for XGBoost (always compute for comparison if not champion)
    if best_freq_name != "XGBoost":
        print(f"  Computing SHAP for XGBoost (champion, for comparison) ...")
        shap_xgb_freq = get_shap_df(xgb_freq_model, X_test, feat_names, True)
    else:
        shap_xgb_freq = shap_freq_df

    # SHAP for best severity model
    print(f"  Computing SHAP for best severity model: {best_sev_name} ...")
    shap_sev_df = get_shap_df(best_sev_model_obj, X_sev_te, feat_names, False)

    # Classify V4 new features
    V4_NEW = {
        "months_since_last_claim_norm", "no_claim_bonus_pct", "policy_tenure_years",
        "fuel_type_Electric","fuel_type_Hybrid","fuel_type_Petrol","fuel_type_Diesel",
        "repair_cost_band_Budget","repair_cost_band_Standard",
        "repair_cost_band_Premium","repair_cost_band_Luxury",
        "vehicle_body_style_Hatchback","vehicle_body_style_Sedan",
        "vehicle_body_style_SUV","vehicle_body_style_MUV","vehicle_body_style_Luxury",
        "parking_type_Garage","parking_type_Open_Compound","parking_type_Street",
        "policy_inception_month","pincode_risk_score",
        "flood_risk_index","theft_risk_index","monsoon_exposure_index","ncb_risk_score",
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write("BEST MODEL SHAP ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best frequency model: {best_freq_name}\n")
        f.write(f"Best severity model:  {best_sev_name}\n")
        f.write(f"SHAP sample size: {N_SAMPLE}\n\n")

        def write_shap_table(f, df, label, total_shap=None):
            if df is None:
                f.write(f"  {label}: SHAP unavailable.\n\n")
                return
            total = df["mean_abs_shap"].sum() if total_shap is None else total_shap
            f.write(f"{label} — Top 30 by Mean |SHAP|\n")
            f.write("-" * 68 + "\n")
            f.write(f"{'Rank':<5} {'Feature':<38} {'Mean|SHAP|':>12} {'Contrib%':>9} {'New?':>6}\n")
            f.write("-" * 68 + "\n")
            for i, row in df.head(30).iterrows():
                pct  = row["mean_abs_shap"] / total * 100
                new  = "[V4]" if any(n in row["feature"] for n in V4_NEW) else ""
                f.write(f"  {i+1:<3} {row['feature']:<38} {row['mean_abs_shap']:>12.6f} "
                        f"{pct:>8.2f}% {new:>6}\n")
            # V4 new feature mass
            new_mass = df[df["feature"].apply(
                lambda x: any(n in x for n in V4_NEW))]["mean_abs_shap"].sum()
            f.write(f"\n  Total SHAP mass:           {total:.6f}\n")
            f.write(f"  V4 new feature SHAP mass:  {new_mass:.6f}  "
                    f"({new_mass/total*100:.1f}% of total)\n\n")

        # Frequency SHAP
        write_shap_table(f, shap_freq_df, f"FREQUENCY — {best_freq_name}")

        # XGB comparison (if best is not XGB)
        if best_freq_name != "XGBoost" and shap_xgb_freq is not None:
            write_shap_table(f, shap_xgb_freq, "FREQUENCY — XGBoost (champion baseline)")

            # Feature rank shift analysis
            if shap_freq_df is not None:
                f.write("FEATURE RANK SHIFT: XGBoost vs Best Model\n")
                f.write("-" * 55 + "\n")
                xgb_ranks = {row["feature"]: i+1 for i, row in shap_xgb_freq.iterrows()}
                best_ranks = {row["feature"]: i+1 for i, row in shap_freq_df.iterrows()}
                f.write(f"{'Feature':<38} {'XGB Rank':>9} {'Best Rank':>10} {'Shift':>7}\n")
                f.write("-" * 66 + "\n")
                common = [feat for feat in shap_freq_df["feature"].head(15).tolist()
                          if feat in xgb_ranks]
                for feat in common:
                    xr = xgb_ranks.get(feat, "N/A")
                    br = best_ranks.get(feat, "N/A")
                    if isinstance(xr, int) and isinstance(br, int):
                        shift = xr - br
                        f.write(f"  {feat:<36} {xr:>9} {br:>10} {shift:>+7}\n")
                f.write("\n")

        # Severity SHAP
        write_shap_table(f, shap_sev_df, f"SEVERITY — {best_sev_name}")

        f.write("KEY SHAP FINDINGS\n")
        f.write("-" * 40 + "\n")
        if shap_freq_df is not None:
            top3 = shap_freq_df.head(3)["feature"].tolist()
            f.write(f"1. Frequency top drivers:    {', '.join(top3)}\n")
        if shap_sev_df is not None:
            top3s = shap_sev_df.head(3)["feature"].tolist()
            f.write(f"2. Severity top drivers:     {', '.join(top3s)}\n")
        if shap_freq_df is not None:
            new_mass_f = shap_freq_df[shap_freq_df["feature"].apply(
                lambda x: any(n in x for n in V4_NEW))]["mean_abs_shap"].sum()
            pct_f = new_mass_f / shap_freq_df["mean_abs_shap"].sum() * 100
            f.write(f"3. V4 new features contribute {pct_f:.1f}% of frequency SHAP\n")
        if shap_sev_df is not None:
            new_mass_s = shap_sev_df[shap_sev_df["feature"].apply(
                lambda x: any(n in x for n in V4_NEW))]["mean_abs_shap"].sum()
            pct_s = new_mass_s / shap_sev_df["mean_abs_shap"].sum() * 100
            f.write(f"4. V4 new features contribute {pct_s:.1f}% of severity SHAP\n")

    print(f"  Written: best_model_shap_report.txt")
    return shap_freq_df, shap_sev_df


# ─────────────────────────────────────────────────────────────
# PHASE 6 — CHAMPION-CHALLENGER DECISION
# ─────────────────────────────────────────────────────────────
def phase6_decision(y_test, y_sev_te,
                    xgb_prob, cat_prob, lgb_prob,
                    xgb_sev_pred, cat_sev_pred, lgb_sev_pred,
                    xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
                    ens_freq, ens_sev,
                    all_freq_sorted, all_sev_sorted,
                    best_freq_name, best_freq_m, best_freq_prob,
                    best_sev_name,  best_sev_m,  best_sev_pred):
    print("\n--- PHASE 6: Champion-Challenger Decision ---")

    # Statistical tests
    xgb_pred_class = (xgb_prob >= 0.5).astype(int)
    cat_pred_class = (cat_prob >= 0.5).astype(int)
    lgb_pred_class = (lgb_prob >= 0.5).astype(int)

    mcn_xgb_cat = mcnemar_test(y_test, xgb_pred_class, cat_pred_class)
    mcn_xgb_lgb = mcnemar_test(y_test, xgb_pred_class, lgb_pred_class)
    mcn_xgb_best_ens = None
    best_ens_prob = ens_freq[max(ens_freq.keys(),
                                  key=lambda k: ens_freq[k]["metrics"]["roc_auc"])]["prob"]
    best_ens_pred_class = (best_ens_prob >= 0.5).astype(int)
    mcn_xgb_best_ens = mcnemar_test(y_test, xgb_pred_class, best_ens_pred_class)

    boot_xgb_cat = bootstrap_r2_diff(y_sev_te, xgb_sev_pred, cat_sev_pred)
    boot_xgb_lgb = bootstrap_r2_diff(y_sev_te, xgb_sev_pred, lgb_sev_pred)

    ALPHA = 0.05

    path = os.path.join(REPORT_DIR, "champion_challenger_decision.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("CHAMPION-CHALLENGER DECISION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CURRENT CHAMPION\n")
        f.write("  XGBoostClassifier (frequency) + XGBoostRegressor (severity)\n")
        f.write(f"  Freq AUC: {xgb_fm['roc_auc']:.4f}  F1: {xgb_fm['f1']:.4f}  "
                f"PR-AUC: {xgb_fm['pr_auc']:.4f}\n")
        f.write(f"  Sev  R²:  {xgb_sm['r2']:.4f}  RMSE: INR {xgb_sm['rmse']:,.0f}\n\n")

        # Q1: Does any challenger beat XGBoost?
        f.write("Q1 — DOES ANY CHALLENGER BEAT XGBoost?\n")
        f.write("-" * 50 + "\n")
        cat_freq_beats = cat_fm["roc_auc"] > xgb_fm["roc_auc"]
        lgb_freq_beats = lgb_fm["roc_auc"] > xgb_fm["roc_auc"]
        cat_sev_beats  = cat_sm["r2"]      > xgb_sm["r2"]
        lgb_sev_beats  = lgb_sm["r2"]      > xgb_sm["r2"]

        f.write(f"  CatBoost Freq AUC: {cat_fm['roc_auc']:.4f}  "
                f"vs XGB {xgb_fm['roc_auc']:.4f}  "
                f"delta={cat_fm['roc_auc']-xgb_fm['roc_auc']:+.4f}  "
                f"{'BEATS XGB' if cat_freq_beats else 'below XGB'}\n")
        f.write(f"  LightGBM Freq AUC: {lgb_fm['roc_auc']:.4f}  "
                f"vs XGB {xgb_fm['roc_auc']:.4f}  "
                f"delta={lgb_fm['roc_auc']-xgb_fm['roc_auc']:+.4f}  "
                f"{'BEATS XGB' if lgb_freq_beats else 'below XGB'}\n")
        f.write(f"  CatBoost Sev R²:   {cat_sm['r2']:.4f}  "
                f"vs XGB {xgb_sm['r2']:.4f}  "
                f"delta={cat_sm['r2']-xgb_sm['r2']:+.4f}  "
                f"{'BEATS XGB' if cat_sev_beats else 'below XGB'}\n")
        f.write(f"  LightGBM Sev R²:   {lgb_sm['r2']:.4f}  "
                f"vs XGB {xgb_sm['r2']:.4f}  "
                f"delta={lgb_sm['r2']-xgb_sm['r2']:+.4f}  "
                f"{'BEATS XGB' if lgb_sev_beats else 'below XGB'}\n\n")

        # Q2: Statistical significance
        f.write("Q2 — IS THE IMPROVEMENT STATISTICALLY MEANINGFUL?\n")
        f.write("-" * 50 + "\n")
        f.write("Frequency — McNemar's Test (H0: classifiers are equally accurate)\n")
        f.write(f"  XGB vs CatBoost: stat={mcn_xgb_cat[0]:.3f}  "
                f"p={mcn_xgb_cat[1]:.4f}  "
                f"{'SIGNIFICANT' if mcn_xgb_cat[1] < ALPHA else 'not significant'} "
                f"(alpha={ALPHA})\n")
        f.write(f"  XGB vs LightGBM: stat={mcn_xgb_lgb[0]:.3f}  "
                f"p={mcn_xgb_lgb[1]:.4f}  "
                f"{'SIGNIFICANT' if mcn_xgb_lgb[1] < ALPHA else 'not significant'}\n")
        if mcn_xgb_best_ens:
            best_ens_name = max(ens_freq.keys(),
                                 key=lambda k: ens_freq[k]["metrics"]["roc_auc"])
            f.write(f"  XGB vs Best Ens ({best_ens_name}): "
                    f"stat={mcn_xgb_best_ens[0]:.3f}  p={mcn_xgb_best_ens[1]:.4f}  "
                    f"{'SIGNIFICANT' if mcn_xgb_best_ens[1] < ALPHA else 'not significant'}\n")

        f.write("\nSeverity — Bootstrap R² Difference (1000 resamples, H1: challenger > XGB)\n")
        f.write(f"  XGB vs CatBoost: mean_diff={boot_xgb_cat[0]:+.4f}  "
                f"p={boot_xgb_cat[1]:.4f}  "
                f"{'SIGNIFICANT' if boot_xgb_cat[1] < ALPHA else 'not significant'}\n")
        f.write(f"  XGB vs LightGBM: mean_diff={boot_xgb_lgb[0]:+.4f}  "
                f"p={boot_xgb_lgb[1]:.4f}  "
                f"{'SIGNIFICANT' if boot_xgb_lgb[1] < ALPHA else 'not significant'}\n\n")

        # Q3: Does ensemble beat all single models?
        f.write("Q3 — DOES ANY ENSEMBLE OUTPERFORM ALL SINGLE MODELS?\n")
        f.write("-" * 50 + "\n")
        best_single_auc = max(xgb_fm["roc_auc"], cat_fm["roc_auc"], lgb_fm["roc_auc"])
        best_single_r2  = max(xgb_sm["r2"],       cat_sm["r2"],       lgb_sm["r2"])
        for ename, eres in ens_freq.items():
            em = eres["metrics"]
            beats = em["roc_auc"] > best_single_auc
            f.write(f"  Freq Ens {ename:<16} AUC={em['roc_auc']:.4f}  "
                    f"vs best single={best_single_auc:.4f}  "
                    f"{'BEATS all singles' if beats else 'below best single'}\n")
        for ename, eres in ens_sev.items():
            em = eres["metrics"]
            beats = em["r2"] > best_single_r2
            f.write(f"  Sev  Ens {ename:<16} R²={em['r2']:.4f}   "
                    f"vs best single={best_single_r2:.4f}  "
                    f"{'BEATS all singles' if beats else 'below best single'}\n")

        # Q4/Q5: Champion and Challenger designation
        f.write("\nQ4 — WHICH MODEL SHOULD BECOME CHAMPION?\n")
        f.write("-" * 50 + "\n")
        # Decision logic: material gain = AUC delta > 0.002 or R² delta > 0.010
        MATERIAL_AUC = 0.002
        MATERIAL_R2  = 0.010
        freq_champ_name = best_freq_name if (best_freq_m["roc_auc"] - xgb_fm["roc_auc"]) > MATERIAL_AUC else "XGBoost"
        sev_champ_name  = best_sev_name  if (best_sev_m["r2"]       - xgb_sm["r2"])       > MATERIAL_R2  else "XGBoost"

        f.write(f"  Frequency champion: {freq_champ_name}\n")
        if freq_champ_name != "XGBoost":
            f.write(f"    AUC gain vs XGB: {best_freq_m['roc_auc']-xgb_fm['roc_auc']:+.4f} "
                    f"(threshold: {MATERIAL_AUC}) — MATERIAL\n")
        else:
            f.write(f"    XGBoost AUC gain vs best challenger: "
                    f"{best_freq_m['roc_auc']-xgb_fm['roc_auc']:+.4f} "
                    f"(below material threshold of {MATERIAL_AUC})\n")
            f.write(f"    XGBoost retains champion status.\n")

        f.write(f"\n  Severity champion: {sev_champ_name}\n")
        if sev_champ_name != "XGBoost":
            f.write(f"    R² gain vs XGB: {best_sev_m['r2']-xgb_sm['r2']:+.4f} "
                    f"(threshold: {MATERIAL_R2}) — MATERIAL\n")
        else:
            f.write(f"    XGBoost R² gain vs best challenger: "
                    f"{best_sev_m['r2']-xgb_sm['r2']:+.4f} "
                    f"(below material threshold of {MATERIAL_R2})\n")
            f.write(f"    XGBoost retains champion status.\n")

        f.write("\nQ5 — WHICH MODELS BECOME CHALLENGERS?\n")
        f.write("-" * 50 + "\n")
        f.write("  All models not designated Champion are assigned Challenger status.\n")
        challengers = set()
        for name, m, _ in all_freq_sorted:
            if name not in (freq_champ_name,):
                challengers.add(name)
        f.write(f"  Challenger models: {', '.join(sorted(challengers))}\n")
        f.write("  These remain available for A/B testing or ensemble deployment.\n\n")

        # Final verdict box
        f.write("=" * 70 + "\n")
        f.write("VERDICT SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Frequency champion:   {freq_champ_name}\n")
        f.write(f"  Severity champion:    {sev_champ_name}\n")
        f.write(f"  Deployment change:    {'YES — promote new champion' if (freq_champ_name != 'XGBoost' or sev_champ_name != 'XGBoost') else 'NO — XGBoost retains champion on both tasks'}\n")
        f.write("=" * 70 + "\n")

    print(f"  Written: champion_challenger_decision.txt")
    return {
        "freq_champion": freq_champ_name,
        "sev_champion": sev_champ_name,
        "mcn_xgb_cat": mcn_xgb_cat,
        "mcn_xgb_lgb": mcn_xgb_lgb,
        "boot_xgb_cat": boot_xgb_cat,
        "boot_xgb_lgb": boot_xgb_lgb,
    }


# ─────────────────────────────────────────────────────────────
# PHASE 7 — PRODUCTION READINESS
# ─────────────────────────────────────────────────────────────
def phase7_production(decision, xgb_fm, xgb_sm, best_freq_m, best_sev_m,
                       best_freq_name, best_sev_name, y_sev_te):
    print("\n--- PHASE 7: Production Readiness ---")

    freq_champ = decision["freq_champion"]
    sev_champ  = decision["sev_champion"]
    any_change = (freq_champ != "XGBoost" or sev_champ != "XGBoost")

    path = os.path.join(REPORT_DIR, "production_impact_assessment.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("PRODUCTION IMPACT ASSESSMENT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if not any_change:
            f.write("OUTCOME: No change in champion model.\n")
            f.write("XGBoost retains champion status on both frequency and severity.\n\n")
            f.write("No champion promotion is triggered.\n")
            f.write("Challenger models (CatBoost, LightGBM, Ensembles) are available\n")
            f.write("for A/B testing or ensemble deployment if business requirements change.\n\n")

            f.write("CHALLENGER OPPORTUNITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write("Best observed frequency AUC: "
                    f"{best_freq_m['roc_auc']:.4f} ({best_freq_name})\n")
            f.write("XGBoost frequency AUC:       "
                    f"{xgb_fm['roc_auc']:.4f}\n")
            f.write("Gap:                         "
                    f"{best_freq_m['roc_auc']-xgb_fm['roc_auc']:+.4f}\n\n")
            f.write("Best observed severity R²:   "
                    f"{best_sev_m['r2']:.4f} ({best_sev_name})\n")
            f.write("XGBoost severity R²:         "
                    f"{xgb_sm['r2']:.4f}\n")
            f.write("Gap:                         "
                    f"{best_sev_m['r2']-xgb_sm['r2']:+.4f}\n\n")

            f.write("RECOMMENDATION\n")
            f.write("-" * 40 + "\n")
            f.write("Continue with XGBoost as champion.\n")
            f.write("Revisit challenger promotion if:\n")
            f.write("  a) More data is collected (>100k rows)\n")
            f.write("  b) New high-signal features are added\n")
            f.write("  c) Calibration requirements change (CatBoost is better calibrated)\n")
            f.write("  d) Inference latency becomes a constraint (LightGBM is fastest)\n")

        else:
            f.write("OUTCOME: Champion promotion triggered.\n")
            if freq_champ != "XGBoost":
                f.write(f"  New frequency champion: {freq_champ}\n")
            if sev_champ != "XGBoost":
                f.write(f"  New severity champion:  {sev_champ}\n")

            f.write("\nPREMIUM IMPACT ESTIMATE\n")
            f.write("-" * 40 + "\n")
            auc_gain = best_freq_m["roc_auc"] - xgb_fm["roc_auc"]
            # Rough conversion: each 0.01 AUC gain ≈ 0.5-1.0% better risk stratification
            prem_impact_pct = auc_gain * 50  # conservative
            f.write(f"  AUC gain: {auc_gain:+.4f}\n")
            f.write(f"  Estimated premium stratification improvement: ~{prem_impact_pct:.1f}%\n")
            f.write("  (Rule of thumb: 0.01 AUC ≈ 0.5% better risk segmentation)\n\n")

            f.write("CLAIM PREDICTION IMPROVEMENT\n")
            f.write("-" * 40 + "\n")
            r2_gain = best_sev_m["r2"] - xgb_sm["r2"]
            f.write(f"  R² gain: {r2_gain:+.4f}\n")
            avg_claim = float(y_sev_te.mean())
            rmse_gain = xgb_sm["rmse"] - best_sev_m["rmse"]
            f.write(f"  RMSE reduction: INR {rmse_gain:,.0f}\n")
            f.write(f"  Mean claim amount: INR {avg_claim:,.0f}\n")
            f.write(f"  Relative RMSE reduction: {rmse_gain/avg_claim*100:.1f}%\n\n")

            f.write("OPERATIONAL COMPLEXITY\n")
            f.write("-" * 40 + "\n")
            comp = {"CatBoost":"MEDIUM — requires catboost library, no native JSON save",
                    "LightGBM":"LOW — similar to XGBoost, .txt save format",
                    "Ens XGB+CAT":"HIGH — two inference calls, two model files",
                    "Ens XGB+LGB":"HIGH — two inference calls",
                    "Ens XGB+CAT+LGB":"VERY HIGH — three model files, three inference calls"}
            f.write(f"  New champion complexity: "
                    f"{comp.get(freq_champ, 'MEDIUM')}\n\n")

            f.write("DEPLOYMENT CHECKLIST\n")
            f.write("-" * 40 + "\n")
            f.write("  [ ] Save challenger model to experiments/outputs/models/\n")
            f.write("  [ ] Update InsurancePreprocessorV4 if needed\n")
            f.write("  [ ] Update /predict endpoint to load new model file\n")
            f.write("  [ ] A/B test: route 10% traffic to new champion\n")
            f.write("  [ ] Monitor claim rate and loss ratio for 30 days\n")
            f.write("  [ ] Conduct model governance sign-off\n")
            f.write("  [ ] IRDAI product filing amendment if pricing changes materially\n")
            f.write("\n  NOTE: This assessment is for dissertation planning only.\n")
            f.write("  DO NOT deploy without supervisor sign-off.\n")

    print(f"  Written: production_impact_assessment.txt")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
def final_summary(xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
                   ens_freq, ens_sev, shap_freq_df, shap_sev_df,
                   decision, all_freq_sorted, all_sev_sorted,
                   elapsed_total):
    print("\n--- Final Summary ---")
    path = os.path.join(REPORT_DIR, "champion_challenger_final_summary.txt")

    freq_champ = decision["freq_champion"]
    sev_champ  = decision["sev_champion"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("CHAMPION-CHALLENGER EXPERIMENT — FINAL SUMMARY\n")
        f.write("Insurance AI Pricing System — Dataset V4\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 72 + "\n\n")

        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Dataset:     car_insurance_portfolio_v4.csv (50,000 rows)\n")
        f.write(f"  Split:       80/20 stratified, seed={RANDOM_SEED}\n")
        f.write(f"  Optuna:      {N_OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV per model\n")
        f.write(f"  Models:      XGBoost / CatBoost / LightGBM + 3 ensembles\n")
        f.write(f"  Runtime:     {elapsed_total:.0f}s\n\n")

        # 1. Baseline
        f.write("1. BASELINE V4 METRICS (XGBoost Champion)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Frequency:  AUC={xgb_fm['roc_auc']:.4f}  "
                f"PR-AUC={xgb_fm['pr_auc']:.4f}  F1={xgb_fm['f1']:.4f}  "
                f"Prec={xgb_fm['precision']:.4f}  Rec={xgb_fm['recall']:.4f}\n")
        f.write(f"  Severity:   R²={xgb_sm['r2']:.4f}  "
                f"RMSE=INR {xgb_sm['rmse']:,.0f}  MAE=INR {xgb_sm['mae']:,.0f}\n\n")

        # 2. Challenger metrics
        f.write("2. CHALLENGER METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("  FREQUENCY:\n")
        for name, m, _ in all_freq_sorted:
            tag = " [champion]" if name == freq_champ else ""
            f.write(f"    {name:<22} AUC={m['roc_auc']:.4f}  "
                    f"F1={m['f1']:.4f}  Rec={m['recall']:.4f}{tag}\n")
        f.write("  SEVERITY:\n")
        for name, m, _ in all_sev_sorted:
            tag = " [champion]" if name == sev_champ else ""
            f.write(f"    {name:<22} R²={m['r2']:.4f}  "
                    f"RMSE=INR {m['rmse']:,.0f}{tag}\n")

        # 3. Ensemble metrics
        f.write("\n3. ENSEMBLE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("  FREQUENCY:\n")
        for ename, eres in ens_freq.items():
            m = eres["metrics"]
            f.write(f"    {ename:<20} AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}\n")
        f.write("  SEVERITY:\n")
        for ename, eres in ens_sev.items():
            m = eres["metrics"]
            f.write(f"    {ename:<20} R²={m['r2']:.4f}  RMSE=INR {m['rmse']:,.0f}\n")

        # 4. SHAP findings
        f.write("\n4. SHAP FINDINGS\n")
        f.write("-" * 40 + "\n")
        if shap_freq_df is not None:
            f.write("  Frequency top 5 features:\n")
            for i, row in shap_freq_df.head(5).iterrows():
                f.write(f"    {i+1}. {row['feature']:<35} {row['mean_abs_shap']:.4f}\n")
        if shap_sev_df is not None:
            f.write("  Severity top 5 features:\n")
            for i, row in shap_sev_df.head(5).iterrows():
                f.write(f"    {i+1}. {row['feature']:<35} {row['mean_abs_shap']:.2f}\n")

        # 5. Statistical significance
        f.write("\n5. STATISTICAL SIGNIFICANCE\n")
        f.write("-" * 40 + "\n")
        ALPHA = 0.05
        f.write("  McNemar's test (frequency classifiers, alpha=0.05):\n")
        f.write(f"    XGB vs CatBoost: p={decision['mcn_xgb_cat'][1]:.4f}  "
                f"{'significant' if decision['mcn_xgb_cat'][1] < ALPHA else 'NOT significant'}\n")
        f.write(f"    XGB vs LightGBM: p={decision['mcn_xgb_lgb'][1]:.4f}  "
                f"{'significant' if decision['mcn_xgb_lgb'][1] < ALPHA else 'NOT significant'}\n")
        f.write("  Bootstrap R² test (severity, 1000 resamples, alpha=0.05):\n")
        f.write(f"    XGB vs CatBoost: mean_diff={decision['boot_xgb_cat'][0]:+.4f}  "
                f"p={decision['boot_xgb_cat'][1]:.4f}  "
                f"{'significant' if decision['boot_xgb_cat'][1] < ALPHA else 'NOT significant'}\n")
        f.write(f"    XGB vs LightGBM: mean_diff={decision['boot_xgb_lgb'][0]:+.4f}  "
                f"p={decision['boot_xgb_lgb'][1]:.4f}  "
                f"{'significant' if decision['boot_xgb_lgb'][1] < ALPHA else 'NOT significant'}\n")

        # 6. Champion recommendation
        f.write("\n6. CHAMPION MODEL RECOMMENDATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Frequency champion: {freq_champ}\n")
        f.write(f"  Severity champion:  {sev_champ}\n")
        no_change = (freq_champ == "XGBoost" and sev_champ == "XGBoost")
        if no_change:
            f.write("  Verdict: XGBoost RETAINS champion on both tasks.\n")
            f.write("  No model promotion warranted at this time.\n")
        else:
            f.write("  Verdict: Champion promotion recommended.\n")

        # 7. Deployment recommendation
        f.write("\n7. DEPLOYMENT RECOMMENDATION\n")
        f.write("-" * 40 + "\n")
        if no_change:
            f.write("  RETAIN XGBoost in production.\n")
            f.write("  Register CatBoost and LightGBM as monitored challengers.\n")
            f.write("  Monitor for regression in loss ratio over next 90 days.\n")
            f.write("  Revisit champion promotion if >100k V4-schema policies accumulate.\n")
        else:
            f.write(f"  PROMOTE {freq_champ} (frequency) and {sev_champ} (severity).\n")
            f.write("  XGBoost demoted to challenger (shadow deployment recommended).\n")
            f.write("  Run A/B test for 30 days before full cutover.\n")

        f.write(f"\n  CRITICAL REMINDER:\n")
        f.write(f"  This experiment is isolated to experiments/. No production\n")
        f.write(f"  models were overwritten. Models at experiments/outputs/models/\n")
        f.write(f"  are NOT deployed. Do NOT promote without governance sign-off.\n")

        f.write("\n\nOUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        for fn in [
            "baseline_v4_metrics.txt",
            "challenger_training_report.txt",
            "ensemble_report.txt",
            "model_comparison_report.txt",
            "best_model_shap_report.txt",
            "champion_challenger_decision.txt",
            "production_impact_assessment.txt",
            "champion_challenger_final_summary.txt",
        ]:
            f.write(f"  experiments/outputs/reports/{fn}\n")

    print(f"  Written: champion_challenger_final_summary.txt")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 60)
    print("CHAMPION-CHALLENGER EXPERIMENT — Dataset V4")
    print("=" * 60)

    (X_train, X_test, y_train, y_test,
     X_sev_tr, X_sev_te, y_sev_tr, y_sev_te,
     feat_names, df, idx_test, claim_te) = load_and_split()

    # Phase 1
    xgb_fm, xgb_sm, xgb_freq_model, xgb_sev_model, xgb_prob, xgb_sev_pred = \
        phase1_baseline(X_train, X_test, y_train, y_test,
                        X_sev_tr, X_sev_te, y_sev_tr, y_sev_te)

    # Phase 2
    ch_results = phase2_challengers(
        X_train, X_test, y_train, y_test,
        X_sev_tr, X_sev_te, y_sev_tr, y_sev_te,
        xgb_fm, xgb_sm
    )
    cat_freq_model = ch_results["catboost_freq"]["model"]
    lgb_freq_model = ch_results["lgbm_freq"]["model"]
    cat_sev_model  = ch_results["catboost_sev"]["model"]
    lgb_sev_model  = ch_results["lgbm_sev"]["model"]

    cat_prob     = ch_results["catboost_freq"]["prob"]
    lgb_prob     = ch_results["lgbm_freq"]["prob"]
    cat_sev_pred = ch_results["catboost_sev"]["pred"]
    lgb_sev_pred = ch_results["lgbm_sev"]["pred"]

    cat_fm = ch_results["catboost_freq"]["metrics"]
    lgb_fm = ch_results["lgbm_freq"]["metrics"]
    cat_sm = ch_results["catboost_sev"]["metrics"]
    lgb_sm = ch_results["lgbm_sev"]["metrics"]

    # Phase 3
    ens_freq, ens_sev = phase3_ensembles(
        y_test, y_sev_te,
        xgb_prob, cat_prob, lgb_prob,
        xgb_sev_pred, cat_sev_pred, lgb_sev_pred,
        xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm
    )

    # Phase 4
    (all_freq_sorted, all_sev_sorted,
     best_freq_name, best_freq_m, best_freq_prob,
     best_sev_name,  best_sev_m,  best_sev_pred) = phase4_comparison(
        y_test, y_sev_te,
        xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
        ens_freq, ens_sev,
        xgb_prob, cat_prob, lgb_prob,
        xgb_sev_pred, cat_sev_pred, lgb_sev_pred
    )

    # Resolve best model objects for SHAP
    model_map_freq = {
        "XGBoost": xgb_freq_model,
        "CatBoost": cat_freq_model,
        "LightGBM": lgb_freq_model,
    }
    model_map_sev = {
        "XGBoost": xgb_sev_model,
        "CatBoost": cat_sev_model,
        "LightGBM": lgb_sev_model,
    }
    # For ensemble best, fall back to XGB for SHAP (no unified SHAP for ensembles)
    shap_freq_model = model_map_freq.get(best_freq_name, xgb_freq_model)
    shap_sev_model  = model_map_sev.get(best_sev_name,  xgb_sev_model)

    # Phase 5
    shap_freq_df, shap_sev_df = phase5_shap(
        best_freq_name, shap_freq_model,
        best_sev_name,  shap_sev_model,
        X_test, X_sev_te, feat_names,
        all_freq_sorted, all_sev_sorted,
        xgb_freq_model, xgb_sev_model,
        cat_freq_model, lgb_freq_model
    )

    # Phase 6
    decision = phase6_decision(
        y_test, y_sev_te,
        xgb_prob, cat_prob, lgb_prob,
        xgb_sev_pred, cat_sev_pred, lgb_sev_pred,
        xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
        ens_freq, ens_sev,
        all_freq_sorted, all_sev_sorted,
        best_freq_name, best_freq_m, best_freq_prob,
        best_sev_name,  best_sev_m,  best_sev_pred
    )

    # Phase 7
    phase7_production(
        decision, xgb_fm, xgb_sm, best_freq_m, best_sev_m,
        best_freq_name, best_sev_name, y_sev_te
    )

    # Final summary
    elapsed = time.time() - t_total
    final_summary(
        xgb_fm, xgb_sm, cat_fm, cat_sm, lgb_fm, lgb_sm,
        ens_freq, ens_sev, shap_freq_df, shap_sev_df,
        decision, all_freq_sorted, all_sev_sorted, elapsed
    )

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"\nFREQUENCY:  XGB={xgb_fm['roc_auc']:.4f}  CAT={cat_fm['roc_auc']:.4f}  LGB={lgb_fm['roc_auc']:.4f}")
    print(f"SEVERITY:   XGB={xgb_sm['r2']:.4f}    CAT={cat_sm['r2']:.4f}    LGB={lgb_sm['r2']:.4f}")
    print(f"Champion:   Freq={decision['freq_champion']}  Sev={decision['sev_champion']}")
    print(f"\nAll reports in experiments/outputs/reports/")
    print(f"No production files were modified.")


if __name__ == "__main__":
    main()
