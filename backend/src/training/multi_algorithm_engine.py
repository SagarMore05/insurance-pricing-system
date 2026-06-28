"""
Enterprise Automatic Multi-Algorithm Model Selection Engine.

Pipeline:
  Load Dataset → Validate → Preprocess → Feature Engineering →
  Train XGBoost / LightGBM / CatBoost (frequency) →
  Train XGBoost / LightGBM / CatBoost (severity) →
  Compare Metrics → Select Winner → Generate SHAP → Serialize →
  Register → Update Dashboard → Generate Reports

Winner selection (frequency):
  1. Highest ROC-AUC  2. Lowest Brier  3. Highest F1
  4. Highest Precision  5. Highest Recall
  Tie-breaker order: XGBoost → LightGBM → CatBoost

Winner selection (severity):
  1. Highest R²  2. Lowest RMSE  3. Lowest MAE  4. Lowest MAPE
  Tie-breaker order: XGBoost → LightGBM → CatBoost
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import shutil
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.pipeline import InsurancePreprocessor
from src.training.dataset_schema import (
    MASTER_DATASET_PATH as _SCHEMA_DATASET_PATH,
    log_dataset_info,
    normalize_columns,
    validate_columns,
    TRAINING_COLS,
)
from src.training.metrics import compute_frequency_metrics, compute_severity_metrics
from src.training.trainers.catboost_trainer import CatBoostTrainer
from src.training.trainers.lightgbm_trainer import LightGBMTrainer
from src.training.trainers.xgboost_trainer import XGBoostTrainer

logger = logging.getLogger("insurance_api.multi_algorithm_engine")

# ── Paths ─────────────────────────────────────────────────────────────────────

_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_DATASET_PATH  = _BACKEND_ROOT / "data" / "master" / "motor_insurance_master_dataset_50000.csv"
_SAVED_DIR     = _BACKEND_ROOT / "models" / "saved"
_RUNS_DIR      = _BACKEND_ROOT / "models" / "training_runs"
_REGISTRY_DIR  = _BACKEND_ROOT / "models" / "registry"
_META_DIR      = _BACKEND_ROOT / "models" / "metadata"
_HISTORY_PATH  = _REGISTRY_DIR / "training_history.json"
_REGISTRY_PATH = _REGISTRY_DIR / "model_registry.json"
_CHAMPION_PATH = _META_DIR / "champion_registry.json"

DATASET_NAME   = "motor_insurance_master_dataset_50000.csv"
ALGORITHMS     = ["xgboost", "lightgbm", "catboost"]
TIE_BREAKER    = {"xgboost": 0, "lightgbm": 1, "catboost": 2}
TOLERANCE      = 0.001   # statistical tie threshold


# ── Main Engine ───────────────────────────────────────────────────────────────

class MultiAlgorithmEngine:
    """
    Orchestrates full multi-algorithm training on motor_insurance_master_dataset_50000.csv.
    Produces trained artifacts for all 6 models (3 freq + 3 sev), selects winners,
    generates SHAP, and updates both registries.
    """

    TRAINER_MAP = {
        "xgboost":  XGBoostTrainer,
        "catboost":  CatBoostTrainer,
        "lightgbm": LightGBMTrainer,
    }

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        test_size: float = 0.2,
        val_fraction: float = 0.5,
        random_state: int = 42,
        shap_sample_size: int = 500,
    ) -> None:
        self.dataset_path   = pathlib.Path(dataset_path) if dataset_path else _DATASET_PATH
        self.test_size      = test_size
        self.val_fraction   = val_fraction
        self.random_state   = random_state
        self.shap_sample_size = shap_sample_size

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, dataset_df: Optional[pd.DataFrame] = None, dataset_label: Optional[str] = None) -> Dict[str, Any]:
        """Execute the complete pipeline. Returns the training run record.

        Parameters
        ----------
        dataset_df : pd.DataFrame, optional
            Pre-loaded DataFrame (e.g., master + production merged). When provided the
            CSV file is NOT loaded; the DataFrame is validated and used directly.
            Columns must follow the master-dataset schema:
            vehicle_brand, previous_claims, actual_claim_amount_inr, etc.
        dataset_label : str, optional
            Human-readable dataset name stored in run records (e.g.
            "master_50000+120_production"). Defaults to DATASET_NAME.
        """
        run_id     = self._generate_run_id()
        run_number = self._next_run_number()
        run_dir    = _RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== MultiAlgorithmEngine — Run %s (run #%d) ===", run_id, run_number)

        start_total = time.perf_counter()

        # 1. Load & validate
        if dataset_df is not None:
            df = dataset_df.copy()
            dataset_hash = hashlib.md5(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest()
            self._validate_dataframe(df)
            used_label = dataset_label or f"merged_dataset_{len(df)}_rows"
            logger.info("Using pre-loaded DataFrame: %d rows, hash %s...", len(df), dataset_hash[:8])
        else:
            df, dataset_hash = self._load_and_validate()
            used_label = dataset_label or DATASET_NAME
            logger.info("Dataset loaded: %d rows, hash %s...", len(df), dataset_hash[:8])

        # 2. Preprocess
        X, y_freq, y_sev_full, feature_names, preprocessor = self._preprocess(df)
        logger.info("Preprocessed: X.shape=%s, %d features", X.shape, len(feature_names))

        # 3. Train/val/test split (stratified on frequency label)
        (X_tr, X_va, X_te,
         yf_tr, yf_va, yf_te,
         ys_tr, ys_va, ys_te) = self._split(X, y_freq, y_sev_full)

        split_info = {
            "n_train": len(X_tr),
            "n_val":   len(X_va),
            "n_test":  len(X_te),
            "n_sev_train": int((yf_tr == 1).sum()),
            "n_sev_val":   int((yf_va == 1).sum()),
            "n_sev_test":  int((yf_te == 1).sum()),
        }
        logger.info("Split: train=%d, val=%d, test=%d", split_info["n_train"], split_info["n_val"], split_info["n_test"])

        # 4. Train all frequency models
        freq_dir = run_dir / "frequency"
        freq_dir.mkdir(exist_ok=True)
        freq_results = self._train_all(
            "frequency", X_tr, yf_tr, X_va, yf_va, X_te, yf_te, feature_names, freq_dir
        )

        # 5. Select frequency winner
        freq_winner, freq_reason = self._select_frequency_winner(freq_results)
        freq_results[freq_winner]["is_winner"] = True
        logger.info("Frequency winner: %s — %s", freq_winner, freq_reason)

        # 6. SHAP for frequency winner
        freq_shap_meta = self._generate_shap(
            freq_results[freq_winner]["trainer"], X_te, feature_names, "frequency", freq_dir
        )

        # 7. Train all severity models (claimants only in each split)
        sev_dir = run_dir / "severity"
        sev_dir.mkdir(exist_ok=True)

        X_tr_s = X_tr[yf_tr == 1];  ys_tr_s = ys_tr[yf_tr == 1]
        X_va_s = X_va[yf_va == 1];  ys_va_s = ys_va[yf_va == 1]
        X_te_s = X_te[yf_te == 1];  ys_te_s = ys_te[yf_te == 1]

        # Winsorize training and validation severity targets at the 99th percentile
        # to reduce the pull of extreme outliers on tree splits.
        # Cap is derived from the training set only (no data leakage).
        # Test targets are kept raw so evaluation reflects real-world claim values.
        _sev_pos = ys_tr_s[ys_tr_s > 0]
        _sev_cap = float(np.percentile(_sev_pos, 99)) if len(_sev_pos) > 0 else float(ys_tr_s.max())
        ys_tr_s_w = np.clip(ys_tr_s, 0.0, _sev_cap)
        ys_va_s_w = np.clip(ys_va_s, 0.0, _sev_cap)
        logger.info(
            "[Severity] Winsorize cap=₹%.0f (99th pct, train claimants); "
            "clipped %d/%d train, %d/%d val severity values",
            _sev_cap,
            int((ys_tr_s > _sev_cap).sum()), len(ys_tr_s),
            int((ys_va_s > _sev_cap).sum()), len(ys_va_s),
        )

        sev_results = self._train_all(
            "severity", X_tr_s, ys_tr_s_w, X_va_s, ys_va_s_w, X_te_s, ys_te_s, feature_names, sev_dir
        )

        # 8. Select severity winner
        sev_winner, sev_reason = self._select_severity_winner(sev_results)
        sev_results[sev_winner]["is_winner"] = True
        logger.info("Severity winner: %s — %s", sev_winner, sev_reason)

        # 9. SHAP for severity winner
        sev_shap_meta = self._generate_shap(
            sev_results[sev_winner]["trainer"], X_te_s, feature_names, "severity", sev_dir
        )

        # 10. Save winner models to production paths
        freq_prod_path = self._save_production(freq_results[freq_winner]["trainer"], "frequency")
        sev_prod_path  = self._save_production(sev_results[sev_winner]["trainer"],  "severity")

        # 11. Save preprocessor (refit on full training set for production)
        prep_path = str(_SAVED_DIR / "preprocessor.pkl")
        joblib.dump(preprocessor, prep_path)

        # 12. Update registries
        version = f"v{run_number}.0.0"
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        total_rows = split_info["n_train"] + split_info["n_val"] + split_info["n_test"]

        self._update_champion_registry(
            run_id, version, ts, freq_winner, sev_winner,
            freq_results, sev_results, feature_names,
            freq_prod_path, sev_prod_path, total_rows, used_label,
        )
        self._update_model_registry(
            run_id, version, ts, freq_winner, sev_winner,
            freq_results, sev_results, feature_names,
            freq_prod_path, sev_prod_path, split_info,
            dataset_hash, used_label,
        )

        # 13. Build & persist run record
        elapsed = round(time.perf_counter() - start_total, 2)
        record  = self._build_run_record(
            run_id, run_number, version, ts, dataset_hash,
            freq_results, sev_results, freq_winner, sev_winner,
            freq_reason, sev_reason, freq_shap_meta, sev_shap_meta,
            freq_prod_path, sev_prod_path, split_info, elapsed, used_label,
        )
        self._append_history(record)
        self._write_run_metadata(run_dir, record)

        # 14. Hot-reload predictor (in-process, best-effort)
        self._try_hot_reload()

        logger.info("=== Run %s complete in %.1fs ===", run_id, elapsed)
        return record

    # ── Step helpers ──────────────────────────────────────────────────────────

    # Required columns for the master dataset (motor_insurance_master_dataset_50000.csv)
    _MASTER_REQUIRED_COLS = [
        "age", "gender", "city", "vehicle_brand", "vehicle_segment",
        "fuel_type", "vehicle_age_years", "vehicle_value_inr",
        "annual_mileage_km", "driving_score", "previous_claims",
        "years_licensed", "annual_income_inr", "claim_occurred", "actual_claim_amount_inr",
    ]

    # Minimum subset required when training on pre-built DataFrames (merged datasets)
    _DATAFRAME_REQUIRED_COLS = [
        "age", "gender", "city",
        "vehicle_age_years", "vehicle_value_inr",
        "annual_mileage_km", "driving_score", "years_licensed",
        "claim_occurred", "actual_claim_amount_inr",
    ]

    def _load_and_validate(self) -> Tuple[pd.DataFrame, str]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Master dataset not found at {self.dataset_path}. "
                "Expected: data/master/motor_insurance_master_dataset_50000.csv"
            )
        logger.info("[Master Dataset Loaded] Path: %s", self.dataset_path)
        df = pd.read_csv(self.dataset_path)

        # Normalize any column aliases before validation
        df = normalize_columns(df)
        logger.info("[Schema Validated] Columns after normalization: %d", len(df.columns))

        # Validate using canonical schema
        validate_columns(df, self._MASTER_REQUIRED_COLS, context="MasterDataset")

        if len(df) < 1000:
            raise ValueError(f"Dataset too small: {len(df)} rows (minimum 1000)")

        log_dataset_info(df, label="Master Dataset")

        dataset_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

        return df, dataset_hash

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate a pre-loaded DataFrame (used when dataset_df is passed to run())."""
        df_norm = normalize_columns(df)
        missing = [c for c in self._DATAFRAME_REQUIRED_COLS if c not in df_norm.columns]
        if missing:
            validate_columns(df_norm, self._DATAFRAME_REQUIRED_COLS, context="MergedDataFrame")
        if len(df) < 1000:
            raise ValueError(f"DataFrame too small: {len(df)} rows (minimum 1000)")
        log_dataset_info(df_norm, label="Merged Dataset")

    def _preprocess(self, df: pd.DataFrame):
        # Normalize aliases so preprocessor receives consistent column names.
        # InsurancePreprocessor.FeatureEngineer also handles this internally;
        # normalizing here makes column names explicit before fitting.
        df_norm = normalize_columns(df, silent=True)

        preprocessor = InsurancePreprocessor()
        y_freq     = df_norm["claim_occurred"].fillna(0).values.astype(int)
        y_sev_full = df_norm["actual_claim_amount_inr"].fillna(0).values.astype(float)

        logger.info("[Columns Normalized] Training on %d rows with %d columns", len(df_norm), len(df_norm.columns))
        X = preprocessor.fit_transform(df_norm)
        feature_names = self._get_feature_names(preprocessor, X)
        logger.info("[Preprocessing Complete] Feature matrix: %s", str(X.shape))
        return X, y_freq, y_sev_full, feature_names, preprocessor

    def _get_feature_names(self, preprocessor: InsurancePreprocessor, X: np.ndarray) -> List[str]:
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass
        try:
            ct = preprocessor.pipeline.named_steps.get("column_transformer")
            if ct is not None:
                return [n.split("__")[-1] for n in ct.get_feature_names_out()]
        except Exception:
            pass
        return [f"feature_{i}" for i in range(X.shape[1])]

    def _split(self, X, y_freq, y_sev_full):
        n = len(X)
        indices = np.arange(n)
        train_idx, temp_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state,
            stratify=y_freq,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=self.val_fraction, random_state=self.random_state,
            stratify=y_freq[temp_idx],
        )
        return (
            X[train_idx], X[val_idx], X[test_idx],
            y_freq[train_idx], y_freq[val_idx], y_freq[test_idx],
            y_sev_full[train_idx], y_sev_full[val_idx], y_sev_full[test_idx],
        )

    def _train_all(
        self,
        model_type: str,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        feature_names: List[str],
        out_dir: pathlib.Path,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for algo in ALGORITHMS:
            logger.info("Training %s / %s ...", model_type, algo)
            trainer_cls = self.TRAINER_MAP[algo]
            trainer     = trainer_cls(model_type=model_type, feature_names=feature_names)
            result      = self._train_one(trainer, model_type, X_tr, y_tr, X_va, y_va, X_te, y_te)
            results[algo] = result

            # Save trainer artifact
            algo_dir = out_dir / algo
            algo_dir.mkdir(exist_ok=True)
            try:
                trainer.save(str(algo_dir))
                result["artifact_path"] = str(algo_dir / "model.pkl")
            except Exception as exc:
                logger.warning("Failed to save %s/%s trainer: %s", model_type, algo, exc)

        return results

    def _train_one(
        self,
        trainer,
        model_type: str,
        X_tr, y_tr, X_va, y_va, X_te, y_te,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "algorithm":  trainer.ALGORITHM,
            "model_type": model_type,
            "status":     "failed",
            "is_winner":  False,
            "metrics":    {},
            "training_time_sec": None,
            "inference_time_ms": None,
            "model_size_mb":     None,
            "artifact_path":     None,
            "trainer":    None,
            "error":      None,
        }
        try:
            t0 = time.perf_counter()
            trainer.train(X_tr, y_tr)
            result["training_time_sec"] = round(time.perf_counter() - t0, 3)

            if model_type == "frequency":
                y_prob = trainer.predict_proba(X_te)
                result["metrics"] = compute_frequency_metrics(y_te, y_prob)
            else:
                y_pred = trainer.predict(X_te)
                result["metrics"] = compute_severity_metrics(y_te, y_pred)

                # 3-fold cross-validated R² on the training set for robust model selection.
                # CV is performed on the winsorized training targets (same scale as model training).
                try:
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(
                        trainer._model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1
                    )
                    result["metrics"]["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
                    result["metrics"]["cv_r2_std"]  = round(float(cv_scores.std()), 4)
                    logger.info(
                        "[Severity CV] %s: R²=%.4f ± %.4f",
                        trainer.ALGORITHM, cv_scores.mean(), cv_scores.std(),
                    )
                except Exception as _cv_exc:
                    logger.warning(
                        "Severity CV R² failed for %s: %s", trainer.ALGORITHM, _cv_exc
                    )
                    result["metrics"]["cv_r2_mean"] = None
                    result["metrics"]["cv_r2_std"]  = None

            # Inference time on val set (or capped subset)
            n_inf = min(500, len(X_va))
            t_inf = time.perf_counter()
            if model_type == "frequency":
                trainer.predict_proba(X_va[:n_inf])
            else:
                trainer.predict(X_va[:n_inf])
            inf_ms = (time.perf_counter() - t_inf) * 1000 / n_inf
            result["inference_time_ms"] = round(inf_ms, 4)

            result["status"]  = "success"
            result["trainer"] = trainer

        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Training failed for %s/%s: %s", model_type, trainer.ALGORITHM, exc)
            logger.debug(traceback.format_exc())

        return result

    # ── Winner selection ──────────────────────────────────────────────────────

    def _select_frequency_winner(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, str]:
        successful = {k: v for k, v in results.items() if v["status"] == "success"}
        if not successful:
            raise RuntimeError("All frequency models failed training.")

        def score(algo: str) -> tuple:
            m = successful[algo]["metrics"]
            return (
                m.get("roc_auc",    -999) or -999,
                -(m.get("brier_score", 999) or 999),
                m.get("f1",         -999) or -999,
                m.get("precision",  -999) or -999,
                m.get("recall",     -999) or -999,
                -TIE_BREAKER[algo],
            )

        winner = max(successful.keys(), key=score)
        m = successful[winner]["metrics"]

        # Build human-readable reason
        others = [a for a in successful if a != winner]
        reasons = []
        auc_w = m.get("roc_auc", 0) or 0
        for other in others:
            m_o = successful[other]["metrics"]
            auc_o = m_o.get("roc_auc", 0) or 0
            if auc_w - auc_o > TOLERANCE:
                reasons.append(f"ROC-AUC {auc_w:.4f} > {other} {auc_o:.4f}")

        if not reasons:
            reasons.append(f"ROC-AUC {auc_w:.4f}, Brier {m.get('brier_score', 0):.4f}, F1 {m.get('f1', 0):.4f}")

        reason = f"{winner.upper()} selected: " + "; ".join(reasons)
        return winner, reason

    def _select_severity_winner(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, str]:
        successful = {k: v for k, v in results.items() if v["status"] == "success"}
        if not successful:
            raise RuntimeError("All severity models failed training.")

        # Prefer cross-validated R² (more robust against lucky/unlucky test splits).
        # Fall back to held-out test R² when CV scores were not computed.
        _any_cv = any(
            successful[a]["metrics"].get("cv_r2_mean") is not None for a in successful
        )

        def score(algo: str) -> tuple:
            m = successful[algo]["metrics"]
            cv_r2   = m.get("cv_r2_mean")
            test_r2 = m.get("r2", -999) or -999
            primary = cv_r2 if (_any_cv and cv_r2 is not None) else test_r2
            return (
                primary,
                test_r2,
                -(m.get("rmse",     999) or 999),
                -(m.get("mae",      999) or 999),
                -(m.get("mape_pct", 999) or 999),
                -TIE_BREAKER[algo],
            )

        winner = max(successful.keys(), key=score)
        m = successful[winner]["metrics"]

        cv_r2_w = m.get("cv_r2_mean")
        r2_w    = m.get("r2", -999) or -999
        metric_label = "CV-R²" if cv_r2_w is not None else "R²"
        primary_w    = cv_r2_w if cv_r2_w is not None else r2_w

        reasons = []
        for other in [a for a in successful if a != winner]:
            m_o    = successful[other]["metrics"]
            cv_r2_o = m_o.get("cv_r2_mean")
            primary_o = cv_r2_o if (_any_cv and cv_r2_o is not None) else (m_o.get("r2", -999) or -999)
            if primary_w - primary_o > TOLERANCE:
                reasons.append(f"{metric_label} {primary_w:.4f} > {other} {primary_o:.4f}")

        if not reasons:
            if cv_r2_w is not None:
                reasons.append(
                    f"CV-R² {cv_r2_w:.4f} (±{m.get('cv_r2_std', 0):.4f}), "
                    f"test R² {r2_w:.4f}, RMSE ₹{m.get('rmse', 0):.0f}"
                )
            else:
                reasons.append(f"R² {r2_w:.4f}, RMSE ₹{m.get('rmse', 0):.0f}, MAE ₹{m.get('mae', 0):.0f}")

        reason = f"{winner.upper()} selected: " + "; ".join(reasons)
        return winner, reason

    # ── SHAP ──────────────────────────────────────────────────────────────────

    def _generate_shap(
        self,
        trainer,
        X_test: np.ndarray,
        feature_names: List[str],
        model_type: str,
        out_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            X_sample = X_test[: self.shap_sample_size]
            raw_model = trainer._model

            explainer   = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_importance = np.abs(shap_values).mean(axis=0).tolist()
            importance_dict = dict(sorted(
                zip(feature_names, mean_importance),
                key=lambda x: -x[1],
            ))

            # Bar plot
            bar_path = str(out_dir / "shap_importance.png")
            top_n = min(20, len(feature_names))
            top_names  = list(importance_dict.keys())[:top_n]
            top_values = list(importance_dict.values())[:top_n]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_names[::-1], top_values[::-1], color="#4f46e5")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{model_type.capitalize()} SHAP Feature Importance ({trainer.ALGORITHM})")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=120, bbox_inches="tight")
            plt.close()

            # Persist importance JSON
            imp_json_path = str(out_dir / "shap_importance.json")
            with open(imp_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "algorithm": trainer.ALGORITHM,
                    "model_type": model_type,
                    "n_samples": len(X_sample),
                    "expected_value": float(explainer.expected_value)
                        if not isinstance(explainer.expected_value, (list, np.ndarray))
                        else float(explainer.expected_value[1]),
                    "feature_importance": importance_dict,
                }, f, indent=2)

            logger.info("SHAP generated for %s/%s (%d samples)", model_type, trainer.ALGORITHM, len(X_sample))
            return {
                "generated": True,
                "algorithm": trainer.ALGORITHM,
                "n_samples": len(X_sample),
                "bar_plot_path": bar_path,
                "importance_json_path": imp_json_path,
                "top_features": {k: round(v, 6) for k, v in list(importance_dict.items())[:10]},
            }

        except Exception as exc:
            logger.warning("SHAP generation failed for %s/%s: %s", model_type,
                           getattr(trainer, "ALGORITHM", "?"), exc)
            return {"generated": False, "error": str(exc)}

    # ── Save production artifacts ─────────────────────────────────────────────

    def _save_production(self, trainer, model_type: str) -> str:
        _SAVED_DIR.mkdir(parents=True, exist_ok=True)
        path = _SAVED_DIR / f"{model_type}_production.pkl"
        joblib.dump(trainer, str(path))
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("Production %s model saved: %s (%.2f MB)", model_type, path, size_mb)
        return str(path)

    # ── Registry updates ──────────────────────────────────────────────────────

    def _update_champion_registry(
        self,
        run_id: str,
        version: str,
        ts: str,
        freq_winner: str,
        sev_winner: str,
        freq_results: Dict,
        sev_results: Dict,
        feature_names: List[str],
        freq_prod_path: str,
        sev_prod_path: str,
        total_rows: int = 50000,
        dataset_label: str = DATASET_NAME,
    ) -> None:
        freq_m = freq_results[freq_winner]["metrics"]
        sev_m  = sev_results[sev_winner]["metrics"]

        registry = {
            "registry_version": "2.1.0",
            "last_updated": ts[:10],
            "created_by":   "multi_algorithm_engine",
            "description":  (
                "Insurance AI Pricing System — Champion Model Registry. "
                f"All champions trained on {DATASET_NAME}. "
                f"Run: {run_id}."
            ),
            "frequency": {
                "champion":           f"{freq_winner.upper()}_MultiRun",
                "algorithm":          self._full_algo_name(freq_winner, "frequency"),
                "metric":             "ROC_AUC",
                "score":              round(freq_m.get("roc_auc", 0) or 0, 4),
                "baseline_score":     round(freq_m.get("roc_auc", 0) or 0, 4),
                "model_path":         self._rel(freq_prod_path),
                "preprocessor_path":  "models/saved/preprocessor.pkl",
                "preprocessor_version": "V1",
                "feature_count":      len(feature_names),
                "version":            version,
                "trained_on_rows":    total_rows,
                "train_dataset":      dataset_label,
                "promotion_date":     ts[:10],
                "promoted_by":        f"multi_algorithm_engine/{run_id}",
                "training_timestamp": ts,
                "run_id":             run_id,
                "eval_metrics": {
                    "roc_auc":    round(freq_m.get("roc_auc", 0) or 0, 4),
                    "pr_auc":     round(freq_m.get("pr_auc", 0) or 0, 4),
                    "f1":         round(freq_m.get("f1", 0) or 0, 4),
                    "precision":  round(freq_m.get("precision", 0) or 0, 4),
                    "recall":     round(freq_m.get("recall", 0) or 0, 4),
                    "brier_score": round(freq_m.get("brier_score", 0) or 0, 4),
                },
                "deployment_status": "active",
                "deployment_note":   (
                    f"Production model. Winner of multi-algorithm comparison (run {run_id}). "
                    f"Algorithm: {freq_winner}. Dataset: {dataset_label}."
                ),
            },
            "severity": {
                "champion":           f"{sev_winner.upper()}_MultiRun",
                "algorithm":          self._full_algo_name(sev_winner, "severity"),
                "metric":             "R2",
                "score":              round(sev_m.get("r2", 0) or 0, 4),
                "baseline_score":     round(sev_m.get("r2", 0) or 0, 4),
                "model_path":         self._rel(sev_prod_path),
                "model_pkl_path":     self._rel(sev_prod_path),
                "preprocessor_path":  "models/saved/preprocessor.pkl",
                "preprocessor_version": "V1",
                "feature_count":      len(feature_names),
                "version":            version,
                "trained_on_rows":    total_rows,
                "train_dataset":      dataset_label,
                "promotion_date":     ts[:10],
                "promoted_by":        f"multi_algorithm_engine/{run_id}",
                "training_timestamp": ts,
                "run_id":             run_id,
                "eval_metrics": {
                    "r2":       round(sev_m.get("r2", 0) or 0, 4),
                    "rmse":     round(sev_m.get("rmse", 0) or 0, 2),
                    "mae":      round(sev_m.get("mae", 0) or 0, 2),
                    "mape_pct": round(sev_m.get("mape_pct", 0) or 0, 3),
                },
                "deployment_status": "active",
                "deployment_note":   (
                    f"Production model. Winner of multi-algorithm comparison (run {run_id}). "
                    f"Algorithm: {sev_winner}. Dataset: {dataset_label}."
                ),
            },
        }

        self._atomic_write(_CHAMPION_PATH, registry)
        logger.info("champion_registry.json updated → freq=%s, sev=%s", freq_winner, sev_winner)

    def _update_model_registry(
        self,
        run_id: str,
        version: str,
        ts: str,
        freq_winner: str,
        sev_winner: str,
        freq_results: Dict,
        sev_results: Dict,
        feature_names: List[str],
        freq_prod_path: str,
        sev_prod_path: str,
        split_info: Dict[str, int],
        dataset_hash: str,
        dataset_label: str = DATASET_NAME,
    ) -> None:
        freq_m = freq_results[freq_winner]["metrics"]
        sev_m  = sev_results[sev_winner]["metrics"]

        def _size_mb(path: str) -> Optional[float]:
            try:
                return round(os.path.getsize(path) / (1024 * 1024), 2)
            except Exception:
                return None

        registry = {
            "registry_version": "2.1.0",
            "schema":           "model_registry_v2",
            "last_updated":     ts,
            "description":      (
                f"Canonical model registry. Winners of multi-algorithm engine run {run_id}. "
                f"Both models trained on {dataset_label}."
            ),
            "frequency": {
                "production": {
                    "model_id":           f"freq_{freq_winner}_{run_id}",
                    "model_name":         f"{freq_winner.capitalize()} Classifier",
                    "model_type":         "frequency",
                    "algorithm":          self._full_algo_name(freq_winner, "frequency"),
                    "version":            version,
                    "status":             "production",
                    "serving_status":     "active",
                    "training_timestamp": ts,
                    "promotion_date":     ts[:10],
                    "promoted_by":        f"multi_algorithm_engine/{run_id}",
                    "dataset":            dataset_label,
                    "dataset_hash":       dataset_hash,
                    "n_total_rows":       split_info["n_train"] + split_info["n_val"] + split_info["n_test"],
                    "n_train_rows":       split_info["n_train"],
                    "n_val_rows":         split_info["n_val"],
                    "n_test_rows":        split_info["n_test"],
                    "feature_count":      len(feature_names),
                    "feature_names":      feature_names,
                    "preprocessor":       "InsurancePreprocessorV1",
                    "preprocessor_path":  "models/saved/preprocessor.pkl",
                    "artifact_path":      self._rel(freq_prod_path),
                    "artifact_size_mb":   _size_mb(freq_prod_path),
                    "run_id":             run_id,
                    "metrics": {
                        "auc_roc":     round(freq_m.get("roc_auc", 0) or 0, 4),
                        "pr_auc":      round(freq_m.get("pr_auc", 0) or 0, 4),
                        "f1":          round(freq_m.get("f1", 0) or 0, 4),
                        "precision":   round(freq_m.get("precision", 0) or 0, 4),
                        "recall":      round(freq_m.get("recall", 0) or 0, 4),
                        "brier_score": round(freq_m.get("brier_score", 0) or 0, 4),
                        "threshold":   0.5,
                    },
                    "all_algorithms": {
                        algo: {
                            "status":  r["status"],
                            "metrics": r["metrics"],
                            "training_time_sec": r["training_time_sec"],
                        }
                        for algo, r in freq_results.items()
                    },
                    "dataset_ceiling": {
                        "theoretical_auc_ceiling": 0.77,
                        "note": "Ceiling imposed by synthetic dataset. Max feature corr=0.42.",
                    },
                    "notes": (
                        f"Winner of 3-way algorithm comparison in run {run_id}. "
                        f"Algorithms evaluated: XGBoost, LightGBM, CatBoost."
                    ),
                },
                "candidate": None,
                "archived":  [],
            },
            "severity": {
                "production": {
                    "model_id":           f"sev_{sev_winner}_{run_id}",
                    "model_name":         f"{sev_winner.capitalize()} Regressor",
                    "model_type":         "severity",
                    "algorithm":          self._full_algo_name(sev_winner, "severity"),
                    "version":            version,
                    "status":             "production",
                    "serving_status":     "active",
                    "training_timestamp": ts,
                    "promotion_date":     ts[:10],
                    "promoted_by":        f"multi_algorithm_engine/{run_id}",
                    "dataset":            dataset_label,
                    "dataset_hash":       dataset_hash,
                    "n_total_rows":       split_info["n_train"] + split_info["n_val"] + split_info["n_test"],
                    "n_train_rows":       split_info["n_sev_train"],
                    "n_val_rows":         split_info["n_sev_val"],
                    "n_test_rows":        split_info["n_sev_test"],
                    "feature_count":      len(feature_names),
                    "preprocessor":       "InsurancePreprocessorV1",
                    "preprocessor_path":  "models/saved/preprocessor.pkl",
                    "artifact_path":      self._rel(sev_prod_path),
                    "artifact_size_mb":   _size_mb(sev_prod_path),
                    "run_id":             run_id,
                    "metrics": {
                        "r2":       round(sev_m.get("r2", 0) or 0, 4),
                        "rmse":     round(sev_m.get("rmse", 0) or 0, 2),
                        "mae":      round(sev_m.get("mae", 0) or 0, 2),
                        "mape_pct": round(sev_m.get("mape_pct", 0) or 0, 3),
                    },
                    "all_algorithms": {
                        algo: {
                            "status":  r["status"],
                            "metrics": r["metrics"],
                            "training_time_sec": r["training_time_sec"],
                        }
                        for algo, r in sev_results.items()
                    },
                    "dataset_ceiling": {
                        "theoretical_r2_ceiling": 0.01,
                        "note": "Claim amounts randomly sampled in synthetic dataset. R2 ceiling ~0.01.",
                    },
                    "notes": (
                        f"Winner of 3-way algorithm comparison in run {run_id}. "
                        f"Trained on claimants only ({split_info['n_sev_train']} train rows). "
                        f"Algorithms evaluated: XGBoost, LightGBM, CatBoost."
                    ),
                },
                "candidate": None,
                "archived":  [],
            },
        }

        self._atomic_write(_REGISTRY_PATH, registry)
        logger.info("model_registry.json updated → freq=%s, sev=%s", freq_winner, sev_winner)

    # ── Training history ──────────────────────────────────────────────────────

    def _build_run_record(
        self,
        run_id, run_number, version, ts, dataset_hash,
        freq_results, sev_results,
        freq_winner, sev_winner,
        freq_reason, sev_reason,
        freq_shap, sev_shap,
        freq_prod_path, sev_prod_path,
        split_info, elapsed_sec,
        dataset_label: str = DATASET_NAME,
    ) -> Dict[str, Any]:
        def _serialise(results: Dict) -> List[Dict]:
            out = []
            for algo, r in results.items():
                out.append({
                    "algorithm":         algo,
                    "model_type":        r["model_type"],
                    "status":            r["status"],
                    "is_winner":         r["is_winner"],
                    "metrics":           r["metrics"],
                    "training_time_sec": r["training_time_sec"],
                    "inference_time_ms": r["inference_time_ms"],
                    "model_size_mb":     r["model_size_mb"],
                    "artifact_path":     r["artifact_path"],
                    "error":             r["error"],
                })
            return out

        return {
            "run_id":               run_id,
            "run_number":           run_number,
            "version":              version,
            "training_timestamp":   ts,
            "dataset":              dataset_label,
            "dataset_hash":         dataset_hash,
            "n_total_rows":         split_info["n_train"] + split_info["n_val"] + split_info["n_test"],
            "n_train_rows":         split_info["n_train"],
            "n_val_rows":           split_info["n_val"],
            "n_test_rows":          split_info["n_test"],
            "n_sev_train_rows":     split_info["n_sev_train"],
            "n_sev_val_rows":       split_info["n_sev_val"],
            "n_sev_test_rows":      split_info["n_sev_test"],
            "feature_count":        None,
            "frequency_results":    _serialise(freq_results),
            "severity_results":     _serialise(sev_results),
            "frequency_winner":     freq_winner,
            "severity_winner":      sev_winner,
            "promotion_reason_frequency": freq_reason,
            "promotion_reason_severity":  sev_reason,
            "frequency_production_path":  self._rel(freq_prod_path),
            "severity_production_path":   self._rel(sev_prod_path),
            "shap_generated_frequency":   freq_shap.get("generated", False),
            "shap_generated_severity":    sev_shap.get("generated", False),
            "shap_path_frequency":        freq_shap.get("importance_json_path"),
            "shap_path_severity":         sev_shap.get("importance_json_path"),
            "total_elapsed_sec":          elapsed_sec,
        }

    def _append_history(self, record: Dict[str, Any]) -> None:
        _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        if _HISTORY_PATH.exists():
            with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {
                "schema_version": "1.0.0",
                "training_runs":  [],
                "latest_run_id":  None,
                "total_runs":     0,
            }

        history["training_runs"].append(record)
        history["latest_run_id"] = record["run_id"]
        history["total_runs"]    = len(history["training_runs"])
        self._atomic_write(_HISTORY_PATH, history)

    def _write_run_metadata(self, run_dir: pathlib.Path, record: Dict[str, Any]) -> None:
        path = run_dir / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_run_id() -> str:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        return f"run_{ts}_{uid}"

    def _next_run_number(self) -> int:
        if not _HISTORY_PATH.exists():
            return 1
        try:
            with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                h = json.load(f)
            return h.get("total_runs", 0) + 1
        except Exception:
            return 1

    @staticmethod
    def _rel(abs_path: str) -> str:
        try:
            return str(pathlib.Path(abs_path).relative_to(_BACKEND_ROOT))
        except ValueError:
            return abs_path

    @staticmethod
    def _full_algo_name(algo: str, model_type: str) -> str:
        names = {
            "xgboost":  "XGBoostClassifier" if model_type == "frequency" else "XGBoostRegressor",
            "lightgbm": "LGBMClassifier"    if model_type == "frequency" else "LGBMRegressor",
            "catboost": "CatBoostClassifier" if model_type == "frequency" else "CatBoostRegressor",
        }
        return names.get(algo, algo)

    @staticmethod
    def _atomic_write(path: pathlib.Path, data: Dict) -> None:
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))

    @staticmethod
    def _try_hot_reload() -> None:
        try:
            from src.models.combined_predictor import reload_predictor
            ok, msg = reload_predictor()
            if ok:
                logger.info("Predictor hot-reloaded: %s", msg)
            else:
                logger.warning("Predictor hot-reload failed: %s", msg)
        except Exception as exc:
            logger.debug("Hot-reload not available: %s", exc)


# ── Convenience function ──────────────────────────────────────────────────────

def run_training(
    dataset_path: Optional[str] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    dataset_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the multi-algorithm engine and return the run record."""
    engine = MultiAlgorithmEngine(dataset_path=dataset_path)
    return engine.run(dataset_df=dataset_df, dataset_label=dataset_label)
