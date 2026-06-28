"""
V5TrainingEngine — Phase 5A Enterprise Training Orchestrator.

Responsibilities:
  1. Load / validate dataset from V4DatasetBuilder
  2. Verify maturity threshold and minimum sample count
  3. Split train / validation / test (70/15/15)
  4. Launch candidate trainers (XGBoost, CatBoost, LightGBM) ×
     model type (frequency, severity)
  5. Collect and persist metrics
  6. Save challenger artifacts under models/challengers/run_<timestamp>/
  7. Record everything in training_runs + candidate_models tables

Safety:
  - NEVER modifies champion_registry.json
  - NEVER overwrites models/champion/ artifacts
  - Does NOT touch production inference paths
  - Does NOT modify ENABLE_SCHEDULED_RETRAINING
  - Produces challenger models ONLY; promotion is Phase 5B
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CandidateModel, TrainingRun
from src.database.session import AsyncSessionLocal
from src.preprocessing.pipeline_v4 import InsurancePreprocessorV4
from src.retraining.dataset_builder_v4 import (
    LABEL_MATURITY_DAYS,
    RETRAIN_MIN_SAMPLES,
    V4DatasetBuilder,
    V4_RAW_FEATURES,
)
from src.training.metrics import compute_frequency_metrics, compute_severity_metrics
from src.training.trainers.catboost_trainer import CatBoostTrainer
from src.training.trainers.lightgbm_trainer import LightGBMTrainer
from src.training.trainers.xgboost_trainer import XGBoostTrainer

logger = logging.getLogger("insurance_api.training")

# Root directory for challenger artifacts
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CHALLENGERS_DIR = _BACKEND_ROOT / "models" / "challengers"

ALGORITHMS = ["xgboost", "catboost", "lightgbm"]

TRAINER_MAP = {
    "xgboost":  XGBoostTrainer,
    "catboost":  CatBoostTrainer,
    "lightgbm": LightGBMTrainer,
}


# ── Validation report ──────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    passed: bool
    dataset_rows: int
    matured_labelled: int
    valid_rows: int
    missing_features: List[str]
    reasons: List[str] = field(default_factory=list)
    maturity_days: int = LABEL_MATURITY_DAYS
    min_samples: int = RETRAIN_MIN_SAMPLES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "dataset_rows": self.dataset_rows,
            "matured_labelled": self.matured_labelled,
            "valid_rows": self.valid_rows,
            "missing_features": self.missing_features,
            "reasons": self.reasons,
            "maturity_days": self.maturity_days,
            "min_samples": self.min_samples,
        }


# ── Training result ────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    algorithm: str
    model_type: str
    status: str             # "COMPLETED" | "FAILED"
    metrics: Dict[str, Any]
    artifact_path: str
    duration_seconds: float
    feature_importance: Dict[str, float]
    error_message: Optional[str] = None
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRunResult:
    run_id: str
    run_tag: str
    status: str
    triggered_by: str
    triggered_at: str
    completed_at: Optional[str]
    dataset_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    feature_count: int
    validation_report: ValidationReport
    candidates: List[CandidateResult]
    summary: Dict[str, Any]
    error_message: Optional[str] = None
    duration_total_seconds: float = 0.0


# ── Main engine ────────────────────────────────────────────────────────────────

class V5TrainingEngine:
    """
    Orchestrates a full V5 challenger training run.

    Usage (from API route via BackgroundTasks):
        engine = V5TrainingEngine(db)
        result = await engine.run(triggered_by="admin")
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def run(
        self,
        triggered_by: str = "admin",
        bypass_maturity_check: bool = False,
        synthetic_df: Optional[pd.DataFrame] = None,
        maturity_days: int = LABEL_MATURITY_DAYS,
        min_samples: int = RETRAIN_MIN_SAMPLES,
        algorithms: Optional[List[str]] = None,
        preexisting_run_id: Optional[str] = None,
        preexisting_run_tag: Optional[str] = None,
    ) -> TrainingRunResult:
        """
        Execute a full training run. Persists results to DB.

        Parameters
        ----------
        triggered_by         : actor label for audit trail
        bypass_maturity_check: skip maturity + sample checks (testing only)
        synthetic_df         : inject pre-built DataFrame (testing only)
        maturity_days        : label maturity threshold
        min_samples          : minimum valid rows required
        algorithms           : subset of ['xgboost','catboost','lightgbm']
        preexisting_run_id   : reuse a DB record pre-created by the API route
        preexisting_run_tag  : run_tag matching the pre-created record
        """
        t0 = time.monotonic()
        now = datetime.utcnow()
        algs = algorithms or ALGORITHMS

        # ── Create or reuse DB record ──────────────────────────────────────────
        if preexisting_run_id is not None and preexisting_run_tag is not None:
            # Background task: route pre-created the record; fetch and update it
            run_id = preexisting_run_id
            run_tag = preexisting_run_tag
            db_run = await self._db.get(TrainingRun, uuid.UUID(run_id))
            if db_run is None:
                # Safety fallback: record missing, create it
                db_run = TrainingRun(
                    run_id=uuid.UUID(run_id),
                    run_tag=run_tag,
                    triggered_by=triggered_by,
                    triggered_at=now,
                    status="RUNNING",
                    maturity_days=maturity_days,
                )
                self._db.add(db_run)
                await self._db.commit()
        else:
            run_id = str(uuid.uuid4())
            run_tag = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            db_run = TrainingRun(
                run_id=uuid.UUID(run_id),
                run_tag=run_tag,
                triggered_by=triggered_by,
                triggered_at=now,
                status="RUNNING",
                maturity_days=maturity_days,
            )
            self._db.add(db_run)
            await self._db.commit()

        validation_report: Optional[ValidationReport] = None
        candidates: List[CandidateResult] = []
        dataset_df: Optional[pd.DataFrame] = None

        try:
            # ── Step 1: Load dataset ───────────────────────────────────────────
            if synthetic_df is not None:
                dataset_df = synthetic_df
                build_result = None
            else:
                builder = V4DatasetBuilder(self._db)
                build_result = await builder.build(
                    maturity_days=maturity_days,
                    include_dataset=True,
                )
                dataset_df = build_result.dataset

            # ── Step 2: Validate ───────────────────────────────────────────────
            validation_report = self._validate_dataset(
                df=dataset_df,
                build_result=build_result,
                bypass=bypass_maturity_check,
                maturity_days=maturity_days,
                min_samples=min_samples,
            )

            if not validation_report.passed:
                db_run.status = "VALIDATION_FAILED"
                db_run.error_message = "; ".join(validation_report.reasons)
                db_run.completed_at = datetime.utcnow()
                db_run.summary = {"validation_report": validation_report.to_dict()}
                await self._db.commit()
                return self._make_result(
                    run_id, run_tag, "VALIDATION_FAILED", triggered_by,
                    now, datetime.utcnow(), 0, 0, 0, 0, 0,
                    validation_report, [], {},
                    error_message="; ".join(validation_report.reasons),
                    duration=time.monotonic() - t0,
                )

            # ── Step 3: Preprocess + split ─────────────────────────────────────
            X, y_freq, y_sev, feature_names, prep = self._preprocess(dataset_df)
            (X_tr, X_val, X_te,
             y_fr_tr, y_fr_val, y_fr_te,
             y_sv_tr, y_sv_val, y_sv_te) = self._split(X, y_freq, y_sev)

            run_dir = _CHALLENGERS_DIR / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save fitted preprocessor for use by Phase 5B promotion
            joblib.dump(prep, run_dir / "preprocessor_v4.pkl")

            # ── Step 4: Train candidates ───────────────────────────────────────
            db_run.dataset_rows = len(dataset_df)
            db_run.feature_count = len(feature_names)
            db_run.train_rows = len(X_tr)
            db_run.val_rows   = len(X_val)
            db_run.test_rows  = len(X_te)
            await self._db.commit()

            for alg in algs:
                for mtype in ("frequency", "severity"):
                    candidate = await self._train_candidate(
                        run_id=run_id,
                        run_tag=run_tag,
                        alg=alg,
                        mtype=mtype,
                        feature_names=feature_names,
                        X_tr=X_tr, X_val=X_val, X_te=X_te,
                        y_freq_tr=y_fr_tr, y_freq_val=y_fr_val, y_freq_te=y_fr_te,
                        y_sev_tr=y_sv_tr,  y_sev_val=y_sv_val,  y_sev_te=y_sv_te,
                        run_dir=run_dir,
                    )
                    candidates.append(candidate)

            # ── Step 5: Build summary ──────────────────────────────────────────
            summary = self._build_summary(candidates, run_tag)
            run_dir_meta = {
                "run_tag": run_tag,
                "run_id": run_id,
                "generated_at": datetime.utcnow().isoformat(),
                "summary": summary,
            }
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(run_dir_meta, f, indent=2)

            db_run.status = "COMPLETED"
            db_run.completed_at = datetime.utcnow()
            db_run.summary = summary
            await self._db.commit()

            return self._make_result(
                run_id, run_tag, "COMPLETED", triggered_by,
                now, db_run.completed_at,
                len(dataset_df), len(X_tr), len(X_val), len(X_te), len(feature_names),
                validation_report, candidates, summary,
                duration=time.monotonic() - t0,
            )

        except Exception as exc:
            logger.exception("Training run %s failed: %s", run_tag, exc)
            db_run.status = "FAILED"
            db_run.error_message = f"{type(exc).__name__}: {exc}"
            db_run.completed_at = datetime.utcnow()
            await self._db.commit()
            return self._make_result(
                run_id, run_tag, "FAILED", triggered_by,
                now, datetime.utcnow(), 0, 0, 0, 0, 0,
                validation_report or ValidationReport(
                    False, 0, 0, 0, [], [str(exc)]
                ),
                candidates, {},
                error_message=str(exc),
                duration=time.monotonic() - t0,
            )

    # ── Validation ─────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dataset(
        df: Optional[pd.DataFrame],
        build_result: Any,
        bypass: bool,
        maturity_days: int,
        min_samples: int,
    ) -> ValidationReport:
        reasons: List[str] = []
        n_total = 0
        n_matured = 0
        n_valid = 0
        missing_feats: List[str] = []

        if df is None or len(df) == 0:
            if not bypass:
                reasons.append(
                    f"No matured labelled records available. Labels mature after "
                    f"{maturity_days} days; first V2 production labels expected late 2026."
                )
                return ValidationReport(
                    passed=False,
                    dataset_rows=0,
                    matured_labelled=0,
                    valid_rows=0,
                    missing_features=[],
                    reasons=reasons,
                    maturity_days=maturity_days,
                    min_samples=min_samples,
                )
            else:
                return ValidationReport(
                    passed=True, dataset_rows=0, matured_labelled=0,
                    valid_rows=0, missing_features=[],
                    reasons=["bypass_maturity_check=True (synthetic data)"],
                    maturity_days=maturity_days, min_samples=min_samples,
                )

        n_total = len(df) if build_result is None else build_result.total_predictions
        n_matured = len(df) if build_result is None else build_result.matured_labelled
        n_valid = len(df)

        if not bypass:
            if n_valid < min_samples:
                reasons.append(
                    f"Insufficient valid rows: {n_valid} < {min_samples} minimum."
                )

        # Check required V4 features
        present = set(df.columns)
        required = {"age", "vehicle_value_inr", "previous_claims_count", "driving_score"}
        for feat in required:
            if feat not in present:
                reasons.append(f"Required V4 feature missing from dataset: {feat!r}")

        # Check labels
        if "claim_occurred" not in df.columns:
            reasons.append("Target column 'claim_occurred' missing from dataset.")

        # Fill rate check
        for feat in V4_RAW_FEATURES:
            if feat in df.columns:
                fill_rate = df[feat].notna().mean() * 100
                if fill_rate < 50.0:
                    missing_feats.append(feat)

        passed = len(reasons) == 0
        return ValidationReport(
            passed=passed,
            dataset_rows=n_total,
            matured_labelled=n_matured,
            valid_rows=n_valid,
            missing_features=missing_feats,
            reasons=reasons,
            maturity_days=maturity_days,
            min_samples=min_samples,
        )

    # ── Preprocessing ──────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], "InsurancePreprocessorV4"]:
        """Apply InsurancePreprocessorV4 and extract targets."""
        prep = InsurancePreprocessorV4()

        # Build feature df with required columns
        feat_cols = [c for c in V4_RAW_FEATURES if c in df.columns]
        X_raw = df[feat_cols].copy()

        # Fill missing optional features with sensible defaults
        for col in InsurancePreprocessorV4.STD_COLS:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        for col in InsurancePreprocessorV4.MM_COLS:
            if col not in X_raw.columns:
                X_raw[col] = 0.0
        for col in InsurancePreprocessorV4.OHE_COLS:
            if col not in X_raw.columns:
                X_raw[col] = "unknown"
        if InsurancePreprocessorV4.LE_COL not in X_raw.columns:
            X_raw[InsurancePreprocessorV4.LE_COL] = "unknown"
        for col in InsurancePreprocessorV4.PASS_COLS:
            if col not in X_raw.columns:
                X_raw[col] = 0.0

        # Ensure months_since_last_claim present for engineer step
        if "months_since_last_claim" not in X_raw.columns:
            X_raw["months_since_last_claim"] = 999  # never claimed

        X = prep.fit_transform(X_raw)
        feature_names = prep.feature_names_ or [f"f{i}" for i in range(X.shape[1])]

        y_freq = df["claim_occurred"].values.astype(int) if "claim_occurred" in df.columns else np.zeros(len(df), dtype=int)
        y_sev  = df["actual_claim_amount_inr"].values.astype(float) if "actual_claim_amount_inr" in df.columns else np.zeros(len(df))

        return X, y_freq, y_sev, feature_names, prep

    # ── Train/val/test split ───────────────────────────────────────────────────

    @staticmethod
    def _split(
        X: np.ndarray,
        y_freq: np.ndarray,
        y_sev: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        random_state: int = 42,
    ):
        test_ratio = 1.0 - train_ratio - val_ratio
        X_tr, X_tmp, yf_tr, yf_tmp, ys_tr, ys_tmp = train_test_split(
            X, y_freq, y_sev,
            test_size=(1.0 - train_ratio),
            random_state=random_state,
            stratify=y_freq if y_freq.sum() > 1 else None,
        )
        rel_val = val_ratio / (val_ratio + test_ratio)
        X_val, X_te, yf_val, yf_te, ys_val, ys_te = train_test_split(
            X_tmp, yf_tmp, ys_tmp,
            test_size=(1.0 - rel_val),
            random_state=random_state,
            stratify=yf_tmp if yf_tmp.sum() > 1 else None,
        )
        return X_tr, X_val, X_te, yf_tr, yf_val, yf_te, ys_tr, ys_val, ys_te

    # ── Single candidate training ──────────────────────────────────────────────

    async def _train_candidate(
        self,
        run_id: str,
        run_tag: str,
        alg: str,
        mtype: str,
        feature_names: List[str],
        X_tr: np.ndarray,
        X_val: np.ndarray,
        X_te: np.ndarray,
        y_freq_tr: np.ndarray,
        y_freq_val: np.ndarray,
        y_freq_te: np.ndarray,
        y_sev_tr: np.ndarray,
        y_sev_val: np.ndarray,
        y_sev_te: np.ndarray,
        run_dir: pathlib.Path,
    ) -> CandidateResult:
        t0 = time.monotonic()
        started_at = datetime.utcnow()
        artifact_dir = str(run_dir / alg / mtype)

        # Create candidate DB record
        db_cand = CandidateModel(
            run_id=uuid.UUID(run_id),
            algorithm=alg,
            model_type=mtype,
            status="TRAINING",
            started_at=started_at,
            train_rows=len(X_tr),
            val_rows=len(X_val),
            test_rows=len(X_te),
        )
        self._db.add(db_cand)
        await self._db.commit()

        try:
            trainer_cls = TRAINER_MAP[alg]
            trainer = trainer_cls(
                model_type=mtype,
                feature_names=feature_names,
                random_state=42,
            )

            if mtype == "frequency":
                y_tr, y_val, y_te = y_freq_tr, y_freq_val, y_freq_te
            else:
                # Severity: train only on rows where a claim occurred
                # (avoids biasing the model with zero-claim rows)
                claim_mask_tr  = y_freq_tr > 0
                claim_mask_val = y_freq_val > 0
                claim_mask_te  = y_freq_te > 0
                X_tr_sev  = X_tr[claim_mask_tr]  if claim_mask_tr.sum()  > 0 else X_tr
                X_val_sev = X_val[claim_mask_val] if claim_mask_val.sum() > 0 else X_val
                X_te_sev  = X_te[claim_mask_te]  if claim_mask_te.sum()  > 0 else X_te
                y_tr  = y_sev_tr[claim_mask_tr]   if claim_mask_tr.sum()  > 0 else y_sev_tr
                y_val = y_sev_val[claim_mask_val]  if claim_mask_val.sum() > 0 else y_sev_val
                y_te  = y_sev_te[claim_mask_te]   if claim_mask_te.sum()  > 0 else y_sev_te

            # CPU-bound training — offload to thread pool so event loop stays responsive
            if mtype == "frequency":
                await asyncio.to_thread(trainer.train, X_tr, y_tr)
            else:
                await asyncio.to_thread(trainer.train, X_tr_sev, y_tr)

            # Evaluate on test split
            if mtype == "frequency":
                y_prob = trainer.predict_proba(X_te)
                metrics = compute_frequency_metrics(y_te, y_prob)
                freq_metrics, sev_metrics = metrics, None
            else:
                y_pred = trainer.predict(X_te_sev) if len(X_te_sev) > 0 else np.zeros(1)
                y_true = y_te if len(y_te) > 0 else np.zeros(1)
                metrics = compute_severity_metrics(y_true, y_pred)
                freq_metrics, sev_metrics = None, metrics

            importance = trainer.get_feature_importance()

            # Save artifacts
            metrics_path = os.path.join(artifact_dir, "metrics.json")
            trainer.save(artifact_dir)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            duration = round(time.monotonic() - t0, 2)

            # Update DB
            db_cand.status = "COMPLETED"
            db_cand.completed_at = datetime.utcnow()
            db_cand.duration_seconds = duration
            db_cand.frequency_metrics = freq_metrics
            db_cand.severity_metrics = sev_metrics
            db_cand.feature_importance = dict(list(importance.items())[:20])  # top-20
            db_cand.artifact_path = artifact_dir
            db_cand.hyperparameters = trainer.hyperparameters
            await self._db.commit()

            return CandidateResult(
                algorithm=alg,
                model_type=mtype,
                status="COMPLETED",
                metrics=metrics,
                artifact_path=artifact_dir,
                duration_seconds=duration,
                feature_importance=importance,
                train_rows=len(X_tr),
                val_rows=len(X_val),
                test_rows=len(X_te),
                hyperparameters=trainer.hyperparameters,
            )

        except Exception as exc:
            logger.warning("Candidate %s/%s failed: %s", alg, mtype, exc)
            duration = round(time.monotonic() - t0, 2)
            db_cand.status = "FAILED"
            db_cand.completed_at = datetime.utcnow()
            db_cand.duration_seconds = duration
            db_cand.error_message = f"{type(exc).__name__}: {exc}"
            await self._db.commit()

            return CandidateResult(
                algorithm=alg,
                model_type=mtype,
                status="FAILED",
                metrics={},
                artifact_path=artifact_dir,
                duration_seconds=duration,
                feature_importance={},
                error_message=str(exc),
            )

    # ── Summary builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(candidates: List[CandidateResult], run_tag: str) -> Dict[str, Any]:
        completed = [c for c in candidates if c.status == "COMPLETED"]
        failed    = [c for c in candidates if c.status == "FAILED"]

        best_freq: Optional[CandidateResult] = None
        best_sev:  Optional[CandidateResult] = None

        for c in completed:
            if c.model_type == "frequency" and c.metrics.get("roc_auc") is not None:
                if best_freq is None or c.metrics["roc_auc"] > best_freq.metrics.get("roc_auc", 0):
                    best_freq = c
            if c.model_type == "severity" and c.metrics.get("r2") is not None:
                if best_sev is None or c.metrics["r2"] > best_sev.metrics.get("r2", -9999):
                    best_sev = c

        return {
            "run_tag": run_tag,
            "total_candidates": len(candidates),
            "completed": len(completed),
            "failed": len(failed),
            "best_frequency_model": {
                "algorithm": best_freq.algorithm,
                "roc_auc": best_freq.metrics.get("roc_auc"),
                "artifact_path": best_freq.artifact_path,
            } if best_freq else None,
            "best_severity_model": {
                "algorithm": best_sev.algorithm,
                "r2": best_sev.metrics.get("r2"),
                "rmse": best_sev.metrics.get("rmse"),
                "artifact_path": best_sev.artifact_path,
            } if best_sev else None,
            "governance_note": (
                "Challenger artifacts saved. No champion modification. "
                "Promotion requires Phase 5B governance approval."
            ),
        }

    # ── Result builder ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_result(
        run_id: str,
        run_tag: str,
        status: str,
        triggered_by: str,
        triggered_at: datetime,
        completed_at: Optional[datetime],
        dataset_rows: int,
        train_rows: int,
        val_rows: int,
        test_rows: int,
        feature_count: int,
        validation_report: ValidationReport,
        candidates: List[CandidateResult],
        summary: Dict[str, Any],
        error_message: Optional[str] = None,
        duration: float = 0.0,
    ) -> TrainingRunResult:
        return TrainingRunResult(
            run_id=run_id,
            run_tag=run_tag,
            status=status,
            triggered_by=triggered_by,
            triggered_at=triggered_at.isoformat(),
            completed_at=completed_at.isoformat() if completed_at else None,
            dataset_rows=dataset_rows,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            feature_count=feature_count,
            validation_report=validation_report,
            candidates=candidates,
            summary=summary,
            error_message=error_message,
            duration_total_seconds=round(duration, 2),
        )

    # ── Background classmethod ─────────────────────────────────────────────────

    @classmethod
    async def run_background(
        cls,
        run_id_placeholder: str,
        triggered_by: str = "admin",
        preexisting_run_tag: Optional[str] = None,
    ) -> None:
        """Entry point for FastAPI BackgroundTasks. Creates its own DB session."""
        try:
            async with AsyncSessionLocal() as session:
                engine = cls(session)
                await engine.run(
                    triggered_by=triggered_by,
                    preexisting_run_id=run_id_placeholder,
                    preexisting_run_tag=preexisting_run_tag,
                )
        except Exception as exc:
            logger.warning("Background training run failed: %s", exc)


# ── Read helpers (called from API routes) ─────────────────────────────────────

async def list_training_runs(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    from sqlalchemy import func, desc
    from sqlalchemy.orm import selectinload

    total_q = await db.execute(
        select(func.count()).select_from(TrainingRun)
    )
    total = total_q.scalar() or 0

    rows_q = await db.execute(
        select(TrainingRun)
        .options(selectinload(TrainingRun.candidates))
        .order_by(desc(TrainingRun.triggered_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    runs = rows_q.scalars().all()

    import math
    return {
        "items": [_run_to_dict(r) for r in runs],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, math.ceil(total / page_size)),
    }


async def get_training_run_by_id(
    db: AsyncSession,
    run_id: str,
) -> Optional[TrainingRun]:
    from sqlalchemy.orm import selectinload
    try:
        uid = uuid.UUID(run_id)
    except ValueError:
        return None
    result = await db.execute(
        select(TrainingRun)
        .options(selectinload(TrainingRun.candidates))
        .where(TrainingRun.run_id == uid)
    )
    return result.scalar_one_or_none()


def _run_to_dict(run: TrainingRun) -> Dict[str, Any]:
    return {
        "run_id": str(run.run_id),
        "run_tag": run.run_tag,
        "triggered_by": run.triggered_by,
        "triggered_at": run.triggered_at.isoformat() if run.triggered_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "status": run.status,
        "dataset_rows": run.dataset_rows,
        "feature_count": run.feature_count,
        "maturity_days": run.maturity_days,
        "train_rows": run.train_rows,
        "val_rows": run.val_rows,
        "test_rows": run.test_rows,
        "error_message": run.error_message,
        "summary": run.summary,
        "candidates": [_candidate_to_dict(c) for c in (run.candidates or [])],
    }


def _candidate_to_dict(c: CandidateModel) -> Dict[str, Any]:
    return {
        "model_id": str(c.model_id),
        "run_id": str(c.run_id),
        "algorithm": c.algorithm,
        "model_type": c.model_type,
        "status": c.status,
        "started_at": c.started_at.isoformat() if c.started_at else None,
        "completed_at": c.completed_at.isoformat() if c.completed_at else None,
        "duration_seconds": float(c.duration_seconds) if c.duration_seconds is not None else None,
        "train_rows": c.train_rows,
        "val_rows": c.val_rows,
        "test_rows": c.test_rows,
        "hyperparameters": c.hyperparameters,
        "frequency_metrics": c.frequency_metrics,
        "severity_metrics": c.severity_metrics,
        "feature_importance": c.feature_importance,
        "artifact_path": c.artifact_path,
        "error_message": c.error_message,
    }
