import logging
import smtplib
import uuid
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from sqlalchemy import select, and_, text
from sqlalchemy.orm import Session

from src.config import settings
from src.database.models import (
    AgentRun, Customer, DrivingProfile, ModelFeedback, ModelVersion,
    Policy, Prediction, Vehicle,
)
from src.preprocessing.pipeline import prepare_training_data
from src.models.frequency_model import FrequencyModel
from src.models.severity_model import SeverityModel
from src.models.combined_predictor import swap_predictor

logger = logging.getLogger("retraining")

AUC_THRESHOLD = 0.72
RMSE_DEGRADATION_PCT = 0.15

# ── Registry integration (optional — degrades gracefully if registry absent) ──
def _try_promotion_engine():
    """Return a PromotionEngine instance, or None if registry not available."""
    try:
        from src.models.promotion_engine import PromotionEngine
        return PromotionEngine()
    except Exception as exc:
        logger.warning("PromotionEngine unavailable (%s) — skipping registry update.", exc)
        return None

# Minimum HITL-reviewed samples before governance metrics are computed
_MIN_HITL_SAMPLES = 5

# Feature columns that must be present for retraining to proceed.
# Column names here must match what the merged training DataFrame actually contains.
# Master dataset uses: vehicle_brand, previous_claims, actual_claim_amount_inr
# DB-joined data uses: car_brand, previous_claims_count, actual_claim_amount_inr
# The enterprise pipeline normalises both to the master dataset schema before training.
_REQUIRED_FEATURE_COLS = [
    "age", "gender", "city",
    "vehicle_age_years", "vehicle_value_inr",
    "driving_score", "annual_mileage_km", "years_licensed",
    "claim_occurred", "actual_claim_amount_inr",
]


class RetrainingPipeline:
    def __init__(self, db_session: Session, model_dir: str = None):
        self.db = db_session
        self.model_dir = model_dir or settings.MODELS_DIR

    # ── Step 1A: Lightweight label-only collect (evaluation only) ─────────────

    def collect_feedback(self, days_back: int = 90) -> pd.DataFrame:
        """
        Returns a minimal DataFrame with labels and model predictions.
        Column names use 'claim_occurred' (not the old 'actual_claim_occurred')
        to match prepare_training_data() expectations.
        Suitable for drift evaluation when full feature reconstruction is not needed.
        """
        cutoff = date.today() - timedelta(days=days_back)
        result = self.db.execute(
            select(
                ModelFeedback.prediction_id,
                ModelFeedback.actual_claim_occurred.label("claim_occurred"),
                ModelFeedback.actual_claim_amount_inr,
                Prediction.claim_probability,
                Prediction.expected_claim_amount_inr,
            )
            .join(Prediction, Prediction.prediction_id == ModelFeedback.prediction_id)
            .where(ModelFeedback.feedback_date >= cutoff)
        )
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=[
            "prediction_id", "claim_occurred", "actual_claim_amount_inr",
            "claim_probability", "expected_claim_amount_inr",
        ])
        logger.info("Collected %d feedback records (label-only)", len(df))
        return df

    # ── Step 1B: Full feature + label reconstruction (retraining + evaluation) ─

    def collect_features_with_labels(self, days_back: int = 90) -> pd.DataFrame:
        """
        Reconstructs the full applicant feature vector for every feedback record
        by joining across the complete data lineage:

          model_feedback
            → predictions      (model output anchor)
            → policies         (binding link)
            → customers        (demographics)
            → vehicles         (vehicle spec)
            → driving_profiles (telematics / claims history)
            ← LEFT JOIN agent_runs (HITL reviewer decision, when available)

        Returns a DataFrame containing:
          - All 12 raw applicant features required by InsurancePreprocessor
          - Labels: claim_occurred (frequency), actual_claim_amount_inr (severity)
          - HITL columns: human_decision, agent_decision (NULL when no review)
          - Model predictions: claim_probability, expected_claim_amount_inr
        """
        cutoff = date.today() - timedelta(days=days_back)

        result = self.db.execute(
            select(
                # Labels (renamed for prepare_training_data compatibility)
                ModelFeedback.actual_claim_occurred.label("claim_occurred"),
                ModelFeedback.actual_claim_amount_inr,
                # HITL governance columns (NULL when no underwriting review occurred)
                AgentRun.review_outcome.label("human_decision"),
                AgentRun.intent.label("agent_decision"),
                # Original applicant features — reconstructed from normalized tables
                Customer.age,
                Customer.gender,
                Customer.city,
                Vehicle.car_brand,
                Vehicle.car_model,
                Vehicle.engine_cc,
                Vehicle.vehicle_age_years,
                Vehicle.vehicle_value_inr,
                DrivingProfile.driving_score,
                DrivingProfile.annual_mileage_km,
                DrivingProfile.previous_claims_count,
                DrivingProfile.years_licensed,
                # Current model predictions (used for drift evaluation)
                Prediction.claim_probability,
                Prediction.expected_claim_amount_inr,
            )
            .join(Prediction, Prediction.prediction_id == ModelFeedback.prediction_id)
            .join(Policy, Policy.policy_id == Prediction.policy_id)
            .join(Customer, Customer.customer_id == Policy.customer_id)
            .join(Vehicle, Vehicle.vehicle_id == Policy.vehicle_id)
            .join(DrivingProfile, DrivingProfile.customer_id == Policy.customer_id)
            # LEFT JOIN: most policies have no HITL review; the join returns NULL
            # for human_decision / agent_decision in those cases.
            .outerjoin(AgentRun, AgentRun.prediction_id == Prediction.prediction_id)
            .where(ModelFeedback.feedback_date >= cutoff)
        )
        rows = result.fetchall()
        columns = [
            "claim_occurred", "actual_claim_amount_inr",
            "human_decision", "agent_decision",
            "age", "gender", "city",
            "car_brand", "car_model", "engine_cc", "vehicle_age_years", "vehicle_value_inr",
            "driving_score", "annual_mileage_km", "previous_claims_count", "years_licensed",
            "claim_probability", "expected_claim_amount_inr",
        ]
        df = pd.DataFrame(rows, columns=columns)
        logger.info(
            "Collected %d enriched feedback records (%d with HITL review)",
            len(df),
            int(df["human_decision"].notna().sum()) if len(df) else 0,
        )
        return df

    # ── Step 2: Model performance evaluation ──────────────────────────────────

    def evaluate_performance(self, feedback_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate current model accuracy and compute HITL governance metrics.

        Accepts DataFrames from either collect_feedback() or
        collect_features_with_labels().  Both use 'claim_occurred' as the
        frequency label column.  Legacy DataFrames with 'actual_claim_occurred'
        are renamed automatically for backward compatibility.
        """
        # Backward-compatibility shim for legacy collect_feedback callers
        if "claim_occurred" not in feedback_df.columns and "actual_claim_occurred" in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={"actual_claim_occurred": "claim_occurred"})

        base_report: Dict[str, Any] = {
            "evaluation_date": str(date.today()),
            "n_samples": len(feedback_df),
            "frequency_auc": None,
            "severity_rmse": None,
            "drift_detected": False,
            "retrain_recommended": False,
            "hitl_reviewed_count": 0,
            "human_override_rate": None,
            "ml_human_agreement_rate": None,
        }

        if len(feedback_df) < settings.RETRAIN_MIN_SAMPLES:
            base_report["reason"] = (
                f"Insufficient samples ({len(feedback_df)} < {settings.RETRAIN_MIN_SAMPLES})"
            )
            # Still compute HITL metrics if the column is present
            self._add_hitl_metrics(feedback_df, base_report)
            return base_report

        # ── Frequency model drift (AUC) ───────────────────────────────────────
        y_true_freq = feedback_df["claim_occurred"].astype(int).values
        y_pred_freq = feedback_df["claim_probability"].astype(float).values
        current_auc = float(roc_auc_score(y_true_freq, y_pred_freq))

        # ── Severity model drift (RMSE) ───────────────────────────────────────
        claims_mask = feedback_df["claim_occurred"] == True
        severity_rmse = None
        if claims_mask.sum() >= 10:
            y_true_sev = feedback_df.loc[claims_mask, "actual_claim_amount_inr"].astype(float).values
            y_pred_sev = feedback_df.loc[claims_mask, "expected_claim_amount_inr"].astype(float).values
            severity_rmse = float(np.sqrt(mean_squared_error(y_true_sev, y_pred_sev)))

        drift_detected = current_auc < AUC_THRESHOLD
        if severity_rmse is not None:
            baseline_rmse = self._get_baseline_rmse()
            if baseline_rmse and severity_rmse > baseline_rmse * (1 + RMSE_DEGRADATION_PCT):
                drift_detected = True

        base_report.update({
            "frequency_auc": round(current_auc, 4),
            "severity_rmse": round(severity_rmse, 2) if severity_rmse is not None else None,
            "drift_detected": drift_detected,
            "retrain_recommended": drift_detected,
        })

        # ── HITL governance metrics ───────────────────────────────────────────
        self._add_hitl_metrics(feedback_df, base_report)

        logger.info("Performance report: %s", base_report)
        return base_report

    def _add_hitl_metrics(self, feedback_df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """
        Compute and insert HITL governance metrics into `report` in-place.

        human_override_rate:    Fraction of HITL-reviewed applications rejected
                                by the human reviewer.
        ml_human_agreement_rate: Fraction of HITL-reviewed applications where
                                the human decision aligned with the ML recommendation:
                                  ML=approve  AND human=approved  → agreement
                                  ML=flag/refer AND human=rejected → agreement
        """
        if "human_decision" not in feedback_df.columns or "agent_decision" not in feedback_df.columns:
            return

        hitl_mask = feedback_df["human_decision"].notna()
        hitl_reviewed_count = int(hitl_mask.sum())
        report["hitl_reviewed_count"] = hitl_reviewed_count

        if hitl_reviewed_count < _MIN_HITL_SAMPLES:
            logger.info(
                "HITL governance metrics not computed: only %d reviewed samples (min %d)",
                hitl_reviewed_count, _MIN_HITL_SAMPLES,
            )
            return

        hitl_df = feedback_df[hitl_mask].copy()

        # human_override_rate: how often did the human reject the application?
        human_override_rate = round(
            float((hitl_df["human_decision"] == "rejected").mean()), 4
        )

        # ml_human_agreement_rate: ML and human reached the same conclusion
        ml_approve_human_approve = (
            (hitl_df["agent_decision"] == "approve") &
            (hitl_df["human_decision"] == "approved")
        )
        ml_flag_human_reject = (
            hitl_df["agent_decision"].isin(["flag", "refer"]) &
            (hitl_df["human_decision"] == "rejected")
        )
        ml_human_agreement_rate = round(
            float((ml_approve_human_approve | ml_flag_human_reject).mean()), 4
        )

        report["human_override_rate"] = human_override_rate
        report["ml_human_agreement_rate"] = ml_human_agreement_rate

        logger.info(
            "HITL governance: reviewed=%d, override_rate=%.4f, agreement_rate=%.4f",
            hitl_reviewed_count, human_override_rate, ml_human_agreement_rate,
        )

    def _get_baseline_rmse(self) -> Optional[float]:
        result = self.db.execute(
            select(ModelVersion)
            .where(and_(ModelVersion.model_type == "severity", ModelVersion.is_active == True))
            .limit(1)
        )
        mv = result.scalar_one_or_none()
        if mv and mv.accuracy_metrics:
            return mv.accuracy_metrics.get("rmse")
        return None

    # ── Step 3: Persist evaluation report ─────────────────────────────────────

    def save_evaluation(self, report: Dict[str, Any]) -> None:
        for model_type in ("frequency", "severity"):
            mv = ModelVersion(
                version_id=uuid.uuid4(),
                version_tag=f"eval-{date.today()}",
                model_type=model_type,
                trained_on=date.today(),
                accuracy_metrics=report,
                is_active=False,
            )
            self.db.add(mv)
        self.db.flush()

    # ── Step 4: Retrain models ─────────────────────────────────────────────────

    def retrain(self, training_df: pd.DataFrame, new_version: str) -> Dict[str, Any]:
        logger.info("Starting retraining with %d rows, version=%s", len(training_df), new_version)

        data = prepare_training_data(training_df, model_dir=self.model_dir)
        splits = data["splits"]

        freq_metrics = {}
        sev_metrics = {}

        if "frequency" in splits:
            freq_model = FrequencyModel(version=new_version, model_dir=self.model_dir)
            s = splits["frequency"]
            freq_metrics = freq_model.train(s["X_train"], s["y_train"], s["X_val"], s["y_val"])
            freq_model.save()
            logger.info("Frequency model trained: %s", freq_metrics)

        if "severity" in splits:
            sev_model = SeverityModel(version=new_version, model_dir=self.model_dir)
            s = splits["severity"]
            sev_metrics = sev_model.train(s["X_train"], s["y_train"], s["X_val"], s["y_val"])
            sev_model.save()
            logger.info("Severity model trained: %s", sev_metrics)

        return {"frequency": freq_metrics, "severity": sev_metrics, "version": new_version}

    # ── Step 5: Calibrate thresholds ──────────────────────────────────────────

    def calibrate_thresholds(self, feedback_df: pd.DataFrame) -> Dict[str, float]:
        probs = feedback_df["claim_probability"].astype(float).values
        p25 = float(np.percentile(probs, 25))
        p70 = float(np.percentile(probs, 70))
        logger.info("Calibrated thresholds: low<%.3f, medium<%.3f, high>=%.3f", p25, p70, p70)
        return {"low_threshold": p25, "high_threshold": p70}

    # ── Step 6: Promote new model ──────────────────────────────────────────────

    def promote_model(self, new_version: str, metrics: Dict[str, Any]) -> None:
        self.db.execute(
            text("UPDATE model_versions SET is_active = false WHERE is_active = true")
        )

        for model_type, type_metrics in [("frequency", metrics.get("frequency", {})),
                                          ("severity", metrics.get("severity", {}))]:
            mv = ModelVersion(
                version_id=uuid.uuid4(),
                version_tag=new_version,
                model_type=model_type,
                trained_on=date.today(),
                accuracy_metrics=type_metrics,
                is_active=True,
            )
            self.db.add(mv)

        self.db.flush()
        swap_predictor(new_version, self.model_dir)
        logger.info("Promoted model version %s", new_version)

    # ── Notifications ──────────────────────────────────────────────────────────

    def send_admin_notification(self, subject: str, body: str) -> None:
        if not settings.SMTP_USER or not settings.ADMIN_EMAIL:
            logger.info("SMTP not configured, skipping notification")
            return
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = settings.SMTP_USER
            msg["To"] = settings.ADMIN_EMAIL
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASS)
                server.send_message(msg)
            logger.info("Notification sent to %s", settings.ADMIN_EMAIL)
        except Exception as e:
            logger.error("Failed to send notification: %s", e)

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def run(self, training_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Full retraining pipeline.

        Phase 1 fix: Attempts to reconstruct the complete applicant feature
        matrix via collect_features_with_labels() so that actual model retraining
        can proceed when drift is detected.  Falls back to the lightweight
        label-only collect if the join fails (e.g. schema not yet migrated).

        Phase 2 fix: HITL governance metrics (human_override_rate,
        ml_human_agreement_rate) are now always computed and included in the
        performance report when sufficient reviewed samples exist.
        """
        logger.info("=== Retraining pipeline started ===")

        # Prefer the enriched DataFrame: needed for both HITL metrics and retraining
        try:
            enriched_df = self.collect_features_with_labels()
        except Exception as exc:
            logger.warning(
                "Feature reconstruction failed (%s) — falling back to label-only collect; "
                "retraining will be skipped if drift is detected",
                exc,
            )
            legacy_df = self.collect_feedback()
            enriched_df = legacy_df  # will lack HITL cols and feature cols

        report = self.evaluate_performance(enriched_df)
        self.save_evaluation(report)

        if not report.get("retrain_recommended"):
            logger.info(
                "Retraining not required: %s",
                report.get("reason", "model performance acceptable"),
            )
            return report

        # Explicit training_df overrides the self-constructed one (e.g. tests)
        df_to_train = training_df if training_df is not None else enriched_df

        # Verify all required columns are present before attempting retraining
        missing_cols = [c for c in _REQUIRED_FEATURE_COLS if c not in df_to_train.columns]
        if missing_cols:
            logger.warning(
                "Training DataFrame missing required columns %s — "
                "drift detected but retraining skipped",
                missing_cols,
            )
            report["retrain_skipped_reason"] = f"Missing columns: {missing_cols}"
            return report

        new_version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        new_metrics = self.retrain(df_to_train, new_version)

        thresholds = self.calibrate_thresholds(enriched_df)
        new_metrics["thresholds"] = thresholds

        freq_auc = new_metrics["frequency"].get("auc_roc", 0)
        sev_r2   = new_metrics["severity"].get("r2", 0)

        if freq_auc >= AUC_THRESHOLD:
            self.promote_model(new_version, new_metrics)
            self.send_admin_notification(
                f"[Insurance AI] New model promoted: {new_version}",
                f"Model {new_version} promoted.\nFrequency AUC: {freq_auc:.4f}\nReport: {new_metrics}",
            )
            report["promoted_version"] = new_version

            # ── Registry champion-challenger evaluation ────────────────────
            engine = _try_promotion_engine()
            if engine is not None:
                freq_model_path = os.path.join(self.model_dir, f"frequency_{new_version}.pkl")
                sev_model_path  = os.path.join(self.model_dir, f"severity_{new_version}.pkl")
                try:
                    # Guard: V1 pipeline must not overwrite V4+ champions.
                    current_freq = engine.registry.get_frequency_champion()
                    current_prep_version = current_freq.get("preprocessor_version", "V1")
                    if current_prep_version != "V1":
                        logger.warning(
                            "V1 retraining pipeline cannot overwrite %s champion — "
                            "skipping registry update to preserve production V4 models.",
                            current_prep_version,
                        )
                    else:
                        reg_result = engine.evaluate_and_promote(
                            freq_metrics=new_metrics["frequency"],
                            sev_metrics=new_metrics["severity"],
                            freq_model_path=freq_model_path,
                            sev_model_path=sev_model_path,
                            freq_algorithm="XGBoost",
                            sev_algorithm="XGBoost",
                            challenger_version=new_version,
                            preprocessor_version="V1",
                            promoted_by="retraining_pipeline",
                        )
                        report["registry_update"] = reg_result
                        logger.info("Registry evaluation: %s", reg_result)
                except Exception as exc:
                    logger.warning("Registry champion evaluation failed (%s) — skipped.", exc)
        else:
            logger.warning(
                "New model AUC %.4f < threshold %.2f — not promoting", freq_auc, AUC_THRESHOLD
            )
            self.send_admin_notification(
                "[Insurance AI] Retraining failed quality gate",
                f"New model {new_version} did not pass AUC threshold "
                f"({freq_auc:.4f} < {AUC_THRESHOLD}).",
            )

        report["retrain_metrics"] = new_metrics
        logger.info("=== Retraining pipeline complete ===")
        return report
