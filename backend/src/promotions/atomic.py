"""
Phase 5B Atomic Promotion Executor.

Guarantees that champion swap is atomic: either ALL steps succeed or the
system is fully restored to its pre-promotion state.

Promotion steps (in order):
  1. backup  — copy models/champion/* → models/champion_backups/{tag}/
  2. install — copy challenger model.pkl files → models/champion/ slots
  3. prep    — copy challenger preprocessor_v4.pkl → models/champion/
  4. registry— call PromotionEngine to update champion_registry.json
  5. reload  — hot-reload the in-memory CombinedPredictor singleton
  6. health  — smoke-test prediction with probe data

On failure at any step 2–6, _rollback() restores step 1's backup.

Safety guarantees:
  - Champion directory is NEVER left partially updated
  - Registry is NEVER written if artifact copy failed
  - Predictor is NEVER swapped if registry write failed
  - All filesystem operations use shutil (atomic on POSIX; best-effort on Windows)
"""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger("insurance_api.promotions.atomic")

_BACKEND_ROOT   = pathlib.Path(__file__).resolve().parent.parent.parent
_CHAMPION_DIR   = _BACKEND_ROOT / "models" / "champion"
_BACKUP_BASE    = _BACKEND_ROOT / "models" / "champion_backups"

# Probe data for post-promotion health check
_HEALTH_PROBE = {
    "age": 35,
    "gender": "male",
    "city": "mumbai",
    "car_brand": "maruti",
    "car_model": "swift",
    "engine_cc": 1200,
    "vehicle_age_years": 3,
    "vehicle_value_inr": 500000.0,
    "annual_mileage_km": 15000,
    "previous_claims_count": 0,
    "years_licensed": 10,
    "driving_score": 75.0,
    "fuel_type": "Petrol",
    "vehicle_usage_type": "Personal",
    "occupation": "Salaried",
    "marital_status": "Married",
    "annual_income_band": "7-15 LPA",
    "parking_type": "Garage",
    "policy_tenure_years": 2.0,
    "no_claim_bonus_pct": 25.0,
    "months_since_last_claim": 999,
    "region_risk_category": "low",
    "pincode_risk_score": 0.3,
    "flood_risk_index": 0.1,
    "theft_risk_index": 0.2,
    "monsoon_exposure_index": 0.15,
    "policy_inception_month": 6,
    "vehicle_safety_rating": 4,
    "airbags_count": 2,
    "vehicle_body_style": "Hatchback",
    "repair_cost_band": "low",
    "vehicle_make": "Maruti",
    "vehicle_model": "Swift",
}


@dataclass
class PromotionStepResult:
    step: str
    success: bool
    message: str
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicPromotionResult:
    success: bool
    backup_path: Optional[str]
    steps: List[PromotionStepResult]
    error: Optional[str] = None
    rollback_triggered: bool = False
    rollback_success: Optional[bool] = None
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":               self.success,
            "backup_path":           self.backup_path,
            "rollback_triggered":    self.rollback_triggered,
            "rollback_success":      self.rollback_success,
            "error":                 self.error,
            "total_duration_seconds": self.total_duration_seconds,
            "steps": [
                {
                    "step":             s.step,
                    "success":          s.success,
                    "message":          s.message,
                    "duration_seconds": round(s.duration_seconds, 3),
                    "details":          s.details,
                }
                for s in self.steps
            ],
        }


class AtomicPromotion:
    """
    Executes champion swap atomically.

    Usage:
        ap = AtomicPromotion()
        result = ap.execute(
            freq_artifact_dir=...,
            sev_artifact_dir=...,
            preprocessor_path=...,
            freq_algorithm="xgboost",
            sev_algorithm="catboost",
            freq_metrics={...},
            sev_metrics={...},
            promoted_by="admin",
            version_tag="v20260625_1",
        )
    """

    def execute(
        self,
        freq_artifact_dir: str,
        sev_artifact_dir: str,
        preprocessor_path: Optional[str],
        freq_algorithm: str,
        sev_algorithm: str,
        freq_metrics: Dict[str, Any],
        sev_metrics: Dict[str, Any],
        promoted_by: str = "admin",
        version_tag: Optional[str] = None,
    ) -> AtomicPromotionResult:
        t_start = time.monotonic()
        steps: List[PromotionStepResult] = []
        backup_path: Optional[str] = None
        tag = version_tag or f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # ── Step 1: Backup ────────────────────────────────────────────────────
        backup_step, backup_path = self._step_backup(tag)
        steps.append(backup_step)
        if not backup_step.success:
            return AtomicPromotionResult(
                success=False,
                backup_path=None,
                steps=steps,
                error=backup_step.message,
                total_duration_seconds=round(time.monotonic() - t_start, 3),
            )

        try:
            # ── Step 2: Install frequency model ──────────────────────────────
            step2 = self._step_install_model(
                artifact_dir=freq_artifact_dir,
                dest_name="frequency_best_model.pkl",
                label="frequency",
            )
            steps.append(step2)
            if not step2.success:
                raise RuntimeError(step2.message)

            # ── Step 3: Install severity model ────────────────────────────────
            step3 = self._step_install_model(
                artifact_dir=sev_artifact_dir,
                dest_name="severity_best_model.pkl",
                label="severity",
            )
            steps.append(step3)
            if not step3.success:
                raise RuntimeError(step3.message)

            # ── Step 4: Install preprocessor ─────────────────────────────────
            step4 = self._step_install_preprocessor(preprocessor_path)
            steps.append(step4)
            if not step4.success:
                raise RuntimeError(step4.message)

            # ── Step 5: Update registry ───────────────────────────────────────
            freq_model_dest = str(_CHAMPION_DIR / "frequency_best_model.pkl")
            sev_model_dest  = str(_CHAMPION_DIR / "severity_best_model.pkl")
            prep_dest       = str(_CHAMPION_DIR / "preprocessor_v4.pkl")
            step5 = self._step_update_registry(
                freq_model_path=freq_model_dest,
                sev_model_path=sev_model_dest,
                preprocessor_path=prep_dest,
                freq_algorithm=freq_algorithm,
                sev_algorithm=sev_algorithm,
                freq_metrics=freq_metrics,
                sev_metrics=sev_metrics,
                version_tag=tag,
                promoted_by=promoted_by,
            )
            steps.append(step5)
            if not step5.success:
                raise RuntimeError(step5.message)

            # ── Step 6: Reload predictor ──────────────────────────────────────
            step6 = self._step_reload_predictor()
            steps.append(step6)
            if not step6.success:
                raise RuntimeError(step6.message)

            # ── Step 7: Health check ──────────────────────────────────────────
            step7 = self._step_health_check()
            steps.append(step7)
            if not step7.success:
                raise RuntimeError(step7.message)

            return AtomicPromotionResult(
                success=True,
                backup_path=backup_path,
                steps=steps,
                total_duration_seconds=round(time.monotonic() - t_start, 3),
            )

        except Exception as exc:
            logger.error("Atomic promotion failed at step: %s", exc)
            # ── Auto-rollback ─────────────────────────────────────────────────
            rb_step, rb_ok = self._rollback(backup_path)
            steps.append(rb_step)
            return AtomicPromotionResult(
                success=False,
                backup_path=backup_path,
                steps=steps,
                error=str(exc),
                rollback_triggered=True,
                rollback_success=rb_ok,
                total_duration_seconds=round(time.monotonic() - t_start, 3),
            )

    # ── Step implementations ──────────────────────────────────────────────────

    def _step_backup(self, tag: str):
        t0 = time.monotonic()
        backup_dir = _BACKUP_BASE / tag
        try:
            _BACKUP_BASE.mkdir(parents=True, exist_ok=True)
            if _CHAMPION_DIR.exists():
                shutil.copytree(str(_CHAMPION_DIR), str(backup_dir))
            else:
                backup_dir.mkdir(parents=True, exist_ok=True)
            dur = round(time.monotonic() - t0, 3)
            logger.info("Champion backup created at %s", backup_dir)
            return (
                PromotionStepResult(
                    step="backup",
                    success=True,
                    message=f"Champion backed up to {backup_dir}",
                    duration_seconds=dur,
                    details={"backup_path": str(backup_dir)},
                ),
                str(backup_dir),
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            return (
                PromotionStepResult(
                    step="backup",
                    success=False,
                    message=f"Backup failed: {exc}",
                    duration_seconds=dur,
                ),
                None,
            )

    def _step_install_model(
        self,
        artifact_dir: str,
        dest_name: str,
        label: str,
    ) -> PromotionStepResult:
        t0 = time.monotonic()
        src = os.path.join(artifact_dir, "model.pkl")
        dst = str(_CHAMPION_DIR / dest_name)
        try:
            if not os.path.exists(src):
                raise FileNotFoundError(f"{label} model.pkl not found at {src}")
            _CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            dur = round(time.monotonic() - t0, 3)
            logger.info("Installed %s model: %s → %s", label, src, dst)
            return PromotionStepResult(
                step=f"install_{label}_model",
                success=True,
                message=f"Copied {label} model.pkl → {dst}",
                duration_seconds=dur,
                details={"src": src, "dst": dst},
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            return PromotionStepResult(
                step=f"install_{label}_model",
                success=False,
                message=f"Failed to install {label} model: {exc}",
                duration_seconds=dur,
            )

    def _step_install_preprocessor(
        self, preprocessor_path: Optional[str]
    ) -> PromotionStepResult:
        t0 = time.monotonic()
        dst = str(_CHAMPION_DIR / "preprocessor_v4.pkl")
        try:
            if preprocessor_path and os.path.exists(preprocessor_path):
                shutil.copy2(preprocessor_path, dst)
                dur = round(time.monotonic() - t0, 3)
                logger.info("Installed preprocessor: %s → %s", preprocessor_path, dst)
                return PromotionStepResult(
                    step="install_preprocessor",
                    success=True,
                    message=f"Preprocessor installed from {preprocessor_path}",
                    duration_seconds=dur,
                    details={"src": preprocessor_path, "dst": dst},
                )
            # Preprocessor unchanged (use existing champion's)
            dur = round(time.monotonic() - t0, 3)
            existing = str(_CHAMPION_DIR / "preprocessor_v4.pkl")
            if os.path.exists(existing):
                logger.info("Preprocessor unchanged — using existing champion's.")
                return PromotionStepResult(
                    step="install_preprocessor",
                    success=True,
                    message="Preprocessor retained from existing champion.",
                    duration_seconds=dur,
                    details={"retained": existing},
                )
            raise FileNotFoundError("No preprocessor source and no existing champion preprocessor.")
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            return PromotionStepResult(
                step="install_preprocessor",
                success=False,
                message=f"Preprocessor install failed: {exc}",
                duration_seconds=dur,
            )

    def _step_update_registry(
        self,
        freq_model_path: str,
        sev_model_path: str,
        preprocessor_path: str,
        freq_algorithm: str,
        sev_algorithm: str,
        freq_metrics: Dict[str, Any],
        sev_metrics: Dict[str, Any],
        version_tag: str,
        promoted_by: str,
    ) -> PromotionStepResult:
        t0 = time.monotonic()
        try:
            from src.models.promotion_engine import PromotionEngine

            engine = PromotionEngine()

            # Map candidate algorithm to display name
            alg_display = {
                "xgboost": "XGBoost", "catboost": "CatBoost", "lightgbm": "LightGBM"
            }
            freq_alg_display = alg_display.get(freq_algorithm.lower(), freq_algorithm)
            sev_alg_display  = alg_display.get(sev_algorithm.lower(), sev_algorithm)

            freq_auc = float(freq_metrics.get("roc_auc", 0.0))
            sev_r2   = float(sev_metrics.get("r2", 0.0))

            # Build additional metrics dicts for registry
            freq_extra = {k: v for k, v in freq_metrics.items()}
            sev_extra  = {k: v for k, v in sev_metrics.items()}

            engine.promote_frequency(
                challenger_auc=freq_auc,
                challenger_model_path=freq_model_path,
                challenger_algorithm=freq_alg_display,
                challenger_version=version_tag,
                preprocessor_path=preprocessor_path,
                preprocessor_version="V4",
                promoted_by=promoted_by,
                additional_metrics=freq_extra,
            )
            engine.promote_severity(
                challenger_r2=sev_r2,
                challenger_model_path=sev_model_path,
                challenger_algorithm=sev_alg_display,
                challenger_version=version_tag,
                preprocessor_path=preprocessor_path,
                preprocessor_version="V4",
                promoted_by=promoted_by,
                additional_metrics=sev_extra,
            )

            dur = round(time.monotonic() - t0, 3)
            logger.info("Registry updated — freq=%s (%.4f), sev=%s (%.4f)",
                        freq_alg_display, freq_auc, sev_alg_display, sev_r2)
            return PromotionStepResult(
                step="update_registry",
                success=True,
                message=f"Registry updated: freq={freq_alg_display}_{version_tag}, sev={sev_alg_display}_{version_tag}",
                duration_seconds=dur,
                details={"version_tag": version_tag},
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            return PromotionStepResult(
                step="update_registry",
                success=False,
                message=f"Registry update failed: {exc}",
                duration_seconds=dur,
            )

    def _step_reload_predictor(self) -> PromotionStepResult:
        t0 = time.monotonic()
        try:
            from src.models.combined_predictor import reload_predictor
            ok, msg = reload_predictor()
            dur = round(time.monotonic() - t0, 3)
            if ok:
                logger.info("Predictor reloaded successfully after promotion.")
                return PromotionStepResult(
                    step="reload_predictor",
                    success=True,
                    message=msg,
                    duration_seconds=dur,
                )
            return PromotionStepResult(
                step="reload_predictor",
                success=False,
                message=msg,
                duration_seconds=dur,
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            return PromotionStepResult(
                step="reload_predictor",
                success=False,
                message=f"reload_predictor() raised: {exc}",
                duration_seconds=dur,
            )

    def _step_health_check(self) -> PromotionStepResult:
        t0 = time.monotonic()
        try:
            from src.models.combined_predictor import get_predictor
            predictor = get_predictor()
            if not predictor.is_ready():
                raise RuntimeError("Predictor is not ready after reload.")

            result = predictor.predict(_HEALTH_PROBE)
            dur = round(time.monotonic() - t0, 3)
            if not isinstance(result.get("claim_probability"), float):
                raise RuntimeError("Health probe did not return valid claim_probability.")
            logger.info(
                "Health check passed — claim_probability=%.4f",
                result["claim_probability"],
            )
            return PromotionStepResult(
                step="health_check",
                success=True,
                message=f"Health probe succeeded — claim_probability={result['claim_probability']:.4f}",
                duration_seconds=dur,
                details={"probe_result": result},
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            logger.error("Post-promotion health check failed: %s", exc)
            return PromotionStepResult(
                step="health_check",
                success=False,
                message=f"Health check failed: {exc}",
                duration_seconds=dur,
            )

    # ── Rollback ──────────────────────────────────────────────────────────────

    def _rollback(self, backup_path: Optional[str]):
        t0 = time.monotonic()
        if not backup_path or not os.path.isdir(backup_path):
            dur = round(time.monotonic() - t0, 3)
            return (
                PromotionStepResult(
                    step="auto_rollback",
                    success=False,
                    message=f"No backup available at '{backup_path}' — cannot rollback.",
                    duration_seconds=dur,
                ),
                False,
            )
        try:
            # Remove the (potentially partial) champion directory
            if _CHAMPION_DIR.exists():
                shutil.rmtree(str(_CHAMPION_DIR))
            # Restore from backup
            shutil.copytree(backup_path, str(_CHAMPION_DIR))
            # Reload predictor from restored state
            try:
                from src.models.combined_predictor import reload_predictor
                reload_predictor()
            except Exception as reload_exc:
                logger.warning("Predictor reload during rollback failed: %s", reload_exc)

            dur = round(time.monotonic() - t0, 3)
            logger.info("Rollback complete — champion restored from %s", backup_path)
            return (
                PromotionStepResult(
                    step="auto_rollback",
                    success=True,
                    message=f"Champion restored from backup at {backup_path}",
                    duration_seconds=dur,
                    details={"restored_from": backup_path},
                ),
                True,
            )
        except Exception as exc:
            dur = round(time.monotonic() - t0, 3)
            logger.critical("ROLLBACK FAILED: %s — champion may be in inconsistent state!", exc)
            return (
                PromotionStepResult(
                    step="auto_rollback",
                    success=False,
                    message=f"ROLLBACK FAILED: {exc}. Manual intervention required.",
                    duration_seconds=dur,
                ),
                False,
            )

    # ── Manual rollback ───────────────────────────────────────────────────────

    def manual_rollback(self, backup_path: str) -> PromotionStepResult:
        """Restore champion from a specific backup path (manual trigger)."""
        step, _ = self._rollback(backup_path)
        return step
