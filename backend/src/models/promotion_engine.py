"""
Champion-Challenger Promotion Engine
======================================
Compares challenger model metrics against the current champion.
Promotes challengers that exceed defined material-improvement thresholds.

Promotion rules (matching champion_challenger_v4.py experiment thresholds):
  Frequency  — metric: ROC_AUC   — threshold: challenger_auc  > champion_auc  + 0.002
  Severity   — metric: R2        — threshold: challenger_r2   > champion_r2   + 0.010

Usage (called from retraining pipeline after new models are trained):
    engine = PromotionEngine()
    result = engine.evaluate_and_promote(
        freq_metrics={"auc_roc": 0.705},
        sev_metrics={"r2": 0.55},
        freq_model_path="models/challengers/new_frequency.pkl",
        sev_model_path="models/challengers/new_severity.pkl",
        promoted_by="retraining_pipeline_v20261001",
    )
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

Tuple_bool_str = Tuple[bool, str]

from src.models.model_registry import (
    ModelRegistry,
    _append_governance_log,
    HISTORY_PATH,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

FREQ_METRIC     = "ROC_AUC"
FREQ_THRESHOLD  = 0.002   # challenger must exceed champion by at least this

SEV_METRIC      = "R2"
SEV_THRESHOLD   = 0.010   # challenger must exceed champion by at least this


class PromotionEngine:
    """
    Evaluates challenger metrics against the registered champion and
    promotes when improvement thresholds are satisfied.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    # ── Core promotion logic ──────────────────────────────────────────────────

    def should_promote_frequency(
        self,
        challenger_auc: float,
        champion_auc: Optional[float] = None,
    ) -> Tuple_bool_str:
        """
        Returns (promote: bool, reason: str).

        Args:
            challenger_auc: ROC-AUC of the challenger model on the test set.
            champion_auc:   Override champion AUC (reads from registry if None).
        """
        if champion_auc is None:
            champion_auc = self.registry.get_frequency_champion()["score"]

        delta = challenger_auc - champion_auc
        if delta > FREQ_THRESHOLD:
            return True, (
                f"Challenger ROC-AUC {challenger_auc:.4f} > champion {champion_auc:.4f} "
                f"+ threshold {FREQ_THRESHOLD} (delta={delta:+.4f}). PROMOTE."
            )
        return False, (
            f"Challenger ROC-AUC {challenger_auc:.4f} did not clear champion {champion_auc:.4f} "
            f"+ threshold {FREQ_THRESHOLD} (delta={delta:+.4f}). RETAIN champion."
        )

    def should_promote_severity(
        self,
        challenger_r2: float,
        champion_r2: Optional[float] = None,
    ) -> Tuple_bool_str:
        """
        Returns (promote: bool, reason: str).

        Args:
            challenger_r2: R² of the challenger model on the test set.
            champion_r2:   Override champion R² (reads from registry if None).
        """
        if champion_r2 is None:
            champion_r2 = self.registry.get_severity_champion()["score"]

        delta = challenger_r2 - champion_r2
        if delta > SEV_THRESHOLD:
            return True, (
                f"Challenger R² {challenger_r2:.4f} > champion {champion_r2:.4f} "
                f"+ threshold {SEV_THRESHOLD} (delta={delta:+.4f}). PROMOTE."
            )
        return False, (
            f"Challenger R² {challenger_r2:.4f} did not clear champion {champion_r2:.4f} "
            f"+ threshold {SEV_THRESHOLD} (delta={delta:+.4f}). RETAIN champion."
        )

    # ── Artifact promotion ────────────────────────────────────────────────────

    def promote_frequency(
        self,
        challenger_auc: float,
        challenger_model_path: str,
        challenger_algorithm: str = "XGBoost",
        challenger_version: str = None,
        preprocessor_path: str = None,
        preprocessor_version: str = "V1",
        feature_count: int = None,
        promoted_by: str = "manual",
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Evaluate and conditionally promote a frequency challenger.

        Side effects on promotion:
          1. Copies challenger_model_path → models/champion/frequency_best_model.pkl
          2. Updates champion_registry.json
          3. Appends to promotion_history.json
          4. Appends to model_governance_log.json

        Returns:
            True if promoted, False if retained.
        """
        promote, reason = self.should_promote_frequency(challenger_auc)
        logger.info("[Frequency] %s", reason)

        champ_info  = self.registry.get_frequency_champion()
        old_score   = champ_info["score"]
        old_champ   = champ_info["champion"]
        version_tag = challenger_version or f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"

        if not promote:
            _append_governance_log({
                "event_type": "rejection",
                "model_type": "frequency",
                "action":     "challenger_rejected",
                "metric":     FREQ_METRIC,
                "old_score":  old_score,
                "new_score":  challenger_auc,
                "decision":   "RETAIN_CHAMPION",
                "challenger": f"{challenger_algorithm}_{version_tag}",
                "incumbent":  old_champ,
                "notes":      reason,
            })
            return False

        # Copy artifact to champion slot
        dest = _resolve_champion_path("frequency_best_model.pkl")
        _safe_copy(challenger_model_path, dest)
        logger.info("Copied challenger → %s", dest)

        # Build updated registry entry
        data = self.registry._load_registry()
        old_entry = dict(data["frequency"])

        # Determine deployment_status
        deploy_status = "active" if preprocessor_version == "V1" else "validated"
        deploy_note   = (
            "" if preprocessor_version == "V1"
            else f"Requires preprocessor V{preprocessor_version}. "
                 "Activate with ModelRegistry.activate_champion() after API upgrade."
        )

        data["frequency"].update({
            "champion":             f"{challenger_algorithm}_{version_tag}",
            "algorithm":            challenger_algorithm,
            "metric":               FREQ_METRIC,
            "score":                round(challenger_auc, 6),
            "baseline_score":       old_score,
            "model_path":           "models/champion/frequency_best_model.pkl",
            "preprocessor_path":    preprocessor_path or old_entry.get("preprocessor_path", ""),
            "preprocessor_version": preprocessor_version,
            "feature_count":        feature_count or old_entry.get("feature_count"),
            "version":              version_tag,
            "promotion_date":       datetime.utcnow().strftime("%Y-%m-%d"),
            "promoted_by":          promoted_by,
            "deployment_status":    deploy_status,
            "deployment_note":      deploy_note,
            "production_fallback":  old_entry.get("production_fallback", V1_FREQ_FALLBACK),
            **({"eval_metrics": additional_metrics} if additional_metrics else {}),
        })
        data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
        self.registry._save_registry(data)

        # History + governance
        self._append_promotion_history(
            model_type="frequency",
            old_champion=old_champ,
            new_champion=data["frequency"]["champion"],
            old_score=old_score,
            new_score=challenger_auc,
            metric=FREQ_METRIC,
            threshold=FREQ_THRESHOLD,
            promoted_by=promoted_by,
            deployment_status=deploy_status,
            notes=reason,
        )
        _append_governance_log({
            "event_type": "promotion",
            "model_type": "frequency",
            "action":     "challenger_promoted",
            "metric":     FREQ_METRIC,
            "old_score":  old_score,
            "new_score":  challenger_auc,
            "decision":   "PROMOTE",
            "challenger": data["frequency"]["champion"],
            "incumbent":  old_champ,
            "notes":      reason,
        })
        logger.info(
            "Frequency champion updated: %s (%.4f) → %s (%.4f)",
            old_champ, old_score, data["frequency"]["champion"], challenger_auc,
        )
        return True

    def promote_severity(
        self,
        challenger_r2: float,
        challenger_model_path: str,
        challenger_algorithm: str = "XGBoost",
        challenger_version: str = None,
        challenger_pkl_path: Optional[str] = None,
        preprocessor_path: str = None,
        preprocessor_version: str = "V1",
        feature_count: int = None,
        promoted_by: str = "manual",
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Evaluate and conditionally promote a severity challenger.

        Side effects on promotion:
          1. Copies challenger_model_path → models/champion/severity_best_model.cbm (or .pkl)
          2. Updates champion_registry.json
          3. Appends to promotion_history.json
          4. Appends to model_governance_log.json

        Returns:
            True if promoted, False if retained.
        """
        promote, reason = self.should_promote_severity(challenger_r2)
        logger.info("[Severity] %s", reason)

        champ_info  = self.registry.get_severity_champion()
        old_score   = champ_info["score"]
        old_champ   = champ_info["champion"]
        version_tag = challenger_version or f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"

        if not promote:
            _append_governance_log({
                "event_type": "rejection",
                "model_type": "severity",
                "action":     "challenger_rejected",
                "metric":     SEV_METRIC,
                "old_score":  old_score,
                "new_score":  challenger_r2,
                "decision":   "RETAIN_CHAMPION",
                "challenger": f"{challenger_algorithm}_{version_tag}",
                "incumbent":  old_champ,
                "notes":      reason,
            })
            return False

        # Detect model type from extension
        ext  = os.path.splitext(challenger_model_path)[1].lower()
        dest_name = "severity_best_model.cbm" if ext == ".cbm" else "severity_best_model.pkl"
        dest      = _resolve_champion_path(dest_name)
        _safe_copy(challenger_model_path, dest)

        # If separate pkl wrapper provided, copy it too
        if challenger_pkl_path and os.path.exists(challenger_pkl_path):
            dest_pkl = _resolve_champion_path("severity_best_model.pkl")
            _safe_copy(challenger_pkl_path, dest_pkl)

        data = self.registry._load_registry()
        old_entry = dict(data["severity"])

        deploy_status = "active" if preprocessor_version == "V1" else "validated"
        deploy_note   = (
            "" if preprocessor_version == "V1"
            else f"Requires preprocessor V{preprocessor_version}. "
                 "Activate with ModelRegistry.activate_champion() after API upgrade."
        )

        data["severity"].update({
            "champion":             f"{challenger_algorithm}_{version_tag}",
            "algorithm":            challenger_algorithm,
            "metric":               SEV_METRIC,
            "score":                round(challenger_r2, 6),
            "baseline_score":       old_score,
            "model_path":           f"models/champion/{dest_name}",
            "model_pkl_path":       "models/champion/severity_best_model.pkl" if challenger_pkl_path else old_entry.get("model_pkl_path", ""),
            "preprocessor_path":    preprocessor_path or old_entry.get("preprocessor_path", ""),
            "preprocessor_version": preprocessor_version,
            "feature_count":        feature_count or old_entry.get("feature_count"),
            "version":              version_tag,
            "promotion_date":       datetime.utcnow().strftime("%Y-%m-%d"),
            "promoted_by":          promoted_by,
            "deployment_status":    deploy_status,
            "deployment_note":      deploy_note,
            "production_fallback":  old_entry.get("production_fallback", V1_SEV_FALLBACK),
            **({"eval_metrics": additional_metrics} if additional_metrics else {}),
        })
        data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
        self.registry._save_registry(data)

        self._append_promotion_history(
            model_type="severity",
            old_champion=old_champ,
            new_champion=data["severity"]["champion"],
            old_score=old_score,
            new_score=challenger_r2,
            metric=SEV_METRIC,
            threshold=SEV_THRESHOLD,
            promoted_by=promoted_by,
            deployment_status=deploy_status,
            notes=reason,
        )
        _append_governance_log({
            "event_type": "promotion",
            "model_type": "severity",
            "action":     "challenger_promoted",
            "metric":     SEV_METRIC,
            "old_score":  old_score,
            "new_score":  challenger_r2,
            "decision":   "PROMOTE",
            "challenger": data["severity"]["champion"],
            "incumbent":  old_champ,
            "notes":      reason,
        })
        logger.info(
            "Severity champion updated: %s (%.4f) → %s (%.4f)",
            old_champ, old_score, data["severity"]["champion"], challenger_r2,
        )
        return True

    # ── Combined evaluation ───────────────────────────────────────────────────

    def evaluate_and_promote(
        self,
        freq_metrics: Dict[str, float],
        sev_metrics: Dict[str, float],
        freq_model_path: str,
        sev_model_path: str,
        freq_algorithm: str = "XGBoost",
        sev_algorithm: str = "XGBoost",
        challenger_version: str = None,
        freq_preprocessor_path: str = None,
        sev_preprocessor_path: str = None,
        preprocessor_version: str = "V1",
        feature_count: int = None,
        sev_model_pkl_path: Optional[str] = None,
        promoted_by: str = "retraining_pipeline",
    ) -> Dict[str, Any]:
        """
        Evaluate both challenger models and promote where thresholds are met.

        Args:
            freq_metrics: dict with at least {"auc_roc": float}
            sev_metrics:  dict with at least {"r2": float}
            freq_model_path: path to challenger frequency model artifact
            sev_model_path:  path to challenger severity model artifact
            ...

        Returns:
            {
              "frequency_promoted": bool,
              "severity_promoted":  bool,
              "frequency_reason":   str,
              "severity_reason":    str,
            }
        """
        version_tag = challenger_version or f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"

        freq_auc = float(freq_metrics.get("auc_roc", 0.0))
        sev_r2   = float(sev_metrics.get("r2", 0.0))

        _, freq_reason = self.should_promote_frequency(freq_auc)
        _, sev_reason  = self.should_promote_severity(sev_r2)

        freq_promoted = self.promote_frequency(
            challenger_auc=freq_auc,
            challenger_model_path=freq_model_path,
            challenger_algorithm=freq_algorithm,
            challenger_version=version_tag,
            preprocessor_path=freq_preprocessor_path,
            preprocessor_version=preprocessor_version,
            feature_count=feature_count,
            promoted_by=promoted_by,
            additional_metrics=freq_metrics,
        )

        sev_promoted = self.promote_severity(
            challenger_r2=sev_r2,
            challenger_model_path=sev_model_path,
            challenger_algorithm=sev_algorithm,
            challenger_version=version_tag,
            challenger_pkl_path=sev_model_pkl_path,
            preprocessor_path=sev_preprocessor_path,
            preprocessor_version=preprocessor_version,
            feature_count=feature_count,
            promoted_by=promoted_by,
            additional_metrics=sev_metrics,
        )

        return {
            "frequency_promoted": freq_promoted,
            "severity_promoted":  sev_promoted,
            "frequency_reason":   freq_reason,
            "severity_reason":    sev_reason,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _append_promotion_history(
        model_type, old_champion, new_champion,
        old_score, new_score, metric, threshold,
        promoted_by, deployment_status, notes,
    ) -> None:
        if not os.path.exists(HISTORY_PATH):
            history = {"schema_version": "1.0.0", "history": []}
        else:
            with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
                history = json.load(fh)

        history["history"].append({
            "timestamp":         datetime.utcnow().isoformat(),
            "model_type":        model_type,
            "action":            "promotion",
            "previous_champion": old_champion,
            "new_champion":      new_champion,
            "previous_score":    old_score,
            "new_score":         new_score,
            "delta":             round(new_score - old_score, 6),
            "metric":            metric,
            "threshold":         threshold,
            "threshold_met":     True,
            "promoted_by":       promoted_by,
            "deployment_status": deployment_status,
            "notes":             notes,
        })
        tmp = HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, HISTORY_PATH)


# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_champion_path(filename: str) -> str:
    return os.path.join(_repo_root(), "models", "champion", filename)


V1_FREQ_FALLBACK = os.path.join(_repo_root(), "models", "saved", "frequency_v1.0.0.pkl")
V1_SEV_FALLBACK  = os.path.join(_repo_root(), "models", "saved", "severity_v1.0.0.pkl")


def _safe_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Copied %s → %s", src, dst)


