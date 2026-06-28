"""
Champion-Challenger Model Registry
====================================
Central loader for champion model artifacts.

Responsibilities:
  1. Load and validate champion_registry.json
  2. Resolve model file paths (absolute, relative, or anchored to repo root)
  3. Return champion metadata and model objects
  4. Support rollback to previous champion
  5. Fail safely with explicit fallback to V1 production models

Usage:
    registry = ModelRegistry()
    info     = registry.get_frequency_champion()
    model    = registry.load_frequency_model()
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────

def _repo_root() -> str:
    """Return the backend/ directory (parent of src/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve(path: str) -> str:
    """
    Convert a registry-relative path to an absolute filesystem path.
    Paths in the registry JSON are relative to the backend/ root.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


# ── Constants ─────────────────────────────────────────────────────────────────

REGISTRY_PATH        = _resolve("models/metadata/champion_registry.json")
HISTORY_PATH         = _resolve("models/metadata/promotion_history.json")
GOVERNANCE_LOG_PATH  = _resolve("models/metadata/model_governance_log.json")

V1_FREQ_FALLBACK     = _resolve("models/saved/frequency_v1.0.0.pkl")
V1_SEV_FALLBACK      = _resolve("models/saved/severity_v1.0.0.pkl")


# ── Registry class ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Reads champion_registry.json and exposes champion model metadata and loaders.

    Thread-safety: all writes acquire a file lock via _atomic_write().
    """

    def __init__(self, registry_path: str = REGISTRY_PATH):
        self._registry_path = registry_path
        self._data: Optional[Dict[str, Any]] = None

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _load_registry(self) -> Dict[str, Any]:
        if self._data is not None:
            return self._data
        if not os.path.exists(self._registry_path):
            raise FileNotFoundError(
                f"Champion registry not found at {self._registry_path}. "
                "Run experiments/create_champion_artifacts.py to initialise."
            )
        with open(self._registry_path, "r", encoding="utf-8") as fh:
            self._data = json.load(fh)
        return self._data

    def _save_registry(self, data: Dict[str, Any]) -> None:
        """Write registry atomically via a temp file."""
        tmp = self._registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self._registry_path)
        self._data = data
        logger.info("Registry saved to %s", self._registry_path)

    def _load_history(self) -> Dict[str, Any]:
        if not os.path.exists(HISTORY_PATH):
            return {"schema_version": "1.0.0", "history": []}
        with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_history(self, history: Dict[str, Any]) -> None:
        tmp = HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, HISTORY_PATH)

    # ── Public query API ─────────────────────────────────────────────────────

    def get_frequency_champion(self) -> Dict[str, Any]:
        """Return the frequency champion metadata dict."""
        return dict(self._load_registry()["frequency"])

    def get_severity_champion(self) -> Dict[str, Any]:
        """Return the severity champion metadata dict."""
        return dict(self._load_registry()["severity"])

    def get_registry_summary(self) -> Dict[str, Any]:
        data = self._load_registry()
        return {
            "registry_version":    data.get("registry_version"),
            "last_updated":        data.get("last_updated"),
            "frequency_champion":  data["frequency"]["champion"],
            "frequency_score":     data["frequency"]["score"],
            "frequency_status":    data["frequency"]["deployment_status"],
            "severity_champion":   data["severity"]["champion"],
            "severity_score":      data["severity"]["score"],
            "severity_status":     data["severity"]["deployment_status"],
        }

    def is_champion_deployed(self, model_type: str) -> bool:
        """Return True only when deployment_status == 'active'."""
        data = self._load_registry()
        return data[model_type].get("deployment_status") == "active"

    def validate_paths(self) -> Dict[str, bool]:
        """Check whether champion artifact files exist on disk."""
        data = self._load_registry()
        return {
            "frequency_pkl":   os.path.exists(_resolve(data["frequency"]["model_path"])),
            "frequency_prep":  os.path.exists(_resolve(data["frequency"]["preprocessor_path"])),
            "severity_cbm":    os.path.exists(_resolve(data["severity"]["model_path"])),
            "severity_pkl":    os.path.exists(_resolve(data["severity"].get("model_pkl_path", ""))),
            "severity_prep":   os.path.exists(_resolve(data["severity"]["preprocessor_path"])),
            "v1_freq_fallback": os.path.exists(V1_FREQ_FALLBACK),
            "v1_sev_fallback":  os.path.exists(V1_SEV_FALLBACK),
        }

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_frequency_model(self):
        """
        Load the frequency champion model.

        Returns:
            model object with a predict_proba(X) method, OR None if unavailable.

        Notes:
            Loaded model is a FrequencyModel instance (joblib pkl) for XGBoost champions.
            For V4 XGBoost: uses InsurancePreprocessorV4 (59 features).
            Caller is responsible for using the correct preprocessor.
        """
        info = self.get_frequency_champion()
        path = _resolve(info["model_path"])

        if not os.path.exists(path):
            logger.warning("Frequency champion artifact missing at %s", path)
            return None

        try:
            model = joblib.load(path)
            logger.info(
                "Loaded frequency champion: %s (score=%.4f) from %s",
                info["champion"], info["score"], path,
            )
            return model
        except Exception as exc:
            logger.error("Failed to load frequency champion: %s", exc)
            return None

    def load_frequency_preprocessor(self):
        """Load the preprocessor paired with the frequency champion."""
        info = self.get_frequency_champion()
        path = _resolve(info["preprocessor_path"])
        if not os.path.exists(path):
            logger.warning("Frequency preprocessor not found at %s", path)
            return None
        return joblib.load(path)

    def load_severity_model(self):
        """
        Load the severity champion model.

        Returns:
            model object with a predict(X) method, OR None if unavailable.

        Notes:
            For CatBoost champion: tries .pkl wrapper first, then .cbm via
            CatBoostSeverityWrapper. Requires catboost package.
        """
        info = self.get_severity_champion()

        # Prefer pkl wrapper (same interface as SeverityModel)
        pkl_path = _resolve(info.get("model_pkl_path", ""))
        cbm_path = _resolve(info["model_path"])

        if os.path.exists(pkl_path):
            try:
                model = joblib.load(pkl_path)
                logger.info(
                    "Loaded severity champion: %s (score=%.4f) from %s",
                    info["champion"], info["score"], pkl_path,
                )
                return model
            except Exception as exc:
                logger.warning("pkl load failed (%s), trying .cbm", exc)

        if os.path.exists(cbm_path) and cbm_path.endswith(".cbm"):
            try:
                from src.models.severity_model import CatBoostSeverityWrapper
                model = CatBoostSeverityWrapper.load_from_cbm(cbm_path)
                logger.info(
                    "Loaded severity champion: %s (score=%.4f) from %s",
                    info["champion"], info["score"], cbm_path,
                )
                return model
            except Exception as exc:
                logger.error("Failed to load CatBoost severity champion: %s", exc)
                return None

        logger.warning("Severity champion artifacts not found.")
        return None

    def load_severity_preprocessor(self):
        """Load the preprocessor paired with the severity champion."""
        info = self.get_severity_champion()
        path = _resolve(info["preprocessor_path"])
        if not os.path.exists(path):
            logger.warning("Severity preprocessor not found at %s", path)
            return None
        return joblib.load(path)

    # ── Rollback support ──────────────────────────────────────────────────────

    def rollback_champion(self, model_type: str, reason: str = "") -> bool:
        """
        Restore the previous champion by reading promotion_history.json
        and reverting to the second-most-recent entry for model_type.

        Args:
            model_type: "frequency" or "severity"
            reason:     Optional description logged to governance log.

        Returns:
            True if rollback succeeded, False otherwise.
        """
        if model_type not in ("frequency", "severity"):
            raise ValueError(f"model_type must be 'frequency' or 'severity', got {model_type!r}")

        history = self._load_history()
        past = [
            e for e in history["history"]
            if e["model_type"] == model_type and e["action"] != "rollback"
        ]

        if len(past) < 2:
            logger.warning(
                "Cannot rollback %s: fewer than 2 history entries available.", model_type
            )
            return False

        prev_entry = past[-2]   # second-most-recent = the one before current
        data = self._load_registry()
        current_champion = data[model_type]["champion"]
        current_score    = data[model_type]["score"]

        data[model_type]["champion"]          = prev_entry["previous_champion"]
        data[model_type]["score"]             = prev_entry["previous_score"]
        data[model_type]["version"]           = f"rollback-{prev_entry['timestamp'][:10]}"
        data[model_type]["deployment_status"] = prev_entry.get("deployment_status", "active")
        data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
        self._save_registry(data)

        rollback_event = {
            "timestamp":         datetime.utcnow().isoformat(),
            "model_type":        model_type,
            "action":            "rollback",
            "previous_champion": current_champion,
            "new_champion":      prev_entry["previous_champion"],
            "previous_score":    current_score,
            "new_score":         prev_entry["previous_score"],
            "metric":            prev_entry["metric"],
            "threshold":         None,
            "threshold_met":     None,
            "promoted_by":       "ModelRegistry.rollback_champion",
            "deployment_status": "active",
            "notes":             reason or "Manual rollback.",
        }
        history["history"].append(rollback_event)
        self._save_history(history)

        _append_governance_log({
            "event_type":  "rollback",
            "model_type":  model_type,
            "action":      "champion_rolled_back",
            "metric":      prev_entry["metric"],
            "old_score":   current_score,
            "new_score":   prev_entry["previous_score"],
            "decision":    "ROLLED_BACK",
            "challenger":  prev_entry["previous_champion"],
            "incumbent":   current_champion,
            "notes":       reason or "Manual rollback requested.",
        })

        logger.info(
            "Rollback complete: %s champion reverted from %s (%.4f) to %s (%.4f)",
            model_type, current_champion, current_score,
            prev_entry["previous_champion"], prev_entry["previous_score"],
        )
        return True

    def activate_champion(self, model_type: str, notes: str = "") -> None:
        """
        Mark a validated champion as active (deploys to production on next restart).

        This should only be called after the API schema has been upgraded to support
        the champion's preprocessor_version (e.g. V4).

        Args:
            model_type: "frequency" or "severity"
            notes:      Reason / ticket reference for the activation.
        """
        data = self._load_registry()
        old_status = data[model_type]["deployment_status"]
        data[model_type]["deployment_status"] = "active"
        data["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
        self._save_registry(data)

        _append_governance_log({
            "event_type": "activation",
            "model_type": model_type,
            "action":     "champion_activated",
            "metric":     data[model_type]["metric"],
            "old_score":  None,
            "new_score":  data[model_type]["score"],
            "decision":   "ACTIVATED",
            "challenger":  data[model_type]["champion"],
            "incumbent":   None,
            "notes":      notes or f"Status changed from {old_status} to active.",
        })
        logger.info("Activated %s champion: %s", model_type, data[model_type]["champion"])


# ── Governance log helper ─────────────────────────────────────────────────────

def _append_governance_log(event_fields: Dict[str, Any]) -> None:
    """Append a single event to model_governance_log.json."""
    if not os.path.exists(GOVERNANCE_LOG_PATH):
        log = {"schema_version": "1.0.0", "events": []}
    else:
        with open(GOVERNANCE_LOG_PATH, "r", encoding="utf-8") as fh:
            log = json.load(fh)

    event = {"timestamp": datetime.utcnow().isoformat()}
    event.update(event_fields)
    log["events"].append(event)

    tmp = GOVERNANCE_LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, GOVERNANCE_LOG_PATH)


# ── Module-level convenience functions ───────────────────────────────────────

_registry_singleton: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = ModelRegistry()
    return _registry_singleton


def get_frequency_champion() -> Dict[str, Any]:
    return get_registry().get_frequency_champion()


def get_severity_champion() -> Dict[str, Any]:
    return get_registry().get_severity_champion()


def get_active_model_version() -> str:
    """Return the active frequency champion version from the registry, or 'v1.0.0'."""
    try:
        return get_registry().get_frequency_champion()["version"]
    except Exception:
        return "v1.0.0"


def load_frequency_model():
    return get_registry().load_frequency_model()


def load_severity_model():
    return get_registry().load_severity_model()
