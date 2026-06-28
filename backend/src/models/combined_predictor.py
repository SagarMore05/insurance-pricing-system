import logging
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple
from src.preprocessing.pipeline import InsurancePreprocessor
from src.models.frequency_model import FrequencyModel
from src.models.severity_model import SeverityModel

logger = logging.getLogger(__name__)


class CombinedPredictor:
    """
    Combines frequency and severity models to compute expected loss.
    expected_loss = P(claim) × E[claim_amount | claim occurred]

    Loading order (each step falls back to the next on failure):
      1. Champion registry  — loads models registered as deployment_status="active"
      2. V1 legacy path     — models/saved/frequency_v1.0.0.pkl + severity_v1.0.0.pkl
    """

    def __init__(self, version: str = "v1.0.0", model_dir: str = "models/saved"):
        self.version = version
        self.model_dir = model_dir
        self.preprocessor: InsurancePreprocessor = None
        self.frequency_model: FrequencyModel = None
        self.severity_model: SeverityModel = None
        self._loaded_from_registry: bool = False

    def load_all(self) -> None:
        """
        Load preprocessor + frequency + severity models.

        Tries the champion registry first (only for models with deployment_status="active").
        Falls back to the versioned V1 legacy path when the registry is unavailable,
        points to undeployed champions, or any artifact is missing.

        Business logic (predict, predict_batch, pricing engine) is unchanged.
        """
        # ── Attempt 1: Champion registry ─────────────────────────────────────
        registry_ok = self._try_load_from_registry()
        if registry_ok:
            return

        # ── Attempt 2: V1 legacy path (original behaviour) ───────────────────
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        freq_path         = os.path.join(self.model_dir, f"frequency_{self.version}.pkl")
        sev_path          = os.path.join(self.model_dir, f"severity_{self.version}.pkl")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        if not os.path.exists(freq_path):
            raise FileNotFoundError(f"Frequency model not found at {freq_path}")
        if not os.path.exists(sev_path):
            raise FileNotFoundError(f"Severity model not found at {sev_path}")

        self.preprocessor     = InsurancePreprocessor.load(preprocessor_path)
        self.frequency_model  = FrequencyModel.load(freq_path)
        self.severity_model   = SeverityModel.load(sev_path)
        self._loaded_from_registry = False
        logger.info("Models loaded from V1 legacy path (version=%s)", self.version)

    def _try_load_from_registry(self) -> bool:
        """
        Attempt to load active champion models from the model registry.

        Returns True only when ALL THREE components are loaded successfully:
          preprocessor + frequency_model + severity_model

        Registry models are only loaded when deployment_status == "active".
        Currently, V4 champions are "validated" (pending API V4 upgrade),
        so this method returns False and the caller falls back to V1 legacy.
        """
        try:
            from src.models.model_registry import ModelRegistry
            registry = ModelRegistry()

            freq_info = registry.get_frequency_champion()
            sev_info  = registry.get_severity_champion()

            freq_active = freq_info.get("deployment_status") == "active"
            sev_active  = sev_info.get("deployment_status") == "active"

            if not freq_active and not sev_active:
                logger.debug(
                    "Registry champions not active (freq=%s, sev=%s) — using V1 fallback.",
                    freq_info.get("deployment_status"),
                    sev_info.get("deployment_status"),
                )
                return False

            loaded_freq = loaded_sev = loaded_prep = False

            # Load frequency champion if active
            if freq_active:
                model = registry.load_frequency_model()
                if model is not None:
                    self.frequency_model = model
                    loaded_freq = True
                    logger.info(
                        "Frequency champion loaded from registry: %s (AUC=%.4f)",
                        freq_info["champion"], freq_info["score"],
                    )

            # Load severity champion if active
            if sev_active:
                model = registry.load_severity_model()
                if model is not None:
                    self.severity_model = model
                    loaded_sev = True
                    logger.info(
                        "Severity champion loaded from registry: %s (R2=%.4f)",
                        sev_info["champion"], sev_info["score"],
                    )

            # Preprocessor: use the frequency champion's preprocessor when active,
            # otherwise use the severity champion's preprocessor.
            prep_info  = freq_info if freq_active else sev_info
            prep_path  = prep_info.get("preprocessor_path", "")
            prep_version = prep_info.get("preprocessor_version", "V1")

            if prep_path and os.path.exists(self._abs(prep_path)):
                self.preprocessor = joblib.load(self._abs(prep_path))
                loaded_prep = True
                logger.info("Preprocessor V%s loaded from registry.", prep_version)

            if loaded_freq and loaded_sev and loaded_prep:
                self._loaded_from_registry = True
                return True

            # Partial load — fall back entirely to V1 for safety
            logger.warning(
                "Partial registry load (freq=%s, sev=%s, prep=%s) — reverting to V1 fallback.",
                loaded_freq, loaded_sev, loaded_prep,
            )
            self.frequency_model = None
            self.severity_model  = None
            self.preprocessor    = None
            return False

        except Exception as exc:
            logger.warning("Registry load failed (%s) — falling back to V1 models.", exc)
            self.frequency_model = None
            self.severity_model  = None
            self.preprocessor    = None
            return False

    @staticmethod
    def _abs(rel_path: str) -> str:
        """Resolve a registry-relative path to absolute."""
        if os.path.isabs(rel_path):
            return rel_path
        backend_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        return os.path.join(backend_root, rel_path)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            input_data: Dict with all raw customer/vehicle/driving fields.
        Returns:
            Dict with claim_probability, expected_claim_amount_inr, expected_loss.
        """
        df = pd.DataFrame([input_data])
        X = self.preprocessor.transform(df)

        claim_probability     = float(self.frequency_model.predict_proba(X)[0])
        expected_claim_amount = float(self.severity_model.predict(X)[0])
        expected_loss         = claim_probability * expected_claim_amount

        return {
            "claim_probability":         round(claim_probability, 5),
            "expected_claim_amount_inr": round(expected_claim_amount, 2),
            "expected_loss_inr":         round(expected_loss, 2),
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = self.preprocessor.transform(df)
        claim_probs      = self.frequency_model.predict_proba(X)
        expected_amounts = self.severity_model.predict(X)
        expected_losses  = claim_probs * expected_amounts

        return {
            "claim_probability":         claim_probs,
            "expected_claim_amount_inr": expected_amounts,
            "expected_loss_inr":         expected_losses,
        }

    def is_ready(self) -> bool:
        return all([
            self.preprocessor    is not None,
            self.frequency_model is not None,
            self.severity_model  is not None,
        ])


# ── Global singleton — loaded at startup ─────────────────────────────────────

_predictor: CombinedPredictor = None


def get_predictor(version: str = None, model_dir: str = "models/saved") -> CombinedPredictor:
    global _predictor
    if _predictor is None or not _predictor.is_ready():
        from src.config import settings
        v = version or settings.MODEL_VERSION
        _predictor = CombinedPredictor(version=v, model_dir=model_dir)
        try:
            _predictor.load_all()
        except FileNotFoundError:
            # Return unloaded predictor in dev if models don't exist yet
            pass
    return _predictor


def swap_predictor(new_version: str, model_dir: str = "models/saved") -> None:
    """Atomically swap the in-memory predictor to a new version."""
    global _predictor
    new = CombinedPredictor(version=new_version, model_dir=model_dir)
    new.load_all()
    _predictor = new


def reload_predictor() -> tuple[bool, str]:
    """
    Hot-reload the in-memory predictor from the current champion registry.

    Called after a Phase 5B champion promotion to activate new models
    without restarting the server.

    Returns:
        (success: bool, message: str)
    """
    global _predictor
    try:
        from src.config import settings
        new = CombinedPredictor(version=settings.MODEL_VERSION, model_dir="models/saved")
        new.load_all()
        if not new.is_ready():
            return False, "New predictor loaded but is_ready() returned False."
        _predictor = new
        logger.info("Predictor hot-reloaded successfully after champion promotion.")
        return True, "Predictor reloaded from updated champion registry."
    except Exception as exc:
        logger.error("reload_predictor() failed: %s", exc)
        return False, f"Reload failed: {exc}"
