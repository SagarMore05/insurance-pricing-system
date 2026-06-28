"""
Abstract base trainer — common interface for XGBoost, CatBoost, and LightGBM.

Every concrete trainer must implement:
  train(X_train, y_train) → self
  predict_proba(X) → np.ndarray   (frequency: probabilities)
  predict(X) → np.ndarray          (severity: amounts)
  get_feature_importance() → Dict[str, float]
  save(directory: str) → str       (returns artifact path)
  load(directory: str) → Trainer   (classmethod)
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import joblib
import numpy as np


class BaseTrainer(ABC):
    """
    Shared contract for frequency + severity candidate trainers.

    Each concrete subclass wraps one algorithm (XGBoost / CatBoost / LightGBM)
    and exposes a uniform interface so V5TrainingEngine can iterate over them.
    """

    #: override in subclass
    ALGORITHM: str = "base"

    def __init__(
        self,
        model_type: str,              # "frequency" | "severity"
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        if model_type not in ("frequency", "severity"):
            raise ValueError(f"model_type must be 'frequency' or 'severity', got {model_type!r}")
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.feature_names = feature_names or []
        self.random_state = random_state
        self._model = None
        self._fitted = False

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseTrainer":
        """Fit the model. Returns self for chaining."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return claim probability for each sample (frequency models)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted claim amount for each sample (severity models)."""

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature name → importance score mapping."""

    @abstractmethod
    def _save_model(self, directory: str) -> None:
        """Save model-specific artifact to `directory`."""

    @classmethod
    @abstractmethod
    def _load_model(cls, directory: str, instance: "BaseTrainer") -> None:
        """Load model-specific artifact from `directory` into `instance`."""

    # ── Shared save/load (metadata + model) ───────────────────────────────────

    def save(self, directory: str) -> str:
        """
        Save model artifact + metadata.json + feature_importance.json to `directory`.
        Returns the directory path.
        """
        os.makedirs(directory, exist_ok=True)
        self._save_model(directory)

        importance = self.get_feature_importance()
        with open(os.path.join(directory, "feature_importance.json"), "w") as f:
            json.dump(importance, f, indent=2)

        metadata = {
            "algorithm": self.ALGORITHM,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "random_state": self.random_state,
        }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return directory

    @classmethod
    def load(cls, directory: str) -> "BaseTrainer":
        """Reconstruct a trainer from a saved directory."""
        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)

        instance = cls(
            model_type=meta["model_type"],
            hyperparameters=meta.get("hyperparameters", {}),
            feature_names=meta.get("feature_names", []),
            random_state=meta.get("random_state", 42),
        )
        cls._load_model(directory, instance)
        instance._fitted = True
        return instance

    # ── Convenience: assert fitted ─────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be trained before inference.")
