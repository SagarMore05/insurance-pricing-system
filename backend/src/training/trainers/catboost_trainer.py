"""CatBoost trainer for frequency (classification) and severity (regression)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.training.trainers.base import BaseTrainer

_FREQ_DEFAULTS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "border_count": 64,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": 0,
    "random_seed": 42,
    "thread_count": -1,
    "auto_class_weights": "Balanced",
}

_SEV_DEFAULTS = {
    "iterations": 600,
    "depth": 3,                 # shallow trees prevent overfitting to noise
    "learning_rate": 0.02,
    "l2_leaf_reg": 8.0,        # strong L2 regularisation
    "border_count": 32,         # fewer candidate split points per feature
    "min_data_in_leaf": 50,    # high leaf floor suppresses noise-driven splits
    "loss_function": "RMSE",
    "verbose": 0,
    "random_seed": 42,
    "thread_count": -1,
}


class CatBoostTrainer(BaseTrainer):
    ALGORITHM = "catboost"

    def __init__(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        defaults = _FREQ_DEFAULTS if model_type == "frequency" else _SEV_DEFAULTS
        merged = {**defaults, **(hyperparameters or {}), "random_seed": random_state}
        super().__init__(model_type, merged, feature_names, random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "CatBoostTrainer":
        from catboost import CatBoostClassifier, CatBoostRegressor

        hp = dict(self.hyperparameters)

        if self.model_type == "frequency":
            self._model = CatBoostClassifier(**hp)
        else:
            hp.pop("auto_class_weights", None)
            self._model = CatBoostRegressor(**hp)

        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return np.maximum(self._model.predict(X), 0.0)

    def get_feature_importance(self) -> Dict[str, float]:
        if self._model is None:
            return {}
        importances = self._model.get_feature_importance()
        names = self.feature_names if self.feature_names else [
            f"f{i}" for i in range(len(importances))
        ]
        pairs = sorted(zip(names, importances.tolist()), key=lambda x: -x[1])
        return {k: round(float(v), 6) for k, v in pairs}

    def _save_model(self, directory: str) -> None:
        joblib.dump(self._model, os.path.join(directory, "model.pkl"))

    @classmethod
    def _load_model(cls, directory: str, instance: "CatBoostTrainer") -> None:
        instance._model = joblib.load(os.path.join(directory, "model.pkl"))
