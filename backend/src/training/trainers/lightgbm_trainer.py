"""LightGBM trainer for frequency (classification) and severity (regression)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.training.trainers.base import BaseTrainer

_FREQ_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

_SEV_DEFAULTS = {
    "n_estimators": 600,
    "num_leaves": 15,           # few leaves prevent noise-overfitting
    "max_depth": 4,
    "learning_rate": 0.02,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_samples": 100,   # minimum samples per leaf
    "min_split_gain": 0.5,      # minimum gain to split a leaf
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "objective": "regression_l2",
    "metric": "rmse",
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


class LightGBMTrainer(BaseTrainer):
    ALGORITHM = "lightgbm"

    def __init__(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        defaults = _FREQ_DEFAULTS if model_type == "frequency" else _SEV_DEFAULTS
        merged = {**defaults, **(hyperparameters or {}), "random_state": random_state}
        super().__init__(model_type, merged, feature_names, random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMTrainer":
        from lightgbm import LGBMClassifier, LGBMRegressor

        hp = dict(self.hyperparameters)

        if self.model_type == "frequency":
            self._model = LGBMClassifier(**hp)
        else:
            hp.pop("is_unbalance", None)
            self._model = LGBMRegressor(**hp)

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
        importances = self._model.feature_importances_
        names = self.feature_names if self.feature_names else [
            f"f{i}" for i in range(len(importances))
        ]
        pairs = sorted(zip(names, importances.tolist()), key=lambda x: -x[1])
        return {k: round(float(v), 6) for k, v in pairs}

    def _save_model(self, directory: str) -> None:
        joblib.dump(self._model, os.path.join(directory, "model.pkl"))

    @classmethod
    def _load_model(cls, directory: str, instance: "LightGBMTrainer") -> None:
        instance._model = joblib.load(os.path.join(directory, "model.pkl"))
