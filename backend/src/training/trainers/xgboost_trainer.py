"""XGBoost trainer for frequency (classification) and severity (regression)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.training.trainers.base import BaseTrainer

# Default hyperparameters tuned for insurance V5 training
_FREQ_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 4,      # ~4:1 class imbalance typical in insurance
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

_SEV_DEFAULTS = {
    "n_estimators": 600,
    "max_depth": 3,            # shallow trees prevent overfitting to noise
    "learning_rate": 0.02,     # slow learning to generalise on low-signal targets
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 50,    # high leaf floor suppresses noise-driven splits
    "gamma": 0.5,              # minimum loss-reduction to make a split
    "reg_alpha": 1.0,          # L1 — pushes irrelevant feature weights to zero
    "reg_lambda": 5.0,         # L2 — shrinks all weights toward zero
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostTrainer(BaseTrainer):
    ALGORITHM = "xgboost"

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostTrainer":
        from xgboost import XGBClassifier, XGBRegressor

        hp = {k: v for k, v in self.hyperparameters.items()
              if k not in ("use_label_encoder", "eval_metric")}

        if self.model_type == "frequency":
            self._model = XGBClassifier(**hp)
        else:
            self._model = XGBRegressor(**hp)

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
    def _load_model(cls, directory: str, instance: "XGBoostTrainer") -> None:
        instance._model = joblib.load(os.path.join(directory, "model.pkl"))
