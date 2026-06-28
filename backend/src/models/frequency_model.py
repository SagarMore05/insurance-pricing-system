import numpy as np
import joblib
import os
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss
from typing import Dict, Any, Tuple


class FrequencyModel:
    """Binary classifier: predicts P(claim occurs) in a policy year."""

    PARAM_GRID = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }

    def __init__(self, version: str = "v1.0.0", model_dir: str = "models/saved"):
        self.version = version
        self.model_dir = model_dir
        self.model: XGBClassifier = None
        self._fitted = False

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        return float(neg / pos) if pos > 0 else 1.0

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        spw = self._compute_scale_pos_weight(y_train)

        base = XGBClassifier(
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
        )

        gs = GridSearchCV(
            base, self.PARAM_GRID,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        self._fitted = True

        metrics = self.evaluate(X_val if X_val is not None else X_train,
                                y_val if y_val is not None else y_train)
        metrics["best_params"] = gs.best_params_
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() or load().")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        return {
            "auc_roc": float(roc_auc_score(y, proba)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y, proba)),
        }

    def get_shap_values(self, X: np.ndarray) -> Tuple[np.ndarray, Any]:
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values, explainer

    def save_feature_importance_plot(self, shap_values: np.ndarray, feature_names: list,
                                      output_path: str = None) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        output_path = output_path or os.path.join(self.model_dir, f"frequency_importance_{self.version}.png")

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-10:][::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in top_idx], mean_shap[top_idx])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Frequency Model Feature Importance ({self.version})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    def get_top_features(self, shap_values: np.ndarray, feature_names: list, n: int = 10) -> Dict[str, float]:
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-n:][::-1]
        return {feature_names[i]: float(mean_shap[i]) for i in top_idx}

    def save(self) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"frequency_{self.version}.pkl")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "FrequencyModel":
        model = joblib.load(path)
        model._fitted = True
        return model
