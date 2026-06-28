import numpy as np
import joblib
import os
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple


class SeverityModel:
    """Regression model: predicts expected claim amount (INR) for claims that occur."""

    PARAM_GRID = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }

    def __init__(self, version: str = "v1.0.0", model_dir: str = "models/saved"):
        self.version = version
        self.model_dir = model_dir
        self.model: XGBRegressor = None
        self._fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        # y_train expected to already be log1p-transformed
        base = XGBRegressor(random_state=42, n_jobs=-1)

        gs = GridSearchCV(
            base, self.PARAM_GRID,
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        self._fitted = True

        metrics = self.evaluate(
            X_val if X_val is not None else X_train,
            y_val if y_val is not None else y_train
        )
        metrics["best_params"] = gs.best_params_
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted claim amount in INR (inverse log1p applied)."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() or load().")
        log_pred = self.model.predict(X)
        return np.expm1(log_pred)

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        """Returns log1p-scale predictions."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() or load().")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_log: np.ndarray) -> Dict[str, float]:
        pred_log = self.predict_log(X)
        y_orig = np.expm1(y_log)
        pred_orig = np.expm1(pred_log)

        rmse = float(np.sqrt(mean_squared_error(y_orig, pred_orig)))
        mae = float(mean_absolute_error(y_orig, pred_orig))
        r2 = float(r2_score(y_orig, pred_orig))

        # MAPE — avoid division by zero
        nonzero = y_orig != 0
        mape = float(np.mean(np.abs((y_orig[nonzero] - pred_orig[nonzero]) / y_orig[nonzero])) * 100) if nonzero.any() else 0.0

        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    def get_shap_values(self, X: np.ndarray) -> Tuple[np.ndarray, Any]:
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values, explainer

    def save_feature_importance_plot(self, shap_values: np.ndarray, feature_names: list,
                                      output_path: str = None) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        output_path = output_path or os.path.join(self.model_dir, f"severity_importance_{self.version}.png")

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-10:][::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in top_idx], mean_shap[top_idx])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Severity Model Feature Importance ({self.version})")
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
        path = os.path.join(self.model_dir, f"severity_{self.version}.pkl")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "SeverityModel":
        model = joblib.load(path)
        model._fitted = True
        return model


class CatBoostSeverityWrapper:
    """
    Drop-in replacement for SeverityModel that wraps a CatBoostRegressor.

    Exposes the same predict(X) interface used by CombinedPredictor so the
    champion registry can swap XGBoost for CatBoost without any other code change.
    """

    def __init__(self, version: str = "v4.0.0"):
        self.version = version
        self._fitted = False
        self.model = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted claim amounts in INR (no log-transform — CatBoost trains on raw INR)."""
        if not self._fitted:
            raise RuntimeError("CatBoostSeverityWrapper not fitted. Load a model first.")
        return self.model.predict(X)

    def save(self, cbm_path: str, pkl_path: str) -> None:
        """Persist as both CatBoost native (.cbm) and joblib (.pkl) formats."""
        os.makedirs(os.path.dirname(cbm_path), exist_ok=True)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        self.model.save_model(cbm_path)
        joblib.dump(self, pkl_path)

    @classmethod
    def load_from_cbm(cls, cbm_path: str, version: str = "v4.0.0") -> "CatBoostSeverityWrapper":
        """Load from CatBoost native binary format."""
        try:
            import catboost as cb
        except ImportError as exc:
            raise ImportError(
                "catboost package is required to load CatBoostSeverityWrapper. "
                "Install with: pip install catboost"
            ) from exc
        wrapper = cls(version=version)
        wrapper.model = cb.CatBoostRegressor()
        wrapper.model.load_model(cbm_path)
        wrapper._fitted = True
        return wrapper

    @classmethod
    def load_from_pkl(cls, pkl_path: str) -> "CatBoostSeverityWrapper":
        """Load from joblib pkl (includes wrapper metadata)."""
        obj = joblib.load(pkl_path)
        obj._fitted = True
        return obj
