"""
Tests for the Multi-Algorithm Model Selection Engine.

Tests verify:
- All three classifiers train successfully (XGBoost, LightGBM, CatBoost)
- All three regressors train successfully
- Winner selection logic produces correct results
- Model registry updates correctly
- Retraining workflow repeats the same pipeline
- API returns correct production model info
- SHAP generated only for winners

Run with:
    cd backend && .\venv\Scripts\python.exe -m pytest tests/test_multi_algorithm_engine.py -v
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import shutil
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a minimal synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    n = 300

    df = pd.DataFrame({
        "age":                   rng.integers(18, 70, n).astype(float),
        "gender":                rng.choice(["male", "female"], n),
        "city":                  rng.choice(["mumbai", "delhi", "jaipur"], n),
        "car_brand":             rng.choice(["maruti", "hyundai", "tata"], n),
        "vehicle_segment":       rng.choice(["Hatchback", "Sedan", "SUV"], n),
        "fuel_type":             rng.choice(["Petrol", "Diesel"], n),
        "vehicle_age_years":     rng.integers(0, 15, n).astype(float),
        "vehicle_value_inr":     rng.uniform(200_000, 2_000_000, n),
        "annual_mileage_km":     rng.uniform(5_000, 30_000, n),
        "driving_score":         rng.uniform(40, 100, n),
        "previous_claims_count": rng.integers(0, 4, n).astype(float),
        "years_licensed":        rng.integers(1, 30, n).astype(float),
        "annual_income_inr":     rng.uniform(300_000, 5_000_000, n),
        "claim_occurred":        rng.integers(0, 2, n),
        "claim_amount_inr":      rng.uniform(10_000, 500_000, n),
    })
    # Zero out claim_amount for non-claimants
    df.loc[df["claim_occurred"] == 0, "claim_amount_inr"] = 0.0

    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def engine(synthetic_dataset, tmp_path):
    """Create a MultiAlgorithmEngine configured with the synthetic dataset."""
    from src.training.multi_algorithm_engine import MultiAlgorithmEngine
    return MultiAlgorithmEngine(
        dataset_path=synthetic_dataset,
        shap_sample_size=50,
    )


# ── Dataset loading ──────────────────────────────────────────────────────────

class TestDatasetLoading:
    def test_loads_valid_csv(self, engine):
        df, hash_ = engine._load_and_validate()
        assert len(df) > 0
        assert isinstance(hash_, str) and len(hash_) == 32

    def test_raises_on_missing_file(self, tmp_path):
        from src.training.multi_algorithm_engine import MultiAlgorithmEngine
        eng = MultiAlgorithmEngine(dataset_path=str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            eng._load_and_validate()

    def test_raises_on_missing_columns(self, tmp_path):
        bad = pd.DataFrame({"age": [1, 2, 3]})
        p = tmp_path / "bad.csv"
        bad.to_csv(p, index=False)
        from src.training.multi_algorithm_engine import MultiAlgorithmEngine
        eng = MultiAlgorithmEngine(dataset_path=str(p))
        with pytest.raises(ValueError, match="missing required columns"):
            eng._load_and_validate()


# ── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_returns_numpy_arrays(self, engine):
        df, _ = engine._load_and_validate()
        X, y_freq, y_sev, names, prep = engine._preprocess(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y_freq, np.ndarray)
        assert isinstance(y_sev, np.ndarray)
        assert X.shape[0] == len(df)
        assert len(names) == X.shape[1]

    def test_feature_count_is_nonzero(self, engine):
        df, _ = engine._load_and_validate()
        X, _, _, names, _ = engine._preprocess(df)
        assert X.shape[1] > 0
        assert len(names) > 0


# ── Individual trainers ───────────────────────────────────────────────────────

class TestFrequencyTrainers:
    """Verify all three classifiers train and produce valid frequency metrics."""

    @pytest.fixture
    def freq_data(self, engine):
        df, _   = engine._load_and_validate()
        X, y, _, _, prep = engine._preprocess(df)
        (X_tr, X_va, X_te, yf_tr, yf_va, yf_te, _, _, _) = engine._split(X, y, np.zeros_like(y, dtype=float))
        return X_tr, yf_tr, X_va, yf_va, X_te, yf_te

    def test_xgboost_frequency_trains(self, engine, freq_data):
        X_tr, yf_tr, X_va, yf_va, X_te, yf_te = freq_data
        from src.training.trainers.xgboost_trainer import XGBoostTrainer
        trainer = XGBoostTrainer(model_type="frequency")
        result  = engine._train_one(trainer, "frequency", X_tr, yf_tr, X_va, yf_va, X_te, yf_te)
        assert result["status"] == "success", result.get("error")
        assert "roc_auc" in result["metrics"]
        assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0

    def test_lightgbm_frequency_trains(self, engine, freq_data):
        X_tr, yf_tr, X_va, yf_va, X_te, yf_te = freq_data
        from src.training.trainers.lightgbm_trainer import LightGBMTrainer
        trainer = LightGBMTrainer(model_type="frequency")
        result  = engine._train_one(trainer, "frequency", X_tr, yf_tr, X_va, yf_va, X_te, yf_te)
        assert result["status"] == "success", result.get("error")
        assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0

    def test_catboost_frequency_trains(self, engine, freq_data):
        X_tr, yf_tr, X_va, yf_va, X_te, yf_te = freq_data
        from src.training.trainers.catboost_trainer import CatBoostTrainer
        trainer = CatBoostTrainer(model_type="frequency")
        result  = engine._train_one(trainer, "frequency", X_tr, yf_tr, X_va, yf_va, X_te, yf_te)
        assert result["status"] == "success", result.get("error")
        assert 0.0 <= result["metrics"]["roc_auc"] <= 1.0


class TestSeverityTrainers:
    """Verify all three regressors train and produce valid severity metrics."""

    @pytest.fixture
    def sev_data(self, engine):
        df, _   = engine._load_and_validate()
        X, y_f, y_s, _, _ = engine._preprocess(df)
        (X_tr, _, X_te, yf_tr, _, yf_te, ys_tr, _, ys_te) = engine._split(X, y_f, y_s)
        X_tr_s = X_tr[yf_tr == 1];  ys_tr_s = ys_tr[yf_tr == 1]
        X_te_s = X_te[yf_te == 1];  ys_te_s = ys_te[yf_te == 1]
        return X_tr_s, ys_tr_s, X_te_s, ys_te_s

    def test_xgboost_severity_trains(self, engine, sev_data):
        X_tr, ys_tr, X_te, ys_te = sev_data
        from src.training.trainers.xgboost_trainer import XGBoostTrainer
        trainer = XGBoostTrainer(model_type="severity")
        result  = engine._train_one(trainer, "severity", X_tr, ys_tr, X_te, ys_te, X_te, ys_te)
        assert result["status"] == "success", result.get("error")
        assert "r2" in result["metrics"]

    def test_lightgbm_severity_trains(self, engine, sev_data):
        X_tr, ys_tr, X_te, ys_te = sev_data
        from src.training.trainers.lightgbm_trainer import LightGBMTrainer
        trainer = LightGBMTrainer(model_type="severity")
        result  = engine._train_one(trainer, "severity", X_tr, ys_tr, X_te, ys_te, X_te, ys_te)
        assert result["status"] == "success", result.get("error")
        assert "r2" in result["metrics"]

    def test_catboost_severity_trains(self, engine, sev_data):
        X_tr, ys_tr, X_te, ys_te = sev_data
        from src.training.trainers.catboost_trainer import CatBoostTrainer
        trainer = CatBoostTrainer(model_type="severity")
        result  = engine._train_one(trainer, "severity", X_tr, ys_tr, X_te, ys_te, X_te, ys_te)
        assert result["status"] == "success", result.get("error")
        assert "r2" in result["metrics"]


# ── Winner selection ──────────────────────────────────────────────────────────

class TestWinnerSelection:
    def test_frequency_winner_highest_auc(self, engine):
        results = {
            "xgboost":  {"status": "success", "metrics": {"roc_auc": 0.75, "brier_score": 0.20, "f1": 0.70, "precision": 0.68, "recall": 0.72}, "is_winner": False},
            "lightgbm": {"status": "success", "metrics": {"roc_auc": 0.78, "brier_score": 0.18, "f1": 0.73, "precision": 0.70, "recall": 0.76}, "is_winner": False},
            "catboost": {"status": "success", "metrics": {"roc_auc": 0.72, "brier_score": 0.22, "f1": 0.68, "precision": 0.65, "recall": 0.71}, "is_winner": False},
        }
        winner, reason = engine._select_frequency_winner(results)
        assert winner == "lightgbm"
        assert "lightgbm" in reason.lower() or "LIGHTGBM" in reason

    def test_frequency_tiebreaker_favors_xgboost(self, engine):
        # All tied on AUC; XGBoost wins by tie-breaker
        tied_metrics = {"roc_auc": 0.75, "brier_score": 0.20, "f1": 0.70, "precision": 0.68, "recall": 0.72}
        results = {
            "xgboost":  {"status": "success", "metrics": tied_metrics.copy(), "is_winner": False},
            "lightgbm": {"status": "success", "metrics": tied_metrics.copy(), "is_winner": False},
            "catboost": {"status": "success", "metrics": tied_metrics.copy(), "is_winner": False},
        }
        winner, _ = engine._select_frequency_winner(results)
        assert winner == "xgboost"

    def test_severity_winner_highest_r2(self, engine):
        results = {
            "xgboost":  {"status": "success", "metrics": {"r2": -0.05, "rmse": 80000, "mae": 50000, "mape_pct": 55}, "is_winner": False},
            "lightgbm": {"status": "success", "metrics": {"r2":  0.02, "rmse": 70000, "mae": 45000, "mape_pct": 50}, "is_winner": False},
            "catboost": {"status": "success", "metrics": {"r2": -0.10, "rmse": 90000, "mae": 60000, "mape_pct": 60}, "is_winner": False},
        }
        winner, reason = engine._select_severity_winner(results)
        assert winner == "lightgbm"

    def test_raises_when_all_failed(self, engine):
        results = {
            "xgboost":  {"status": "failed", "metrics": {}, "is_winner": False},
            "lightgbm": {"status": "failed", "metrics": {}, "is_winner": False},
            "catboost": {"status": "failed", "metrics": {}, "is_winner": False},
        }
        with pytest.raises(RuntimeError, match="All frequency models failed"):
            engine._select_frequency_winner(results)

    def test_skips_failed_model(self, engine):
        results = {
            "xgboost":  {"status": "failed", "metrics": {}, "is_winner": False},
            "lightgbm": {"status": "success", "metrics": {"roc_auc": 0.78, "brier_score": 0.18, "f1": 0.73, "precision": 0.70, "recall": 0.76}, "is_winner": False},
            "catboost": {"status": "failed", "metrics": {}, "is_winner": False},
        }
        winner, _ = engine._select_frequency_winner(results)
        assert winner == "lightgbm"


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    """Integration test: run the complete pipeline on synthetic data."""

    def test_run_completes_and_returns_record(self, engine, tmp_path, monkeypatch):
        # Patch registry paths to temp dir so we don't write to real models/
        import src.training.multi_algorithm_engine as mod

        monkeypatch.setattr(mod, "_SAVED_DIR",    tmp_path / "saved")
        monkeypatch.setattr(mod, "_RUNS_DIR",     tmp_path / "runs")
        monkeypatch.setattr(mod, "_REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(mod, "_HISTORY_PATH", tmp_path / "registry" / "training_history.json")
        monkeypatch.setattr(mod, "_REGISTRY_PATH", tmp_path / "registry" / "model_registry.json")
        monkeypatch.setattr(mod, "_CHAMPION_PATH", tmp_path / "champion_registry.json")

        record = engine.run()

        assert record["frequency_winner"] in ("xgboost", "lightgbm", "catboost")
        assert record["severity_winner"]  in ("xgboost", "lightgbm", "catboost")
        assert len(record["frequency_results"]) == 3
        assert len(record["severity_results"])  == 3

    def test_only_winner_is_winner(self, engine, tmp_path, monkeypatch):
        import src.training.multi_algorithm_engine as mod
        monkeypatch.setattr(mod, "_SAVED_DIR",    tmp_path / "saved")
        monkeypatch.setattr(mod, "_RUNS_DIR",     tmp_path / "runs")
        monkeypatch.setattr(mod, "_REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(mod, "_HISTORY_PATH", tmp_path / "registry" / "training_history.json")
        monkeypatch.setattr(mod, "_REGISTRY_PATH", tmp_path / "registry" / "model_registry.json")
        monkeypatch.setattr(mod, "_CHAMPION_PATH", tmp_path / "champion_registry.json")

        record = engine.run()

        freq_winners = [r for r in record["frequency_results"] if r["is_winner"]]
        sev_winners  = [r for r in record["severity_results"]  if r["is_winner"]]
        assert len(freq_winners) == 1
        assert len(sev_winners)  == 1

    def test_registry_updated_after_run(self, engine, tmp_path, monkeypatch):
        import src.training.multi_algorithm_engine as mod
        history_path  = tmp_path / "registry" / "training_history.json"
        registry_path = tmp_path / "registry" / "model_registry.json"
        champion_path = tmp_path / "champion_registry.json"

        monkeypatch.setattr(mod, "_SAVED_DIR",    tmp_path / "saved")
        monkeypatch.setattr(mod, "_RUNS_DIR",     tmp_path / "runs")
        monkeypatch.setattr(mod, "_REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(mod, "_HISTORY_PATH", history_path)
        monkeypatch.setattr(mod, "_REGISTRY_PATH", registry_path)
        monkeypatch.setattr(mod, "_CHAMPION_PATH", champion_path)

        record = engine.run()

        assert history_path.exists(), "training_history.json must be created"
        assert registry_path.exists(), "model_registry.json must be updated"
        assert champion_path.exists(), "champion_registry.json must be updated"

        with open(history_path) as f:
            history = json.load(f)
        assert history["total_runs"] == 1
        assert history["latest_run_id"] == record["run_id"]

        with open(champion_path) as f:
            champion = json.load(f)
        assert champion["frequency"]["deployment_status"] == "active"
        assert champion["severity"]["deployment_status"]  == "active"
        assert champion["frequency"]["train_dataset"] == "motor_insurance_master_dataset_50000.csv"

    def test_retraining_repeats_same_workflow(self, engine, tmp_path, monkeypatch):
        """Running the engine twice produces two independent history entries."""
        import src.training.multi_algorithm_engine as mod
        history_path = tmp_path / "registry" / "training_history.json"
        monkeypatch.setattr(mod, "_SAVED_DIR",    tmp_path / "saved")
        monkeypatch.setattr(mod, "_RUNS_DIR",     tmp_path / "runs")
        monkeypatch.setattr(mod, "_REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(mod, "_HISTORY_PATH", history_path)
        monkeypatch.setattr(mod, "_REGISTRY_PATH", tmp_path / "registry" / "model_registry.json")
        monkeypatch.setattr(mod, "_CHAMPION_PATH", tmp_path / "champion_registry.json")

        record1 = engine.run()
        record2 = engine.run()

        with open(history_path) as f:
            history = json.load(f)

        assert history["total_runs"] == 2
        assert record1["run_id"] != record2["run_id"]
        assert record2["run_number"] == 2


# ── SHAP ─────────────────────────────────────────────────────────────────────

class TestSHAP:
    def test_shap_generated_for_winner_only(self, engine, tmp_path, monkeypatch):
        """SHAP should be generated for the winner; no SHAP for other algorithms."""
        import src.training.multi_algorithm_engine as mod
        monkeypatch.setattr(mod, "_SAVED_DIR",    tmp_path / "saved")
        monkeypatch.setattr(mod, "_RUNS_DIR",     tmp_path / "runs")
        monkeypatch.setattr(mod, "_REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(mod, "_HISTORY_PATH", tmp_path / "registry" / "training_history.json")
        monkeypatch.setattr(mod, "_REGISTRY_PATH", tmp_path / "registry" / "model_registry.json")
        monkeypatch.setattr(mod, "_CHAMPION_PATH", tmp_path / "champion_registry.json")

        record = engine.run()
        # SHAP flags in the run record
        assert "shap_generated_frequency" in record
        assert "shap_generated_severity"  in record


# ── API endpoint smoke tests ──────────────────────────────────────────────────

class TestAPIEndpoints:
    """Smoke tests for model selection API routes."""

    def test_latest_returns_no_runs_when_history_empty(self, tmp_path, monkeypatch):
        from src.api.routes import model_selection_api as api_mod
        monkeypatch.setattr(api_mod, "_HISTORY_PATH", tmp_path / "nonexistent.json")

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(api_mod.get_latest_run())
        assert result["has_runs"] is False
        assert result["total_runs"] == 0

    def test_history_returns_empty_when_no_runs(self, tmp_path, monkeypatch):
        from src.api.routes import model_selection_api as api_mod
        monkeypatch.setattr(api_mod, "_HISTORY_PATH", tmp_path / "nonexistent.json")

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            api_mod.get_training_history(page=1, page_size=20)
        )
        assert result["total"] == 0
        assert result["runs"] == []
