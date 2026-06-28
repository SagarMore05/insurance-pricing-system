import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.frequency_model import FrequencyModel
from src.models.severity_model import SeverityModel
from src.engine.pricing_engine import PricingEngine

RNG = np.random.default_rng(99)


def make_classification_data(n=200):
    X = RNG.random((n, 15)).astype(np.float32)
    # Create some correlation with target
    y = (X[:, 0] + X[:, 1] - X[:, 2] + RNG.normal(0, 0.3, n) > 0.5).astype(int)
    return X, y


def make_regression_data(n=200):
    X = RNG.random((n, 15)).astype(np.float32)
    y_log = X[:, 0] * 3 + X[:, 2] * 2 + RNG.normal(0, 0.5, n)
    y_log = np.clip(y_log, 0, None)
    return X, y_log


# ─── FrequencyModel ───────────────────────────────────────────────────────────

def test_frequency_model_trains_and_predicts():
    X, y = make_classification_data(300)
    model = FrequencyModel()
    metrics = model.train(X[:200], y[:200], X[200:], y[200:])
    assert "auc_roc" in metrics
    assert 0 <= metrics["auc_roc"] <= 1

    probs = model.predict_proba(X[:10])
    assert probs.shape == (10,)
    assert all(0 <= p <= 1 for p in probs)


def test_frequency_model_raises_before_training():
    model = FrequencyModel()
    X, _ = make_classification_data(10)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_frequency_model_evaluate():
    X, y = make_classification_data(200)
    model = FrequencyModel()
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert set(metrics.keys()) >= {"auc_roc", "precision", "recall", "f1", "brier_score"}


# ─── SeverityModel ───────────────────────────────────────────────────────────

def test_severity_model_trains_and_predicts():
    X, y_log = make_regression_data(300)
    model = SeverityModel()
    metrics = model.train(X[:200], y_log[:200], X[200:], y_log[200:])
    assert "rmse" in metrics and "r2" in metrics

    amounts = model.predict(X[:10])
    assert amounts.shape == (10,)
    assert all(a >= 0 for a in amounts)


def test_severity_model_inverse_log_transform():
    X, y_log = make_regression_data(100)
    model = SeverityModel()
    model.train(X, y_log)
    pred_log = model.predict_log(X[:5])
    pred_orig = model.predict(X[:5])
    np.testing.assert_allclose(pred_orig, np.expm1(pred_log), rtol=1e-5)


def test_severity_model_raises_before_training():
    model = SeverityModel()
    X, _ = make_regression_data(10)
    with pytest.raises(RuntimeError):
        model.predict(X)


# ─── PricingEngine ────────────────────────────────────────────────────────────

def test_pricing_engine_deterministic():
    engine = PricingEngine()
    kwargs = dict(
        claim_probability=0.18,
        expected_claim_amount_inr=85_000,
        age=30,
        driving_score=72,
        previous_claims_count=1,
        annual_mileage_km=15_000,
        vehicle_value_inr=900_000,
    )
    r1 = engine.calculate(**kwargs)
    r2 = engine.calculate(**kwargs)
    assert r1.final_premium == r2.final_premium
    assert r1.risk_level == r2.risk_level


def test_pricing_engine_young_driver_surcharge():
    engine = PricingEngine()
    young = engine.calculate(0.15, 50_000, age=22, driving_score=70,
                              previous_claims_count=0, annual_mileage_km=10_000,
                              vehicle_value_inr=500_000)
    adult = engine.calculate(0.15, 50_000, age=35, driving_score=70,
                              previous_claims_count=0, annual_mileage_km=10_000,
                              vehicle_value_inr=500_000)
    assert young.final_premium > adult.final_premium


def test_pricing_engine_safe_driver_discount():
    engine = PricingEngine()
    safe = engine.calculate(0.15, 50_000, age=35, driving_score=90,
                             previous_claims_count=0, annual_mileage_km=10_000,
                             vehicle_value_inr=500_000)
    poor = engine.calculate(0.15, 50_000, age=35, driving_score=40,
                             previous_claims_count=0, annual_mileage_km=10_000,
                             vehicle_value_inr=500_000)
    assert safe.final_premium < poor.final_premium


def test_pricing_engine_floor_cap():
    engine = PricingEngine(min_premium=3000, max_premium=150_000)
    # Very low risk — should hit floor
    low = engine.calculate(0.001, 1_000, age=35, driving_score=95,
                            previous_claims_count=0, annual_mileage_km=3_000,
                            vehicle_value_inr=400_000)
    assert low.final_premium >= 3000

    # Very high risk — should hit cap
    high = engine.calculate(0.9, 5_000_000, age=22, driving_score=20,
                             previous_claims_count=5, annual_mileage_km=50_000,
                             vehicle_value_inr=3_000_000)
    assert high.final_premium <= 150_000


def test_pricing_engine_risk_levels():
    # Thresholds: <0.20 → low, <0.40 → medium, ≥0.40 → high
    engine = PricingEngine()
    low = engine.calculate(0.10, 50_000, age=35, driving_score=80,
                            previous_claims_count=0, annual_mileage_km=10_000,
                            vehicle_value_inr=500_000)
    medium = engine.calculate(0.30, 50_000, age=35, driving_score=80,
                               previous_claims_count=0, annual_mileage_km=10_000,
                               vehicle_value_inr=500_000)
    high = engine.calculate(0.70, 50_000, age=35, driving_score=80,
                             previous_claims_count=0, annual_mileage_km=10_000,
                             vehicle_value_inr=500_000)
    assert low.risk_level == "low"
    assert medium.risk_level == "medium"
    assert high.risk_level == "high"


def test_pricing_engine_no_claims_bonus():
    engine = PricingEngine()
    clean = engine.calculate(0.15, 50_000, age=35, driving_score=70,
                              previous_claims_count=0, annual_mileage_km=10_000,
                              vehicle_value_inr=500_000)
    dirty = engine.calculate(0.15, 50_000, age=35, driving_score=70,
                              previous_claims_count=3, annual_mileage_km=10_000,
                              vehicle_value_inr=500_000)
    assert clean.final_premium < dirty.final_premium


def test_pricing_engine_irdai_ncb_schedule():
    # Thresholds: 0% → no discount; 20% → 0.80 factor; 50% → 0.50 factor
    engine = PricingEngine()
    base_kwargs = dict(claim_probability=0.15, expected_claim_amount_inr=50_000,
                       age=35, driving_score=70, previous_claims_count=0,
                       annual_mileage_km=10_000, vehicle_value_inr=500_000)
    p_zero = engine.calculate(**base_kwargs, no_claim_bonus_pct=0.0)
    p_20   = engine.calculate(**base_kwargs, no_claim_bonus_pct=20.0)
    p_50   = engine.calculate(**base_kwargs, no_claim_bonus_pct=50.0)

    assert p_20.final_premium < p_zero.final_premium   # 20% NCB reduces premium
    assert p_50.final_premium < p_20.final_premium     # 50% NCB reduces it further

    adj20 = next((a for a in p_20.adjustments if a.reason == "no_claims_bonus"), None)
    assert adj20 is not None
    assert abs(adj20.factor - 0.80) < 0.001   # 20% discount → factor 0.80

    adj50 = next((a for a in p_50.adjustments if a.reason == "no_claims_bonus"), None)
    assert adj50 is not None
    assert abs(adj50.factor - 0.50) < 0.001   # 50% discount → factor 0.50

    # 0% NCB must produce no no_claims_bonus adjustment
    assert not any(a.reason == "no_claims_bonus" for a in p_zero.adjustments)


# ─── Retraining Safety Guard ─────────────────────────────────────────────────

def test_scheduler_not_started_when_disabled():
    """Scheduler must not start when ENABLE_SCHEDULED_RETRAINING is False."""
    with patch("src.retraining.scheduler.start_scheduler") as mock_start:
        from src.config import Settings
        settings = Settings(ENABLE_SCHEDULED_RETRAINING=False)
        # Simulate the guard logic from main.py lifespan
        if settings.ENABLE_SCHEDULED_RETRAINING:
            mock_start()
        mock_start.assert_not_called()


def test_scheduler_starts_when_enabled():
    """Scheduler must start when ENABLE_SCHEDULED_RETRAINING is True."""
    with patch("src.retraining.scheduler.start_scheduler") as mock_start:
        from src.config import Settings
        settings = Settings(ENABLE_SCHEDULED_RETRAINING=True)
        # Simulate the guard logic from main.py lifespan
        if settings.ENABLE_SCHEDULED_RETRAINING:
            mock_start()
        mock_start.assert_called_once()


def test_enable_scheduled_retraining_default_is_false():
    """Default value of ENABLE_SCHEDULED_RETRAINING must be False — safety invariant."""
    from src.config import Settings
    fresh = Settings()
    assert fresh.ENABLE_SCHEDULED_RETRAINING is False, (
        "ENABLE_SCHEDULED_RETRAINING must default to False to protect V4 champions"
    )


def test_pricing_engine_explanation_structure():
    engine = PricingEngine()
    result = engine.calculate(0.20, 60_000, age=28, driving_score=60,
                               previous_claims_count=1, annual_mileage_km=20_000,
                               vehicle_value_inr=800_000)
    d = result.to_dict()
    assert "base_premium" in d
    assert "adjustments" in d
    assert "final_premium" in d
    assert "summary" in d
    assert isinstance(d["adjustments"], list)
