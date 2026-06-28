"""
Unit tests for V4DatasetBuilder.

Uses AsyncMock to simulate the DB session without a real PostgreSQL connection.
Tests cover:
  - Empty DB (no predictions)
  - Predictions with no labels (Day 0 scenario)
  - Predictions with labels but not yet matured
  - Matured labelled predictions — valid training rows
  - Rows dropped for missing JSON
  - Rows dropped for failed required-field validation
  - Feature fill-rate computation
  - Readiness threshold logic (< 500 rows = not ready)
"""
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retraining.dataset_builder_v4 import (
    V4DatasetBuilder,
    V4_RAW_FEATURES,
    LABEL_MATURITY_DAYS,
    RETRAIN_MIN_SAMPLES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow():
    return datetime.utcnow()


def _make_row(
    *,
    features: dict | None = None,
    actual_claim_occurred=True,
    actual_claim_amount_inr=50000.0,
    policy_age_days: int = 100,   # > 90 = matured by default
    claim_probability: float = 0.08,
    expected_claim_amount_inr: float = 40000.0,
):
    """Build a mock row that mimics a SQLAlchemy Row returned by db.execute()."""
    now = _utcnow()
    r = MagicMock()
    r.prediction_id = uuid.uuid4()
    r.inference_features_json = features
    r.claim_probability = claim_probability
    r.expected_claim_amount_inr = expected_claim_amount_inr
    r.prediction_ts = now
    r.policy_ts = now - timedelta(days=policy_age_days)
    r.actual_claim_occurred = actual_claim_occurred
    r.label_amount = actual_claim_amount_inr
    return r


def _minimal_features() -> dict:
    """Minimal valid inference_features_json covering all required fields."""
    return {
        "age": 35,
        "gender": "Male",
        "occupation": "Salaried",
        "marital_status": "Married",
        "annual_income_band": "7-15 LPA",
        "city": "Mumbai",
        "years_licensed": 10,
        "previous_claims_count": 0,
        "months_since_last_claim": 0,
        "no_claim_bonus_pct": 25.0,
        "policy_tenure_years": 2.0,
        "vehicle_make": "Maruti Suzuki",
        "vehicle_model": "Swift",
        "fuel_type": "Petrol",
        "vehicle_usage_type": "Personal",
        "annual_mileage_km": 15000,
        "parking_type": "Garage",
        "driving_score": 82.5,
        "pincode_risk_score": 4.2,
        "region_risk_category": "2",
        "flood_risk_index": 3.1,
        "theft_risk_index": 2.8,
        "monsoon_exposure_index": 5.0,
        "vehicle_value_inr": 700000.0,
        "vehicle_age_years": 3,
        "engine_cc": 1197,
        "vehicle_safety_rating": 4,
        "airbags_count": 2,
        "repair_cost_band": "Standard",
        "vehicle_body_style": "Sedan",
        "policy_inception_month": 6,
    }


def _make_db(rows):
    """Return a mock AsyncSession whose execute() returns given rows."""
    db = AsyncMock()
    execute_result = MagicMock()
    execute_result.all.return_value = rows
    db.execute = AsyncMock(return_value=execute_result)
    return db


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_db_returns_zeros():
    """No predictions → all counts zero, not ready."""
    db = _make_db([])
    builder = V4DatasetBuilder(db)
    result = await builder.build()

    assert result.total_predictions == 0
    assert result.labelled_predictions == 0
    assert result.matured_predictions == 0
    assert result.matured_labelled == 0
    assert result.rows_valid == 0
    assert result.is_ready_for_retraining is False
    assert "No matured labelled records" in result.readiness_reason
    assert result.dataset is None


@pytest.mark.asyncio
async def test_unlabelled_predictions_not_included():
    """Predictions with no feedback (actual_claim_occurred=None) are not training rows."""
    rows = [
        _make_row(features=_minimal_features(), actual_claim_occurred=None),
        _make_row(features=_minimal_features(), actual_claim_occurred=None),
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.total_predictions == 2
    assert result.labelled_predictions == 0
    assert result.rows_valid == 0


@pytest.mark.asyncio
async def test_immature_labelled_not_training_rows():
    """Labelled predictions from policies < 90 days old are excluded from training."""
    rows = [
        _make_row(features=_minimal_features(), policy_age_days=30),
        _make_row(features=_minimal_features(), policy_age_days=45),
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.total_predictions == 2
    assert result.labelled_predictions == 2
    assert result.matured_predictions == 0
    assert result.matured_labelled == 0
    assert result.rows_valid == 0
    assert result.is_ready_for_retraining is False


@pytest.mark.asyncio
async def test_valid_matured_labelled_row_included():
    """A matured labelled row with complete features should pass validation."""
    rows = [_make_row(features=_minimal_features(), policy_age_days=100)]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build(include_dataset=True)

    assert result.rows_valid == 1
    assert result.rows_dropped_missing_json == 0
    assert result.rows_dropped_validation == 0
    assert result.dataset is not None
    assert len(result.dataset) == 1
    # Verify label columns present
    assert "claim_occurred" in result.dataset.columns
    assert "actual_claim_amount_inr" in result.dataset.columns
    # Verify prediction_id column present
    assert "prediction_id" in result.dataset.columns


@pytest.mark.asyncio
async def test_missing_json_row_dropped():
    """A prediction with null/unparsable inference_features_json is dropped."""
    rows = [
        _make_row(features=None, policy_age_days=100),          # null JSONB
        _make_row(features="not-valid-json", policy_age_days=100),  # invalid str
        _make_row(features=_minimal_features(), policy_age_days=100),  # valid
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.rows_dropped_missing_json == 2
    assert result.rows_valid == 1


@pytest.mark.asyncio
async def test_missing_required_field_drops_row():
    """Rows missing a required field (e.g. gender) are dropped at validation."""
    bad_features = _minimal_features()
    del bad_features["gender"]

    rows = [
        _make_row(features=bad_features, policy_age_days=100),
        _make_row(features=_minimal_features(), policy_age_days=100),
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.rows_dropped_validation == 1
    assert result.rows_valid == 1


@pytest.mark.asyncio
async def test_feature_fill_rates_computed():
    """Feature fill rates should reflect actual presence across valid rows."""
    feat_with_null = _minimal_features()
    feat_with_null["pincode_risk_score"] = None  # optional — not in required

    rows = [
        _make_row(features=_minimal_features(), policy_age_days=100),
        _make_row(features=feat_with_null, policy_age_days=100),
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.rows_valid == 2
    # All required features present in both rows → 100%
    assert result.feature_fill_rates["age"] == 100.0
    # pincode_risk_score null in 1 of 2 rows → 50%
    assert result.feature_fill_rates["pincode_risk_score"] == 50.0
    # 50% fill means pincode_risk_score flagged as missing (< 80%)
    assert "pincode_risk_score" in result.missing_features


@pytest.mark.asyncio
async def test_not_ready_below_minimum_threshold():
    """Dataset with valid rows but below RETRAIN_MIN_SAMPLES is not ready."""
    rows = [_make_row(features=_minimal_features(), policy_age_days=100) for _ in range(10)]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.rows_valid == 10
    assert result.is_ready_for_retraining is False
    assert str(RETRAIN_MIN_SAMPLES) in result.readiness_reason


@pytest.mark.asyncio
async def test_ready_at_or_above_minimum_threshold():
    """Dataset with >= RETRAIN_MIN_SAMPLES valid rows is marked ready."""
    rows = [
        _make_row(features=_minimal_features(), policy_age_days=100)
        for _ in range(RETRAIN_MIN_SAMPLES)
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.rows_valid == RETRAIN_MIN_SAMPLES
    assert result.is_ready_for_retraining is True
    assert "Ready" in result.readiness_reason


@pytest.mark.asyncio
async def test_build_result_to_dict_excludes_dataset():
    """to_dict() should not include the dataset DataFrame."""
    rows = [_make_row(features=_minimal_features(), policy_age_days=100)]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build(include_dataset=True)
    d = result.to_dict()

    assert "dataset" not in d
    assert "rows_valid" in d
    assert "feature_fill_rates" in d


@pytest.mark.asyncio
async def test_v4_raw_features_count():
    """Sanity: V4_RAW_FEATURES must contain exactly 31 entries."""
    assert len(V4_RAW_FEATURES) == 31


@pytest.mark.asyncio
async def test_matured_predictions_count():
    """matured_predictions counts all V2 predictions where policy age >= 90d."""
    rows = [
        _make_row(features=_minimal_features(), policy_age_days=100, actual_claim_occurred=None),
        _make_row(features=_minimal_features(), policy_age_days=45,  actual_claim_occurred=None),
        _make_row(features=_minimal_features(), policy_age_days=95,  actual_claim_occurred=True),
    ]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build()

    assert result.total_predictions == 3
    assert result.matured_predictions == 2   # age 100 + age 95
    assert result.matured_labelled == 1      # only age 95 has label


@pytest.mark.asyncio
async def test_days_back_filter_applied(monkeypatch):
    """days_back parameter should add a where clause (stmt is passed to db.execute)."""
    captured_stmts = []

    async def fake_execute(stmt):
        captured_stmts.append(stmt)
        result = MagicMock()
        result.all.return_value = []
        return result

    db = AsyncMock()
    db.execute = fake_execute
    builder = V4DatasetBuilder(db)
    await builder.build(days_back=30)

    # One execute call should have been made
    assert len(captured_stmts) == 1


@pytest.mark.asyncio
async def test_include_dataset_false_returns_none():
    """include_dataset=False → result.dataset is None even when rows exist."""
    rows = [_make_row(features=_minimal_features(), policy_age_days=100)]
    db = _make_db(rows)
    result = await V4DatasetBuilder(db).build(include_dataset=False)

    assert result.rows_valid == 1
    assert result.dataset is None
