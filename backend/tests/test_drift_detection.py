"""
Unit tests for the drift detection service.

All tests use pure Python — no database, no file I/O (where possible).
PSI computation is the critical path so tests focus on:
  - Correct PSI formula
  - Severity classification
  - Small-sample guard
  - Categorical handling (new categories, missing categories)
  - History persistence helpers
  - Overall drift aggregation
"""
import json
import math
import pathlib
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.drift_detection import (
    DriftDetectionService,
    DriftSnapshot,
    FeatureDriftResult,
    _compute_numeric_psi,
    _compute_categorical_psi,
    _psi_value,
    _severity,
    _DEFAULT_PSI_LOW,
    _DEFAULT_PSI_HIGH,
    _NUMERIC_REF_PCTS,
    _MIN_PROD_SAMPLE,
    save_drift_snapshot,
    load_drift_history,
)


# ─── PSI formula unit tests ────────────────────────────────────────────────────

def test_psi_identical_distributions_is_zero():
    """Perfect match → PSI = 0."""
    pcts = [10.0, 15.0, 25.0, 25.0, 15.0, 9.0, 1.0]
    assert _psi_value(pcts, pcts) == pytest.approx(0.0, abs=1e-6)


def test_psi_symmetric_difference_positive():
    """PSI is always ≥ 0 regardless of direction of shift."""
    ref = [20.0, 30.0, 50.0]
    prod = [50.0, 30.0, 20.0]
    result = _psi_value(ref, prod)
    assert result > 0


def test_psi_small_shift_below_low_threshold():
    """A slight shift should produce PSI < 0.10."""
    ref  = [10.0, 15.0, 25.0, 25.0, 15.0, 9.0, 1.0]
    prod = [10.5, 14.5, 25.0, 25.0, 15.0, 9.0, 1.0]
    assert _psi_value(ref, prod) < _DEFAULT_PSI_LOW


def test_psi_large_shift_exceeds_high_threshold():
    """A population completely flipped produces PSI well above 0.25."""
    ref  = [70.0, 20.0, 10.0]
    prod = [10.0, 20.0, 70.0]
    assert _psi_value(ref, prod) > _DEFAULT_PSI_HIGH


def test_psi_epsilon_guard_no_divide_by_zero():
    """Zero production percentage uses ε and should not raise."""
    ref  = [50.0, 50.0]
    prod = [100.0, 0.0]
    # Should not raise
    result = _psi_value(ref, prod)
    assert math.isfinite(result)
    assert result > 0


# ─── Severity classification ───────────────────────────────────────────────────

def test_severity_low():
    assert _severity(0.05, 0.10, 0.25) == "low"


def test_severity_boundary_low_to_medium():
    assert _severity(0.10, 0.10, 0.25) == "medium"


def test_severity_medium():
    assert _severity(0.15, 0.10, 0.25) == "medium"


def test_severity_boundary_medium_to_high():
    assert _severity(0.25, 0.10, 0.25) == "high"


def test_severity_high():
    assert _severity(0.40, 0.10, 0.25) == "high"


def test_severity_custom_thresholds():
    assert _severity(0.05, 0.03, 0.08) == "medium"
    assert _severity(0.09, 0.03, 0.08) == "high"


# ─── Numeric PSI ──────────────────────────────────────────────────────────────

def _make_ref_stats(**kwargs):
    defaults = dict(count=50000, mean=40.0, std=10.0, min=18.0, max=73.0,
                    p10=26.0, p25=32.0, p50=39.0, p75=46.0, p90=53.0, p99=63.0)
    defaults.update(kwargs)
    return defaults


def test_numeric_psi_insufficient_data():
    """Fewer than MIN_PROD_SAMPLE values → severity='insufficient_data'."""
    result = _compute_numeric_psi("age", _make_ref_stats(), [], 0.10, 0.25)
    assert result.severity == "insufficient_data"
    assert result.psi == 0.0


def test_numeric_psi_exact_reference_distribution():
    """
    Generate prod values that match the reference percentile structure exactly
    → PSI should be close to 0.
    """
    stats = _make_ref_stats()
    # Produce values spread exactly over the bins: 10% < p10, 15% in [p10, p25), etc.
    # Simple approach: generate uniform samples within each bin
    prod_values = (
        [20.0] * 10   +   # < p10=26 → bin 0 (10%)
        [28.0] * 15   +   # p10-p25 → bin 1 (15%)
        [35.0] * 25   +   # p25-p50 → bin 2 (25%)
        [42.0] * 25   +   # p50-p75 → bin 3 (25%)
        [49.0] * 15   +   # p75-p90 → bin 4 (15%)
        [55.0] * 9    +   # p90-p99 → bin 5 (9%)
        [70.0] * 1        # ≥ p99   → bin 6 (1%)
    )
    result = _compute_numeric_psi("age", stats, prod_values, 0.10, 0.25)
    assert result.severity == "low"
    assert result.psi < _DEFAULT_PSI_LOW
    assert result.feature == "age"
    assert result.feature_type == "numeric"
    assert result.prod_n == 100
    assert len(result.bin_labels) == 7


def test_numeric_psi_high_drift():
    """All values concentrated in one tail → high drift."""
    stats = _make_ref_stats()
    prod_values = [65.0] * 100  # all values ≥ p99
    result = _compute_numeric_psi("age", stats, prod_values, 0.10, 0.25)
    assert result.severity == "high"
    assert result.psi > _DEFAULT_PSI_HIGH


def test_numeric_psi_bins_sum_to_100():
    """Production bin percentages should sum to 100 (within float tolerance)."""
    stats = _make_ref_stats()
    prod_values = list(range(18, 74)) * 2  # 112 values spread across range
    result = _compute_numeric_psi("age", stats, prod_values, 0.10, 0.25)
    assert result.prod_n > 0
    assert abs(sum(result.prod_bins) - 100.0) < 0.01


def test_numeric_ref_bins_unchanged():
    """Reference bins must always equal the fixed 7-bucket reference percentages."""
    stats = _make_ref_stats()
    prod_values = [40.0] * 50
    result = _compute_numeric_psi("age", stats, prod_values, 0.10, 0.25)
    assert result.ref_bins == _NUMERIC_REF_PCTS


# ─── Categorical PSI ──────────────────────────────────────────────────────────

def _make_cat_stats(**kwargs):
    defaults = dict(
        count=50000,
        value_counts={"Male": 29139, "Female": 19850, "Other": 1011},
        value_pcts={"Male": 58.278, "Female": 39.7, "Other": 2.022},
    )
    defaults.update(kwargs)
    return defaults


def test_categorical_psi_insufficient_data():
    result = _compute_categorical_psi("gender", _make_cat_stats(), [], 0.10, 0.25)
    assert result.severity == "insufficient_data"


def test_categorical_psi_matching_distribution():
    """Production distribution matching reference → near-zero PSI."""
    # Generate 1000 values proportional to reference
    prod_values = (
        ["Male"] * 583 +
        ["Female"] * 397 +
        ["Other"] * 20
    )
    result = _compute_categorical_psi("gender", _make_cat_stats(), prod_values, 0.10, 0.25)
    assert result.severity == "low"
    assert result.psi < _DEFAULT_PSI_LOW


def test_categorical_psi_high_drift():
    """All production values are a single category → high drift."""
    prod_values = ["Female"] * 100
    result = _compute_categorical_psi("gender", _make_cat_stats(), prod_values, 0.10, 0.25)
    assert result.severity == "high"
    assert result.psi > _DEFAULT_PSI_HIGH


def test_categorical_psi_new_category_added():
    """New category not in reference gets '<other>' bin."""
    prod_values = ["Male"] * 80 + ["Non-Binary"] * 20
    result = _compute_categorical_psi("gender", _make_cat_stats(), prod_values, 0.10, 0.25)
    assert "<other>" in result.bin_labels
    other_idx = result.bin_labels.index("<other>")
    assert result.prod_bins[other_idx] == pytest.approx(20.0, abs=0.01)


def test_categorical_feature_type():
    prod_values = ["Male"] * 50 + ["Female"] * 50
    result = _compute_categorical_psi("gender", _make_cat_stats(), prod_values, 0.10, 0.25)
    assert result.feature_type == "categorical"


# ─── History persistence ───────────────────────────────────────────────────────

def _make_minimal_snapshot(overall_psi=0.05) -> DriftSnapshot:
    return DriftSnapshot(
        computed_at="2026-06-25T12:00:00",
        window_days=30,
        prod_sample_n=100,
        psi_low_threshold=0.10,
        psi_high_threshold=0.25,
        feature_results=[],
        overall_psi=overall_psi,
        overall_severity="low",
        recommendation="Monitor",
        alerts=[],
        high_drift_features=[],
        medium_drift_features=[],
        features_computed=0,
        features_skipped=18,
    )


def test_save_and_load_drift_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_path = pathlib.Path(tmpdir) / "drift_history.json"
        snapshot = _make_minimal_snapshot(overall_psi=0.07)

        with patch("src.services.drift_detection._HISTORY_PATH", hist_path):
            save_drift_snapshot(snapshot)
            history = load_drift_history(limit=10)

        assert len(history) == 1
        assert history[0]["overall_psi"] == pytest.approx(0.07)
        assert history[0]["recommendation"] == "Monitor"


def test_history_rotation_at_max_entries():
    """History file should not grow beyond _HISTORY_MAX_ENTRIES."""
    from src.services.drift_detection import _HISTORY_MAX_ENTRIES
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_path = pathlib.Path(tmpdir) / "drift_history.json"
        snapshot = _make_minimal_snapshot()

        with patch("src.services.drift_detection._HISTORY_PATH", hist_path):
            for _ in range(_HISTORY_MAX_ENTRIES + 5):
                save_drift_snapshot(snapshot)
            history = load_drift_history(limit=_HISTORY_MAX_ENTRIES + 10)

        assert len(history) == _HISTORY_MAX_ENTRIES


def test_load_history_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_path = pathlib.Path(tmpdir) / "nonexistent.json"
        with patch("src.services.drift_detection._HISTORY_PATH", hist_path):
            history = load_drift_history(limit=10)
        assert history == []


def test_load_history_limit_respected():
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_path = pathlib.Path(tmpdir) / "drift_history.json"
        snapshot = _make_minimal_snapshot()

        with patch("src.services.drift_detection._HISTORY_PATH", hist_path):
            for _ in range(20):
                save_drift_snapshot(snapshot)
            history = load_drift_history(limit=5)

        assert len(history) == 5


# ─── DriftDetectionService integration (mocked DB) ───────────────────────────

def _make_db(json_rows):
    """Return AsyncMock DB that returns the given JSON rows from scalars().all()."""
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = json_rows
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_mock
    db = AsyncMock()
    db.execute = AsyncMock(return_value=execute_result)
    return db


def _minimal_feature_row():
    return {
        "age": 35, "gender": "Male", "occupation": "Salaried",
        "marital_status": "Married", "annual_income_band": "7-15 LPA",
        "city": "Mumbai", "years_licensed": 10, "previous_claims_count": 0,
        "months_since_last_claim": 0, "no_claim_bonus_pct": 25.0,
        "policy_tenure_years": 2.0, "vehicle_make": "Maruti",
        "vehicle_model": "Swift", "fuel_type": "Petrol",
        "vehicle_usage_type": "Personal", "annual_mileage_km": 15000,
        "parking_type": "Garage", "driving_score": 82.5,
        "pincode_risk_score": 4.2, "region_risk_category": "2",
        "flood_risk_index": 3.1, "theft_risk_index": 2.8,
        "monsoon_exposure_index": 5.0, "vehicle_value_inr": 700000.0,
        "vehicle_age_years": 3, "engine_cc": 1197,
        "vehicle_safety_rating": 4, "airbags_count": 2,
        "repair_cost_band": "Standard", "vehicle_body_style": "Sedan",
        "policy_inception_month": 6,
    }


@pytest.mark.asyncio
async def test_service_empty_db_returns_no_data():
    """Empty DB → all features skipped → severity='no_data'."""
    db = _make_db([])

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": {}, "categorical": {}}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    assert snapshot.overall_severity == "no_data"
    assert snapshot.prod_sample_n == 0
    assert snapshot.recommendation == "Insufficient Data — accumulate ≥30 V2 quotes"


@pytest.mark.asyncio
async def test_service_insufficient_samples_skip_all():
    """10 rows < MIN_PROD_SAMPLE → all features skipped."""
    rows = [_minimal_feature_row() for _ in range(10)]
    db = _make_db(rows)

    ref_numeric = {"age": {"count": 50000, "mean": 39.0, "std": 10.0,
                            "min": 18.0, "max": 73.0, "p10": 26.0,
                            "p25": 32.0, "p50": 39.0, "p75": 46.0,
                            "p90": 53.0, "p99": 63.0}}
    ref_categorical = {}

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": ref_numeric, "categorical": ref_categorical}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    assert snapshot.features_skipped == 1
    assert snapshot.features_computed == 0
    assert snapshot.overall_severity == "no_data"


@pytest.mark.asyncio
async def test_service_sufficient_samples_compute():
    """30+ rows → numeric PSI computed, not skipped."""
    rows = [_minimal_feature_row() for _ in range(30)]
    db = _make_db(rows)

    ref_numeric = {"age": {"count": 50000, "mean": 39.0, "std": 10.0,
                            "min": 18.0, "max": 73.0, "p10": 26.0,
                            "p25": 32.0, "p50": 39.0, "p75": 46.0,
                            "p90": 53.0, "p99": 63.0}}
    ref_categorical = {}

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": ref_numeric, "categorical": ref_categorical}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    assert snapshot.features_computed == 1
    assert snapshot.features_skipped == 0
    assert snapshot.overall_severity in ("low", "medium", "high")
    assert snapshot.prod_sample_n == 30


@pytest.mark.asyncio
async def test_service_json_string_rows_parsed():
    """asyncpg may return JSONB as a JSON string — service must parse it."""
    rows = [json.dumps(_minimal_feature_row()) for _ in range(30)]
    db = _make_db(rows)

    ref_numeric = {"age": {"count": 50000, "mean": 39.0, "std": 10.0,
                            "min": 18.0, "max": 73.0, "p10": 26.0,
                            "p25": 32.0, "p50": 39.0, "p75": 46.0,
                            "p90": 53.0, "p99": 63.0}}

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": ref_numeric, "categorical": {}}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    assert snapshot.features_computed == 1


@pytest.mark.asyncio
async def test_service_alert_generated_for_high_drift():
    """All values concentrated at extreme → alert is generated."""
    rows = []
    for _ in range(30):
        r = _minimal_feature_row()
        r["age"] = 73  # max value → all in bin 6 → high PSI
        rows.append(r)
    db = _make_db(rows)

    ref_numeric = {"age": {"count": 50000, "mean": 39.0, "std": 10.0,
                            "min": 18.0, "max": 73.0, "p10": 26.0,
                            "p25": 32.0, "p50": 39.0, "p75": 46.0,
                            "p90": 53.0, "p99": 63.0}}

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": ref_numeric, "categorical": {}}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    assert len(snapshot.alerts) > 0
    assert "age" in snapshot.high_drift_features


@pytest.mark.asyncio
async def test_service_to_dict_serializable():
    """DriftSnapshot.to_dict() must be JSON-serializable."""
    rows = [_minimal_feature_row() for _ in range(30)]
    db = _make_db(rows)

    with patch("src.services.drift_detection._load_reference") as mock_ref:
        mock_ref.return_value = {"numeric": {}, "categorical": {}}
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)

    d = snapshot.to_dict()
    json_str = json.dumps(d)  # must not raise
    assert isinstance(json_str, str)

