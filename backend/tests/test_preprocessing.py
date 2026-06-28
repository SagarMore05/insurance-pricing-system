import pytest
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import InsurancePreprocessor, DataCleaner, FeatureEngineer

BASE_ROW = {
    "customer_id": "test-uuid",
    "age": 35,
    "gender": "male",
    "city": "mumbai",
    "car_brand": "maruti",
    "car_model": "Swift",
    "engine_cc": 1200,
    "vehicle_age_years": 3,
    "vehicle_value_inr": 600_000,
    "driving_score": 75.0,
    "annual_mileage_km": 12000,
    "previous_claims_count": 0,
    "years_licensed": 10,
}


def make_df(n=5, **overrides) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {**BASE_ROW, "customer_id": f"uuid-{i}"}
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── DataCleaner ─────────────────────────────────────────────────────────────

def test_cleaner_handles_missing_numerics():
    df = make_df(3)
    df.loc[0, "age"] = np.nan
    df.loc[1, "driving_score"] = np.nan
    cleaner = DataCleaner()
    cleaner.fit(df)
    result = cleaner.transform(df)
    assert result["age"].isna().sum() == 0
    assert result["driving_score"].isna().sum() == 0


def test_cleaner_clips_mileage():
    df = make_df(2)
    df.loc[0, "annual_mileage_km"] = -500
    df.loc[1, "annual_mileage_km"] = 999_999
    cleaner = DataCleaner()
    cleaner.fit(df)
    result = cleaner.transform(df)
    assert result["annual_mileage_km"].min() >= 0
    assert result["annual_mileage_km"].max() <= 200_000


def test_cleaner_clips_age():
    df = make_df(2)
    df.loc[0, "age"] = 10
    df.loc[1, "age"] = 100
    cleaner = DataCleaner()
    cleaner.fit(df)
    result = cleaner.transform(df)
    assert result["age"].min() >= 18
    assert result["age"].max() <= 85


def test_cleaner_removes_duplicate_customers():
    df = make_df(3)
    df.loc[1, "customer_id"] = "uuid-0"  # duplicate
    cleaner = DataCleaner()
    cleaner.fit(df)
    result = cleaner.transform(df)
    assert len(result) == 2


# ─── FeatureEngineer ──────────────────────────────────────────────────────────

def test_feature_engineer_creates_derived_features():
    df = make_df(2)
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert "risk_age_ratio" in result.columns
    assert "vehicle_depreciation_factor" in result.columns
    assert "normalized_mileage" in result.columns
    assert "high_value_vehicle" in result.columns
    assert "city_tier" in result.columns
    assert "driving_risk_score" in result.columns


def test_high_value_vehicle_flag():
    df = make_df(2)
    df.loc[0, "vehicle_value_inr"] = 2_000_000
    df.loc[1, "vehicle_value_inr"] = 500_000
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert result.loc[0, "high_value_vehicle"] == 1
    assert result.loc[1, "high_value_vehicle"] == 0


def test_unknown_city_defaults_to_tier3():
    df = make_df(1)
    df.loc[0, "city"] = "unknowncity123"
    fe = FeatureEngineer()
    result = fe.transform(df)
    assert result.loc[0, "city_tier"] == 3


# ─── InsurancePreprocessor ────────────────────────────────────────────────────

def test_preprocessor_fit_transform_shape():
    df = make_df(20)
    proc = InsurancePreprocessor()
    X = proc.fit_transform(df)
    assert X.shape[0] == 20
    assert X.shape[1] > 10  # should have many features after encoding


def test_preprocessor_transform_single_row():
    df = make_df(20)
    proc = InsurancePreprocessor()
    proc.fit(df)
    single = df.iloc[:1]
    X = proc.transform(single)
    assert X.shape[0] == 1


def test_preprocessor_handles_all_nulls():
    df = make_df(5)
    df["age"] = np.nan
    df["driving_score"] = np.nan
    proc = InsurancePreprocessor()
    X = proc.fit_transform(df)
    assert not np.isnan(X).any()


def test_preprocessor_unknown_brand():
    df_train = make_df(20)
    proc = InsurancePreprocessor()
    proc.fit(df_train)

    df_test = make_df(1)
    df_test.loc[0, "car_brand"] = "lamborghini"  # unknown brand
    X = proc.transform(df_test)
    assert X.shape[0] == 1


def test_preprocessor_raises_before_fit():
    proc = InsurancePreprocessor()
    df = make_df(2)
    with pytest.raises(RuntimeError):
        proc.transform(df)
