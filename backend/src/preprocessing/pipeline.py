import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, Any

CITY_TIER_MAP = {
    # Tier 1
    "mumbai": 1, "delhi": 1, "bangalore": 1, "hyderabad": 1, "chennai": 1,
    "kolkata": 1, "pune": 1, "ahmedabad": 1,
    # Tier 2
    "jaipur": 2, "lucknow": 2, "kanpur": 2, "nagpur": 2, "indore": 2,
    "bhopal": 2, "patna": 2, "vadodara": 2, "coimbatore": 2, "agra": 2,
    "visakhapatnam": 2, "surat": 2,
    # Tier 3 (default for unlisted)
}

TOP_BRANDS = [
    "maruti", "hyundai", "tata", "mahindra", "honda",
    "toyota", "ford", "volkswagen", "kia", "renault",
    "skoda", "mg", "nissan", "jeep", "bmw",
]

# Canonical mappings for new categorical features (case-insensitive → master dataset values)
_SEGMENT_MAP = {
    "hatchback": "Hatchback", "sedan": "Sedan", "suv": "SUV",
    "luxury": "Luxury", "muv": "SUV", "mpv": "SUV",
}
_FUEL_MAP = {
    "petrol": "Petrol", "diesel": "Diesel", "cng": "CNG",
    "ev": "EV", "electric": "EV", "hybrid": "Petrol",
}

_INCOME_DEFAULT = 2_596_312.0   # dataset mean; used as inference default


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates derived features from raw input. Handles optional master-dataset columns."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Accept both master-dataset names and legacy internal names ─────────
        if "vehicle_brand" in df.columns and "car_brand" not in df.columns:
            df = df.rename(columns={"vehicle_brand": "car_brand"})
        if "previous_claims" in df.columns and "previous_claims_count" not in df.columns:
            df = df.rename(columns={"previous_claims": "previous_claims_count"})

        # ── Normalise gender to lowercase (training data is lowercase after rename) ──
        if "gender" in df.columns:
            df["gender"] = df["gender"].str.lower()

        # ── Optional columns: add defaults when arriving from legacy API calls ──
        if "vehicle_segment" not in df.columns:
            df["vehicle_segment"] = "Sedan"
        else:
            df["vehicle_segment"] = (
                df["vehicle_segment"]
                .fillna("Sedan")
                .apply(lambda v: _SEGMENT_MAP.get(str(v).lower(), "Sedan"))
            )

        if "fuel_type" not in df.columns:
            df["fuel_type"] = "Petrol"
        else:
            df["fuel_type"] = (
                df["fuel_type"]
                .fillna("Petrol")
                .apply(lambda v: _FUEL_MAP.get(str(v).lower(), "Petrol"))
            )

        if "annual_income_inr" not in df.columns:
            df["annual_income_inr"] = _INCOME_DEFAULT
        else:
            df["annual_income_inr"] = df["annual_income_inr"].fillna(_INCOME_DEFAULT)

        # ── Derived risk features ──────────────────────────────────────────────
        df["risk_age_ratio"] = df["previous_claims_count"] / df["years_licensed"].clip(lower=1)
        df["vehicle_depreciation_factor"] = 1 / (1 + df["vehicle_age_years"] * 0.15)
        df["normalized_mileage"] = df["annual_mileage_km"] / 15000
        df["high_value_vehicle"] = (df["vehicle_value_inr"] > 1_500_000).astype(int)
        df["city_tier"] = df["city"].str.lower().map(CITY_TIER_MAP).fillna(3).astype(int)
        df["driving_risk_score"] = (
            (100 - df["driving_score"]) / 100 * (1 + df["previous_claims_count"] * 0.2)
        )

        # ── Severity-predictive interaction features (actuarial) ───────────────
        # Current market value proxy: newer/more expensive vehicles → higher repair cost
        df["vehicle_value_per_age"] = (
            df["vehicle_value_inr"] / (1.0 + df["vehicle_age_years"])
        )
        # Underinsurance ratio: high vehicle value relative to income → moral hazard
        df["income_to_vehicle_ratio"] = (
            df["annual_income_inr"] / df["vehicle_value_inr"].clip(lower=1.0)
        )
        # Exposure-weighted claim risk: mileage × claim history interaction
        df["exposure_risk_index"] = (
            df["normalized_mileage"] * (1.0 + df["previous_claims_count"] * 0.2)
        )

        # ── V2 derived features (append-only; existing indices unchanged) ───────
        # Proportion of adult life spent driving — 1.0 = highly experienced
        df["driver_experience_ratio"] = (
            df["years_licensed"] / (df["age"] - 18).clip(lower=1)
        ).clip(upper=1.0)

        # Annualised claim rate — high values signal serial claimants
        df["claim_frequency_rate"] = (
            df["previous_claims_count"] / df["years_licensed"].clip(lower=1)
        )

        # Vehicle-to-income stress — proxy for underinsurance and moral hazard
        df["vehicle_premium_stress"] = (
            df["vehicle_value_inr"] / df["annual_income_inr"].clip(lower=1.0)
        ).clip(upper=10.0)

        # Normalize car_brand → car_brand_encoded (for label encoder)
        df["car_brand_encoded"] = df["car_brand"].str.lower().apply(
            lambda b: b if b in TOP_BRANDS else "other"
        )

        return df


class DataCleaner(BaseEstimator, TransformerMixin):
    """Handles missing values, outliers, and duplicates."""

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_medians_ = {}
        self.categorical_modes_ = {}

        numeric_cols = [
            "age", "annual_mileage_km", "driving_score",
            "vehicle_value_inr", "vehicle_age_years", "years_licensed",
            "previous_claims_count", "annual_income_inr",
        ]
        categorical_cols = ["city", "gender", "vehicle_segment", "fuel_type"]

        for col in numeric_cols:
            if col in X.columns:
                median_val = X[col].median()
                self.numeric_medians_[col] = median_val if pd.notna(median_val) else 0.0

        for col in categorical_cols:
            if col in X.columns:
                self.categorical_modes_[col] = (
                    X[col].mode()[0] if not X[col].mode().empty else "unknown"
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop duplicates on customer-level features if customer_id present
        if "customer_id" in df.columns:
            df = df.drop_duplicates(subset=["customer_id"])

        # Fill missing numerics with fitted medians
        for col, median in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # Fill missing categoricals with fitted modes
        for col, mode in self.categorical_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)

        # Clip outliers
        if "annual_mileage_km" in df.columns:
            df["annual_mileage_km"] = df["annual_mileage_km"].clip(0, 200_000)
        if "age" in df.columns:
            df["age"] = df["age"].clip(18, 85)
        if "driving_score" in df.columns:
            df["driving_score"] = df["driving_score"].clip(0, 100)
        if "annual_income_inr" in df.columns:
            df["annual_income_inr"] = df["annual_income_inr"].clip(0, 50_000_000)

        return df


class InsurancePreprocessor:
    """Full preprocessing pipeline: clean → engineer → encode → scale.

    Features (33 total):
      StandardScaled  (5): age, annual_mileage_km, vehicle_value_inr, driving_score,
                           annual_income_inr
      MinMaxScaled    (2): vehicle_age_years, years_licensed
      OneHotEncoded  (13): gender(2), city_tier(3), vehicle_segment(4), fuel_type(4)
      LabelEncoded    (1): car_brand_encoded
      Passthrough    (12): previous_claims_count, risk_age_ratio,
                           vehicle_depreciation_factor, normalized_mileage,
                           high_value_vehicle, driving_risk_score,
                           vehicle_value_per_age, income_to_vehicle_ratio,
                           exposure_risk_index,
                           driver_experience_ratio, claim_frequency_rate,
                           vehicle_premium_stress
    """

    STANDARD_SCALE_COLS = [
        "age", "annual_mileage_km", "vehicle_value_inr",
        "driving_score", "annual_income_inr",
    ]
    MINMAX_SCALE_COLS = ["vehicle_age_years", "years_licensed"]
    # OHE columns in order; FeatureEngineer ensures they are present
    OHE_COLS = ["gender", "city_tier", "vehicle_segment", "fuel_type"]
    LABEL_ENCODE_COL = "car_brand_encoded"

    FEATURE_COLS = [
        "age", "annual_mileage_km", "vehicle_value_inr", "driving_score",
        "annual_income_inr", "vehicle_age_years", "years_licensed",
        "previous_claims_count", "risk_age_ratio", "vehicle_depreciation_factor",
        "normalized_mileage", "high_value_vehicle", "driving_risk_score",
        "city_tier", "gender", "vehicle_segment", "fuel_type", "car_brand_encoded",
    ]

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.brand_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "InsurancePreprocessor":
        df_clean = self.cleaner.fit_transform(df)
        df_engineered = self.feature_engineer.transform(df_clean)

        self.standard_scaler.fit(df_engineered[self.STANDARD_SCALE_COLS])
        self.minmax_scaler.fit(df_engineered[self.MINMAX_SCALE_COLS])

        # Build OHE input: gender (already normalised), city_tier as string,
        # vehicle_segment and fuel_type already normalised by FeatureEngineer
        ohe_input = pd.DataFrame({
            "gender":          df_engineered["gender"],
            "city_tier":       df_engineered["city_tier"].astype(str),
            "vehicle_segment": df_engineered["vehicle_segment"],
            "fuel_type":       df_engineered["fuel_type"],
        })
        self.ohe.fit(ohe_input)

        brands = df_engineered["car_brand_encoded"].fillna("other")
        self.brand_encoder.fit(pd.concat([brands, pd.Series(["other"])], ignore_index=True))
        self._brand_classes = set(self.brand_encoder.classes_)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        df_clean = self.cleaner.transform(df)
        df_eng = self.feature_engineer.transform(df_clean)

        standard_scaled = self.standard_scaler.transform(df_eng[self.STANDARD_SCALE_COLS])
        minmax_scaled = self.minmax_scaler.transform(df_eng[self.MINMAX_SCALE_COLS])

        ohe_input = pd.DataFrame({
            "gender":          df_eng["gender"],
            "city_tier":       df_eng["city_tier"].astype(str),
            "vehicle_segment": df_eng["vehicle_segment"],
            "fuel_type":       df_eng["fuel_type"],
        })
        ohe_encoded = self.ohe.transform(ohe_input)

        brands = df_eng["car_brand_encoded"].apply(
            lambda b: b if b in self._brand_classes else "other"
        )
        brand_encoded = self.brand_encoder.transform(brands).reshape(-1, 1)

        passthrough_cols = [
            "previous_claims_count", "risk_age_ratio",
            "vehicle_depreciation_factor", "normalized_mileage",
            "high_value_vehicle", "driving_risk_score",
            "vehicle_value_per_age", "income_to_vehicle_ratio", "exposure_risk_index",
            "driver_experience_ratio", "claim_frequency_rate", "vehicle_premium_stress",
        ]
        passthrough = df_eng[passthrough_cols].values

        return np.hstack([
            standard_scaled, minmax_scaled, ohe_encoded,
            brand_encoded, passthrough,
        ])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str = None) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        save_path = path or os.path.join(self.model_dir, "preprocessor.pkl")
        joblib.dump(self, save_path)
        return save_path

    @classmethod
    def load(cls, path: str) -> "InsurancePreprocessor":
        return joblib.load(path)


def prepare_training_data(
    df: pd.DataFrame,
    target_frequency: str = "claim_occurred",
    target_severity: str = "actual_claim_amount_inr",
    model_dir: str = "models/saved",
) -> Dict[str, Any]:
    preprocessor = InsurancePreprocessor(model_dir=model_dir)
    X = preprocessor.fit_transform(df)
    preprocessor.save()

    y_freq = df[target_frequency].values if target_frequency in df.columns else None
    y_sev  = np.log1p(df[target_severity].values) if target_severity in df.columns else None

    splits = {}
    stratify_col = df.get("risk_level") if "risk_level" in df.columns else None

    if y_freq is not None:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y_freq, test_size=0.15, stratify=stratify_col, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.15 / 0.85, random_state=42
        )
        splits["frequency"] = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }

    if y_sev is not None:
        mask = (df[target_frequency] == 1).values if target_frequency in df.columns \
               else np.ones(len(df), dtype=bool)
        # For severity target, keep only claimants with positive amounts
        sev_mask = mask & (df[target_severity].values > 0)
        X_sev      = X[sev_mask]
        y_sev_filt = y_sev[sev_mask]
        X_stmp, X_stest, y_stmp, y_stest = train_test_split(
            X_sev, y_sev_filt, test_size=0.15, random_state=42
        )
        X_strain, X_sval, y_strain, y_sval = train_test_split(
            X_stmp, y_stmp, test_size=0.15 / 0.85, random_state=42
        )
        splits["severity"] = {
            "X_train": X_strain, "X_val": X_sval, "X_test": X_stest,
            "y_train": y_strain, "y_val": y_sval, "y_test": y_stest,
        }

    return {"preprocessor": preprocessor, "splits": splits}
