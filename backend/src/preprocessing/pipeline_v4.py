"""
InsurancePreprocessorV4
========================
Production preprocessing pipeline for the V4 (50k-row) model family.

Feature groups (59 features total):
  StandardScaled (9): age, annual_mileage_km, vehicle_value_inr, driving_score,
                       vehicle_safety_rating, pincode_risk_score, flood_risk_index,
                       theft_risk_index, monsoon_exposure_index
  MinMaxScaled   (3): vehicle_age_years, years_licensed, policy_tenure_years
  OneHotEncoded (~37): gender, city_tier_str, fuel_type, vehicle_usage_type,
                       occupation, marital_status, annual_income_band,
                       parking_type, repair_cost_band, vehicle_body_style
  LabelEncoded   (1): vehicle_make
  Passthrough    (8): previous_claims_count, months_since_last_claim_norm,
                       no_claim_bonus_pct, airbags_count, policy_inception_month,
                       risk_age_ratio, ncb_risk_score, engine_cc_norm

Engineered features (created inside _engineer()):
  city_tier_str               — mapped from city name to "1"/"2" (Tier 1/2 cities)
  risk_age_ratio              — previous_claims_count / years_licensed
  ncb_risk_score              — (50 - no_claim_bonus_pct) / 50
  months_since_last_claim_norm — 0 for never-claimed; raw / 120 otherwise
  engine_cc_norm              — engine_cc / 3500

This class is serialised with joblib into models/champion/preprocessor_v4.pkl.
Placing it here (not in __main__) ensures it can be unpickled from any context.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


CITY_TIER_MAP = {
    "Mumbai": 1, "Delhi": 1, "Bangalore": 1, "Hyderabad": 1, "Chennai": 1,
    "Kolkata": 1, "Pune": 1, "Ahmedabad": 1,
    "Jaipur": 2, "Lucknow": 2, "Kanpur": 2, "Nagpur": 2, "Indore": 2,
    "Bhopal": 2, "Patna": 2, "Vadodara": 2, "Coimbatore": 2, "Surat": 2,
    "Agra": 2, "Visakhapatnam": 2,
}


class InsurancePreprocessorV4:
    """Full V4 preprocessing pipeline: engineer → standardise → encode."""

    STD_COLS = [
        "age", "annual_mileage_km", "vehicle_value_inr", "driving_score",
        "vehicle_safety_rating", "pincode_risk_score",
        "flood_risk_index", "theft_risk_index", "monsoon_exposure_index",
    ]
    MM_COLS  = ["vehicle_age_years", "years_licensed", "policy_tenure_years"]
    OHE_COLS = [
        "gender", "city_tier_str", "fuel_type", "vehicle_usage_type",
        "occupation", "marital_status", "annual_income_band",
        "parking_type", "repair_cost_band", "vehicle_body_style",
    ]
    PASS_COLS = [
        "previous_claims_count", "months_since_last_claim_norm",
        "no_claim_bonus_pct", "airbags_count", "policy_inception_month",
        "risk_age_ratio", "ncb_risk_score", "engine_cc_norm",
    ]
    LE_COL = "vehicle_make"

    def __init__(self):
        self.std_scaler     = StandardScaler()
        self.mm_scaler      = MinMaxScaler()
        self.ohe            = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.le             = LabelEncoder()
        self.feature_names_ = None
        self._fitted        = False

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["city_tier_str"] = (
            out["city"].str.lower()
            .map({k.lower(): v for k, v in CITY_TIER_MAP.items()})
            .fillna(2).astype(int).astype(str)
        )
        out["risk_age_ratio"] = out["previous_claims_count"] / out["years_licensed"].clip(lower=1)
        out["ncb_risk_score"] = (50.0 - out["no_claim_bonus_pct"]) / 50.0
        out["months_since_last_claim_norm"] = np.where(
            out["months_since_last_claim"] == 999,
            0.0,
            out["months_since_last_claim"] / 120.0,
        )
        out["engine_cc_norm"] = out["engine_cc"] / 3500.0
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df2      = self._engineer(df)
        std_arr  = self.std_scaler.fit_transform(df2[self.STD_COLS])
        mm_arr   = self.mm_scaler.fit_transform(df2[self.MM_COLS])
        ohe_arr  = self.ohe.fit_transform(df2[self.OHE_COLS])
        le_vals  = self.le.fit_transform(df2[self.LE_COL])
        pass_arr = df2[self.PASS_COLS].values
        X = np.hstack([std_arr, mm_arr, ohe_arr, le_vals.reshape(-1, 1), pass_arr])
        self.feature_names_ = (
            self.STD_COLS
            + self.MM_COLS
            + list(self.ohe.get_feature_names_out(self.OHE_COLS))
            + [self.LE_COL + "_encoded"]
            + self.PASS_COLS
        )
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("InsurancePreprocessorV4 must be fit_transform()'d before transform().")
        df2      = self._engineer(df)
        std_arr  = self.std_scaler.transform(df2[self.STD_COLS])
        mm_arr   = self.mm_scaler.transform(df2[self.MM_COLS])
        ohe_arr  = self.ohe.transform(df2[self.OHE_COLS])
        le_vals  = self.le.transform(df2[self.LE_COL])
        pass_arr = df2[self.PASS_COLS].values
        return np.hstack([std_arr, mm_arr, ohe_arr, le_vals.reshape(-1, 1), pass_arr])

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "InsurancePreprocessorV4":
        obj = joblib.load(path)
        return obj
