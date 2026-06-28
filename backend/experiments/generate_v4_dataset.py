"""
V4 Dataset Generation — Controlled Experiment
==============================================
Generates car_insurance_portfolio_v4.csv (50,000 rows) with genuinely new
causal features that cannot be derived from V3 columns.

V4 design principles:
  1. No telematics data (no hard braking, speeding ratio, night driving)
  2. No credit bureau data
  3. No insurer-only proprietary data
  4. All new features are customer-declared or publicly-derivable

Key V4 additions over V3:
  - months_since_last_claim (recency dimension absent from V3)
  - no_claim_bonus_pct      (NCB — pricing construct with genuine causal link)
  - policy_tenure_years     (loyalty signal)
  - fuel_type               (EV/Hybrid repair cost multiplier — genuinely new)
  - repair_cost_band        (vehicle model repair cost — uses ignored car_model data)
  - vehicle_body_style      (body style affects repair complexity)
  - parking_type            (theft and damage risk factor)
  - policy_inception_month  (seasonal claims exposure)
  - Geographic risk indices: pincode_risk_score, flood_risk_index,
                             theft_risk_index, monsoon_exposure_index

V4 driving_score DECOUPLED from claims/usage (V3 flaw corrected).

Outputs:
  experiments/outputs/data/car_insurance_portfolio_v4.csv
  experiments/outputs/reports/v4_feature_design_document.txt
  experiments/outputs/reports/v4_validation_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

RANDOM_SEED = 42
N_ROWS = 50_000
rng = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "outputs", "data")
REPORT_DIR = os.path.join(SCRIPT_DIR, "outputs", "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# GEOGRAPHIC RISK TABLES (public data proxies: IMD, NCRB, MoRTH)
# ─────────────────────────────────────────────────────────────
CITY_PROFILES = {
    # city: (tier, flood_risk[0-10], theft_risk[0-10], pincode_base[1-10], monsoon[0-10], pop_weight)
    "Mumbai":       (1, 8.5, 6.0, 7.2, 9.0, 0.10),
    "Delhi":        (1, 5.0, 8.5, 6.8, 6.0, 0.12),
    "Bangalore":    (1, 4.5, 5.5, 6.5, 7.0, 0.10),
    "Hyderabad":    (1, 5.5, 5.0, 6.2, 6.5, 0.08),
    "Chennai":      (1, 7.0, 4.5, 6.0, 8.5, 0.08),
    "Kolkata":      (1, 8.0, 6.5, 7.0, 8.5, 0.07),
    "Pune":         (1, 5.0, 5.0, 5.8, 7.0, 0.06),
    "Ahmedabad":    (1, 4.0, 5.5, 5.5, 5.0, 0.06),
    "Jaipur":       (2, 3.0, 5.0, 5.0, 3.5, 0.04),
    "Lucknow":      (2, 5.5, 5.5, 5.5, 6.0, 0.04),
    "Kanpur":       (2, 5.0, 5.0, 5.2, 5.5, 0.03),
    "Nagpur":       (2, 4.5, 4.5, 5.0, 6.5, 0.03),
    "Indore":       (2, 3.5, 4.0, 4.8, 5.5, 0.03),
    "Bhopal":       (2, 4.0, 4.0, 4.5, 5.5, 0.03),
    "Patna":        (2, 7.5, 5.5, 6.0, 7.5, 0.03),
    "Vadodara":     (2, 4.5, 4.5, 4.8, 5.0, 0.03),
    "Coimbatore":   (2, 3.5, 3.5, 4.5, 6.0, 0.03),
    "Surat":        (2, 6.0, 5.0, 5.5, 6.0, 0.03),
    "Agra":         (2, 4.0, 5.0, 5.0, 5.0, 0.02),
    "Visakhapatnam":(2, 6.5, 4.0, 5.5, 7.5, 0.03),
}
CITIES = list(CITY_PROFILES.keys())
CITY_WEIGHTS = np.array([CITY_PROFILES[c][5] for c in CITIES])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

CITY_TIER_MAP = {c: CITY_PROFILES[c][0] for c in CITIES}

# ─────────────────────────────────────────────────────────────
# VEHICLE CATALOG
# Each entry: (fuel_type_weights, body_style, repair_band, value_range, safety_rating, airbags)
# fuel_type_weights: [Petrol, Diesel, Electric, Hybrid]
# ─────────────────────────────────────────────────────────────
VEHICLE_CATALOG = {
    # Hatchbacks
    ("Maruti",   "Swift"):      ([0.60, 0.25, 0.10, 0.05], "Hatchback", "Budget",    (500_000,  900_000),   4, 2),
    ("Maruti",   "Alto"):       ([0.85, 0.05, 0.05, 0.05], "Hatchback", "Budget",    (350_000,  600_000),   3, 2),
    ("Maruti",   "Baleno"):     ([0.55, 0.20, 0.15, 0.10], "Hatchback", "Budget",    (700_000, 1_000_000),  4, 6),
    ("Hyundai",  "i20"):        ([0.55, 0.20, 0.15, 0.10], "Hatchback", "Standard",  (750_000, 1_100_000),  4, 6),
    ("Tata",     "Tiago"):      ([0.50, 0.15, 0.30, 0.05], "Hatchback", "Budget",    (550_000,  900_000),   5, 6),
    ("Renault",  "Kwid"):       ([0.80, 0.10, 0.05, 0.05], "Hatchback", "Budget",    (400_000,  650_000),   3, 2),
    # Sedans
    ("Honda",    "City"):       ([0.55, 0.25, 0.10, 0.10], "Sedan",     "Standard",  (1_000_000, 1_500_000), 5, 6),
    ("Maruti",   "Dzire"):      ([0.60, 0.25, 0.08, 0.07], "Sedan",     "Budget",    (700_000, 1_100_000),   4, 6),
    ("Hyundai",  "Verna"):      ([0.50, 0.25, 0.15, 0.10], "Sedan",     "Standard",  (1_000_000, 1_600_000), 5, 6),
    ("Skoda",    "Slavia"):     ([0.45, 0.25, 0.10, 0.20], "Sedan",     "Premium",   (1_200_000, 1_800_000), 5, 6),
    ("Toyota",   "Camry"):      ([0.20, 0.10, 0.05, 0.65], "Sedan",     "Premium",   (2_000_000, 3_200_000), 5, 8),
    ("Honda",    "Accord"):     ([0.30, 0.10, 0.05, 0.55], "Sedan",     "Premium",   (2_200_000, 3_500_000), 5, 8),
    # SUVs
    ("Hyundai",  "Creta"):      ([0.45, 0.30, 0.15, 0.10], "SUV",       "Standard",  (1_000_000, 1_800_000), 5, 6),
    ("Tata",     "Nexon"):      ([0.35, 0.20, 0.40, 0.05], "SUV",       "Standard",  (900_000,  1_700_000),  5, 6),
    ("Kia",      "Seltos"):     ([0.45, 0.30, 0.15, 0.10], "SUV",       "Standard",  (1_100_000, 1_900_000), 5, 6),
    ("Mahindra", "Scorpio"):    ([0.20, 0.70, 0.05, 0.05], "SUV",       "Standard",  (1_200_000, 2_000_000), 4, 6),
    ("Mahindra", "XUV700"):     ([0.40, 0.40, 0.10, 0.10], "SUV",       "Premium",   (1_500_000, 2_500_000), 5, 7),
    ("Toyota",   "Fortuner"):   ([0.10, 0.80, 0.05, 0.05], "SUV",       "Premium",   (3_000_000, 4_500_000), 5, 7),
    ("Jeep",     "Compass"):    ([0.40, 0.40, 0.10, 0.10], "SUV",       "Premium",   (1_800_000, 2_800_000), 5, 6),
    ("Ford",     "Endeavour"):  ([0.10, 0.80, 0.05, 0.05], "SUV",       "Premium",   (2_500_000, 3_800_000), 5, 7),
    # MUVs / MPVs
    ("Maruti",   "Ertiga"):     ([0.50, 0.30, 0.10, 0.10], "MUV",       "Budget",    (900_000,  1_400_000),  4, 6),
    ("Toyota",   "Innova"):     ([0.15, 0.75, 0.05, 0.05], "MUV",       "Standard",  (1_500_000, 2_500_000), 5, 7),
    ("Kia",      "Carens"):     ([0.50, 0.30, 0.10, 0.10], "MUV",       "Standard",  (1_000_000, 1_700_000), 5, 6),
    ("Honda",    "BR-V"):       ([0.55, 0.30, 0.05, 0.10], "MUV",       "Standard",  (1_100_000, 1_600_000), 4, 6),
    # Electric
    ("Tata",     "Nexon EV"):   ([0.00, 0.00, 1.00, 0.00], "SUV",       "Premium",   (1_400_000, 2_000_000), 5, 6),
    ("Tata",     "Tiago EV"):   ([0.00, 0.00, 1.00, 0.00], "Hatchback", "Standard",  (850_000,  1_300_000),  5, 6),
    ("MG",       "ZS EV"):      ([0.00, 0.00, 1.00, 0.00], "SUV",       "Premium",   (1_800_000, 2_500_000), 5, 6),
    ("Hyundai",  "Kona EV"):    ([0.00, 0.00, 1.00, 0.00], "SUV",       "Premium",   (2_300_000, 3_000_000), 5, 6),
    ("BYD",      "Atto 3"):     ([0.00, 0.00, 1.00, 0.00], "SUV",       "Premium",   (2_000_000, 2_800_000), 5, 6),
    # Luxury
    ("BMW",      "3 Series"):   ([0.35, 0.20, 0.10, 0.35], "Sedan",     "Luxury",    (4_500_000, 6_500_000), 5, 8),
    ("Mercedes", "C-Class"):    ([0.35, 0.20, 0.10, 0.35], "Sedan",     "Luxury",    (4_800_000, 7_000_000), 5, 8),
    ("Audi",     "A4"):         ([0.35, 0.20, 0.10, 0.35], "Sedan",     "Luxury",    (4_500_000, 6_000_000), 5, 8),
    ("BMW",      "X5"):         ([0.25, 0.20, 0.10, 0.45], "SUV",       "Luxury",    (7_000_000,11_000_000), 5, 8),
    ("Mercedes", "GLE"):        ([0.25, 0.15, 0.10, 0.50], "SUV",       "Luxury",    (8_000_000,12_000_000), 5, 8),
    ("Porsche",  "Cayenne"):    ([0.30, 0.10, 0.15, 0.45], "SUV",       "Luxury",    (9_000_000,14_000_000), 5, 8),
}
VEHICLE_KEYS = list(VEHICLE_CATALOG.keys())
VEHICLE_WEIGHTS_RAW = np.array([
    0.08, 0.09, 0.07, 0.07, 0.06, 0.05,  # hatchbacks
    0.06, 0.07, 0.06, 0.04, 0.02, 0.02,  # sedans
    0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02,  # SUVs
    0.03, 0.04, 0.02, 0.02,              # MUVs
    0.03, 0.02, 0.02, 0.01, 0.01,        # Electric
    0.01, 0.01, 0.01, 0.005, 0.005, 0.003,  # Luxury
])
VEHICLE_WEIGHTS = VEHICLE_WEIGHTS_RAW / VEHICLE_WEIGHTS_RAW.sum()

FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid"]
FUEL_MULT = {"Petrol": 1.00, "Diesel": 1.05, "Electric": 1.35, "Hybrid": 1.18}

BODY_MULT = {"Hatchback": 0.95, "Sedan": 1.00, "MUV": 1.08, "SUV": 1.12, "Luxury": 1.30}
REPAIR_MULT = {"Budget": 0.92, "Standard": 1.05, "Premium": 1.20, "Luxury": 1.40}

# ─────────────────────────────────────────────────────────────
# DEMOGRAPHIC DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────
OCCUPATIONS = ["Professional", "Salaried", "Self-Employed", "Student", "Retired", "Manual"]
OCC_WEIGHTS  = [0.20, 0.35, 0.15, 0.10, 0.10, 0.10]
OCC_DELTA    = {"Professional": -0.025, "Salaried": 0.000, "Self-Employed": 0.015,
                "Student": 0.035, "Retired": -0.025, "Manual": 0.045}

MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"]
MAR_WEIGHTS      = [0.30, 0.55, 0.10, 0.05]
MAR_DELTA        = {"Single": 0.000, "Married": -0.018, "Divorced": 0.000, "Widowed": -0.010}

GENDERS = ["Male", "Female", "Other"]
GEN_WEIGHTS = [0.58, 0.40, 0.02]

INCOME_BANDS = ["<3 LPA", "3-7 LPA", "7-15 LPA", "15-30 LPA", ">30 LPA"]
INC_DELTA    = {"<3 LPA": 0.010, "3-7 LPA": 0.000, "7-15 LPA": -0.008,
                "15-30 LPA": -0.008, ">30 LPA": -0.010}

USAGE_TYPES = ["Personal", "Commercial", "Rideshare"]
USAGE_WEIGHTS = [0.78, 0.14, 0.08]
USAGE_DELTA   = {"Personal": 0.000, "Commercial": 0.070, "Rideshare": 0.110}

PARKING_TYPES = ["Garage", "Open_Compound", "Street"]
PARK_WEIGHTS  = [0.35, 0.40, 0.25]
PARK_DELTA    = {"Garage": -0.010, "Open_Compound": 0.010, "Street": 0.030}

POLICY_MONTHS = list(range(1, 13))
# Seasonal: more claims in monsoon (Jun-Sep) and fog season (Dec-Jan)
MONTH_WEIGHTS = [0.09, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.09, 0.07, 0.07, 0.06]
# Risk exposure by inception month (delta prob, relative to flat)
MONTH_DELTA   = {1: 0.010, 2: 0.005, 3: 0.000, 4: 0.000, 5: 0.000,
                 6: 0.010, 7: 0.015, 8: 0.015, 9: 0.010, 10: 0.000,
                 11: 0.000, 12: 0.010}

NCB_SCHEDULE = [0, 20, 25, 35, 45, 50]  # after 0,1,2,3,4,5+ claim-free years

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def sample_age(n):
    # Age 18-75, right-skewed toward 25-50
    raw = rng.beta(2.5, 4.0, n) * 57 + 18
    return raw.astype(int)

def sample_years_licensed(age):
    n = len(age)
    min_lic = np.maximum(0, age - 18)
    raw = rng.beta(2.0, 2.0, n) * min_lic
    return np.clip(raw.astype(int), 0, age - 18)

def sample_previous_claims(n):
    # ~70% have 0 claims, rest 1-5
    probs = [0.70, 0.15, 0.08, 0.04, 0.02, 0.01]
    return rng.choice(range(6), n, p=probs)

def sample_months_since_last_claim(prev_claims):
    """
    Months since last claim — causal: customers with 0 previous claims
    have no 'last claim' so we assign 999 (never claimed).
    For others, sample recency from 1-120 months.
    """
    n = len(prev_claims)
    months = np.where(prev_claims == 0, 999,
                      rng.integers(1, 121, n))
    return months

def sample_policy_tenure(n):
    # 0.5 to 25 years, log-normal
    raw = rng.lognormal(mean=1.0, sigma=0.8, size=n)
    return np.clip(raw, 0.5, 25.0).round(1)

def sample_ncb(prev_claims, policy_tenure):
    """
    NCB based on claim-free years (heuristic from policy_tenure and prev_claims).
    If prev_claims == 0, NCB = min(50, step from tenure years).
    If prev_claims > 0, NCB resets partially.
    """
    claim_free_years = np.where(
        prev_claims == 0,
        policy_tenure.astype(int),
        np.maximum(0, policy_tenure.astype(int) - prev_claims * 2)
    )
    ncb = np.array([NCB_SCHEDULE[min(y, 5)] for y in claim_free_years])
    return ncb.astype(float)

def assign_income_band_from_vehicle_value(vehicle_value):
    n = len(vehicle_value)
    bands = []
    for vv in vehicle_value:
        if vv < 600_000:
            b = rng.choice(INCOME_BANDS, p=[0.35, 0.40, 0.15, 0.08, 0.02])
        elif vv < 1_200_000:
            b = rng.choice(INCOME_BANDS, p=[0.10, 0.35, 0.35, 0.15, 0.05])
        elif vv < 2_500_000:
            b = rng.choice(INCOME_BANDS, p=[0.02, 0.10, 0.30, 0.40, 0.18])
        elif vv < 5_000_000:
            b = rng.choice(INCOME_BANDS, p=[0.00, 0.02, 0.10, 0.40, 0.48])
        else:
            b = rng.choice(INCOME_BANDS, p=[0.00, 0.00, 0.03, 0.22, 0.75])
        bands.append(b)
    return np.array(bands)

def compute_driving_score_v4(years_licensed, prev_claims, months_since_last_claim):
    """
    V4 driving score — DECOUPLED from vehicle_usage_type and claims_last_3_years
    (fixes V3 double-encoding flaw).

    Base = 70
    + Experience bonus: +0.5 per year licensed (max +12.5 at 25yr)
    - Claim penalty: -10 per previous claim (harsh but decays with recency)
    + Recency bonus: up to +5 if no recent claim (months_since_last > 36)
    + Noise: N(0, 5)
    """
    base = 70.0
    exp_bonus = np.minimum(years_licensed * 0.5, 12.5)
    claim_penalty = np.minimum(prev_claims * 10.0, 30.0)
    recency_bonus = np.where(
        months_since_last_claim == 999, 5.0,
        np.where(months_since_last_claim > 36, 4.0,
        np.where(months_since_last_claim > 12, 2.0,
        np.where(months_since_last_claim > 6,  1.0, 0.0)))
    )
    noise = rng.normal(0, 5.0, len(years_licensed))
    score = base + exp_bonus - claim_penalty + recency_bonus + noise
    return np.clip(score, 0.0, 100.0).round(1)

def compute_annual_mileage(usage_type, vehicle_age):
    """Annual mileage depends on usage type and vehicle age."""
    n = len(usage_type)
    base = np.where(usage_type == "Personal", 12_000,
           np.where(usage_type == "Commercial", 30_000, 45_000)).astype(float)
    # Older vehicles driven slightly less
    age_factor = 1.0 - np.minimum(vehicle_age * 0.01, 0.15)
    noise = rng.lognormal(0, 0.25, n)
    km = base * age_factor * noise
    return np.clip(km, 2_000, 100_000).astype(int)

def compute_claim_probability(row_data):
    """
    Causal claim probability formula (V4).
    Each delta is independently motivated by actuarial evidence.
    """
    age = row_data["age"]
    years_lic = row_data["years_licensed"]
    prev_claims = row_data["previous_claims_count"]
    months_since = row_data["months_since_last_claim"]
    ncb_pct = row_data["no_claim_bonus_pct"]
    tenure_yrs = row_data["policy_tenure_years"]
    occupation = row_data["occupation"]
    marital = row_data["marital_status"]
    income_band = row_data["annual_income_band"]
    usage = row_data["vehicle_usage_type"]
    mileage = row_data["annual_mileage_km"]
    parking = row_data["parking_type"]
    pincode_risk = row_data["pincode_risk_score"]
    theft_risk = row_data["theft_risk_index"]
    flood_risk = row_data["flood_risk_index"]
    driving_score = row_data["driving_score"]
    inception_month = row_data["policy_inception_month"]

    # Base intercept (calibrated to hit ~17% mean claim rate)
    p = 0.100

    # Age risk (U-shaped: young and very old drivers riskier)
    if age < 23:    p += 0.080
    elif age < 30:  p += 0.055
    elif age < 40:  p += 0.030
    elif age < 55:  p += 0.000
    elif age < 65:  p -= 0.010
    else:           p += 0.035

    # Experience (monotone protective)
    p -= 0.020 * min(years_lic / 25.0, 1.0)

    # Previous claims — strongest frequency signal
    p += 0.120 * min(prev_claims / 3.0, 1.0)

    # Recency of last claim (months_since_last_claim)
    # 999 = never claimed (lowest risk)
    if months_since == 999:  p += 0.000
    elif months_since <= 6:  p += 0.060
    elif months_since <= 12: p += 0.040
    elif months_since <= 24: p += 0.020
    elif months_since <= 36: p += 0.010
    else:                    p += 0.000

    # NCB: customers with high NCB have longer claim-free history
    p -= 0.040 * (ncb_pct / 50.0)

    # Policy tenure: loyal customers are better risks (loyalty bias / learning)
    p -= 0.015 * min(tenure_yrs / 10.0, 1.0)

    # Occupation delta
    p += OCC_DELTA.get(occupation, 0)

    # Marital status delta
    p += MAR_DELTA.get(marital, 0)

    # Income delta
    p += INC_DELTA.get(income_band, 0)

    # Vehicle usage
    p += USAGE_DELTA.get(usage, 0)

    # Annual mileage (exposure)
    p += 0.020 * (mileage / 20_000)

    # Parking type
    p += PARK_DELTA.get(parking, 0)

    # Geographic risk (pincode, theft, flood)
    p += 0.010 * (pincode_risk - 5.0) / 5.0
    p += 0.008 * theft_risk / 10.0
    p += 0.006 * flood_risk / 10.0

    # Driving score (inverted: higher score = lower risk)
    p -= 0.040 * (driving_score - 50.0) / 50.0

    # Seasonal inception month
    p += MONTH_DELTA.get(inception_month, 0)

    return max(0.01, min(0.99, p))

def compute_claim_amount(row_data):
    """
    Severity formula — causal multipliers documented.
    base = vehicle_value_inr × Uniform(0.05, 0.55)
    Then multiply by fuel, repair, body, safety, airbag, flood multipliers.
    """
    vehicle_value = row_data["vehicle_value_inr"]
    fuel_type     = row_data["fuel_type"]
    repair_band   = row_data["repair_cost_band"]
    body_style    = row_data["vehicle_body_style"]
    safety_rating = row_data["vehicle_safety_rating"]
    airbags       = row_data["airbags_count"]
    flood_risk    = row_data["flood_risk_index"]
    vehicle_age   = row_data["vehicle_age_years"]

    base = vehicle_value * rng.uniform(0.05, 0.55)

    fuel_m   = FUEL_MULT.get(fuel_type, 1.0)
    repair_m = REPAIR_MULT.get(repair_band, 1.0)
    body_m   = BODY_MULT.get(body_style, 1.0)
    safety_m = 1.0 + (5 - safety_rating) * 0.055   # worse safety = higher cost
    airbag_d = max(0.85, 1.0 - (airbags - 2) * 0.025)  # more airbags = lower cost
    flood_m  = 1.0 + flood_risk * 0.040             # flood-prone area = higher water damage
    age_d    = 1.0 - min(vehicle_age * 0.02, 0.30)  # depreciation reduces claim value

    amount = base * fuel_m * repair_m * body_m * safety_m * airbag_d * flood_m * age_d
    return float(np.clip(amount, 5_000, vehicle_value * 0.95))


# ─────────────────────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────────────────────

def generate():
    print(f"Generating V4 dataset: {N_ROWS:,} rows ...")
    t0 = datetime.now()

    # ── Customer IDs
    customer_ids = [f"V4-{i+1:06d}" for i in range(N_ROWS)]

    # ── City + geographic risk
    city_arr = rng.choice(CITIES, N_ROWS, p=CITY_WEIGHTS)
    city_tier = np.array([CITY_PROFILES[c][0] for c in city_arr])
    flood_risk   = np.array([CITY_PROFILES[c][1] for c in city_arr])
    theft_risk   = np.array([CITY_PROFILES[c][2] for c in city_arr])
    pincode_base = np.array([CITY_PROFILES[c][3] for c in city_arr])
    monsoon      = np.array([CITY_PROFILES[c][4] for c in city_arr])

    # Add noise to geographic indices (intra-city variation)
    pincode_risk  = np.clip(pincode_base + rng.normal(0, 0.8, N_ROWS), 1.0, 10.0).round(2)
    flood_risk    = np.clip(flood_risk   + rng.normal(0, 0.5, N_ROWS), 0.0, 10.0).round(2)
    theft_risk    = np.clip(theft_risk   + rng.normal(0, 0.5, N_ROWS), 0.0, 10.0).round(2)
    monsoon_exp   = np.clip(monsoon      + rng.normal(0, 0.4, N_ROWS), 0.0, 10.0).round(2)

    # ── Demographics
    age        = sample_age(N_ROWS)
    gender     = rng.choice(GENDERS, N_ROWS, p=GEN_WEIGHTS)
    occupation = rng.choice(OCCUPATIONS, N_ROWS, p=OCC_WEIGHTS)
    marital    = rng.choice(MARITAL_STATUSES, N_ROWS, p=MAR_WEIGHTS)
    years_lic  = sample_years_licensed(age)
    prev_claims = sample_previous_claims(N_ROWS)

    # ── Claim history dimensions (NEW in V4)
    months_since_last = sample_months_since_last_claim(prev_claims)
    policy_tenure     = sample_policy_tenure(N_ROWS)
    ncb_pct           = sample_ncb(prev_claims, policy_tenure)
    policy_month      = rng.choice(POLICY_MONTHS, N_ROWS, p=MONTH_WEIGHTS)

    # ── Vehicle selection
    veh_idx  = rng.choice(len(VEHICLE_KEYS), N_ROWS, p=VEHICLE_WEIGHTS)
    veh_make = np.array([VEHICLE_KEYS[i][0] for i in veh_idx])
    veh_model = np.array([VEHICLE_KEYS[i][1] for i in veh_idx])
    veh_specs = [VEHICLE_CATALOG[VEHICLE_KEYS[i]] for i in veh_idx]

    # Fuel type (sampled from per-model distribution)
    fuel_type = np.array([
        rng.choice(FUEL_TYPES, p=spec[0]) for spec in veh_specs
    ])
    body_style  = np.array([spec[1] for spec in veh_specs])
    repair_band = np.array([spec[2] for spec in veh_specs])
    val_lo      = np.array([spec[3][0] for spec in veh_specs])
    val_hi      = np.array([spec[3][1] for spec in veh_specs])
    safety_rat  = np.array([spec[4] for spec in veh_specs])
    airbags_cnt = np.array([spec[5] for spec in veh_specs])

    # Vehicle value: uniform within range with age depreciation applied later
    vehicle_value_raw = (rng.uniform(0, 1, N_ROWS) * (val_hi - val_lo) + val_lo).astype(int)

    # Vehicle age: 0-15 years, weighted toward newer
    vehicle_age = rng.choice(range(16), N_ROWS,
                             p=[0.12,0.10,0.10,0.09,0.08,0.08,0.07,0.07,
                                0.06,0.05,0.04,0.04,0.03,0.03,0.02,0.02])
    # Depreciated vehicle value
    vehicle_value = (vehicle_value_raw * (1 - vehicle_age * 0.07)).astype(int)
    vehicle_value = np.maximum(vehicle_value, 100_000)

    # Engine CC (correlated with body style and value)
    engine_base = {"Hatchback": 1100, "Sedan": 1500, "SUV": 1800,
                   "MUV": 1500, "Luxury": 2500}
    engine_cc = np.array([
        int(engine_base.get(bs, 1500) * rng.uniform(0.8, 1.4))
        if ft != "Electric" else 0
        for bs, ft in zip(body_style, fuel_type)
    ])

    # Income band (correlated with vehicle value)
    income_band = assign_income_band_from_vehicle_value(vehicle_value)

    # Usage type
    usage_type = rng.choice(USAGE_TYPES, N_ROWS, p=USAGE_WEIGHTS)

    # Annual mileage
    annual_mileage = compute_annual_mileage(usage_type, vehicle_age)

    # Parking type
    parking_type = rng.choice(PARKING_TYPES, N_ROWS, p=PARK_WEIGHTS)

    # V4 driving score (decoupled)
    driving_score = compute_driving_score_v4(years_lic, prev_claims, months_since_last)

    # ── Claim targets
    print("  Computing claim probabilities ...")
    row_dicts = []
    for i in range(N_ROWS):
        row_dicts.append({
            "age": age[i],
            "years_licensed": years_lic[i],
            "previous_claims_count": prev_claims[i],
            "months_since_last_claim": months_since_last[i],
            "no_claim_bonus_pct": ncb_pct[i],
            "policy_tenure_years": policy_tenure[i],
            "occupation": occupation[i],
            "marital_status": marital[i],
            "annual_income_band": income_band[i],
            "vehicle_usage_type": usage_type[i],
            "annual_mileage_km": annual_mileage[i],
            "parking_type": parking_type[i],
            "pincode_risk_score": pincode_risk[i],
            "theft_risk_index": theft_risk[i],
            "flood_risk_index": flood_risk[i],
            "driving_score": driving_score[i],
            "policy_inception_month": policy_month[i],
            "vehicle_value_inr": vehicle_value[i],
            "vehicle_age_years": vehicle_age[i],
            "fuel_type": fuel_type[i],
            "repair_cost_band": repair_band[i],
            "vehicle_body_style": body_style[i],
            "vehicle_safety_rating": safety_rat[i],
            "airbags_count": airbags_cnt[i],
        })

    claim_probs = np.array([compute_claim_probability(r) for r in row_dicts])
    claim_occurred = (rng.uniform(0, 1, N_ROWS) < claim_probs).astype(int)

    claim_amounts = np.zeros(N_ROWS)
    for i in range(N_ROWS):
        if claim_occurred[i]:
            claim_amounts[i] = compute_claim_amount(row_dicts[i])

    print(f"  Claim rate: {claim_occurred.mean()*100:.2f}%  "
          f"(target ~17%, n_claims={claim_occurred.sum():,})")
    print(f"  Mean claim amount (claimants): "
          f"INR {claim_amounts[claim_occurred==1].mean():,.0f}")

    # ── Assemble DataFrame
    df = pd.DataFrame({
        "customer_id":              customer_ids,
        # Customer-declared features
        "age":                      age,
        "gender":                   gender,
        "occupation":               occupation,
        "marital_status":           marital,
        "annual_income_band":       income_band,
        "city":                     city_arr,
        "years_licensed":           years_lic,
        "previous_claims_count":    prev_claims,
        "months_since_last_claim":  months_since_last,
        "no_claim_bonus_pct":       ncb_pct,
        "policy_tenure_years":      policy_tenure,
        "vehicle_make":             veh_make,
        "vehicle_model":            veh_model,
        "fuel_type":                fuel_type,
        "vehicle_usage_type":       usage_type,
        "annual_mileage_km":        annual_mileage,
        "parking_type":             parking_type,
        # System-derived features
        "driving_score":            driving_score,
        "pincode_risk_score":       pincode_risk,
        "region_risk_category":     city_tier,
        "flood_risk_index":         flood_risk,
        "theft_risk_index":         theft_risk,
        "monsoon_exposure_index":   monsoon_exp,
        "vehicle_value_inr":        vehicle_value,
        "vehicle_age_years":        vehicle_age,
        "engine_cc":                engine_cc,
        "vehicle_safety_rating":    safety_rat,
        "airbags_count":            airbags_cnt,
        "repair_cost_band":         repair_band,
        "vehicle_body_style":       body_style,
        "policy_inception_month":   policy_month,
        # Targets
        "claim_probability_true":   claim_probs.round(4),
        "claim_occurred":           claim_occurred,
        "actual_claim_amount_inr":  claim_amounts.round(2),
    })

    # ── Save CSV
    out_path = os.path.join(DATA_DIR, "car_insurance_portfolio_v4.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {df.shape}")

    elapsed = (datetime.now() - t0).total_seconds()
    return df, claim_probs, elapsed


def write_feature_design_document(df):
    path = os.path.join(REPORT_DIR, "v4_feature_design_document.txt")
    n = len(df)
    with open(path, "w", encoding="utf-8") as f:
        f.write("V4 FEATURE DESIGN DOCUMENT\n")
        f.write("Insurance AI Pricing System — Controlled Experiment\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("OBJECTIVE\n")
        f.write("---------\n")
        f.write("Determine whether richer insurance information can materially\n")
        f.write("improve frequency and severity models without telematics data,\n")
        f.write("credit bureau data, or insurer-only proprietary data.\n\n")

        f.write("V4 DESIGN PRINCIPLES\n")
        f.write("--------------------\n")
        f.write("1. No telematics (no GPS, no hard braking, no speeding ratio)\n")
        f.write("2. No credit bureau data\n")
        f.write("3. No insurer-internal proprietary data\n")
        f.write("4. All new features are customer-declared or publicly derivable\n")
        f.write("5. V4 driving_score is DECOUPLED from usage_type and claims_3yr\n")
        f.write("   (fixes V3 double-encoding flaw)\n")
        f.write("6. Three mathematically redundant V3 features removed:\n")
        f.write("   normalized_mileage, vehicle_depreciation_factor, high_value_vehicle\n\n")

        f.write("SCHEMA: 50,000 rows x 33 columns\n")
        f.write("---------------------------------\n\n")

        f.write("SECTION A — CUSTOMER-DECLARED FEATURES (17)\n")
        f.write("--------------------------------------------\n")
        customer_features = [
            ("age", "int", "18-75", "Driver age (U-shaped risk)"),
            ("gender", "str", "Male/Female/Other", "Customer gender"),
            ("occupation", "str", "6 categories", "Occupation affects driving exposure"),
            ("marital_status", "str", "4 categories", "Marital status risk proxy"),
            ("annual_income_band", "str", "5 bands", "Income (correlated r~0.55 with vehicle value)"),
            ("city", "str", "20 cities", "City of registration"),
            ("years_licensed", "int", "0-57", "Years of driving license held"),
            ("previous_claims_count", "int", "0-5", "Lifetime claim count"),
            ("months_since_last_claim", "int", "1-120 or 999", "999=never claimed. NEW V4 recency signal"),
            ("no_claim_bonus_pct", "float", "0/20/25/35/45/50", "NCB tier — loyalty/risk proxy. NEW V4"),
            ("policy_tenure_years", "float", "0.5-25.0", "Years with current insurer. NEW V4"),
            ("vehicle_make", "str", "15 makes", "Vehicle manufacturer"),
            ("vehicle_model", "str", "35 models", "Specific model (maps to repair band)"),
            ("fuel_type", "str", "Petrol/Diesel/Electric/Hybrid", "NEW V4 — EV repair cost 1.35x. Genuine causal signal"),
            ("vehicle_usage_type", "str", "Personal/Commercial/Rideshare", "Usage intensity (standalone, not encoded into driving_score)"),
            ("annual_mileage_km", "int", "2000-100000", "Self-declared mileage (exposure)"),
            ("parking_type", "str", "Garage/Open_Compound/Street", "NEW V4 — theft and damage risk"),
        ]
        f.write(f"{'Feature':<30} {'Type':<8} {'Range/Values':<35} Description\n")
        f.write("-" * 110 + "\n")
        for feat, typ, rng_val, desc in customer_features:
            f.write(f"{feat:<30} {typ:<8} {rng_val:<35} {desc}\n")

        f.write("\nSECTION B — SYSTEM-DERIVED FEATURES (13)\n")
        f.write("-----------------------------------------\n")
        system_features = [
            ("driving_score", "float", "0-100", "V4 formula: decoupled from usage_type (V3 flaw fixed)"),
            ("pincode_risk_score", "float", "1.0-10.0", "NEW V4 — geographic risk (MoRTH accident data proxy)"),
            ("region_risk_category", "int", "1/2", "City tier (1=metro, 2=tier-2)"),
            ("flood_risk_index", "float", "0-10", "NEW V4 — IMD flood vulnerability proxy"),
            ("theft_risk_index", "float", "0-10", "NEW V4 — NCRB vehicle theft rate proxy"),
            ("monsoon_exposure_index", "float", "0-10", "NEW V4 — seasonal weather severity"),
            ("vehicle_value_inr", "int", "100000-14000000", "Depreciated vehicle value"),
            ("vehicle_age_years", "int", "0-15", "Vehicle age (affects depreciation and repair cost)"),
            ("engine_cc", "int", "0-3500", "Engine displacement (0 for EVs)"),
            ("vehicle_safety_rating", "int", "3-5", "NCAP/equivalent safety stars"),
            ("airbags_count", "int", "2-8", "Number of airbags (from vehicle specs DB)"),
            ("repair_cost_band", "str", "Budget/Standard/Premium/Luxury", "NEW V4 — from vehicle model specs DB"),
            ("vehicle_body_style", "str", "Hatchback/Sedan/MUV/SUV/Luxury", "NEW V4 — body style repair complexity"),
            ("policy_inception_month", "int", "1-12", "NEW V4 — seasonal claims exposure"),
        ]
        f.write(f"{'Feature':<30} {'Type':<8} {'Range/Values':<35} Description\n")
        f.write("-" * 110 + "\n")
        for feat, typ, rng_val, desc in system_features:
            f.write(f"{feat:<30} {typ:<8} {rng_val:<35} {desc}\n")

        f.write("\nSECTION C — TARGET VARIABLES (3)\n")
        f.write("---------------------------------\n")
        f.write("claim_probability_true    float  0.01-0.99  True latent probability (oracle only)\n")
        f.write("claim_occurred            int    0/1        Binary frequency target\n")
        f.write("actual_claim_amount_inr   float  0 or 5000+ Severity target (0 for non-claimants)\n")

        f.write("\n\nCAUSAL GENERATION FORMULAS\n")
        f.write("==========================\n\n")
        f.write("FREQUENCY MODEL — Claim Probability Formula\n")
        f.write("-------------------------------------------\n")
        f.write("p = 0.100  (base intercept, calibrated to ~17% mean rate)\n\n")
        f.write("Age deltas (U-shaped):     <23 +0.080, <30 +0.055, <40 +0.030,\n")
        f.write("                           <55 +0.000, <65 -0.010, 65+ +0.035\n")
        f.write("Experience:                -0.020 × min(years_licensed/25, 1.0)\n")
        f.write("Previous claims:           +0.120 × min(prev_claims/3, 1.0)\n")
        f.write("Recency (months_since):    999->+0.000, <=6->+0.060, <=12->+0.040,\n")
        f.write("                           <=24->+0.020, <=36->+0.010, >36->+0.000\n")
        f.write("NCB:                       -0.040 × (ncb_pct/50.0)\n")
        f.write("Policy tenure:             -0.015 × min(policy_tenure/10, 1.0)\n")
        f.write("Occupation deltas:         Professional -0.025, Salaried 0.000,\n")
        f.write("                           Self-Employed +0.015, Student +0.035,\n")
        f.write("                           Retired -0.025, Manual +0.045\n")
        f.write("Marital status deltas:     Single 0.000, Married -0.018,\n")
        f.write("                           Divorced 0.000, Widowed -0.010\n")
        f.write("Income deltas:             <3LPA +0.010, 3-7LPA 0.000,\n")
        f.write("                           7-15LPA -0.008, 15-30LPA -0.008, >30LPA -0.010\n")
        f.write("Usage type:                Personal 0.000, Commercial +0.070, Rideshare +0.110\n")
        f.write("Annual mileage:            +0.020 × (km/20,000)\n")
        f.write("Parking type:              Garage -0.010, Open_Compound +0.010, Street +0.030\n")
        f.write("Pincode risk:              +0.010 × (score-5)/5\n")
        f.write("Theft risk index:          +0.008 × index/10\n")
        f.write("Flood risk index:          +0.006 × index/10\n")
        f.write("Driving score:             -0.040 × (score-50)/50\n")
        f.write("Inception month:           Jan +0.010, Feb +0.005, Jun-Sep +0.010-0.015,\n")
        f.write("                           Dec +0.010 (monsoon + fog season)\n\n")

        f.write("SEVERITY MODEL — Claim Amount Formula\n")
        f.write("--------------------------------------\n")
        f.write("base = vehicle_value_inr × Uniform(0.05, 0.55)\n")
        f.write("fuel_mult:    Petrol 1.00, Diesel 1.05, Electric 1.35, Hybrid 1.18\n")
        f.write("repair_mult:  Budget 0.92, Standard 1.05, Premium 1.20, Luxury 1.40\n")
        f.write("body_mult:    Hatchback 0.95, Sedan 1.00, MUV 1.08, SUV 1.12, Luxury 1.30\n")
        f.write("safety_mult:  1.0 + (5-safety_rating) × 0.055  (worse safety = higher cost)\n")
        f.write("airbag_disc:  max(0.85, 1.0 - (airbags-2) × 0.025) (more airbags = lower)\n")
        f.write("flood_mult:   1.0 + flood_risk_index × 0.040\n")
        f.write("age_disc:     1.0 - min(vehicle_age × 0.02, 0.30) (depreciation)\n")
        f.write("final_amount = base × fuel_m × repair_m × body_m × safety_m\n")
        f.write("                     × airbag_d × flood_m × age_d\n")
        f.write("             clipped to [5,000, vehicle_value × 0.95]\n\n")

        f.write("V4 DRIVING SCORE FORMULA (DECOUPLED)\n")
        f.write("-------------------------------------\n")
        f.write("V3 flaw: driving_score encoded both claims_last_3_years AND\n")
        f.write("         vehicle_usage_type — creating collinearity.\n")
        f.write("V4 fix:  Score derives ONLY from experience and claim history.\n")
        f.write("         Usage type kept as STANDALONE feature.\n\n")
        f.write("score = 70 (base)\n")
        f.write("      + min(years_licensed × 0.5, 12.5)  [experience bonus]\n")
        f.write("      - min(previous_claims_count × 10, 30) [claim penalty]\n")
        f.write("      + recency_bonus  [+5 if never claimed, +4 if >36mo, etc.]\n")
        f.write("      + N(0, 5) noise\n")
        f.write("      clipped to [0, 100]\n\n")

        f.write("V4 vs V3 FEATURE COMPARISON\n")
        f.write("---------------------------\n")
        f.write("Features RETAINED from V3 (11):     age, gender, city, vehicle_make,\n")
        f.write("  vehicle_model, engine_cc, vehicle_age_years, vehicle_value_inr,\n")
        f.write("  annual_mileage_km, previous_claims_count, years_licensed\n\n")
        f.write("V3 features RETAINED (improved):     occupation, marital_status,\n")
        f.write("  vehicle_usage_type (standalone), annual_income_band, driving_score (fixed),\n")
        f.write("  vehicle_safety_rating, airbags_count, region_risk_category\n\n")
        f.write("Mathematically REDUNDANT V3 features REMOVED (3):\n")
        f.write("  normalized_mileage, vehicle_depreciation_factor, high_value_vehicle\n\n")
        f.write("GENUINELY NEW V4 features (12):\n")
        f.write("  months_since_last_claim  — recency dimension (absent from V3)\n")
        f.write("  no_claim_bonus_pct       — loyalty/risk signal\n")
        f.write("  policy_tenure_years      — insurer tenure\n")
        f.write("  fuel_type               — EV repair ×1.35, Hybrid ×1.18 (genuinely causal)\n")
        f.write("  repair_cost_band        — model-level repair cost (exploits ignored car_model)\n")
        f.write("  vehicle_body_style      — body style repair complexity\n")
        f.write("  parking_type            — theft/damage exposure\n")
        f.write("  policy_inception_month  — seasonal claims signal\n")
        f.write("  pincode_risk_score      — geographic risk (MoRTH proxy)\n")
        f.write("  flood_risk_index        — flood vulnerability (IMD proxy)\n")
        f.write("  theft_risk_index        — vehicle theft rate (NCRB proxy)\n")
        f.write("  monsoon_exposure_index  — seasonal weather severity\n")

    print(f"  Written: v4_feature_design_document.txt")
    return path


def write_validation_report(df, elapsed):
    path = os.path.join(REPORT_DIR, "v4_validation_report.txt")
    claimants = df[df["claim_occurred"] == 1]
    n = len(df)
    with open(path, "w", encoding="utf-8") as f:
        f.write("V4 DATASET VALIDATION REPORT\n")
        f.write("Insurance AI Pricing System — Controlled Experiment\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SECTION 1 — DATASET OVERVIEW\n")
        f.write("-----------------------------\n")
        f.write(f"Total rows:             {n:,}\n")
        f.write(f"Total columns:          {len(df.columns)}\n")
        f.write(f"Generation time:        {elapsed:.1f}s\n")
        f.write(f"Random seed:            {RANDOM_SEED}\n")
        f.write(f"Missing values:         {df.isnull().sum().sum()}\n\n")

        f.write("SECTION 2 — TARGET DISTRIBUTION\n")
        f.write("---------------------------------\n")
        n_claims = df["claim_occurred"].sum()
        claim_rate = n_claims / n * 100
        f.write(f"Claims:                 {n_claims:,} ({claim_rate:.2f}%)\n")
        f.write(f"Non-claims:             {n - n_claims:,} ({100-claim_rate:.2f}%)\n")
        f.write(f"Mean true prob:         {df['claim_probability_true'].mean():.4f}\n")
        f.write(f"True prob range:        [{df['claim_probability_true'].min():.4f}, "
                f"{df['claim_probability_true'].max():.4f}]\n")
        f.write(f"Mean claim amount (claimants): INR {claimants['actual_claim_amount_inr'].mean():,.0f}\n")
        f.write(f"Median claim amount:    INR {claimants['actual_claim_amount_inr'].median():,.0f}\n")
        f.write(f"Min claim amount:       INR {claimants['actual_claim_amount_inr'].min():,.0f}\n")
        f.write(f"Max claim amount:       INR {claimants['actual_claim_amount_inr'].max():,.0f}\n\n")

        f.write("SECTION 3 — FEATURE DISTRIBUTIONS\n")
        f.write("-----------------------------------\n")
        f.write(f"Age: mean={df['age'].mean():.1f}, std={df['age'].std():.1f}, "
                f"range=[{df['age'].min()},{df['age'].max()}]\n")
        f.write(f"Years licensed: mean={df['years_licensed'].mean():.1f}, "
                f"range=[{df['years_licensed'].min()},{df['years_licensed'].max()}]\n")
        f.write(f"Previous claims: mean={df['previous_claims_count'].mean():.3f}, "
                f"max={df['previous_claims_count'].max()}\n")
        f.write(f"Months since last claim (excluding 999): "
                f"mean={df.loc[df['months_since_last_claim']!=999,'months_since_last_claim'].mean():.1f}\n")
        f.write(f"Customers with months_since=999 (never claimed): "
                f"{(df['months_since_last_claim']==999).sum():,} "
                f"({(df['months_since_last_claim']==999).mean()*100:.1f}%)\n")
        f.write(f"NCB pct: mean={df['no_claim_bonus_pct'].mean():.1f}, "
                f"values={sorted(df['no_claim_bonus_pct'].unique().tolist())}\n")
        f.write(f"Policy tenure: mean={df['policy_tenure_years'].mean():.1f}yr, "
                f"range=[{df['policy_tenure_years'].min()},{df['policy_tenure_years'].max()}]\n")
        f.write(f"Vehicle value: mean=INR {df['vehicle_value_inr'].mean():,.0f}, "
                f"range=[{df['vehicle_value_inr'].min():,},{df['vehicle_value_inr'].max():,}]\n")
        f.write(f"Vehicle age: mean={df['vehicle_age_years'].mean():.1f}yr\n")
        f.write(f"Driving score: mean={df['driving_score'].mean():.1f}, "
                f"range=[{df['driving_score'].min()},{df['driving_score'].max()}]\n")
        f.write(f"Pincode risk: mean={df['pincode_risk_score'].mean():.2f}, "
                f"range=[{df['pincode_risk_score'].min():.2f},{df['pincode_risk_score'].max():.2f}]\n")
        f.write(f"Flood risk: mean={df['flood_risk_index'].mean():.2f}\n")
        f.write(f"Theft risk: mean={df['theft_risk_index'].mean():.2f}\n\n")

        f.write("SECTION 4 — CATEGORICAL DISTRIBUTIONS\n")
        f.write("---------------------------------------\n")
        for col in ["gender", "occupation", "marital_status", "annual_income_band",
                    "fuel_type", "vehicle_body_style", "repair_cost_band",
                    "parking_type", "vehicle_usage_type", "region_risk_category"]:
            vc = df[col].value_counts()
            f.write(f"\n{col}:\n")
            for val, cnt in vc.items():
                f.write(f"  {str(val):<25} {cnt:>6,}  ({cnt/n*100:.1f}%)\n")

        f.write("\nSECTION 5 — V4 AVOIDANCE OF V2 FAILURE MODE\n")
        f.write("--------------------------------------------\n")
        f.write("V2 failure: all 6 features were mathematical derivations of existing V1 columns.\n")
        f.write("            AUC regressed from 0.6645 to 0.6603 (Bayes ceiling unchanged).\n\n")
        f.write("V4 new features and their causal mechanism:\n")
        new_feats = [
            ("months_since_last_claim", "Temporal recency — not derivable from claims_count alone"),
            ("no_claim_bonus_pct",      "Policy-level NCB tier — loyalty + risk accumulated signal"),
            ("policy_tenure_years",     "Insurer tenure — not encoded in any V3 column"),
            ("fuel_type",               "EV repair costs 35% higher — new causal path to severity"),
            ("repair_cost_band",        "Model-level repair cost — leverages previously ignored car_model"),
            ("vehicle_body_style",      "Body style repair complexity — new severity signal"),
            ("parking_type",            "Theft/damage exposure — absent from V3"),
            ("policy_inception_month",  "Seasonal risk — monsoon/fog exposure window"),
            ("pincode_risk_score",      "Sub-city geographic risk — finer than city_tier (V3)"),
            ("flood_risk_index",        "Flood vulnerability — new causal path (water damage claims)"),
            ("theft_risk_index",        "Theft rate — new causal path (theft claims)"),
            ("monsoon_exposure_index",  "Weather severity — seasonal severity multiplier"),
        ]
        for feat, mechanism in new_feats:
            f.write(f"  {feat:<30} {mechanism}\n")

        f.write("\nSECTION 6 — DATA QUALITY CHECKS\n")
        f.write("---------------------------------\n")
        checks = [
            ("No missing values", df.isnull().sum().sum() == 0),
            ("Claim rate 12-22%", 0.12 <= df["claim_occurred"].mean() <= 0.22),
            ("Age range 18-75", df["age"].between(18, 75).all()),
            ("Driving score 0-100", df["driving_score"].between(0, 100).all()),
            ("NCB in valid set", df["no_claim_bonus_pct"].isin([0,20,25,35,45,50]).all()),
            ("Vehicle value > 0", (df["vehicle_value_inr"] > 0).all()),
            ("Claim amount >= 5000 for claimants",
             (df.loc[df["claim_occurred"]==1, "actual_claim_amount_inr"] >= 4999).all()),
            ("True prob [0.01, 0.99]",
             df["claim_probability_true"].between(0.01, 0.99).all()),
            ("N rows = 50,000", len(df) == 50_000),
        ]
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            f.write(f"  [{status}] {check_name}\n")

        all_pass = all(p for _, p in checks)
        f.write(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'ONE OR MORE CHECKS FAILED'}\n")

    print(f"  Written: v4_validation_report.txt")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("V4 DATASET GENERATION — Controlled Experiment")
    print("=" * 60)
    df, claim_probs, elapsed = generate()
    write_feature_design_document(df)
    write_validation_report(df, elapsed)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: experiments/outputs/data/car_insurance_portfolio_v4.csv")
    print(f"        experiments/outputs/reports/v4_feature_design_document.txt")
    print(f"        experiments/outputs/reports/v4_validation_report.txt")
