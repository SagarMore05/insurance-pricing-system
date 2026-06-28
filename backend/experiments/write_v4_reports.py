"""Helper: write Phase 2 and Phase 5 reports from the already-generated V4 CSV."""
import os
import sys
import pandas as pd
from datetime import datetime

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reports")
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "data")

df = pd.read_csv(os.path.join(DATA_DIR, "car_insurance_portfolio_v4.csv"))
print(f"Loaded V4 CSV: {df.shape}")

# ── Phase 2: Feature Design Document
with open(os.path.join(REPORT_DIR, "v4_feature_design_document.txt"), "w", encoding="utf-8") as f:
    f.write("V4 FEATURE DESIGN DOCUMENT\n")
    f.write("Insurance AI Pricing System - Controlled Experiment\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("Determine whether richer insurance information can materially improve\n")
    f.write("frequency and severity models without telematics, credit bureau, or\n")
    f.write("insurer-only proprietary data.\n\n")

    f.write("SCHEMA: 50,000 rows x 35 columns\n")
    f.write("=" * 40 + "\n\n")

    f.write("A. CUSTOMER-DECLARED FEATURES (17)\n")
    f.write("-" * 40 + "\n")
    feats_a = [
        ("age",                      "int",   "18-75",              "Driver age (U-shaped risk)"),
        ("gender",                   "str",   "Male/Female/Other",  "Customer gender"),
        ("occupation",               "str",   "6 categories",       "Driving exposure proxy"),
        ("marital_status",           "str",   "4 categories",       "Risk proxy"),
        ("annual_income_band",       "str",   "5 bands",            "Income vs vehicle value (r~0.55)"),
        ("city",                     "str",   "20 cities",          "Registration city"),
        ("years_licensed",           "int",   "0-57",               "Driving experience"),
        ("previous_claims_count",    "int",   "0-5",                "Lifetime claim count"),
        ("months_since_last_claim",  "int",   "1-120 or 999",       "[NEW V4] 999=never. Recency signal."),
        ("no_claim_bonus_pct",       "float", "0/20/25/35/45/50",   "[NEW V4] NCB tier - loyalty/risk"),
        ("policy_tenure_years",      "float", "0.5-25.0",           "[NEW V4] Years with insurer"),
        ("vehicle_make",             "str",   "15 makes",           "Manufacturer"),
        ("vehicle_model",            "str",   "35 models",          "Model -> repair_cost_band"),
        ("fuel_type",                "str",   "P/D/E/H",            "[NEW V4] EV repair x1.35, Hybrid x1.18"),
        ("vehicle_usage_type",       "str",   "Per/Com/Ride",       "Standalone (NOT in driving_score)"),
        ("annual_mileage_km",        "int",   "2000-100000",        "Exposure"),
        ("parking_type",             "str",   "Gar/Open/Street",    "[NEW V4] Theft/damage risk"),
    ]
    f.write(f"{'Feature':<28} {'Type':<7} {'Values':<22} Description\n")
    f.write("-" * 90 + "\n")
    for feat, typ, vals, desc in feats_a:
        f.write(f"{feat:<28} {typ:<7} {vals:<22} {desc}\n")

    f.write("\nB. SYSTEM-DERIVED FEATURES (15)\n")
    f.write("-" * 40 + "\n")
    feats_b = [
        ("driving_score",            "float", "0-100",              "V4 decoupled formula (V3 flaw fixed)"),
        ("pincode_risk_score",       "float", "1.0-10.0",           "[NEW V4] MoRTH accident data proxy"),
        ("region_risk_category",     "int",   "1/2",                "City tier"),
        ("flood_risk_index",         "float", "0-10",               "[NEW V4] IMD flood vulnerability"),
        ("theft_risk_index",         "float", "0-10",               "[NEW V4] NCRB theft rate proxy"),
        ("monsoon_exposure_index",   "float", "0-10",               "[NEW V4] Seasonal weather severity"),
        ("vehicle_value_inr",        "int",   "100k-14M",           "Depreciated value"),
        ("vehicle_age_years",        "int",   "0-15",               "Vehicle age"),
        ("engine_cc",                "int",   "0-3500",             "Displacement (0 for EV)"),
        ("vehicle_safety_rating",    "int",   "3-5",                "NCAP stars"),
        ("airbags_count",            "int",   "2-8",                "Airbag count"),
        ("repair_cost_band",         "str",   "Bdg/Std/Prm/Lux",   "[NEW V4] Model repair cost band"),
        ("vehicle_body_style",       "str",   "Hat/Sed/MUV/SUV/Lux","[NEW V4] Body style complexity"),
        ("policy_inception_month",   "int",   "1-12",               "[NEW V4] Seasonal risk window"),
    ]
    for feat, typ, vals, desc in feats_b:
        f.write(f"{feat:<28} {typ:<7} {vals:<22} {desc}\n")

    f.write("\nC. TARGET VARIABLES (3)\n")
    f.write("-" * 40 + "\n")
    f.write("claim_probability_true    float  0.01-0.99  Oracle (generation formula output)\n")
    f.write("claim_occurred            int    0/1        Frequency target\n")
    f.write("actual_claim_amount_inr   float  0 or 5000+ Severity target (0 for non-claimants)\n")

    f.write("\n\nCAUSAL GENERATION FORMULAS\n")
    f.write("=" * 40 + "\n\n")

    f.write("FREQUENCY: p = 0.100 (base, calibrated to ~17% mean claim rate)\n")
    f.write("  + age delta (U-shaped: <23=+0.080, <30=+0.055, <40=+0.030, 55-65=-0.010, 65+=+0.035)\n")
    f.write("  + experience:   -0.020 * min(years_licensed/25, 1.0)\n")
    f.write("  + prev_claims:  +0.120 * min(n/3, 1.0)  [strongest signal]\n")
    f.write("  + recency:      999->+0.000, <=6->+0.060, <=12->+0.040, <=24->+0.020, <=36->+0.010\n")
    f.write("  + NCB:          -0.040 * (ncb_pct/50.0)\n")
    f.write("  + tenure:       -0.015 * min(policy_tenure/10, 1.0)\n")
    f.write("  + occupation:   Professional=-0.025, Salaried=0, SE=+0.015, Student=+0.035, Retired=-0.025, Manual=+0.045\n")
    f.write("  + marital:      Single=0, Married=-0.018, Divorced=0, Widowed=-0.010\n")
    f.write("  + income:       <3LPA=+0.010, 3-7=0, 7-15=-0.008, 15-30=-0.008, >30=-0.010\n")
    f.write("  + usage:        Personal=0, Commercial=+0.070, Rideshare=+0.110\n")
    f.write("  + mileage:      +0.020 * (km/20000)\n")
    f.write("  + parking:      Garage=-0.010, Open=+0.010, Street=+0.030\n")
    f.write("  + pincode:      +0.010 * (score-5)/5\n")
    f.write("  + theft:        +0.008 * index/10\n")
    f.write("  + flood:        +0.006 * index/10\n")
    f.write("  + driving:      -0.040 * (score-50)/50\n")
    f.write("  + seasonal:     Jan/Jun-Sep/Dec +0.005-0.015\n\n")

    f.write("SEVERITY: base = vehicle_value_inr * Uniform(0.05, 0.55)\n")
    f.write("  * fuel:    Petrol=1.00, Diesel=1.05, Electric=1.35, Hybrid=1.18\n")
    f.write("  * repair:  Budget=0.92, Standard=1.05, Premium=1.20, Luxury=1.40\n")
    f.write("  * body:    Hatchback=0.95, Sedan=1.00, MUV=1.08, SUV=1.12, Luxury=1.30\n")
    f.write("  * safety:  1 + (5-rating)*0.055  (worse safety => higher cost)\n")
    f.write("  * airbag:  max(0.85, 1-(airbags-2)*0.025)  (more airbags => lower cost)\n")
    f.write("  * flood:   1 + flood_risk_index*0.040\n")
    f.write("  * age:     1 - min(vehicle_age*0.02, 0.30)  (depreciation)\n")
    f.write("  clipped to [5000, vehicle_value*0.95]\n\n")

    f.write("V4 DRIVING SCORE - DECOUPLED FORMULA (V3 FLAW FIXED)\n")
    f.write("  V3 flaw: driving_score = f(claims_last_3yr, usage_type, score) => double-encoding\n")
    f.write("  V4 fix:  score = 70 + min(years_lic*0.5,12.5) - min(prev_claims*10,30)\n")
    f.write("                 + recency_bonus + N(0,5), clipped [0,100]\n")
    f.write("  vehicle_usage_type remains STANDALONE - no longer encoded into score\n\n")

    f.write("REDUNDANT V3 FEATURES REMOVED IN V4 (Type A - mathematical)\n")
    f.write("  normalized_mileage        = annual_mileage_km / 15000  (r=1.0)\n")
    f.write("  vehicle_depreciation_factor = 1/(1+vehicle_age*0.15)   (monotone)\n")
    f.write("  high_value_vehicle         = int(vehicle_value > 1.5M) (threshold)\n\n")

    f.write("V4 vs V3 FEATURE COUNT\n")
    f.write("  V3 raw features:    25  |  V3 preprocessed: 40\n")
    f.write("  V4 raw features:    30  |  V4 preprocessed: ~60\n")
    f.write("  Net new causal features: 12 (listed above with [NEW V4])\n")

print("v4_feature_design_document.txt written")

# ── Phase 5: Validation Report
claimants = df[df["claim_occurred"] == 1]
with open(os.path.join(REPORT_DIR, "v4_validation_report.txt"), "w", encoding="utf-8") as f:
    f.write("V4 DATASET VALIDATION REPORT\n")
    f.write("Insurance AI Pricing System - Controlled Experiment\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("SECTION 1 - DATASET OVERVIEW\n")
    f.write(f"Total rows:    {len(df):,}\n")
    f.write(f"Total cols:    {len(df.columns)}\n")
    f.write(f"Missing vals:  {int(df.isnull().sum().sum())}\n\n")

    f.write("SECTION 2 - TARGET DISTRIBUTION\n")
    n_cl = int(df["claim_occurred"].sum())
    f.write(f"Claims:        {n_cl:,} ({n_cl/len(df)*100:.2f}%)\n")
    f.write(f"Non-claims:    {len(df)-n_cl:,} ({(len(df)-n_cl)/len(df)*100:.2f}%)\n")
    f.write(f"True prob mean:  {df['claim_probability_true'].mean():.4f}\n")
    f.write(f"True prob range: [{df['claim_probability_true'].min():.4f}, {df['claim_probability_true'].max():.4f}]\n")
    f.write(f"Mean claim (claimants):   INR {claimants['actual_claim_amount_inr'].mean():,.0f}\n")
    f.write(f"Median claim (claimants): INR {claimants['actual_claim_amount_inr'].median():,.0f}\n")
    f.write(f"Min claim:     INR {claimants['actual_claim_amount_inr'].min():,.0f}\n")
    f.write(f"Max claim:     INR {claimants['actual_claim_amount_inr'].max():,.0f}\n\n")

    f.write("SECTION 3 - FEATURE DISTRIBUTIONS\n")
    f.write(f"age:           mean={df['age'].mean():.1f}  std={df['age'].std():.1f}  range=[{df['age'].min()},{df['age'].max()}]\n")
    f.write(f"years_lic:     mean={df['years_licensed'].mean():.1f}  max={df['years_licensed'].max()}\n")
    f.write(f"prev_claims:   mean={df['previous_claims_count'].mean():.3f}  max={df['previous_claims_count'].max()}\n")
    never = int((df["months_since_last_claim"] == 999).sum())
    f.write(f"months_since=999 (never claimed): {never:,} ({never/len(df)*100:.1f}%)\n")
    active = df.loc[df["months_since_last_claim"] != 999, "months_since_last_claim"]
    f.write(f"months_since (claimants): mean={active.mean():.1f}  range=[{active.min()},{active.max()}]\n")
    f.write(f"NCB pct values:  {sorted(df['no_claim_bonus_pct'].unique().tolist())}\n")
    f.write(f"policy_tenure:   mean={df['policy_tenure_years'].mean():.1f}yr  range=[{df['policy_tenure_years'].min():.1f},{df['policy_tenure_years'].max():.1f}]\n")
    f.write(f"driving_score:   mean={df['driving_score'].mean():.1f}  range=[{df['driving_score'].min():.1f},{df['driving_score'].max():.1f}]\n")
    f.write(f"vehicle_value:   mean=INR {df['vehicle_value_inr'].mean():,.0f}  range=[{df['vehicle_value_inr'].min():,},{df['vehicle_value_inr'].max():,}]\n")
    f.write(f"pincode_risk:    mean={df['pincode_risk_score'].mean():.2f}\n")
    f.write(f"flood_risk:      mean={df['flood_risk_index'].mean():.2f}\n")
    f.write(f"theft_risk:      mean={df['theft_risk_index'].mean():.2f}\n\n")

    f.write("SECTION 4 - CATEGORICAL DISTRIBUTIONS\n")
    for col in ["gender", "occupation", "marital_status", "annual_income_band",
                "fuel_type", "vehicle_body_style", "repair_cost_band",
                "parking_type", "vehicle_usage_type", "region_risk_category"]:
        vc = df[col].value_counts()
        f.write(f"\n{col}:\n")
        for val, cnt in vc.items():
            f.write(f"  {str(val):<25} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)\n")

    f.write("\n\nSECTION 5 - V4 vs V2 FAILURE MODE COMPARISON\n")
    f.write("V2 failure: all 6 features were mathematical derivations of V1 columns.\n")
    f.write("            AUC regressed 0.6645 -> 0.6603. Bayes ceiling unchanged.\n\n")
    f.write("V4 new features and independent causal mechanisms:\n")
    new_feats = [
        ("months_since_last_claim", "Temporal recency NOT derivable from claim count alone"),
        ("no_claim_bonus_pct",      "Policy NCB tier -- accumulated loyalty + risk signal"),
        ("policy_tenure_years",     "Insurer tenure -- not encoded in any V3 column"),
        ("fuel_type",               "EV repair 35% higher -- genuine new causal path to severity"),
        ("repair_cost_band",        "Model-level repair cost -- exploits previously ignored car_model"),
        ("vehicle_body_style",      "Body style repair complexity -- new severity signal"),
        ("parking_type",            "Theft/damage exposure -- absent from V3"),
        ("policy_inception_month",  "Seasonal risk window -- monsoon/fog exposure"),
        ("pincode_risk_score",      "Sub-city geographic risk -- finer than city_tier"),
        ("flood_risk_index",        "Flood vulnerability -- new causal path (water damage)"),
        ("theft_risk_index",        "Theft rate -- new causal path (theft claims)"),
        ("monsoon_exposure_index",  "Weather severity -- seasonal severity multiplier"),
    ]
    for feat, mech in new_feats:
        f.write(f"  {feat:<30}  {mech}\n")

    f.write("\n\nSECTION 6 - DATA QUALITY CHECKS\n")
    checks = [
        ("No missing values", int(df.isnull().sum().sum()) == 0),
        ("Claim rate 10-25%", 0.10 <= float(df["claim_occurred"].mean()) <= 0.25),
        ("Age 18-75", bool(df["age"].between(18, 75).all())),
        ("Driving score 0-100", bool(df["driving_score"].between(0, 100).all())),
        ("NCB in {0,20,25,35,45,50}", bool(df["no_claim_bonus_pct"].isin([0, 20, 25, 35, 45, 50]).all())),
        ("Vehicle value > 0", bool((df["vehicle_value_inr"] > 0).all())),
        ("Claim amount >= 4999 for claimants",
         bool((df.loc[df["claim_occurred"] == 1, "actual_claim_amount_inr"] >= 4999).all())),
        ("True prob in [0.01, 0.99]", bool(df["claim_probability_true"].between(0.01, 0.99).all())),
        ("N rows = 50,000", len(df) == 50_000),
        ("35 columns present", len(df.columns) == 35),
    ]
    for name, passed in checks:
        f.write(f"  [{'PASS' if passed else 'FAIL'}] {name}\n")
    all_pass = all(p for _, p in checks)
    f.write(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}\n")

print("v4_validation_report.txt written")
print("Done.")
