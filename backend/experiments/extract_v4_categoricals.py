import pandas as pd

df = pd.read_csv('experiments/outputs/data/car_insurance_portfolio_v4.csv')

cat_cols = ['gender', 'fuel_type', 'vehicle_usage_type', 'occupation',
            'marital_status', 'annual_income_band', 'parking_type',
            'region_risk_category']

for col in cat_cols:
    if col in df.columns:
        vals = sorted(df[col].dropna().unique().tolist())
        print("{}: {}".format(col, vals))

print()
print("months_since_last_claim: min={} max={} unique_sample={}".format(
    df['months_since_last_claim'].min(),
    df['months_since_last_claim'].max(),
    sorted(df['months_since_last_claim'].dropna().unique().tolist())[:20]))

print("no_claim_bonus_pct: min={} max={} unique={}".format(
    df['no_claim_bonus_pct'].min(),
    df['no_claim_bonus_pct'].max(),
    sorted(df['no_claim_bonus_pct'].dropna().unique().tolist())))

print("policy_tenure_years: min={} max={} unique={}".format(
    df['policy_tenure_years'].min(),
    df['policy_tenure_years'].max(),
    sorted(df['policy_tenure_years'].dropna().unique().tolist())))

print("policy_inception_month: unique={}".format(
    sorted(df['policy_inception_month'].dropna().unique().tolist())))

print("vehicle_make: {}".format(sorted(df['vehicle_make'].dropna().unique().tolist())))
print("vehicle_model sample: {}".format(sorted(df['vehicle_model'].dropna().unique().tolist())[:10]))
