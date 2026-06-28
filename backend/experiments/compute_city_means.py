import pandas as pd
import json

df = pd.read_csv('experiments/outputs/data/car_insurance_portfolio_v4.csv')

print("=== CITY MEANS (for JSON anchor values) ===")
city_means = df.groupby('city').agg(
    region_risk_category=('region_risk_category', 'first'),
    flood_risk_index=('flood_risk_index', 'mean'),
    theft_risk_index=('theft_risk_index', 'mean'),
    monsoon_exposure_index=('monsoon_exposure_index', 'mean'),
    count=('city', 'count')
).reset_index().sort_values('city')

for _, row in city_means.iterrows():
    cat = row['region_risk_category']
    print("  {:20} | cat={} | flood={:.2f} theft={:.2f} monsoon={:.2f} | n={}".format(
        row['city'], int(cat),
        row['flood_risk_index'], row['theft_risk_index'], row['monsoon_exposure_index'],
        int(row['count'])))

print()
print("=== REGION RISK CATEGORY MAPPING ===")
cat_data = df.groupby(['city', 'region_risk_category']).size().reset_index(name='count')
cat_by_city = {}
for _, row in cat_data.iterrows():
    cat_by_city[row['city']] = row['region_risk_category']
    print("  {:20} -> {}".format(row['city'], row['region_risk_category']))
