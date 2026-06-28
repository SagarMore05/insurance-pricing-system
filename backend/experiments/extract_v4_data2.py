import pandas as pd
import json

df = pd.read_csv('experiments/outputs/data/car_insurance_portfolio_v4.csv')

print("=== MAKE+MODEL COMBOS ===")
combos = df.groupby(['vehicle_make', 'vehicle_model']).size().reset_index(name='count')
print("Total:", len(combos))
print(combos[['vehicle_make', 'vehicle_model']].to_string(index=False))

print()
print("=== VEHICLE SPECS (unique per make+model) ===")
spec_cols = ['vehicle_make', 'vehicle_model', 'vehicle_safety_rating', 'airbags_count',
             'vehicle_body_style', 'repair_cost_band']
specs = df[spec_cols].drop_duplicates(['vehicle_make', 'vehicle_model']).sort_values(['vehicle_make', 'vehicle_model'])
for _, row in specs.iterrows():
    print("  {} | {} | rating={} airbags={} body={} repair={}".format(
        row['vehicle_make'], row['vehicle_model'],
        row['vehicle_safety_rating'], int(row['airbags_count']),
        row['vehicle_body_style'], row['repair_cost_band']))

print()
print("=== CITY RISK DATA (unique per city) ===")
city_cols = ['city', 'region_risk_category', 'flood_risk_index', 'theft_risk_index',
             'monsoon_exposure_index']
city_data = df[city_cols].drop_duplicates('city').sort_values('city')
for _, row in city_data.iterrows():
    print("  {:20} | {:10} | flood={} theft={} monsoon={}".format(
        row['city'], row['region_risk_category'],
        row['flood_risk_index'], row['theft_risk_index'],
        row['monsoon_exposure_index']))

print()
print("=== DISTINCT VALUES ===")
print("body_style:", sorted(df['vehicle_body_style'].dropna().unique().tolist()))
print("repair_band:", sorted(df['repair_cost_band'].dropna().unique().tolist()))
print("region_risk:", sorted(df['region_risk_category'].dropna().unique().tolist()))
print("safety_rating:", sorted(df['vehicle_safety_rating'].dropna().unique().tolist()))
print("airbags:", sorted(df['airbags_count'].dropna().astype(int).unique().tolist()))
print("flood_idx:", sorted(df['flood_risk_index'].dropna().unique().tolist()))
print("theft_idx:", sorted(df['theft_risk_index'].dropna().unique().tolist()))
print("monsoon_idx:", sorted(df['monsoon_exposure_index'].dropna().unique().tolist()))
