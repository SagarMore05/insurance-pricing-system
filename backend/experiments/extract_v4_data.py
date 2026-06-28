import pandas as pd

df = pd.read_csv('experiments/outputs/data/car_insurance_portfolio_v4.csv')

print("=== ALL MAKE+MODEL COMBINATIONS ===")
combos = df.groupby(['vehicle_make', 'vehicle_model']).size().reset_index(name='count')
combos = combos.sort_values(['vehicle_make', 'vehicle_model'])
for _, row in combos.iterrows():
    print("  {:12} | {:25} | {:5} records".format(
        row['vehicle_make'], row['vehicle_model'], row['count']))

print()
print("Total unique make+model combos: {}".format(len(combos)))
print()

print("=== VEHICLE SPECS PER MAKE+MODEL (from dataset) ===")
spec_cols = ['vehicle_make', 'vehicle_model', 'vehicle_safety_rating', 'airbags_count',
             'vehicle_body_style', 'repair_cost_band']
specs = df[spec_cols].drop_duplicates().sort_values(['vehicle_make', 'vehicle_model'])
for _, row in specs.iterrows():
    print("  {:12} | {:25} | rating={} airbags={} body={} repair={}".format(
        row['vehicle_make'], row['vehicle_model'],
        row['vehicle_safety_rating'], int(row['airbags_count']),
        row['vehicle_body_style'], row['repair_cost_band']))

print()
print("=== CITY RISK STATS ===")
city_cols = ['city', 'region_risk_category', 'flood_risk_index', 'theft_risk_index',
             'monsoon_exposure_index', 'pincode_risk_score']
city_data = df[city_cols].drop_duplicates().sort_values('city')
for _, row in city_data.iterrows():
    print("  {:20} | {} | flood={} theft={} monsoon={} pincode={}".format(
        row['city'], row['region_risk_category'],
        row['flood_risk_index'], row['theft_risk_index'],
        row['monsoon_exposure_index'], row['pincode_risk_score']))

print()
print("Total unique cities: {}".format(len(city_data)))
print()

print("=== VEHICLE_BODY_STYLE values ===")
print(sorted(df['vehicle_body_style'].dropna().unique().tolist()))

print("=== REPAIR_COST_BAND values ===")
print(sorted(df['repair_cost_band'].dropna().unique().tolist()))

print("=== REGION_RISK_CATEGORY values ===")
print(sorted(df['region_risk_category'].dropna().unique().tolist()))

print("=== SAFETY RATING range ===")
print("  Unique:", sorted(df['vehicle_safety_rating'].dropna().unique().tolist()))

print("=== AIRBAGS range ===")
print("  Unique:", sorted(df['airbags_count'].dropna().unique().tolist()))

print("=== FLOOD/THEFT/MONSOON unique values ===")
for col in ['flood_risk_index', 'theft_risk_index', 'monsoon_exposure_index']:
    print("  {}: {}".format(col, sorted(df[col].dropna().unique().tolist())))
