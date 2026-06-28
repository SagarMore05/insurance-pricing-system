import sys
sys.path.insert(0, ".")
from src.services.lookup_loader import (
    get_vehicle_specs, get_city_risk, validate_lookup_tables, _vehicle_key, _city_key
)

result = validate_lookup_tables()
print("=== Validation ===")
print(f"  vehicle_specs_count : {result['vehicle_specs_count']}")
print(f"  city_risk_count     : {result['city_risk_count']}")
print(f"  vehicle_specs_valid : {result['vehicle_specs_valid']}")
print(f"  city_risk_valid     : {result['city_risk_valid']}")
print(f"  errors              : {result['errors']}")
assert result["vehicle_specs_valid"], "vehicle_specs FAILED"
assert result["city_risk_valid"], "city_geo_risk FAILED"
assert not result["errors"], f"Errors: {result['errors']}"

print()
print("=== Vehicle Lookups ===")
cases = [
    ("Maruti",    "Swift",     2.0, 2,  "Hatchback", "Low"),
    ("TATA",      "Nexon",     5.0, 6,  "SUV",       "Medium"),
    ("Hyundai",   "Creta",     3.0, 6,  "SUV",       "Medium"),
    ("BMW",       "3 Series",  5.0, 8,  "Sedan",     "Very High"),
    ("Honda",     "City",      5.0, 6,  "Sedan",     "Medium"),
    ("Mahindra",  "XUV700",    5.0, 7,  "SUV",       "High"),
]
for brand, model, exp_r, exp_a, exp_b, exp_c in cases:
    r = get_vehicle_specs(brand, model)
    print(f"  {brand} {model}: rating={r['vehicle_safety_rating']} airbags={r['airbags_count']} "
          f"body={r['vehicle_body_style']} repair={r['repair_cost_band']}")
    assert r["vehicle_safety_rating"] == exp_r, f"rating mismatch for {brand} {model}"
    assert r["airbags_count"] == exp_a, f"airbags mismatch for {brand} {model}"
    assert r["vehicle_body_style"] == exp_b, f"body style mismatch for {brand} {model}"
    assert r["repair_cost_band"] == exp_c, f"repair band mismatch for {brand} {model}"

print()
print("=== Unknown Vehicle Fallback ===")
r = get_vehicle_specs("Unknown", "ModelX", vehicle_value_inr=2500000)
print(f"  Unknown ModelX (value=25L): rating={r['vehicle_safety_rating']} airbags={r['airbags_count']} repair={r['repair_cost_band']}")
assert r["repair_cost_band"] == "High", f"Expected High for 25L vehicle, got {r['repair_cost_band']}"
assert r["vehicle_safety_rating"] == 3.0
assert r["airbags_count"] == 2

print()
print("=== City Lookups ===")
city_cases = [
    ("Mumbai",      "Very High", 0.85, 0.65),
    ("Delhi",       "High",      0.45, 0.85),
    ("Bangalore",   "High",      0.55, 0.50),
    ("Bengaluru",   "High",      0.55, 0.50),  # alias
    ("Bombay",      "Very High", 0.85, 0.65),  # alias -> Mumbai
    ("Gurgaon",     "High",      0.40, 0.65),  # alias -> Gurugram
    ("Jaipur",      "Medium",    0.15, 0.55),
    ("Chennai",     "High",      0.80, 0.40),
]
for city, exp_cat, exp_flood, exp_theft in city_cases:
    r = get_city_risk(city)
    print(f"  {city}: cat={r['region_risk_category']} flood={r['flood_risk_index']} "
          f"theft={r['theft_risk_index']} pincode_risk={r['pincode_risk_score']}")
    assert r["region_risk_category"] == exp_cat, f"cat mismatch for {city}: {r['region_risk_category']} != {exp_cat}"
    assert abs(r["flood_risk_index"] - exp_flood) < 0.001, f"flood mismatch for {city}"
    assert abs(r["theft_risk_index"] - exp_theft) < 0.001, f"theft mismatch for {city}"
    ps = r["pincode_risk_score"]
    assert 0.0 <= ps <= 1.0, f"pincode_risk_score out of range for {city}: {ps}"

print()
print("=== Unknown City Fallback ===")
r = get_city_risk("Atlantis")
print(f"  Atlantis: cat={r['region_risk_category']} pincode_risk={r['pincode_risk_score']}")
assert r["region_risk_category"] == "Medium"
assert r["pincode_risk_score"] == 0.325  # 0.35*0.30 + 0.40*0.30 + 0.25*0.40

print()
print("=== Key Normalisation ===")
assert _vehicle_key("Maruti", "Alto K10") == "maruti_alto_k10"
assert _vehicle_key("maruti", "swift dzire") == "maruti_swift_dzire"
assert _city_key("Bengaluru") == "bangalore"
assert _city_key("New Delhi") == "delhi"
assert _city_key("Bombay") == "mumbai"
print("  All key normalisation checks PASSED")

print()
print("=== Pincode Risk Score Computation ===")
from src.services.lookup_loader import _pincode_risk_score
score = _pincode_risk_score(0.85, 0.65, 0.95)
print(f"  Mumbai composite: 0.35*0.85 + 0.40*0.65 + 0.25*0.95 = {score}")
expected = round(0.35*0.85 + 0.40*0.65 + 0.25*0.95, 4)
assert abs(score - expected) < 0.0001

print()
print("="*50)
print("ALL VALIDATION TESTS PASSED")
