"""
Tests for lookup_loader.py

Covers:
  - Known vehicle (exact match)
  - Known city (exact match)
  - Unknown vehicle make
  - Unknown vehicle model
  - Unknown city
  - None / empty inputs
  - validate_lookup_tables integrity
  - Default value correctness
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.lookup_loader import (
    get_vehicle_specs,
    get_city_risk,
    validate_lookup_tables,
    clear_cache,
    VEHICLE_DEFAULTS,
    CITY_DEFAULTS,
)

PASS = "PASS"
FAIL = "FAIL"
_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = "  [{}] {}".format(status, label)
    if detail:
        msg += "  ({})".format(detail)
    print(msg)
    if not condition:
        _failures.append(label)


# ===========================================================================
# SECTION 1 — Known vehicles (exact match from V4 dataset)
# ===========================================================================
print()
print("=" * 60)
print("SECTION 1 — Known Vehicle Lookups")
print("=" * 60)

known_vehicles = [
    ("Maruti",   "Alto",      3, 2,  "Hatchback", "Budget"),
    ("Maruti",   "Swift",     4, 2,  "Hatchback", "Budget"),
    ("Maruti",   "Baleno",    4, 6,  "Hatchback", "Budget"),
    ("Maruti",   "Dzire",     4, 6,  "Sedan",     "Budget"),
    ("Maruti",   "Ertiga",    4, 6,  "MUV",       "Budget"),
    ("Hyundai",  "Creta",     5, 6,  "SUV",       "Standard"),
    ("Hyundai",  "Verna",     5, 6,  "Sedan",     "Standard"),
    ("Hyundai",  "i20",       4, 6,  "Hatchback", "Standard"),
    ("Hyundai",  "Kona EV",   5, 6,  "SUV",       "Premium"),
    ("Tata",     "Nexon",     5, 6,  "SUV",       "Standard"),
    ("Tata",     "Nexon EV",  5, 6,  "SUV",       "Premium"),
    ("Tata",     "Tiago",     5, 6,  "Hatchback", "Budget"),
    ("Tata",     "Tiago EV",  5, 6,  "Hatchback", "Standard"),
    ("Honda",    "City",      5, 6,  "Sedan",     "Standard"),
    ("Honda",    "Accord",    5, 8,  "Sedan",     "Premium"),
    ("Honda",    "BR-V",      4, 6,  "MUV",       "Standard"),
    ("Toyota",   "Fortuner",  5, 7,  "SUV",       "Premium"),
    ("Toyota",   "Innova",    5, 7,  "MUV",       "Standard"),
    ("Toyota",   "Camry",     5, 8,  "Sedan",     "Premium"),
    ("Kia",      "Seltos",    5, 6,  "SUV",       "Standard"),
    ("Kia",      "Carens",    5, 6,  "MUV",       "Standard"),
    ("Mahindra", "XUV700",    5, 7,  "SUV",       "Premium"),
    ("Mahindra", "Scorpio",   4, 6,  "SUV",       "Standard"),
    ("BMW",      "3 Series",  5, 8,  "Sedan",     "Luxury"),
    ("BMW",      "X5",        5, 8,  "SUV",       "Luxury"),
    ("Audi",     "A4",        5, 8,  "Sedan",     "Luxury"),
    ("Mercedes", "C-Class",   5, 8,  "Sedan",     "Luxury"),
    ("Mercedes", "GLE",       5, 8,  "SUV",       "Luxury"),
    ("Porsche",  "Cayenne",   5, 8,  "SUV",       "Luxury"),
    ("Renault",  "Kwid",      3, 2,  "Hatchback", "Budget"),
    ("Skoda",    "Slavia",    5, 6,  "Sedan",     "Premium"),
    ("Jeep",     "Compass",   5, 6,  "SUV",       "Premium"),
    ("MG",       "ZS EV",     5, 6,  "SUV",       "Premium"),
    ("BYD",      "Atto 3",    5, 6,  "SUV",       "Premium"),
    ("Ford",     "Endeavour", 5, 7,  "SUV",       "Premium"),
]

for make, model, exp_rating, exp_airbags, exp_body, exp_repair in known_vehicles:
    result = get_vehicle_specs(make, model)
    check(
        f"{make} {model}: rating={exp_rating} airbags={exp_airbags} body={exp_body} repair={exp_repair}",
        result["vehicle_safety_rating"] == exp_rating
        and result["airbags_count"] == exp_airbags
        and result["vehicle_body_style"] == exp_body
        and result["repair_cost_band"] == exp_repair,
        repr(result)
    )

# ===========================================================================
# SECTION 2 — Known cities (exact match from V4 dataset)
# ===========================================================================
print()
print("=" * 60)
print("SECTION 2 — Known City Lookups")
print("=" * 60)

known_cities = [
    ("Mumbai",        "High",   8.5, 6.0,  9.0),
    ("Delhi",         "High",   5.0, 8.5,  6.0),
    ("Bangalore",     "High",   4.5, 5.5,  7.0),
    ("Chennai",       "High",   7.0, 4.5,  8.5),
    ("Kolkata",       "High",   8.0, 6.5,  8.5),
    ("Hyderabad",     "High",   5.5, 5.0,  6.5),
    ("Pune",          "High",   5.0, 5.0,  7.0),
    ("Ahmedabad",     "High",   4.0, 5.5,  5.0),
    ("Jaipur",        "Medium", 3.0, 5.0,  3.5),
    ("Agra",          "Medium", 4.0, 5.0,  5.0),
    ("Bhopal",        "Medium", 4.0, 4.0,  5.5),
    ("Coimbatore",    "Medium", 3.5, 3.5,  6.0),
    ("Indore",        "Medium", 3.5, 4.0,  5.5),
    ("Kanpur",        "Medium", 5.0, 5.0,  5.5),
    ("Lucknow",       "Medium", 5.5, 5.5,  6.0),
    ("Nagpur",        "Medium", 4.5, 4.5,  6.5),
    ("Patna",         "Medium", 7.5, 5.5,  7.5),
    ("Surat",         "Medium", 6.0, 5.0,  6.0),
    ("Vadodara",      "Medium", 4.5, 4.5,  5.0),
    ("Visakhapatnam", "Medium", 6.5, 4.0,  7.5),
]

for city, exp_cat, exp_flood, exp_theft, exp_monsoon in known_cities:
    result = get_city_risk(city)
    check(
        f"{city}: cat={exp_cat} flood={exp_flood} theft={exp_theft} monsoon={exp_monsoon}",
        result["region_risk_category"] == exp_cat
        and result["flood_risk_index"] == exp_flood
        and result["theft_risk_index"] == exp_theft
        and result["monsoon_exposure_index"] == exp_monsoon,
        repr(result)
    )

# ===========================================================================
# SECTION 3 — Unknown vehicle inputs
# ===========================================================================
print()
print("=" * 60)
print("SECTION 3 — Unknown Vehicle Inputs")
print("=" * 60)

unknown_make = get_vehicle_specs("Lamborghini", "Urus")
check("Unknown make returns VEHICLE_DEFAULTS",
      unknown_make == VEHICLE_DEFAULTS,
      repr(unknown_make))

unknown_model = get_vehicle_specs("Maruti", "Baleno RS")
check("Unknown model (known make) returns VEHICLE_DEFAULTS",
      unknown_model == VEHICLE_DEFAULTS,
      repr(unknown_model))

nonexistent_both = get_vehicle_specs("Tesla", "Model S")
check("Tesla Model S (not in dataset) returns VEHICLE_DEFAULTS",
      nonexistent_both == VEHICLE_DEFAULTS,
      repr(nonexistent_both))

# ===========================================================================
# SECTION 4 — Unknown city inputs
# ===========================================================================
print()
print("=" * 60)
print("SECTION 4 — Unknown City Inputs")
print("=" * 60)

unknown_city = get_city_risk("Atlantis")
check("Unknown city returns CITY_DEFAULTS",
      unknown_city == CITY_DEFAULTS,
      repr(unknown_city))

unrecognised_city = get_city_risk("XYZ City")
check("Nonsense city returns CITY_DEFAULTS",
      unrecognised_city == CITY_DEFAULTS,
      repr(unrecognised_city))

# ===========================================================================
# SECTION 5 — None / empty / missing inputs
# ===========================================================================
print()
print("=" * 60)
print("SECTION 5 — None and Empty Inputs")
print("=" * 60)

none_make = get_vehicle_specs(None, "Creta")
check("None make returns VEHICLE_DEFAULTS",
      none_make == VEHICLE_DEFAULTS,
      repr(none_make))

none_model = get_vehicle_specs("Hyundai", None)
check("None model returns VEHICLE_DEFAULTS",
      none_model == VEHICLE_DEFAULTS,
      repr(none_model))

none_both = get_vehicle_specs(None, None)
check("None make+model returns VEHICLE_DEFAULTS",
      none_both == VEHICLE_DEFAULTS,
      repr(none_both))

empty_make = get_vehicle_specs("", "Creta")
check("Empty string make returns VEHICLE_DEFAULTS",
      empty_make == VEHICLE_DEFAULTS,
      repr(empty_make))

empty_model = get_vehicle_specs("Hyundai", "")
check("Empty string model returns VEHICLE_DEFAULTS",
      empty_model == VEHICLE_DEFAULTS,
      repr(empty_model))

none_city = get_city_risk(None)
check("None city returns CITY_DEFAULTS",
      none_city == CITY_DEFAULTS,
      repr(none_city))

empty_city = get_city_risk("")
check("Empty string city returns CITY_DEFAULTS",
      empty_city == CITY_DEFAULTS,
      repr(empty_city))

# ===========================================================================
# SECTION 6 — validate_lookup_tables integrity
# ===========================================================================
print()
print("=" * 60)
print("SECTION 6 — validate_lookup_tables")
print("=" * 60)

clear_cache()
report = validate_lookup_tables()

check("vehicle_specs_count == 35",
      report["vehicle_specs_count"] == 35,
      f"actual={report['vehicle_specs_count']}")

check("city_risk_count == 20",
      report["city_risk_count"] == 20,
      f"actual={report['city_risk_count']}")

check("vehicle_specs_valid == True",
      report["vehicle_specs_valid"] is True,
      repr(report.get("errors")))

check("city_risk_valid == True",
      report["city_risk_valid"] is True,
      repr(report.get("errors")))

check("No errors",
      len(report["errors"]) == 0,
      repr(report["errors"]))

print()
print("  vehicle_specs_path :", report["vehicle_specs_path"])
print("  city_risk_path     :", report["city_risk_path"])
print("  vehicle_count      :", report["vehicle_specs_count"])
print("  city_count         :", report["city_risk_count"])
print("  errors             :", report["errors"])

# ===========================================================================
# SECTION 7 — Return shape correctness
# ===========================================================================
print()
print("=" * 60)
print("SECTION 7 — Return Schema Correctness")
print("=" * 60)

vehicle_keys = {"vehicle_safety_rating", "airbags_count", "vehicle_body_style", "repair_cost_band"}
city_keys = {"region_risk_category", "flood_risk_index", "theft_risk_index", "monsoon_exposure_index"}

sample_v = get_vehicle_specs("Toyota", "Fortuner")
check("Vehicle result has exactly 4 expected keys",
      set(sample_v.keys()) == vehicle_keys,
      str(set(sample_v.keys())))

sample_c = get_city_risk("Mumbai")
check("City result has exactly 4 expected keys",
      set(sample_c.keys()) == city_keys,
      str(set(sample_c.keys())))

default_v = get_vehicle_specs(None, None)
check("VEHICLE_DEFAULTS has exactly 4 expected keys",
      set(default_v.keys()) == vehicle_keys,
      str(set(default_v.keys())))

default_c = get_city_risk(None)
check("CITY_DEFAULTS has exactly 4 expected keys",
      set(default_c.keys()) == city_keys,
      str(set(default_c.keys())))

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print()
print("=" * 60)
total = (len(known_vehicles) + len(known_cities)
         + 3 + 2 + 7 + 5 + 4)
passed = total - len(_failures)
print(f"RESULT: {passed}/{total} tests PASSED")
if _failures:
    print("FAILED tests:")
    for f in _failures:
        print(f"  {FAIL} {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
