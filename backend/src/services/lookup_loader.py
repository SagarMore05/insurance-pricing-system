"""
lookup_loader.py
================
Cached, fault-tolerant lookup service for V4 champion model enrichment.

Two static JSON tables (loaded once per process, cached indefinitely):
  data/lookups/vehicle_specs.json   — vehicle_safety_rating, airbags_count,
                                       vehicle_body_style, repair_cost_band
  data/lookups/city_geo_risk.json   — region_risk_category, flood_risk_index,
                                       theft_risk_index, monsoon_exposure_index

JSON structure
--------------
  vehicle_specs  : {Vehicle_Make: {Vehicle_Model: {spec_fields}}}
  city_geo_risk  : {City: {risk_fields}}

Public API
----------
  get_vehicle_specs(make, model)  -> dict
  get_city_risk(city)             -> dict
  validate_lookup_tables()        -> dict
  clear_cache()                   -> None

Both lookup functions:
  - Never raise.
  - Return documented default values for unknown / null inputs.
  - Are thread-safe (GIL + lru_cache).

V4 Preprocessor Column Mapping
-------------------------------
  get_vehicle_specs() → InsurancePreprocessorV4:
    vehicle_safety_rating  -> STD_COLS (StandardScaled)
    airbags_count          -> PASS_COLS (passthrough)
    vehicle_body_style     -> OHE_COLS (OneHotEncoded)
    repair_cost_band       -> OHE_COLS (OneHotEncoded)

  get_city_risk() → InsurancePreprocessorV4:
    region_risk_category   -> categorical label
    flood_risk_index       -> STD_COLS (StandardScaled)
    theft_risk_index       -> STD_COLS (StandardScaled)
    monsoon_exposure_index -> STD_COLS (StandardScaled)

Risk index scale: 1.0 (lowest risk) to 10.0 (highest risk).
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_LOOKUPS_DIR = os.path.join(_BACKEND_ROOT, "data", "lookups")

VEHICLE_SPECS_PATH = os.path.join(_LOOKUPS_DIR, "vehicle_specs.json")
CITY_RISK_PATH = os.path.join(_LOOKUPS_DIR, "city_geo_risk.json")

# ---------------------------------------------------------------------------
# Defaults — returned when key not found or on any error
# ---------------------------------------------------------------------------

VEHICLE_DEFAULTS: Dict[str, Any] = {
    "vehicle_safety_rating": 4,
    "airbags_count": 6,
    "vehicle_body_style": "Hatchback",
    "repair_cost_band": "Standard",
}

CITY_DEFAULTS: Dict[str, Any] = {
    "region_risk_category": "Medium",
    "flood_risk_index": 5.0,
    "theft_risk_index": 5.0,
    "monsoon_exposure_index": 5.5,
}

# Valid values for integrity checks
_VALID_BODY_STYLES = {"Hatchback", "Sedan", "SUV", "MUV"}
_VALID_REPAIR_BANDS = {"Budget", "Standard", "Premium", "Luxury"}
_VALID_RISK_CATEGORIES = {"High", "Medium"}

# ---------------------------------------------------------------------------
# Internal JSON loaders (cached)
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("_metadata", None)
    return data


@lru_cache(maxsize=1)
def _load_vehicle_specs() -> Dict[str, Any]:
    """Load vehicle_specs.json. Cached after first call."""
    if not os.path.exists(VEHICLE_SPECS_PATH):
        logger.warning("vehicle_specs.json not found at %s", VEHICLE_SPECS_PATH)
        return {}
    data = _load_json(VEHICLE_SPECS_PATH)
    logger.info("vehicle_specs.json loaded: %d makes", len(data))
    return data


@lru_cache(maxsize=1)
def _load_city_risk() -> Dict[str, Any]:
    """Load city_geo_risk.json. Cached after first call."""
    if not os.path.exists(CITY_RISK_PATH):
        logger.warning("city_geo_risk.json not found at %s", CITY_RISK_PATH)
        return {}
    data = _load_json(CITY_RISK_PATH)
    logger.info("city_geo_risk.json loaded: %d cities", len(data))
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vehicle_specs(make: Optional[str], model: Optional[str]) -> Dict[str, Any]:
    """
    Return vehicle specification features for the given make+model.

    Lookup key: specs[make][model]

    Args:
        make  : Vehicle manufacturer (e.g. "Hyundai"). Exact match.
        model : Vehicle model (e.g. "Creta"). Exact match.

    Returns:
        dict with keys: vehicle_safety_rating (int), airbags_count (int),
                        vehicle_body_style (str), repair_cost_band (str)

    Never raises. Returns VEHICLE_DEFAULTS for unknown or null inputs.
    """
    if not make or not model:
        logger.warning("get_vehicle_specs called with null/empty make=%r model=%r", make, model)
        return dict(VEHICLE_DEFAULTS)

    try:
        table = _load_vehicle_specs()

        # Direct match
        make_data = table.get(make)
        if make_data is None:
            # Try stripped variant
            make_data = table.get(make.strip())
        if make_data is None:
            logger.warning("Unknown vehicle make: %r — returning defaults", make)
            return dict(VEHICLE_DEFAULTS)

        model_data = make_data.get(model)
        if model_data is None:
            model_data = make_data.get(model.strip())
        if model_data is None:
            logger.warning("Unknown vehicle model: %r %r — returning defaults", make, model)
            return dict(VEHICLE_DEFAULTS)

        return {
            "vehicle_safety_rating": int(model_data["vehicle_safety_rating"]),
            "airbags_count":         int(model_data["airbags_count"]),
            "vehicle_body_style":    str(model_data["vehicle_body_style"]),
            "repair_cost_band":      str(model_data["repair_cost_band"]),
        }
    except Exception as exc:
        logger.error("get_vehicle_specs(%r, %r) error: %s", make, model, exc)
        return dict(VEHICLE_DEFAULTS)


def get_city_risk(city: Optional[str]) -> Dict[str, Any]:
    """
    Return geographic risk features for the given city.

    Lookup key: risks[city] (exact match, then stripped match).

    Args:
        city : City name (e.g. "Mumbai"). Matches the V4 dataset city names.

    Returns:
        dict with keys: region_risk_category (str), flood_risk_index (float),
                        theft_risk_index (float), monsoon_exposure_index (float)
        Risk indices are on a 1.0-10.0 scale.

    Never raises. Returns CITY_DEFAULTS for unknown or null inputs.
    """
    if not city:
        logger.warning("get_city_risk called with null/empty city=%r", city)
        return dict(CITY_DEFAULTS)

    try:
        table = _load_city_risk()

        city_data = table.get(city)
        if city_data is None:
            city_data = table.get(city.strip())
        if city_data is None:
            # Try title-case as fallback (e.g. "mumbai" -> "Mumbai")
            city_data = table.get(city.strip().title())
        if city_data is None:
            logger.warning("Unknown city: %r — returning defaults", city)
            return dict(CITY_DEFAULTS)

        return {
            "region_risk_category":  str(city_data["region_risk_category"]),
            "flood_risk_index":       float(city_data["flood_risk_index"]),
            "theft_risk_index":       float(city_data["theft_risk_index"]),
            "monsoon_exposure_index": float(city_data["monsoon_exposure_index"]),
        }
    except Exception as exc:
        logger.error("get_city_risk(%r) error: %s", city, exc)
        return dict(CITY_DEFAULTS)


def validate_lookup_tables() -> Dict[str, Any]:
    """
    Run integrity checks on both lookup tables.

    Returns:
        dict:
          vehicle_specs_path   str
          city_risk_path       str
          vehicle_specs_count  int   total make+model combinations
          city_risk_count      int   total city records
          vehicle_specs_valid  bool
          city_risk_valid      bool
          errors               list[str]
    """
    errors: List[str] = []

    # ---- vehicle specs ----
    vehicle_count = 0
    vehicle_valid = False
    try:
        raw = _load_json(VEHICLE_SPECS_PATH)
        seen_combos: set = set()
        for make, models in raw.items():
            if not isinstance(models, dict):
                errors.append(f"vehicle_specs: make '{make}' value is not a dict")
                continue
            for model, specs in models.items():
                combo = (make, model)
                if combo in seen_combos:
                    errors.append(f"vehicle_specs: duplicate ({make}, {model})")
                seen_combos.add(combo)
                vehicle_count += 1
                for field in ["vehicle_safety_rating", "airbags_count",
                              "vehicle_body_style", "repair_cost_band"]:
                    if specs.get(field) is None:
                        errors.append(f"vehicle_specs: null '{field}' for ({make}, {model})")
                if specs.get("vehicle_body_style") not in _VALID_BODY_STYLES:
                    errors.append(
                        f"vehicle_specs: invalid body_style "
                        f"'{specs.get('vehicle_body_style')}' for ({make}, {model})"
                    )
                if specs.get("repair_cost_band") not in _VALID_REPAIR_BANDS:
                    errors.append(
                        f"vehicle_specs: invalid repair_band "
                        f"'{specs.get('repair_cost_band')}' for ({make}, {model})"
                    )
                rating = specs.get("vehicle_safety_rating")
                if rating is not None and not (1 <= rating <= 5):
                    errors.append(
                        f"vehicle_specs: safety_rating={rating} out of [1-5] for ({make}, {model})"
                    )
        vehicle_valid = vehicle_count > 0 and not any("vehicle_specs" in e for e in errors)
    except Exception as exc:
        errors.append(f"vehicle_specs load failed: {exc}")

    # ---- city geo risk ----
    city_count = 0
    city_valid = False
    try:
        raw_c = _load_json(CITY_RISK_PATH)
        seen_cities: set = set()
        for city, risk in raw_c.items():
            if city in seen_cities:
                errors.append(f"city_geo_risk: duplicate city '{city}'")
            seen_cities.add(city)
            city_count += 1
            for field in ["region_risk_category", "flood_risk_index",
                          "theft_risk_index", "monsoon_exposure_index"]:
                if risk.get(field) is None:
                    errors.append(f"city_geo_risk: null '{field}' for city '{city}'")
            if risk.get("region_risk_category") not in _VALID_RISK_CATEGORIES:
                errors.append(
                    f"city_geo_risk: invalid category "
                    f"'{risk.get('region_risk_category')}' for '{city}'"
                )
            for idx_f in ["flood_risk_index", "theft_risk_index", "monsoon_exposure_index"]:
                val = risk.get(idx_f)
                if val is not None and not (1.0 <= val <= 10.0):
                    errors.append(
                        f"city_geo_risk: {idx_f}={val} out of 1-10 range for '{city}'"
                    )
        city_valid = city_count > 0 and not any("city_geo_risk" in e for e in errors)
    except Exception as exc:
        errors.append(f"city_geo_risk load failed: {exc}")

    return {
        "vehicle_specs_path":  VEHICLE_SPECS_PATH,
        "city_risk_path":      CITY_RISK_PATH,
        "vehicle_specs_count": vehicle_count,
        "city_risk_count":     city_count,
        "vehicle_specs_valid": vehicle_valid,
        "city_risk_valid":     city_valid,
        "errors":              errors,
    }


def clear_cache() -> None:
    """Invalidate both lru_cache instances (testing / post-JSON-update)."""
    _load_vehicle_specs.cache_clear()
    _load_city_risk.cache_clear()
    logger.info("Lookup table cache cleared.")
