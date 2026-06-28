"""
Driving Safety Score — weighted multi-factor scoring model.

Score range: 0–100
  0–50   Very High Risk
  50–65  High Risk
  65–80  Average Driver
  80–90  Good Driver
  90–96  Excellent Driver
  96–100 Exceptional Driver

Base score: 50 — bonuses and deductions are applied from there.

The first four parameters are mandatory (used by both V1 and V2 paths).
The remaining keyword-only parameters are optional and are passed by the
V2 enrichment service for a richer, more personalised score.
"""

from __future__ import annotations


def calculate_driving_score(
    annual_mileage: int,
    prior_claims: int,
    years_licensed: int,
    age: int,
    *,
    occupation: str | None = None,
    marital_status: str | None = None,
    vehicle_age: int | None = None,
    vehicle_value: float | None = None,
    fuel_type: str | None = None,
    vehicle_segment: str | None = None,
    vehicle_usage_type: str | None = None,
    parking_type: str | None = None,
    ncb_pct: float | None = None,
    city_risk_category: str | None = None,
) -> dict:
    """
    Compute a transparent driving safety score (0–100).

    Returns:
        {"score": int, "factors": list[str]}

    Each item in ``factors`` is a human-readable string of the form
    ``"Description (+N)"`` or ``"Description (-N)"``.
    """
    score = 50
    factors: list[str] = []

    # ── 1. Prior claims (most influential factor) ────────────────────────────
    _claims_map = {0: +10, 1: -5, 2: -12, 3: -17}
    claims_delta = _claims_map.get(prior_claims, -22)  # 4+ → -22
    score += claims_delta
    if prior_claims == 0:
        factors.append(f"No previous claims (+{claims_delta})")
    else:
        claim_word = "claim" if prior_claims == 1 else "claims"
        factors.append(f"{prior_claims} prior {claim_word} ({claims_delta})")

    # ── 2. Annual mileage ────────────────────────────────────────────────────
    if annual_mileage <= 5_000:
        mileage_delta, mileage_label = +6, "Very low annual mileage"
    elif annual_mileage <= 10_000:
        mileage_delta, mileage_label = +3, "Low annual mileage"
    elif annual_mileage <= 20_000:
        mileage_delta, mileage_label = 0, None
    elif annual_mileage <= 30_000:
        mileage_delta, mileage_label = -5, "High annual mileage"
    else:
        mileage_delta, mileage_label = -9, "Very high annual mileage"

    score += mileage_delta
    if mileage_label and mileage_delta != 0:
        sign = "+" if mileage_delta > 0 else ""
        factors.append(f"{mileage_label} ({sign}{mileage_delta})")

    # ── 3. Years licensed ────────────────────────────────────────────────────
    if years_licensed == 0:
        yrs_delta, yrs_label = 0, None
    elif years_licensed <= 2:
        yrs_delta = +3
        yrs_label = f"{years_licensed} yr{'s' if years_licensed > 1 else ''} licensed"
    elif years_licensed <= 5:
        yrs_delta, yrs_label = +6, f"{years_licensed} years licensed"
    elif years_licensed <= 10:
        yrs_delta, yrs_label = +9, f"{years_licensed} years licensed"
    elif years_licensed <= 20:
        yrs_delta, yrs_label = +12, f"{years_licensed} years licensed"
    else:
        yrs_delta, yrs_label = +14, f"{years_licensed}+ years licensed"

    score += yrs_delta
    if yrs_label and yrs_delta > 0:
        factors.append(f"{yrs_label} (+{yrs_delta})")

    # ── 4. Driver age ────────────────────────────────────────────────────────
    if 18 <= age <= 22:
        age_delta, age_label = -8, "Young driver"
    elif 23 <= age <= 25:
        age_delta, age_label = -4, "Inexperienced age group"
    elif 26 <= age <= 55:
        age_delta, age_label = +4, "Prime driving age"
    elif 56 <= age <= 65:
        age_delta, age_label = +1, "Experienced driver"
    else:
        age_delta, age_label = -4, "Senior driver"

    score += age_delta
    sign = "+" if age_delta > 0 else ""
    factors.append(f"{age_label} ({sign}{age_delta})")

    # ── 5. Occupation (optional) ─────────────────────────────────────────────
    if occupation is not None:
        _occ_map: dict[str, tuple[int, str | None]] = {
            "Professional":   (+3, "Professional occupation"),
            "Salaried":       (+1, "Salaried employee"),
            "Self-Employed":  ( 0, None),
            "Manual":         (-1, "Manual worker"),
            "Retired":        (-1, "Retired"),
            "Student":        (-3, "Student"),
        }
        occ_delta, occ_label = _occ_map.get(occupation, (0, None))
        score += occ_delta
        if occ_label and occ_delta != 0:
            sign = "+" if occ_delta > 0 else ""
            factors.append(f"{occ_label} ({sign}{occ_delta})")

    # ── 6. Marital status (optional) ─────────────────────────────────────────
    if marital_status == "Married":
        score += 2
        factors.append("Married driver (+2)")

    # ── 7. Vehicle age (optional) ────────────────────────────────────────────
    if vehicle_age is not None:
        if vehicle_age <= 3:
            vage_delta, vage_label = +3, "New vehicle"
        elif vehicle_age <= 7:
            vage_delta, vage_label = +1, "Modern vehicle"
        elif vehicle_age <= 12:
            vage_delta, vage_label = -1, "Older vehicle"
        else:
            vage_delta, vage_label = -3, "Aged vehicle"

        score += vage_delta
        if vage_delta != 0:
            sign = "+" if vage_delta > 0 else ""
            factors.append(f"{vage_label} ({sign}{vage_delta})")

    # ── 8. Fuel type (optional) ──────────────────────────────────────────────
    if fuel_type is not None:
        _fuel_map: dict[str, tuple[int, str | None]] = {
            "Electric": (+3, "Electric vehicle"),
            "EV":       (+3, "Electric vehicle"),
            "Hybrid":   (+2, "Hybrid vehicle"),
            "Petrol":   ( 0, None),
            "Diesel":   (-1, "Diesel vehicle"),
            "CNG":      (-1, "CNG vehicle"),
        }
        fuel_delta, fuel_label = _fuel_map.get(fuel_type, (0, None))
        score += fuel_delta
        if fuel_label and fuel_delta != 0:
            sign = "+" if fuel_delta > 0 else ""
            factors.append(f"{fuel_label} ({sign}{fuel_delta})")

    # ── 9. Vehicle segment / body style (optional) ───────────────────────────
    if vehicle_segment is not None:
        _seg_map: dict[str, tuple[int, str | None]] = {
            "Hatchback": (+1, "Compact vehicle"),
            "Sedan":     (+1, "Sedan"),
            "SUV":       ( 0, None),
            "MUV":       ( 0, None),
            "Luxury":    (-1, "Luxury vehicle"),
        }
        seg_delta, seg_label = _seg_map.get(vehicle_segment, (0, None))
        score += seg_delta
        if seg_label and seg_delta != 0:
            sign = "+" if seg_delta > 0 else ""
            factors.append(f"{seg_label} ({sign}{seg_delta})")

    # ── 10. Vehicle market value (optional) ──────────────────────────────────
    if vehicle_value is not None:
        if vehicle_value < 300_000:
            vval_delta, vval_label = -1, "Budget vehicle"
        elif vehicle_value <= 800_000:
            vval_delta, vval_label = 0, None
        elif vehicle_value <= 2_000_000:
            vval_delta, vval_label = +2, "Safety-rated vehicle"
        else:
            vval_delta, vval_label = +3, "Premium vehicle"

        score += vval_delta
        if vval_label and vval_delta != 0:
            sign = "+" if vval_delta > 0 else ""
            factors.append(f"{vval_label} ({sign}{vval_delta})")

    # ── 11. Vehicle usage type (optional) ────────────────────────────────────
    if vehicle_usage_type is not None:
        _usage_map: dict[str, tuple[int, str | None]] = {
            "Personal":   (+3, "Personal use"),
            "Rideshare":  (-3, "Rideshare use"),
            "Commercial": (-5, "Commercial use"),
        }
        usage_delta, usage_label = _usage_map.get(vehicle_usage_type, (0, None))
        score += usage_delta
        if usage_label and usage_delta != 0:
            sign = "+" if usage_delta > 0 else ""
            factors.append(f"{usage_label} ({sign}{usage_delta})")

    # ── 12. Parking type (optional) ──────────────────────────────────────────
    if parking_type is not None:
        _park_map: dict[str, tuple[int, str | None]] = {
            "Garage":        (+4, "Garage parking"),
            "Open_Compound": ( 0, None),
            "Street":        (-3, "Street parking"),
        }
        park_delta, park_label = _park_map.get(parking_type, (0, None))
        score += park_delta
        if park_label and park_delta != 0:
            sign = "+" if park_delta > 0 else ""
            factors.append(f"{park_label} ({sign}{park_delta})")

    # ── 13. No-claim bonus (optional) ────────────────────────────────────────
    if ncb_pct is not None and ncb_pct > 0:
        if ncb_pct >= 50:
            ncb_delta = +9
        elif ncb_pct >= 45:
            ncb_delta = +7
        elif ncb_pct >= 35:
            ncb_delta = +5
        elif ncb_pct >= 25:
            ncb_delta = +3
        else:
            ncb_delta = +2  # 20%
        score += ncb_delta
        factors.append(f"No-claim bonus {int(ncb_pct)}% (+{ncb_delta})")

    # ── 14. City / area risk (optional) ──────────────────────────────────────
    if city_risk_category is not None:
        _city_map: dict[str, tuple[int, str | None]] = {
            "High":   (-3, "High-risk city"),
            "Medium": ( 0, None),
            "Low":    (+2, "Low-risk city"),
        }
        city_delta, city_label = _city_map.get(city_risk_category, (0, None))
        score += city_delta
        if city_label and city_delta != 0:
            sign = "+" if city_delta > 0 else ""
            factors.append(f"{city_label} ({sign}{city_delta})")

    return {"score": max(0, min(100, score)), "factors": factors}
