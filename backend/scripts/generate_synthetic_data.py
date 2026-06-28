"""
Generate 10,000 rows of realistic synthetic insurance data.
Distribution: 60% no claims, 30% 1 claim, 10% 2+ claims.
"""
import numpy as np
import pandas as pd
import uuid
from pathlib import Path

RNG = np.random.default_rng(42)

CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "jaipur", "lucknow",
    "surat", "indore", "bhopal", "nagpur", "coimbatore",
    "patna", "vadodara", "agra", "visakhapatnam", "kanpur",
]
GENDERS = ["male", "female", "other"]
GENDER_WEIGHTS = [0.55, 0.43, 0.02]

CAR_BRANDS = [
    "maruti", "hyundai", "tata", "mahindra", "honda",
    "toyota", "ford", "volkswagen", "kia", "renault",
    "skoda", "mg", "nissan", "jeep", "bmw", "audi", "other",
]
BRAND_WEIGHTS = [0.25, 0.15, 0.12, 0.10, 0.07, 0.06, 0.04, 0.04, 0.04, 0.03,
                  0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]

CAR_MODELS = {
    "maruti": ["Swift", "Baleno", "Dzire", "Alto", "WagonR"],
    "hyundai": ["i20", "Creta", "Verna", "Venue", "Tucson"],
    "tata": ["Nexon", "Harrier", "Safari", "Punch", "Altroz"],
    "mahindra": ["Scorpio", "XUV700", "Bolero", "Thar", "XUV300"],
    "honda": ["City", "Amaze", "Jazz", "WR-V", "CR-V"],
    "toyota": ["Fortuner", "Innova", "Glanza", "Urban Cruiser", "Camry"],
    "ford": ["EcoSport", "Endeavour", "Figo", "Freestyle", "Mustang"],
    "volkswagen": ["Polo", "Vento", "Tiguan", "Taigun", "Virtus"],
    "kia": ["Seltos", "Sonet", "Carnival", "EV6", "Carens"],
    "renault": ["Kwid", "Triber", "Kiger", "Duster", "Captur"],
    "skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
    "mg": ["Hector", "ZS EV", "Astor", "Gloster", "Comet"],
    "nissan": ["Magnite", "Kicks", "GT-R", "Terrano", "Micra"],
    "jeep": ["Compass", "Meridian", "Wrangler", "Grand Cherokee", "Renegade"],
    "bmw": ["3 Series", "5 Series", "X1", "X3", "X5"],
    "audi": ["A4", "A6", "Q3", "Q5", "Q7"],
    "other": ["Generic Model A", "Generic Model B", "Generic Model C"],
}


def sample_engine_cc(brand: str) -> int:
    luxury = {"bmw", "audi", "jeep"}
    mid = {"toyota", "volkswagen", "skoda", "honda", "ford"}
    if brand in luxury:
        return int(RNG.integers(2000, 4000))
    elif brand in mid:
        return int(RNG.integers(1200, 2500))
    return int(RNG.integers(800, 1800))


def sample_vehicle_value(brand: str, engine_cc: int, age_years: int) -> float:
    luxury = {"bmw", "audi"}
    premium = {"jeep", "toyota", "volkswagen", "skoda"}
    if brand in luxury:
        base = RNG.integers(3_000_000, 8_000_000)
    elif brand in premium:
        base = RNG.integers(1_500_000, 3_500_000)
    else:
        base = RNG.integers(400_000, 1_800_000)

    depreciation = max(0.3, 1 - age_years * 0.12)
    return round(float(base * depreciation), -3)  # round to nearest 1000


def compute_claim_probability(row: dict) -> float:
    """Actuarial-style claim probability based on risk factors."""
    base = 0.15

    age = row["age"]
    if age < 25:
        base += 0.10
    elif age > 70:
        base += 0.07
    elif 30 <= age <= 55:
        base -= 0.03

    score = row["driving_score"]
    base += (50 - score) * 0.003  # higher score → lower probability

    base += row["previous_claims_count"] * 0.08

    if row["annual_mileage_km"] > 30_000:
        base += 0.07
    elif row["annual_mileage_km"] < 5_000:
        base -= 0.03

    if row["vehicle_age_years"] > 8:
        base += 0.04

    return float(np.clip(base + RNG.normal(0, 0.02), 0.02, 0.65))


def generate_dataset(n: int = 10_000) -> pd.DataFrame:
    rows = []
    claim_count_dist = RNG.choice([0, 1, 2, 3], size=n, p=[0.60, 0.30, 0.07, 0.03])

    for i in range(n):
        customer_id = str(uuid.uuid4())
        age = int(RNG.integers(18, 86))
        gender = RNG.choice(GENDERS, p=GENDER_WEIGHTS)
        city = RNG.choice(CITIES)
        years_licensed = min(age - 18, int(RNG.integers(1, max(2, age - 17))))

        brand = RNG.choice(CAR_BRANDS, p=BRAND_WEIGHTS)
        model = RNG.choice(CAR_MODELS.get(brand, ["Generic"]))
        veh_age = int(RNG.integers(0, 16))
        engine_cc = sample_engine_cc(brand)
        vehicle_value = sample_vehicle_value(brand, engine_cc, veh_age)

        driving_score = float(np.clip(RNG.normal(68, 15), 0, 100))
        annual_mileage = int(np.clip(RNG.lognormal(9.6, 0.5), 1000, 200_000))
        prev_claims = claim_count_dist[i]

        row_dict = {
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "city": city,
            "car_brand": brand,
            "car_model": model,
            "engine_cc": engine_cc,
            "vehicle_age_years": veh_age,
            "vehicle_value_inr": vehicle_value,
            "driving_score": round(driving_score, 1),
            "annual_mileage_km": annual_mileage,
            "previous_claims_count": prev_claims,
            "years_licensed": years_licensed,
        }

        claim_prob = compute_claim_probability(row_dict)
        claim_occurred = int(RNG.random() < claim_prob)

        actual_claim_amount = None
        if claim_occurred:
            base_claim = vehicle_value * RNG.uniform(0.05, 0.60)
            actual_claim_amount = round(float(np.clip(base_claim, 5_000, vehicle_value)), -2)

        # Derive risk level
        if claim_prob < 0.10:
            risk_level = "low"
        elif claim_prob < 0.25:
            risk_level = "medium"
        else:
            risk_level = "high"

        row_dict.update({
            "claim_occurred": claim_occurred,
            "actual_claim_amount_inr": actual_claim_amount,
            "claim_probability_true": round(claim_prob, 4),
            "risk_level": risk_level,
        })
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} rows")
    print(f"Claim rate: {df['claim_occurred'].mean():.1%}")
    print(f"Risk distribution:\n{df['risk_level'].value_counts()}")
    print(f"Avg claim amount (when claimed): ₹{df.loc[df['claim_occurred']==1, 'actual_claim_amount_inr'].mean():,.0f}")
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_dataset(10_000)
    out_path = out_dir / "synthetic_insurance_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
