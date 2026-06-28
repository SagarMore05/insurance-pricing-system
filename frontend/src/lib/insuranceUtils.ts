import type { Occupation } from '../types/v2'

// ─── Occupation: 8 customer-facing display values → 6 backend enum values ────

export const OCCUPATION_BACKEND_MAP: Record<string, Occupation> = {
  'Salaried':            'Salaried',
  'Self-Employed':       'Self-Employed',
  'Business Owner':      'Self-Employed',
  'Government Employee': 'Salaried',
  'Professional':        'Professional',
  'Student':             'Student',
  'Retired':             'Retired',
  'Other':               'Manual',
}

// ─── Vehicle base ex-showroom prices (INR) for IRDAI IDV estimation ──────────

export const VEHICLE_BASE_INR: Record<string, Record<string, number>> = {
  Audi:     { 'A4': 4_700_000 },
  BMW:      { '3 Series': 5_000_000, 'X5': 9_500_000 },
  BYD:      { 'Atto 3': 3_300_000 },
  Ford:     { 'Endeavour': 3_500_000 },
  Honda:    { 'Accord': 4_500_000, 'BR-V': 950_000, 'City': 1_200_000 },
  Hyundai:  { 'Creta': 1_100_000, 'i20': 700_000, 'Kona EV': 2_390_000, 'Verna': 1_100_000 },
  Jeep:     { 'Compass': 1_950_000 },
  Kia:      { 'Carens': 1_000_000, 'Seltos': 1_100_000 },
  MG:       { 'ZS EV': 2_200_000 },
  Mahindra: { 'Scorpio': 1_390_000, 'XUV700': 1_400_000 },
  Maruti:   { 'Alto': 350_000, 'Baleno': 660_000, 'Dzire': 680_000, 'Ertiga': 840_000, 'Swift': 650_000 },
  Mercedes: { 'C-Class': 5_700_000, 'GLE': 9_500_000 },
  Porsche:  { 'Cayenne': 13_400_000 },
  Renault:  { 'Kwid': 470_000 },
  Skoda:    { 'Slavia': 1_150_000 },
  Tata:     { 'Nexon': 810_000, 'Nexon EV': 1_470_000, 'Tiago': 590_000, 'Tiago EV': 850_000 },
  Toyota:   { 'Camry': 4_500_000, 'Fortuner': 3_300_000, 'Innova': 1_900_000 },
}

/**
 * IRDAI IDV depreciation schedule.
 * Returns estimated vehicle value in INR, rounded to nearest ₹1000.
 * Returns null when make/model is not in the lookup table.
 */
export function estimateVehicleValue(make: string, model: string, ageYears: number): number | null {
  const basePrice = VEHICLE_BASE_INR[make]?.[model]
  if (!basePrice) return null
  let factor: number
  if (ageYears <= 0)     factor = 0.95  // brand-new / current year
  else if (ageYears < 1) factor = 0.85  // < 6 months old
  else if (ageYears < 2) factor = 0.80  // 1–2 years
  else if (ageYears < 3) factor = 0.70  // 2–3 years
  else if (ageYears < 4) factor = 0.60  // 3–4 years
  else                   factor = 0.50  // 5+ years: IRDAI cap
  return Math.round(basePrice * factor / 1000) * 1000
}

/**
 * IRDAI NCB slab derivation — mirrors backend derive_ncb_pct exactly.
 * Slabs: 0 / 20 / 25 / 35 / 45 / 50%
 */
export function deriveNcbPct(prevClaims: number, tenureYears: number): number {
  if (prevClaims > 0) return 0
  const tenure = Math.floor(tenureYears)
  if (tenure <= 0) return 0
  if (tenure === 1) return 20
  if (tenure === 2) return 25
  if (tenure === 3) return 35
  if (tenure === 4) return 45
  return 50  // 5+ years
}
