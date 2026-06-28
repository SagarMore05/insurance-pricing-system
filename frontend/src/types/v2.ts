// ─── V2 Enums ────────────────────────────────────────────────────────────────
// Values must exactly match the Python backend Enum values in schemas.py

export type GenderV2 = 'Male' | 'Female' | 'Other'

export type FuelType = 'Petrol' | 'Diesel' | 'EV' | 'CNG'

export type VehicleUsageType = 'Personal' | 'Commercial' | 'Rideshare'

export type Occupation =
  | 'Salaried'
  | 'Self-Employed'
  | 'Professional'
  | 'Retired'
  | 'Student'
  | 'Manual'

export type MaritalStatus = 'Single' | 'Married' | 'Divorced' | 'Widowed'

export type AnnualIncomeBand =
  | '<3 LPA'
  | '3-7 LPA'
  | '7-15 LPA'
  | '15-30 LPA'
  | '>30 LPA'

export type ParkingType = 'Garage' | 'Street' | 'Open_Compound'

// ─── V2 Request ───────────────────────────────────────────────────────────────

export interface CustomerQuoteRequestV2 {
  // Personal
  age: number
  gender: GenderV2
  occupation: Occupation
  marital_status: MaritalStatus
  annual_income_band: AnnualIncomeBand

  // Location
  city: string

  // Driving history
  years_licensed: number
  previous_claims_count: number
  months_since_last_claim: number  // 0 = never claimed
  no_claim_bonus_pct?: number      // auto-derived by backend enrichment service; do not send from UI
  policy_tenure_years: number

  // Vehicle identification
  vehicle_make: string
  vehicle_model: string
  vehicle_age_years: number
  vehicle_value_inr: number
  engine_cc: number

  // Vehicle usage
  fuel_type: FuelType
  vehicle_usage_type: VehicleUsageType
  annual_mileage_km: number
  parking_type: ParkingType
}

// ─── V2 Enriched Feature Audit ────────────────────────────────────────────────

export interface DrivingScoreEnrichment {
  score: number
  factors: string[]
}

export interface VehicleEnrichment {
  vehicle_safety_rating: number
  airbags_count: number
  vehicle_body_style: string
  repair_cost_band: string
  lookup_hit: boolean
}

export interface CityEnrichment {
  region_risk_category: string
  flood_risk_index: number
  theft_risk_index: number
  monsoon_exposure_index: number
  pincode_risk_score: number
  lookup_hit: boolean
}

export interface EnrichedFeaturesV2 {
  driving_score: DrivingScoreEnrichment
  vehicle: VehicleEnrichment
  city: CityEnrichment
  policy_inception_month: number
  no_claim_bonus_pct: number  // auto-derived by backend enrichment service
}

// ─── V2 Champion Metadata ─────────────────────────────────────────────────────

export interface ChampionMetadataV2 {
  frequency_model: string
  severity_model: string
  model_version: string
  v4_ready: boolean
}

// ─── V2 Response ─────────────────────────────────────────────────────────────

import type { RiskLevel, ExplanationDetail, DrivingScoreResult } from './index'

export interface QuoteResponseV2 {
  // V1-compatible core fields
  prediction_id: string
  premium_amount_inr: number
  risk_level: RiskLevel
  claim_probability: number
  expected_claim_amount_inr: number
  explanation: ExplanationDetail
  model_version: string
  created_at: string
  driving_score_info?: DrivingScoreResult

  // V2 additions
  enriched_features: EnrichedFeaturesV2
  champion_metadata: ChampionMetadataV2
  api_version: string
}

// ─── V2 Preview Response ──────────────────────────────────────────────────────

export interface QuotePreviewResponseV2 {
  enriched_features: EnrichedFeaturesV2
  champion_metadata: ChampionMetadataV2
  request_summary: Record<string, string | number>
  api_version: string
  message: string
}

// ─── Lookup data constants (mirrors vehicle_specs.json coverage) ──────────────

export const V4_MAKES_MODELS: Record<string, string[]> = {
  Audi:     ['A4'],
  BMW:      ['3 Series', 'X5'],
  BYD:      ['Atto 3'],
  Ford:     ['Endeavour'],
  Honda:    ['Accord', 'BR-V', 'City'],
  Hyundai:  ['Creta', 'i20', 'Kona EV', 'Verna'],
  Jeep:     ['Compass'],
  Kia:      ['Carens', 'Seltos'],
  MG:       ['ZS EV'],
  Mahindra: ['Scorpio', 'XUV700'],
  Maruti:   ['Alto', 'Baleno', 'Dzire', 'Ertiga', 'Swift'],
  Mercedes: ['C-Class', 'GLE'],
  Porsche:  ['Cayenne'],
  Renault:  ['Kwid'],
  Skoda:    ['Slavia'],
  Tata:     ['Nexon', 'Nexon EV', 'Tiago', 'Tiago EV'],
  Toyota:   ['Camry', 'Fortuner', 'Innova'],
}

export const V4_CITIES = [
  'Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai',
  'Coimbatore', 'Delhi', 'Hyderabad', 'Indore', 'Jaipur',
  'Kanpur', 'Kolkata', 'Lucknow', 'Mumbai', 'Nagpur',
  'Patna', 'Pune', 'Surat', 'Vadodara', 'Visakhapatnam',
]
