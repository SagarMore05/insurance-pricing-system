import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import QuoteResultV2 from '../QuoteResultV2'
import type { QuoteResponseV2, CustomerQuoteRequestV2 } from '../../../types/v2'

vi.mock('../../../api/insurance', () => ({
  insuranceApi: {
    createPolicy: vi.fn().mockResolvedValue({ policy_id: 'pol-001' }),
  },
}))

// ─── Fixture data ─────────────────────────────────────────────────────────────

const mockRequest: CustomerQuoteRequestV2 = {
  age: 32,
  gender: 'Male',
  occupation: 'Salaried',
  marital_status: 'Single',
  annual_income_band: '7-15 LPA',
  city: 'Mumbai',
  years_licensed: 8,
  previous_claims_count: 0,
  months_since_last_claim: 0,
  policy_tenure_years: 3,
  vehicle_make: 'Maruti',
  vehicle_model: 'Swift',
  vehicle_age_years: 2,
  vehicle_value_inr: 520_000,
  engine_cc: 1197,
  fuel_type: 'Petrol',
  vehicle_usage_type: 'Personal',
  annual_mileage_km: 12000,
  parking_type: 'Garage',
}

const mockQuote: QuoteResponseV2 = {
  prediction_id: 'pred-abc123',
  premium_amount_inr: 24_800,
  risk_level: 'medium',
  claim_probability: 0.14,
  expected_claim_amount_inr: 68_000,
  explanation: {
    base_premium: 18_000,
    expected_loss: 12_600,
    adjustments: [
      { reason: 'luxury_vehicle_surcharge', factor: 1.15, impact_inr: 4_200 },
      { reason: 'young_driver_surcharge',   factor: 1.10, impact_inr: 2_600 },
      { reason: 'no_claims_bonus',          factor: 0.87, impact_inr: -3_100 },
      { reason: 'low_mileage_discount',     factor: 0.92, impact_inr: -900 },
    ],
    final_premium: 24_800,
    risk_level: 'medium',
    claim_probability: 0.14,
    summary: 'Your premium reflects a moderate risk profile with a no-claim bonus applied.',
  },
  model_version: 'v4.2.0',
  created_at: '2026-06-24T00:00:00Z',
  enriched_features: {
    driving_score: { score: 82, factors: ['Long tenure (+)', 'No prior claims (+)'] },
    vehicle: {
      vehicle_safety_rating: 4,
      airbags_count: 2,
      vehicle_body_style: 'Hatchback',
      repair_cost_band: 'Budget',
      lookup_hit: true,
    },
    city: {
      region_risk_category: 'Moderate',
      flood_risk_index: 4.0,
      theft_risk_index: 5.5,
      monsoon_exposure_index: 7.0,
      pincode_risk_score: 5.2,
      lookup_hit: true,
    },
    policy_inception_month: 7,
    no_claim_bonus_pct: 35,
  },
  champion_metadata: {
    frequency_model: 'FrequencyModel',
    severity_model: 'CatBoostSeverityWrapper',
    model_version: 'v4.2.0',
    v4_ready: true,
  },
  api_version: 'v2',
}

function renderResult(quoteOverrides: Partial<QuoteResponseV2> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <QuoteResultV2
          quote={{ ...mockQuote, ...quoteOverrides }}
          requestData={mockRequest}
          onBack={vi.fn()}
        />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// ─── Part 9: Risk terminology ─────────────────────────────────────────────────

describe('QuoteResultV2 — Part 9: risk terminology', () => {
  it('shows "Accident Risk" label (not "Likelihood of Making a Claim")', () => {
    renderResult()
    expect(screen.getByText('Accident Risk')).toBeInTheDocument()
    expect(screen.queryByText('Likelihood of Making a Claim')).not.toBeInTheDocument()
  })

  it('shows "Estimated repair cost for your profile" (not "Average claim cost")', () => {
    renderResult()
    expect(screen.getByText('Estimated repair cost for your profile')).toBeInTheDocument()
    expect(screen.queryByText(/Average claim cost/)).not.toBeInTheDocument()
  })

  it('does not expose model_version string "v4.2.0" to customer', () => {
    renderResult()
    // model_version should not appear in the summary tab
    expect(screen.queryByText(/v4\.2\.0/)).not.toBeInTheDocument()
  })
})

// ─── Part 11: Always-visible premium explainability ───────────────────────────

describe('QuoteResultV2 — Part 11: premium explainability', () => {
  it('shows "Why this premium?" heading immediately (no click required)', () => {
    renderResult()
    expect(screen.getByText('Why this premium?')).toBeInTheDocument()
  })

  it('shows "Starting premium" base amount', () => {
    renderResult()
    expect(screen.getByText('Starting premium')).toBeInTheDocument()
  })

  it('shows "Increasing your premium" section for positive adjustments', () => {
    renderResult()
    expect(screen.getByText(/Increasing your premium/i)).toBeInTheDocument()
  })

  it('shows "Reducing your premium" section for negative adjustments', () => {
    renderResult()
    expect(screen.getByText(/Reducing your premium/i)).toBeInTheDocument()
  })

  it('shows ↑ arrow for positive adjustments', () => {
    renderResult()
    const upArrows = screen.getAllByText('↑')
    expect(upArrows.length).toBeGreaterThan(0)
  })

  it('shows ↓ arrow for negative adjustments', () => {
    renderResult()
    const downArrows = screen.getAllByText('↓')
    expect(downArrows.length).toBeGreaterThan(0)
  })

  it('shows "Your Annual Premium" total', () => {
    renderResult()
    // Appears in hero card label AND in the breakdown total row — both are correct
    const els = screen.getAllByText('Your Annual Premium')
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it('does NOT show old formula text "expected loss × 1.35"', () => {
    renderResult()
    expect(screen.queryByText(/expected loss/i)).not.toBeInTheDocument()
  })

  it('does NOT show factor multipliers like "(×1.15)"', () => {
    renderResult()
    expect(screen.queryByText(/×1\.\d+/)).not.toBeInTheDocument()
  })
})

// ─── Part 13: Quote result page cleanup ──────────────────────────────────────

describe('QuoteResultV2 — Part 13: result page cleanup', () => {
  it('shows premium amount in hero card', () => {
    renderResult()
    // ₹24,800 appears both in the hero display and the breakdown total — both correct
    const els = screen.getAllByText('₹24,800')
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it('shows risk level badge', () => {
    renderResult()
    expect(screen.getByText('Medium Risk')).toBeInTheDocument()
  })

  it('human-readable adjustment labels (no raw reason keys)', () => {
    renderResult()
    // luxury_vehicle_surcharge → "High-Value Vehicle Loading"
    expect(screen.getByText('High-Value Vehicle Loading')).toBeInTheDocument()
    // no_claims_bonus → "No-Claims Bonus Applied"
    expect(screen.getByText('No-Claims Bonus Applied')).toBeInTheDocument()
  })

  it('tabs are present: Breakdown, Your Profile, About Pricing', () => {
    renderResult()
    expect(screen.getByRole('button', { name: /Breakdown/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Your Profile/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /About Pricing/i })).toBeInTheDocument()
  })

  it('switching to "Your Profile" tab shows enriched features panel', async () => {
    renderResult()
    await userEvent.click(screen.getByRole('button', { name: /Your Profile/i }))
    // DerivedFeaturesPanel renders "Driving Safety Score"
    expect(screen.getByText('Driving Safety Score')).toBeInTheDocument()
  })

  it('switching to "About Pricing" tab shows governance panel', async () => {
    renderResult()
    await userEvent.click(screen.getByRole('button', { name: /About Pricing/i }))
    expect(screen.getByText('About This Quote')).toBeInTheDocument()
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument()
  })
})

// ─── Quote result summary text ────────────────────────────────────────────────

describe('QuoteResultV2 — summary explanation text', () => {
  it('shows the explanation summary from backend', () => {
    renderResult()
    expect(screen.getByText(/moderate risk profile with a no-claim bonus/)).toBeInTheDocument()
  })
})
