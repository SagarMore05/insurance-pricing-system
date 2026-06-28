import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import DerivedFeaturesPanel from '../DerivedFeaturesPanel'
import type { EnrichedFeaturesV2 } from '../../../types/v2'

// ─── Shared fixture builder ───────────────────────────────────────────────────

function makeFeatures(overrides: Partial<EnrichedFeaturesV2> = {}): EnrichedFeaturesV2 {
  return {
    driving_score: {
      score: 88,
      factors: ['Driving Experience (+)', 'Long tenure (+)'],
    },
    vehicle: {
      vehicle_safety_rating: 4,
      airbags_count: 6,
      vehicle_body_style: 'SUV',
      repair_cost_band: 'Standard',
      lookup_hit: true,
    },
    city: {
      region_risk_category: 'Moderate',
      flood_risk_index: 2.5,        // Low tier (≤ 3.5)
      theft_risk_index: 5.5,        // Moderate tier (3.5–6.5)
      monsoon_exposure_index: 8.0,  // High tier (> 6.5)
      pincode_risk_score: 5.0,
      lookup_hit: true,
    },
    policy_inception_month: 6,
    no_claim_bonus_pct: 25,
    ...overrides,
  }
}

// ─── Part 4: Driving Score UX ────────────────────────────────────────────────

describe('DerivedFeaturesPanel — Part 4: driving score UX', () => {
  it('shows "Excellent Driver" label when score ≥ 85', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 88, factors: [] } })} />)
    expect(screen.getByText('Excellent Driver')).toBeInTheDocument()
  })

  it('shows "Good Driver" label when score is 60–84', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 72, factors: [] } })} />)
    expect(screen.getByText('Good Driver')).toBeInTheDocument()
  })

  it('shows "Needs Improvement" label when score < 60', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 45, factors: [] } })} />)
    expect(screen.getByText('Needs Improvement')).toBeInTheDocument()
  })

  it('does NOT show bare "Excellent" or "Good" labels (old labels removed)', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 90, factors: [] } })} />)
    // "Excellent Driver" should exist, but bare "Excellent" alone should not
    const elements = screen.queryAllByText('Excellent')
    expect(elements).toHaveLength(0)
  })

  it('shows preferential-rates tagline for score ≥ 85', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 90, factors: [] } })} />)
    expect(screen.getByText(/preferential rates/)).toBeInTheDocument()
  })

  it('shows claim-free improvement tagline for score 60–84', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 70, factors: [] } })} />)
    expect(screen.getByText(/claim-free year can improve/)).toBeInTheDocument()
  })

  it('shows higher-risk tagline for score < 60', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 40, factors: [] } })} />)
    expect(screen.getByText(/Higher-risk profile/)).toBeInTheDocument()
  })

  it('displays the numeric score', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ driving_score: { score: 88, factors: [] } })} />)
    expect(screen.getByText('88')).toBeInTheDocument()
  })
})

// ─── Part 5: NCB display ──────────────────────────────────────────────────────

describe('DerivedFeaturesPanel — Part 5: NCB display', () => {
  it('displays no_claim_bonus_pct value', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ no_claim_bonus_pct: 45 })} />)
    expect(screen.getByText('45%')).toBeInTheDocument()
  })

  it('shows "IRDAI NCB applied" text when NCB > 0', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ no_claim_bonus_pct: 20 })} />)
    expect(screen.getByText('IRDAI NCB applied to your premium')).toBeInTheDocument()
  })

  it('shows "no bonus yet" when NCB = 0', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures({ no_claim_bonus_pct: 0 })} />)
    expect(screen.getByText('no bonus yet')).toBeInTheDocument()
  })
})

// ─── Part 10: Area risk descriptions ─────────────────────────────────────────

describe('DerivedFeaturesPanel — Part 10: area risk explanations', () => {
  describe('Flood exposure', () => {
    it('shows low description for flood_risk_index ≤ 3.5', () => {
      const features = makeFeatures()
      features.city.flood_risk_index = 2.5
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/rarely affected by flooding/)).toBeInTheDocument()
    })

    it('shows moderate description for flood_risk_index 3.5–6.5', () => {
      const features = makeFeatures()
      features.city.flood_risk_index = 5.0
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/seasonal flooding may occur/)).toBeInTheDocument()
    })

    it('shows high description for flood_risk_index > 6.5', () => {
      const features = makeFeatures()
      features.city.flood_risk_index = 8.5
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/prone to seasonal flooding/)).toBeInTheDocument()
    })
  })

  describe('Theft risk', () => {
    it('shows low description for theft_risk_index ≤ 3.5', () => {
      const features = makeFeatures()
      features.city.theft_risk_index = 2.0
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/vehicle theft is uncommon/)).toBeInTheDocument()
    })

    it('shows moderate description for theft_risk_index 3.5–6.5', () => {
      const features = makeFeatures()
      features.city.theft_risk_index = 5.5
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/consider additional security measures/)).toBeInTheDocument()
    })

    it('shows high description for theft_risk_index > 6.5', () => {
      const features = makeFeatures()
      features.city.theft_risk_index = 8.0
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/enhanced vehicle security is strongly advisable/)).toBeInTheDocument()
    })
  })

  describe('Monsoon impact', () => {
    it('shows high description for monsoon_exposure_index > 6.5', () => {
      const features = makeFeatures()
      features.city.monsoon_exposure_index = 8.0
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/heavy seasonal rainfall/)).toBeInTheDocument()
    })

    it('shows moderate description for monsoon_exposure_index 3.5–6.5', () => {
      const features = makeFeatures()
      features.city.monsoon_exposure_index = 5.0
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/periodic heavy rainfall expected/)).toBeInTheDocument()
    })

    it('shows low description for monsoon_exposure_index ≤ 3.5', () => {
      const features = makeFeatures()
      features.city.monsoon_exposure_index = 1.5
      render(<DerivedFeaturesPanel enrichedFeatures={features} />)
      expect(screen.getByText(/light seasonal rainfall/)).toBeInTheDocument()
    })
  })

  it('does not render bare numeric indices without explanation', () => {
    render(<DerivedFeaturesPanel enrichedFeatures={makeFeatures()} />)
    // The risk bars show "Low" / "Moderate" / "High" labels, not just numbers
    expect(screen.queryByText('2.5')).not.toBeInTheDocument()
  })
})
