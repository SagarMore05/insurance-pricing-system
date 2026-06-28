import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import GovernancePanel from '../GovernancePanel'
import type { ChampionMetadataV2 } from '../../../types/v2'

const metaV4: ChampionMetadataV2 = {
  frequency_model: 'FrequencyModel',
  severity_model:  'CatBoostSeverityWrapper',
  model_version:   'v4.2.0',
  v4_ready:        true,
}

const metaV1: ChampionMetadataV2 = {
  frequency_model: 'XGBClassifier-V1',
  severity_model:  'XGBRegressor-V1',
  model_version:   'v1.0.0',
  v4_ready:        false,
}

const unknownMeta: ChampionMetadataV2 = {
  frequency_model: 'UnknownFrequencyEngine',
  severity_model:  'UnknownSeverityEngine',
  model_version:   'v99',
  v4_ready:        true,
}

// ─── Part 9: Risk Terminology ────────────────────────────────────────────────

describe('GovernancePanel — Part 9: risk terminology', () => {
  it('displays "Risk Assessment" for V4 frequency model (not "Risk Prediction AI")', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument()
    expect(screen.queryByText('Risk Prediction AI')).not.toBeInTheDocument()
  })

  it('displays "Repair Cost Analysis" for V4 severity model (not "Cost Estimation AI")', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.getByText('Repair Cost Analysis')).toBeInTheDocument()
    expect(screen.queryByText('Cost Estimation AI')).not.toBeInTheDocument()
  })

  it('chip label uses "Accident likelihood" (not "Claim likelihood")', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.getByText('Accident likelihood')).toBeInTheDocument()
    expect(screen.queryByText('Claim likelihood')).not.toBeInTheDocument()
  })

  it('chip label uses "Repair cost estimate" (not "Claim cost estimate")', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.getByText('Repair cost estimate')).toBeInTheDocument()
    expect(screen.queryByText('Claim cost estimate')).not.toBeInTheDocument()
  })

  it('model_version string "v4.2.0" is NOT shown to the customer', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.queryByText(/v4\.2\.0/)).not.toBeInTheDocument()
  })
})

describe('GovernancePanel — V1 fallback terminology', () => {
  it('shows "Risk Assessment (V1)" for XGBClassifier-V1 frequency model', () => {
    render(<GovernancePanel championMetadata={metaV1} />)
    expect(screen.getByText('Risk Assessment (V1)')).toBeInTheDocument()
  })

  it('shows "Repair Cost Analysis (V1)" for XGBRegressor-V1 severity model', () => {
    render(<GovernancePanel championMetadata={metaV1} />)
    expect(screen.getByText('Repair Cost Analysis (V1)')).toBeInTheDocument()
  })
})

describe('GovernancePanel — unknown model name fallback', () => {
  it('falls back to "Risk Assessment" for unknown frequency model name', () => {
    render(<GovernancePanel championMetadata={unknownMeta} />)
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument()
  })

  it('falls back to "Repair Cost Analysis" for unknown severity model name', () => {
    render(<GovernancePanel championMetadata={unknownMeta} />)
    expect(screen.getByText('Repair Cost Analysis')).toBeInTheDocument()
  })
})

describe('GovernancePanel — status badge', () => {
  it('shows "Enhanced AI" when v4_ready is true', () => {
    render(<GovernancePanel championMetadata={metaV4} />)
    expect(screen.getByText('Enhanced AI')).toBeInTheDocument()
  })

  it('shows "Standard AI" when v4_ready is false', () => {
    render(<GovernancePanel championMetadata={metaV1} />)
    expect(screen.getByText('Standard AI')).toBeInTheDocument()
  })
})
