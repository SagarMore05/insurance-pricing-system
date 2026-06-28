import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import QuoteWizardV2 from '../QuoteWizardV2'
import type { QuotePreviewResponseV2 } from '../../../types/v2'

vi.mock('../../../api/insuranceV2', () => ({
  insuranceApiV2: {
    getQuoteV2:     vi.fn(),
    previewQuoteV2: vi.fn(),
  },
}))

vi.mock('../../../api/insurance', () => ({
  insuranceApi: { createPolicy: vi.fn() },
}))

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
}

function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={makeQueryClient()}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

async function fillStep0() {
  await userEvent.clear(screen.getByLabelText(/age/i))
  await userEvent.type(screen.getByLabelText(/age/i), '35')
  await userEvent.selectOptions(screen.getByLabelText(/gender/i), 'Male')
  await userEvent.selectOptions(screen.getByLabelText(/occupation/i), 'Salaried')
  await userEvent.selectOptions(screen.getByLabelText(/marital status/i), 'Single')
  await userEvent.selectOptions(screen.getByLabelText(/annual income/i), '7-15 LPA')
  await userEvent.click(screen.getByRole('button', { name: /next/i }))
}

// ─── Section 1: Step 0 — Customer Information ────────────────────────────────

describe('QuoteWizardV2 — Step 0: Customer Information', () => {
  beforeEach(() => render(<QuoteWizardV2 />, { wrapper: Wrapper }))

  it('renders step 0 with all required personal fields', () => {
    expect(screen.getByLabelText(/age/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/gender/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/occupation/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/marital status/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/annual income/i)).toBeInTheDocument()
  })

  // Part 15: Accessibility — aria-labels
  describe('Part 15: aria-labels on Step 0', () => {
    it('Age input has aria-label', () => expect(screen.getByLabelText('Age')).toBeInTheDocument())
    it('Gender select has aria-label', () => expect(screen.getByLabelText('Gender')).toBeInTheDocument())
    it('Occupation select has aria-label', () => expect(screen.getByLabelText('Occupation')).toBeInTheDocument())
    it('Marital status select has aria-label', () => expect(screen.getByLabelText('Marital status')).toBeInTheDocument())
    it('Annual income band has aria-label', () => expect(screen.getByLabelText('Annual income band')).toBeInTheDocument())
  })

  // Part 7: Occupation options
  describe('Part 7: Occupation options', () => {
    it('shows 8 occupation value options (excluding placeholder)', () => {
      const sel = screen.getByLabelText('Occupation') as HTMLSelectElement
      expect(sel.options.length - 1).toBe(8)  // minus placeholder
    })
    it('includes "Salaried Employee"', () => expect(screen.getByRole('option', { name: 'Salaried Employee' })).toBeInTheDocument())
    it('includes "Government / Public Sector Employee"', () => expect(screen.getByRole('option', { name: /Government \/ Public Sector/ })).toBeInTheDocument())
    it('includes "Business Owner"', () => expect(screen.getByRole('option', { name: 'Business Owner' })).toBeInTheDocument())
    it('includes "Other" occupation', () => {
      const occupSel = screen.getByLabelText('Occupation') as HTMLSelectElement
      expect(Array.from(occupSel.options).some(o => o.value === 'Other')).toBe(true)
    })
    it('does NOT include "Manual" occupation', () => expect(screen.queryByRole('option', { name: 'Manual' })).not.toBeInTheDocument())
  })

  // Part 6: Marital status simplification
  describe('Part 6: Marital Status (Single/Married/Other)', () => {
    it('has exactly 3 value options', () => {
      const sel = screen.getByLabelText('Marital status') as HTMLSelectElement
      expect(Array.from(sel.options).filter(o => o.value !== '').length).toBe(3)
    })
    it('includes "Single"', () => expect(screen.getByRole('option', { name: 'Single' })).toBeInTheDocument())
    it('includes "Married"', () => expect(screen.getByRole('option', { name: 'Married' })).toBeInTheDocument())
    it('includes "Other" marital status', () => {
      const sel = screen.getByLabelText('Marital status') as HTMLSelectElement
      expect(Array.from(sel.options).some(o => o.value === 'Other')).toBe(true)
    })
    it('does NOT include "Divorced"', () => expect(screen.queryByRole('option', { name: 'Divorced' })).not.toBeInTheDocument())
    it('does NOT include "Widowed"', () => expect(screen.queryByRole('option', { name: 'Widowed' })).not.toBeInTheDocument())
  })

  // Validation
  describe('Zod validation', () => {
    it('blocks Next when age is missing', async () => {
      await userEvent.click(screen.getByRole('button', { name: /next/i }))
      expect(screen.getByText(/required/i)).toBeInTheDocument()
    })
    it('blocks Next when age < 18', async () => {
      await userEvent.type(screen.getByLabelText('Age'), '16')
      await userEvent.click(screen.getByRole('button', { name: /next/i }))
      expect(screen.getByText(/minimum 18/i)).toBeInTheDocument()
    })
    it('blocks Next when age > 100', async () => {
      await userEvent.type(screen.getByLabelText('Age'), '105')
      await userEvent.click(screen.getByRole('button', { name: /next/i }))
      expect(screen.getByText(/maximum 100/i)).toBeInTheDocument()
    })
    it('advances to step 1 on valid step 0 data', async () => {
      await fillStep0()
      await waitFor(() => expect(screen.getByLabelText('City')).toBeInTheDocument())
    })
  })
})

// ─── Section 2: Step 1 — Driving History ─────────────────────────────────────

describe('QuoteWizardV2 — Step 1: Driving History', () => {
  beforeEach(async () => {
    render(<QuoteWizardV2 />, { wrapper: Wrapper })
    await fillStep0()
    await waitFor(() => expect(screen.getByLabelText('City')).toBeInTheDocument())
  })

  // Part 8: "Years Insured" rename
  describe('Part 8: Policy Tenure → Years Insured', () => {
    it('shows "Years Insured" label', () => {
      expect(screen.getByText('Years Insured')).toBeInTheDocument()
    })
    it('does NOT show "Policy Tenure" anywhere', () => {
      expect(screen.queryByText(/Policy Tenure/i)).not.toBeInTheDocument()
    })
  })

  // Part 2: Conditional months-since-last-claim
  describe('Part 2: Conditional claim history', () => {
    it('shows "No previous claims recorded" banner when Previous Claims = 0 (default)', () => {
      expect(screen.getByText('No previous claims recorded')).toBeInTheDocument()
    })

    it('does NOT show Months Since Last Claim input when Previous Claims = 0', () => {
      expect(screen.queryByLabelText('Months since last claim')).not.toBeInTheDocument()
    })

    it('shows Months Since Last Claim input when Previous Claims > 0', async () => {
      await userEvent.clear(screen.getByLabelText('Previous claims count'))
      await userEvent.type(screen.getByLabelText('Previous claims count'), '1')
      await waitFor(() => expect(screen.getByLabelText('Months since last claim')).toBeInTheDocument())
    })

    it('hides the "No previous claims" banner when Previous Claims > 0', async () => {
      await userEvent.clear(screen.getByLabelText('Previous claims count'))
      await userEvent.type(screen.getByLabelText('Previous claims count'), '2')
      await waitFor(() => expect(screen.queryByText('No previous claims recorded')).not.toBeInTheDocument())
    })
  })

  // Part 15: aria-labels step 1
  describe('Part 15: aria-labels on Step 1', () => {
    it('City has aria-label', () => expect(screen.getByLabelText('City')).toBeInTheDocument())
    it('Years licensed has aria-label', () => expect(screen.getByLabelText('Years licensed')).toBeInTheDocument())
    it('Previous claims has aria-label', () => expect(screen.getByLabelText('Previous claims count')).toBeInTheDocument())
    it('Years Insured field has aria-label', () => expect(screen.getByLabelText('Years insured')).toBeInTheDocument())
  })
})

// ─── Section 3: Wizard structure ──────────────────────────────────────────────

describe('QuoteWizardV2 — Wizard structure', () => {
  it('shows 5-step progress bar on mount', () => {
    render(<QuoteWizardV2 />, { wrapper: Wrapper })
    // "Customer Info" appears as both the progress label AND the step heading — getAllByText handles this
    expect(screen.getAllByText('Customer Info').length).toBeGreaterThanOrEqual(1)
    // The remaining step labels appear only in the progress bar (unique)
    expect(screen.getByText('Driving History')).toBeInTheDocument()
    expect(screen.getByText('Vehicle Details')).toBeInTheDocument()
    expect(screen.getByText('Review')).toBeInTheDocument()
    expect(screen.getByText('Your Quote')).toBeInTheDocument()
  })

  it('Back button does not appear on step 0', () => {
    render(<QuoteWizardV2 />, { wrapper: Wrapper })
    expect(screen.queryByRole('button', { name: /back/i })).not.toBeInTheDocument()
  })

  it('Back button appears after advancing to step 1', async () => {
    render(<QuoteWizardV2 />, { wrapper: Wrapper })
    await fillStep0()
    await waitFor(() => expect(screen.getByRole('button', { name: /back/i })).toBeInTheDocument())
  })

  it('step subtitle is customer-friendly (not technical)', () => {
    render(<QuoteWizardV2 />, { wrapper: Wrapper })
    expect(screen.getByText('Tell us about yourself')).toBeInTheDocument()
  })
})

// ─── Section 4: Data integrity ────────────────────────────────────────────────

describe('V4_MAKES_MODELS data coverage', () => {
  it('has 17 makes', async () => {
    const { V4_MAKES_MODELS } = await import('../../../types/v2')
    expect(Object.keys(V4_MAKES_MODELS)).toHaveLength(17)
  })

  it('has 35 total models', async () => {
    const { V4_MAKES_MODELS } = await import('../../../types/v2')
    const total = Object.values(V4_MAKES_MODELS).reduce((s, m) => s + m.length, 0)
    expect(total).toBe(35)
  })

  it('includes Mahindra with Scorpio and XUV700', async () => {
    const { V4_MAKES_MODELS } = await import('../../../types/v2')
    expect(V4_MAKES_MODELS['Mahindra']).toContain('Scorpio')
    expect(V4_MAKES_MODELS['Mahindra']).toContain('XUV700')
  })
})

describe('V4_CITIES coverage', () => {
  it('has 20 cities', async () => {
    const { V4_CITIES } = await import('../../../types/v2')
    expect(V4_CITIES).toHaveLength(20)
  })

  it('includes major metros', async () => {
    const { V4_CITIES } = await import('../../../types/v2')
    expect(V4_CITIES).toContain('Mumbai')
    expect(V4_CITIES).toContain('Delhi')
    expect(V4_CITIES).toContain('Bangalore')
    expect(V4_CITIES).toContain('Hyderabad')
  })
})

// ─── Section 5: API client endpoints ─────────────────────────────────────────

describe('API client configuration', () => {
  it('V1 api client uses /api/v1 baseURL', async () => {
    const { apiClient } = await import('../../../api/client')
    expect(apiClient.defaults.baseURL).toBe('/api/v1')
  })

  it('V2 api client uses /api/v2 baseURL', async () => {
    const { apiClientV2 } = await import('../../../api/clientV2')
    expect(apiClientV2.defaults.baseURL).toBe('/api/v2')
  })
})
