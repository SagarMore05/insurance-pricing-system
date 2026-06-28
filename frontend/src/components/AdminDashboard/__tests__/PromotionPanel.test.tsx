import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import PromotionPanel from '../PromotionPanel'
import type {
  PromotionDetail,
  PromotionListResponse,
  RollbackRecord,
} from '../../../types'

vi.mock('../../../api/insurance', () => ({
  insuranceApi: {
    listPromotions: vi.fn(),
    getPromotion: vi.fn(),
    executePromotion: vi.fn(),
    rollbackPromotion: vi.fn(),
    listRollbacks: vi.fn(),
  },
}))

import { insuranceApi } from '../../../api/insurance'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function renderPanel() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <PromotionPanel />
    </QueryClientProvider>,
  )
}

function shortId(id: string) {
  return `#${id.slice(-8).toUpperCase()}`
}

// ─── Fixtures ─────────────────────────────────────────────────────────────────

const APPROVED_PROMO: PromotionDetail = {
  promotion_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
  run_id: 'run-0000-0000-0000-000000000001',
  frequency_candidate_id: 'cand-0000-0000-0000-000000000001',
  severity_candidate_id: 'sev-00000-0000-0000-000000000001',
  status: 'APPROVED',
  all_gates_passed: true,
  evaluation_report: {
    gates_summary: {
      gates: {
        human_approval:   { gate: 'human_approval',   passed: true, reason: 'Approved.',  details: {} },
        shadow_completed: { gate: 'shadow_completed',  passed: true, reason: 'Bypass.',    details: {} },
        shadow_count:     { gate: 'shadow_count',      passed: true, reason: 'Bypass.',    details: {} },
        no_critical_drift:{ gate: 'no_critical_drift', passed: true, reason: 'No drift.',  details: {} },
        performance:      { gate: 'performance',       passed: true, reason: 'Better.',    details: {} },
        artifacts_exist:  { gate: 'artifacts_exist',   passed: true, reason: 'Found.',     details: {} },
        registry_valid:   { gate: 'registry_valid',    passed: true, reason: 'Valid.',     details: {} },
      },
    },
  },
  gates_passed: { human_approval: {} },
  gates_failed: {},
  old_frequency_champion: { champion: 'XGBoost_v1', score: 0.715 },
  old_severity_champion: null,
  new_frequency_champion: null,
  new_severity_champion: null,
  promoted_by: null,
  promoted_at: null,
  promotion_duration_seconds: null,
  backup_path: null,
  error_message: null,
  notes: 'Test notes.',
  created_at: '2026-06-25T10:00:00',
  updated_at: '2026-06-25T10:01:00',
  rollback_records: [],
}

const ACTIVE_PROMO: PromotionDetail = {
  ...APPROVED_PROMO,
  promotion_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
  status: 'ACTIVE',
  promoted_by: 'admin',
  promoted_at: '2026-06-25T10:02:00',
  promotion_duration_seconds: 4.5,
  backup_path: '/models/champion_backups/v20260625',
  new_frequency_champion: { champion: 'XGBoost_v20260625', score: 0.720 },
  new_severity_champion: { champion: 'CatBoost_v20260625', score: 0.680 },
}

const REJECTED_PROMO: PromotionDetail = {
  ...APPROVED_PROMO,
  promotion_id: 'cccccccc-cccc-cccc-cccc-cccccccccccc',
  status: 'REJECTED',
  all_gates_passed: false,
  gates_failed: {
    human_approval: { gate: 'human_approval', passed: false, reason: 'No approval_id.', details: {} },
  },
  evaluation_report: null,
}

const RB_RECORD: RollbackRecord = {
  rollback_id: 'rbrbrbr0-rbr0-rbr0-rbr0-rbr0rbr0rbr0',
  promotion_id: ACTIVE_PROMO.promotion_id,
  rollback_reason: 'health check failed after promotion',
  rollback_trigger: 'health_check',
  rollback_status: 'SUCCESS',
  rollback_duration_seconds: 1.2,
  rolled_back_by: 'system',
  rolled_back_at: '2026-06-25T10:30:00',
  error_message: null,
  restored_frequency_champion: { champion: 'XGBoost_v1' },
  restored_severity_champion: null,
}

const emptyList: PromotionListResponse = {
  items: [],
  total: 0,
  page: 1,
  page_size: 15,
  total_pages: 0,
}

const approvedOnlyList: PromotionListResponse = {
  items: [APPROVED_PROMO],
  total: 1,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

const activeOnlyList: PromotionListResponse = {
  items: [ACTIVE_PROMO],
  total: 1,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

const mixedList: PromotionListResponse = {
  items: [APPROVED_PROMO, ACTIVE_PROMO, REJECTED_PROMO],
  total: 3,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('PromotionPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // ── Sub-nav ───────────────────────────────────────────────────────────────

  it('renders three sub-nav buttons', () => {
    vi.mocked(insuranceApi.listPromotions).mockReturnValue(new Promise(() => {}))
    renderPanel()
    expect(screen.getByRole('button', { name: /Promotion Queue/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'History' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Rollback History' })).toBeInTheDocument()
  })

  it('defaults to Promotion Queue view showing queue heading', () => {
    vi.mocked(insuranceApi.listPromotions).mockReturnValue(new Promise(() => {}))
    renderPanel()
    expect(screen.getByText('APPROVED evaluations awaiting execution')).toBeInTheDocument()
  })

  it('switches to History view on click', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => {
      expect(screen.getByText('Promotion History')).toBeInTheDocument()
    })
  })

  it('switches to Rollback History view on click', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.listRollbacks).mockResolvedValue([])
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'Rollback History' }))
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Rollback History' })).toBeInTheDocument()
    })
  })

  // ── Queue view ────────────────────────────────────────────────────────────

  it('queue badge shows count when APPROVED items exist', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(mixedList)
    renderPanel()
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Promotion Queue (1)' })).toBeInTheDocument()
    })
  })

  it('queue shows only APPROVED promotion rows', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(mixedList)
    renderPanel()
    await waitFor(() => {
      expect(screen.getAllByTestId('promotion-row')).toHaveLength(1)
    })
    expect(screen.getByText(shortId(APPROVED_PROMO.promotion_id))).toBeInTheDocument()
  })

  it('queue shows empty message when no APPROVED items', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue({
      ...emptyList,
      items: [ACTIVE_PROMO, REJECTED_PROMO],
    })
    renderPanel()
    await waitFor(() => {
      expect(screen.getByText('No promotions found.')).toBeInTheDocument()
    })
  })

  // ── History view ──────────────────────────────────────────────────────────

  it('history shows all promotion status badges', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(mixedList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => {
      expect(screen.getAllByTestId('promotion-row')).toHaveLength(3)
    })
    expect(screen.getAllByText('APPROVED').length).toBeGreaterThan(0)
    expect(screen.getAllByText('ACTIVE').length).toBeGreaterThan(0)
    expect(screen.getAllByText('REJECTED').length).toBeGreaterThan(0)
  })

  it('history shows gates result column', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => {
      expect(screen.getAllByText('✓ All').length).toBeGreaterThan(0)
    })
  })

  it('history shows status filter dropdown', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })
    expect(screen.getByText('All statuses')).toBeInTheDocument()
  })

  // ── Rollback History ──────────────────────────────────────────────────────

  it('rollback history renders records', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.listRollbacks).mockResolvedValue([RB_RECORD])
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'Rollback History' }))
    await waitFor(() => {
      expect(screen.getByText('health_check')).toBeInTheDocument()
    })
    expect(screen.getByText('health check failed after promotion')).toBeInTheDocument()
  })

  it('rollback history shows empty message when no records', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.listRollbacks).mockResolvedValue([])
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'Rollback History' }))
    await waitFor(() => {
      expect(screen.getByText('No rollback events recorded.')).toBeInTheDocument()
    })
  })

  // ── Detail drawer ─────────────────────────────────────────────────────────

  it('clicking a queue row opens the detail drawer', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    renderPanel()
    await waitFor(() => {
      expect(screen.getAllByTestId('promotion-row')).toHaveLength(1)
    })
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByTestId('promotion-drawer')).toBeInTheDocument()
    })
    expect(screen.getByText('Promotion Detail')).toBeInTheDocument()
  })

  it('drawer shows governance gates section', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    renderPanel()
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByText('Governance Gates')).toBeInTheDocument()
    })
    expect(screen.getByText('human approval')).toBeInTheDocument()
  })

  it('APPROVED drawer shows Execute Promotion button', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    renderPanel()
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Execute Promotion' })).toBeInTheDocument()
    })
  })

  it('ACTIVE drawer shows Rollback Champion button', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(activeOnlyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument()
    })
  })

  it('REJECTED drawer does NOT show Execute or Rollback buttons', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue({
      ...emptyList,
      items: [REJECTED_PROMO],
      total: 1,
    })
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByTestId('promotion-drawer')).toBeInTheDocument()
    })
    expect(screen.queryByRole('button', { name: 'Execute Promotion' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Rollback Champion' })).not.toBeInTheDocument()
  })

  it('Close button removes drawer from DOM', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    renderPanel()
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => expect(screen.getByTestId('promotion-drawer')).toBeInTheDocument())
    fireEvent.click(screen.getByText('×'))
    await waitFor(() => {
      expect(screen.queryByTestId('promotion-drawer')).not.toBeInTheDocument()
    })
  })

  // ── Execute promotion ─────────────────────────────────────────────────────

  it('Execute Promotion calls executePromotion with correct id and "admin"', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(approvedOnlyList)
    vi.mocked(insuranceApi.executePromotion).mockResolvedValue(ACTIVE_PROMO)
    renderPanel()
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Execute Promotion' })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Execute Promotion' }))
    await waitFor(() => {
      expect(insuranceApi.executePromotion).toHaveBeenCalledWith(
        APPROVED_PROMO.promotion_id, 'admin',
      )
    })
  })

  // ── Rollback flow ─────────────────────────────────────────────────────────

  it('Rollback Champion button reveals reason input', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(activeOnlyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Rollback Champion' }))
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Rollback reason…')).toBeInTheDocument()
    })
  })

  it('Confirm Rollback calls rollbackPromotion with typed reason', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(activeOnlyList)
    vi.mocked(insuranceApi.rollbackPromotion).mockResolvedValue({
      ...ACTIVE_PROMO,
      status: 'ROLLED_BACK',
    })
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'Rollback Champion' }))
    await waitFor(() => expect(screen.getByPlaceholderText('Rollback reason…')).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText('Rollback reason…'), {
      target: { value: 'health metrics degraded' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Confirm Rollback' }))
    await waitFor(() => {
      expect(insuranceApi.rollbackPromotion).toHaveBeenCalledWith(
        ACTIVE_PROMO.promotion_id,
        'health metrics degraded',
        'admin',
      )
    })
  })

  it('Confirm Rollback is disabled when reason is empty', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(activeOnlyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'Rollback Champion' }))
    await waitFor(() => expect(screen.getByPlaceholderText('Rollback reason…')).toBeInTheDocument())

    const confirmBtn = screen.getByRole('button', { name: 'Confirm Rollback' })
    expect(confirmBtn).toBeDisabled()
    expect(insuranceApi.rollbackPromotion).not.toHaveBeenCalled()
  })

  it('Cancel hides rollback reason form and restores Rollback Champion button', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(activeOnlyList)
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'History' }))
    await waitFor(() => expect(screen.getAllByTestId('promotion-row')).toHaveLength(1))
    fireEvent.click(screen.getAllByTestId('promotion-row')[0])
    await waitFor(() => expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: 'Rollback Champion' }))
    await waitFor(() => expect(screen.getByPlaceholderText('Rollback reason…')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByPlaceholderText('Rollback reason…')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Rollback Champion' })).toBeInTheDocument()
  })

  // ── API call patterns ─────────────────────────────────────────────────────

  it('listPromotions is called immediately on mount', () => {
    vi.mocked(insuranceApi.listPromotions).mockReturnValue(new Promise(() => {}))
    renderPanel()
    expect(insuranceApi.listPromotions).toHaveBeenCalledTimes(1)
  })

  it('listRollbacks is NOT called until Rollback History tab is selected', async () => {
    vi.mocked(insuranceApi.listPromotions).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.listRollbacks).mockResolvedValue([])
    renderPanel()
    await waitFor(() => {
      expect(screen.getByText('No promotions found.')).toBeInTheDocument()
    })
    expect(insuranceApi.listRollbacks).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Rollback History' }))
    await waitFor(() => {
      expect(insuranceApi.listRollbacks).toHaveBeenCalledTimes(1)
    })
  })
})
