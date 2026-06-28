import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import ShadowPanel from '../ShadowPanel'
import { insuranceApi } from '../../../api/insurance'
import type {
  ShadowStatusResponse,
  ShadowStatisticsResponse,
  ShadowHistoryResponse,
  ShadowPredictionRecord,
} from '../../../types'

vi.mock('../../../api/insurance')

// ─── Test fixtures ──────────────────────────────────────────────────────────────

const mockStatus: ShadowStatusResponse = {
  framework_status: 'OPERATIONAL',
  challenger_available: false,
  challenger_status: 'UNAVAILABLE',
  total_shadow_predictions: 42,
  waiting_for_challenger: 42,
  completed_comparisons: 0,
  failed_comparisons: 0,
  champion_model: 'XGBoost_V4 + CatBoost_V4',
  champion_version: 'v4.0.0',
  last_recorded_at: '2026-06-25T10:00:00',
  message: 'Shadow framework is operational. Recording all V2 predictions. Awaiting V5 challenger model registration.',
}

const mockStats: ShadowStatisticsResponse = {
  window_days: 30,
  total_shadow_predictions: 42,
  completed_comparisons: 0,
  waiting_for_challenger: 42,
  completion_rate_pct: 0.0,
  avg_premium_difference_inr: null,
  avg_prediction_latency_ms: null,
  challenger_available: false,
}

const mockRecord: ShadowPredictionRecord = {
  id: 'rec-uuid-1',
  request_id: 'req-abc-123',
  quote_id: 'quote-uuid-1',
  created_at: '2026-06-25T09:30:00',
  champion_model_name: 'XGBoost_V4+CatBoost_V4',
  champion_version: 'v4.0.0',
  challenger_model_name: null,
  challenger_version: null,
  champion_prediction_json: { final_premium_inr: 45000, risk_level: 'low' },
  challenger_prediction_json: null,
  premium_difference: null,
  risk_difference: null,
  prediction_latency_ms: 87,
  status: 'WAITING_FOR_CHALLENGER',
  comparison_result: null,
  notes: 'Framework operational — awaiting V5 challenger registration.',
}

const mockHistory: ShadowHistoryResponse = {
  items: [mockRecord],
  total: 1,
  page: 1,
  page_size: 20,
  total_pages: 1,
}

const emptyHistory: ShadowHistoryResponse = {
  items: [],
  total: 0,
  page: 1,
  page_size: 20,
  total_pages: 1,
}

function setupMocks(historyOverride?: ShadowHistoryResponse) {
  vi.mocked(insuranceApi.getShadowStatus).mockResolvedValue(mockStatus)
  vi.mocked(insuranceApi.getShadowStatistics).mockResolvedValue(mockStats)
  vi.mocked(insuranceApi.getShadowHistory).mockResolvedValue(historyOverride ?? mockHistory)
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

describe('ShadowPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading spinner initially', () => {
    setupMocks()
    render(<ShadowPanel />)
    expect(screen.getByTestId('shadow-loading')).toBeInTheDocument()
  })

  it('renders panel after data loads', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-panel')).toBeInTheDocument())
  })

  it('shows framework banner with OPERATIONAL status', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-framework-banner')).toBeInTheDocument())
    expect(screen.getByText(/OPERATIONAL/)).toBeInTheDocument()
  })

  it('shows challenger status as UNAVAILABLE', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-framework-banner')).toBeInTheDocument())
    expect(screen.getByText(/UNAVAILABLE/)).toBeInTheDocument()
  })

  it('shows governance alert when no challenger', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-governance-alert')).toBeInTheDocument())
    expect(screen.getByText(/No V5 challenger/i)).toBeInTheDocument()
  })

  it('shows status overview with correct prediction count', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-status-overview')).toBeInTheDocument())
    // 42 appears in both overview cards and stats — use getAllByText
    const items = screen.getAllByText('42')
    expect(items.length).toBeGreaterThan(0)
  })

  it('shows statistics panel', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-statistics')).toBeInTheDocument())
    expect(screen.getByText(/Last 30 Days/i)).toBeInTheDocument()
  })

  it('shows completion rate as 0% when no challenger', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-statistics')).toBeInTheDocument())
    expect(screen.getByText('0%')).toBeInTheDocument()
  })

  it('shows avg premium diff as — when null', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-statistics')).toBeInTheDocument())
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThan(0)
  })

  it('renders history table', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-table')).toBeInTheDocument())
  })

  it('shows WAITING FOR CHALLENGER status chip in history', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    // Status chip renders status with underscores replaced by spaces (exact casing)
    const chips = screen.getAllByText('WAITING FOR CHALLENGER')
    expect(chips.length).toBeGreaterThan(0)
  })

  it('shows empty message when no history', async () => {
    setupMocks(emptyHistory)
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-empty')).toBeInTheDocument())
    expect(screen.getByText(/No predictions recorded yet/i)).toBeInTheDocument()
  })

  it('opens detail drawer on row click', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-history-row'))
    expect(screen.getByTestId('shadow-detail-drawer')).toBeInTheDocument()
  })

  it('opens detail drawer on detail button click', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-detail-btn')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-detail-btn'))
    expect(screen.getByTestId('shadow-detail-drawer')).toBeInTheDocument()
  })

  it('detail drawer shows champion prediction JSON', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-history-row'))
    expect(screen.getByTestId('shadow-champion-json')).toBeInTheDocument()
    expect(screen.getByTestId('shadow-champion-json').textContent).toContain('45000')
  })

  it('closes detail drawer on close button', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-history-row'))
    expect(screen.getByTestId('shadow-detail-drawer')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('shadow-detail-close'))
    expect(screen.queryByTestId('shadow-detail-drawer')).not.toBeInTheDocument()
  })

  it('status filter triggers re-fetch', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-status-filter')).toBeInTheDocument())
    fireEvent.change(screen.getByTestId('shadow-status-filter'), {
      target: { value: 'WAITING_FOR_CHALLENGER' },
    })
    await waitFor(() => expect(insuranceApi.getShadowHistory).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'WAITING_FOR_CHALLENGER' }),
    ))
  })

  it('window day selector changes stats request', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-window-select')).toBeInTheDocument())
    fireEvent.change(screen.getByTestId('shadow-window-select'), {
      target: { value: '7' },
    })
    await waitFor(() => expect(insuranceApi.getShadowStatistics).toHaveBeenCalledWith(7))
  })

  it('refresh button reloads data', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-refresh-btn')).toBeInTheDocument())
    vi.clearAllMocks()
    setupMocks()
    fireEvent.click(screen.getByTestId('shadow-refresh-btn'))
    await waitFor(() => expect(insuranceApi.getShadowStatus).toHaveBeenCalledTimes(1))
  })

  it('shows error state on API failure', async () => {
    vi.mocked(insuranceApi.getShadowStatus).mockRejectedValue(new Error('Network error'))
    vi.mocked(insuranceApi.getShadowStatistics).mockRejectedValue(new Error('Network error'))
    vi.mocked(insuranceApi.getShadowHistory).mockRejectedValue(new Error('Network error'))
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-error')).toBeInTheDocument())
    expect(screen.getByText(/Network error/i)).toBeInTheDocument()
  })

  it('shows last_recorded_at when available', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-framework-banner')).toBeInTheDocument())
    // Last recorded_at is formatted via toLocaleString — just verify it rendered
    expect(screen.getByText(/Last recorded:/i)).toBeInTheDocument()
  })

  it('shows champion model in banner', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-framework-banner')).toBeInTheDocument())
    expect(screen.getByText(/XGBoost_V4/)).toBeInTheDocument()
  })

  it('no challenger does not render challenger JSON section in drawer', async () => {
    setupMocks()
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-history-row'))
    expect(screen.queryByTestId('shadow-challenger-json')).not.toBeInTheDocument()
  })
})

// ─── With completed comparison ────────────────────────────────────────────────

describe('ShadowPanel — with challenger comparison', () => {
  const completedRecord: ShadowPredictionRecord = {
    id: 'rec-uuid-2',
    request_id: 'req-xyz-456',
    quote_id: 'quote-uuid-2',
    created_at: '2026-06-25T11:00:00',
    champion_model_name: 'XGBoost_V4+CatBoost_V4',
    champion_version: 'v4.0.0',
    challenger_model_name: 'XGBoost_V5+CatBoost_V5',
    challenger_version: 'v5.0.0',
    champion_prediction_json: { final_premium_inr: 50000, risk_level: 'medium' },
    challenger_prediction_json: { final_premium_inr: 55000, risk_level: 'high' },
    premium_difference: 5000,
    risk_difference: 'medium→high',
    prediction_latency_ms: 142,
    status: 'COMPLETED',
    comparison_result: 'CHALLENGER_HIGHER',
    notes: 'Shadow comparison completed. Result: CHALLENGER_HIGHER',
  }

  const completedHistory: ShadowHistoryResponse = {
    items: [completedRecord],
    total: 1,
    page: 1,
    page_size: 20,
    total_pages: 1,
  }

  const activeStatus: ShadowStatusResponse = {
    ...mockStatus,
    challenger_available: true,
    challenger_status: 'AVAILABLE',
    completed_comparisons: 1,
    waiting_for_challenger: 0,
    message: 'Shadow comparison active — challenger model available.',
  }

  const activeStats: ShadowStatisticsResponse = {
    ...mockStats,
    completed_comparisons: 1,
    waiting_for_challenger: 0,
    completion_rate_pct: 100.0,
    avg_premium_difference_inr: 5000,
    avg_prediction_latency_ms: 142,
    challenger_available: true,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(insuranceApi.getShadowStatus).mockResolvedValue(activeStatus)
    vi.mocked(insuranceApi.getShadowStatistics).mockResolvedValue(activeStats)
    vi.mocked(insuranceApi.getShadowHistory).mockResolvedValue(completedHistory)
  })

  it('hides governance alert when challenger is available', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-panel')).toBeInTheDocument())
    expect(screen.queryByTestId('shadow-governance-alert')).not.toBeInTheDocument()
  })

  it('shows COMPLETED status chip', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    // Status chip shows exact uppercased text; card label shows "Completed Comparisons"
    const chips = screen.getAllByText('COMPLETED')
    expect(chips.length).toBeGreaterThan(0)
  })

  it('shows CHALLENGER HIGHER comparison chip', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    expect(screen.getByText(/CHALLENGER HIGHER/i)).toBeInTheDocument()
  })

  it('shows premium difference in history row', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    // ₹5000 appears in both history row and stats — verify at least one exists
    const items = screen.getAllByText('₹5000')
    expect(items.length).toBeGreaterThan(0)
  })

  it('detail drawer shows both champion and challenger JSON', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-history-row')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('shadow-history-row'))
    expect(screen.getByTestId('shadow-champion-json')).toBeInTheDocument()
    expect(screen.getByTestId('shadow-challenger-json')).toBeInTheDocument()
  })

  it('shows avg premium diff in statistics', async () => {
    render(<ShadowPanel />)
    await waitFor(() => expect(screen.getByTestId('shadow-statistics')).toBeInTheDocument())
    // ₹5000 appears in both history row and statistics panel
    const items = screen.getAllByText('₹5000')
    expect(items.length).toBeGreaterThan(0)
  })
})
