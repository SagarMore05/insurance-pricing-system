import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ApprovalPanel from '../ApprovalPanel'
import type {
  ApprovalRequest,
  ApprovalRequestDetail,
  ApprovalRequestListResponse,
  ApprovalAuditLogEntry,
  ModelCard,
} from '../../../types'

vi.mock('../../../api/insurance', () => ({
  insuranceApi: {
    listApprovalRequests: vi.fn(),
    getApprovalRequest: vi.fn(),
    createApprovalRequest: vi.fn(),
    approveApprovalRequest: vi.fn(),
    rejectApprovalRequest: vi.fn(),
    getApprovalAuditLog: vi.fn(),
  },
}))

import { insuranceApi } from '../../../api/insurance'

// ─── Fixtures ─────────────────────────────────────────────────────────────────

const MOCK_CARD: ModelCard = {
  model_type: 'frequency',
  model_version: 'v4.0.0',
  algorithm: 'XGBoost',
  training_date: '2026-06-22',
  dataset_size: 50000,
  preprocessor_version: 'V4',
  champion_since: '2026-06-22',
  frequency_metrics: { roc_auc: 0.6897, pr_auc: 0.31, f1: 0.28 },
  severity_metrics: null,
  feature_importance_summary: { driving_score: 0.1843, age: 0.0874 },
  drift_summary: {
    computed_at: '2026-06-25T12:00:00',
    overall_psi: 0.05,
    overall_severity: 'low',
    high_drift_features: [],
    medium_drift_features: [],
    features_computed: 18,
    recommendation: 'Monitor',
  },
  recommendation: 'Monitor — Stable distribution',
}

const PENDING_REQUEST: ApprovalRequest = {
  request_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
  model_type: 'frequency',
  model_version: 'v4.0.0',
  status: 'pending',
  submitted_by: 'admin',
  submitted_at: '2026-06-25T10:00:00',
  reviewed_by: null,
  reviewed_at: null,
  reviewer_note: null,
  model_card: MOCK_CARD,
  recommendation: 'Monitor — Stable distribution',
}

const APPROVED_REQUEST: ApprovalRequest = {
  ...PENDING_REQUEST,
  request_id: 'ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj',
  status: 'approved',
  reviewed_by: 'admin',
  reviewed_at: '2026-06-25T11:00:00',
  reviewer_note: 'LGTM',
}

const AUDIT_LOGS: ApprovalAuditLogEntry[] = [
  {
    log_id: 'log-001',
    request_id: PENDING_REQUEST.request_id,
    event_type: 'created',
    actor: 'admin',
    note: null,
    event_data: null,
    created_at: '2026-06-25T10:00:00',
  },
]

const PENDING_DETAIL: ApprovalRequestDetail = {
  ...PENDING_REQUEST,
  audit_logs: AUDIT_LOGS,
}

const EMPTY_LIST: ApprovalRequestListResponse = {
  items: [],
  total: 0,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

const PENDING_LIST: ApprovalRequestListResponse = {
  items: [PENDING_REQUEST],
  total: 1,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

const HISTORY_LIST: ApprovalRequestListResponse = {
  items: [APPROVED_REQUEST],
  total: 1,
  page: 1,
  page_size: 15,
  total_pages: 1,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
}

function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={makeQueryClient()}>
      {children}
    </QueryClientProvider>
  )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('ApprovalPanel — loading state', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockReturnValue(new Promise(() => {}))
  })

  it('renders loading skeleton while pending', () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(screen.getByTestId('approval-loading')).toBeInTheDocument()
  })
})

describe('ApprovalPanel — empty state', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockResolvedValue(EMPTY_LIST)
  })

  it('renders approval panel container', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByTestId('approval-panel')).toBeInTheDocument()
  })

  it('shows empty state message for pending', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByTestId('approval-empty')).toBeInTheDocument()
    expect(await screen.findByText(/No pending approval requests/)).toBeInTheDocument()
  })

  it('shows the new request button', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByTestId('new-request-btn')).toBeInTheDocument()
  })

  it('shows governance safety notice', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText(/Governance only/)).toBeInTheDocument()
  })

  it('shows Model Approval Workflow heading', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText('Model Approval Workflow')).toBeInTheDocument()
  })

  it('shows Pending and History tab buttons', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText('Pending')).toBeInTheDocument()
    expect(await screen.findByText('History')).toBeInTheDocument()
  })
})

describe('ApprovalPanel — with pending requests', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockResolvedValue(PENDING_LIST)
    vi.mocked(insuranceApi.getApprovalRequest).mockResolvedValue(PENDING_DETAIL)
  })

  it('renders approval row', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByTestId('approval-row')).toBeInTheDocument()
  })

  it('shows model version in row', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText('v4.0.0')).toBeInTheDocument()
  })

  it('shows pending status badge', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText('pending')).toBeInTheDocument()
  })

  it('shows Approve and Reject buttons for pending request', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByTestId('approve-btn')).toBeInTheDocument()
    expect(await screen.findByTestId('reject-btn')).toBeInTheDocument()
  })

  it('shows Frequency model type label', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    expect(await screen.findByText('Frequency')).toBeInTheDocument()
  })

  it('shows pending count badge on tab', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    // pending count badge should appear (value 1)
    const badge = await screen.findByText('1')
    expect(badge).toBeInTheDocument()
  })
})

describe('ApprovalPanel — approve flow', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockResolvedValue(PENDING_LIST)
    vi.mocked(insuranceApi.getApprovalRequest).mockResolvedValue(PENDING_DETAIL)
  })

  it('opens review dialog on Approve click', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    const btn = await screen.findByTestId('approve-btn')
    fireEvent.click(btn)
    expect(await screen.findByTestId('review-dialog')).toBeInTheDocument()
  })

  it('shows approve confirmation text in dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    const btn = await screen.findByTestId('approve-btn')
    fireEvent.click(btn)
    expect(await screen.findByText('Approve Request')).toBeInTheDocument()
  })

  it('shows governance disclaimer in approve dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('approve-btn'))
    expect(await screen.findByText(/No model promotion/)).toBeInTheDocument()
  })

  it('shows Confirm Approve button in dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('approve-btn'))
    expect(await screen.findByTestId('confirm-review-btn')).toBeInTheDocument()
    expect(screen.getByText('Confirm Approve')).toBeInTheDocument()
  })

  it('closes dialog on Cancel', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('approve-btn'))
    await screen.findByTestId('review-dialog')
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByTestId('review-dialog')).not.toBeInTheDocument()
  })
})

describe('ApprovalPanel — reject flow', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockResolvedValue(PENDING_LIST)
    vi.mocked(insuranceApi.getApprovalRequest).mockResolvedValue(PENDING_DETAIL)
  })

  it('opens review dialog on Reject click', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('reject-btn'))
    expect(await screen.findByTestId('review-dialog')).toBeInTheDocument()
  })

  it('shows Reject Request title in dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('reject-btn'))
    expect(await screen.findByText('Reject Request')).toBeInTheDocument()
  })

  it('shows Confirm Reject button', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('reject-btn'))
    await screen.findByTestId('review-dialog')
    expect(screen.getByText('Confirm Reject')).toBeInTheDocument()
  })
})

describe('ApprovalPanel — create request dialog', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockResolvedValue(EMPTY_LIST)
  })

  it('opens create dialog on New Request click', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('new-request-btn'))
    expect(await screen.findByTestId('create-dialog')).toBeInTheDocument()
  })

  it('shows model type selector in create dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('new-request-btn'))
    expect(await screen.findByTestId('model-type-select')).toBeInTheDocument()
  })

  it('shows Submit Request button in create dialog', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('new-request-btn'))
    expect(await screen.findByTestId('submit-request-btn')).toBeInTheDocument()
  })

  it('closes create dialog on Cancel', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByTestId('new-request-btn'))
    await screen.findByTestId('create-dialog')
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByTestId('create-dialog')).not.toBeInTheDocument()
  })
})

describe('ApprovalPanel — history view', () => {
  beforeEach(() => {
    vi.mocked(insuranceApi.listApprovalRequests).mockImplementation((params) => {
      if (!params?.status) {
        return Promise.resolve(HISTORY_LIST)
      }
      return Promise.resolve(EMPTY_LIST)
    })
  })

  it('switches to history view on History tab click', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByText('History'))
    expect(await screen.findByText('approved')).toBeInTheDocument()
  })

  it('shows no Approve/Reject buttons for approved items', async () => {
    render(<ApprovalPanel />, { wrapper: Wrapper })
    fireEvent.click(await screen.findByText('History'))
    await screen.findByText('approved')
    expect(screen.queryByTestId('approve-btn')).not.toBeInTheDocument()
    expect(screen.queryByTestId('reject-btn')).not.toBeInTheDocument()
  })
})
