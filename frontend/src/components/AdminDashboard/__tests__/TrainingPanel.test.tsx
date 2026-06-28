import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import TrainingPanel from '../TrainingPanel'
import { insuranceApi } from '../../../api/insurance'
import type {
  TrainingRunListResponse,
  TrainingRun,
  CandidateModel,
  TrainingTriggerResponse,
} from '../../../types'

vi.mock('../../../api/insurance')

// ─── Fixtures ────────────────────────────────────────────────────────────────────

const freqCandidate: CandidateModel = {
  model_id: 'cand-freq-001',
  run_id: 'run-001',
  algorithm: 'xgboost',
  model_type: 'frequency',
  status: 'COMPLETED',
  started_at: '2026-06-25T10:00:00',
  completed_at: '2026-06-25T10:00:42',
  duration_seconds: 42.1,
  train_rows: 420,
  val_rows: 90,
  test_rows: 90,
  hyperparameters: { n_estimators: 300 },
  frequency_metrics: {
    roc_auc: 0.7812,
    pr_auc: 0.6543,
    precision: 0.72,
    recall: 0.68,
    f1: 0.70,
    brier_score: 0.14,
    threshold: 0.5,
    n_samples: 90,
    n_positive: 18,
    positive_rate: 0.2,
  },
  severity_metrics: null,
  feature_importance: { age: 0.12, vehicle_age: 0.09 },
  artifact_path: 'models/challengers/run_20260625_100000_abc123/xgboost/frequency',
  error_message: null,
}

const sevCandidate: CandidateModel = {
  model_id: 'cand-sev-001',
  run_id: 'run-001',
  algorithm: 'catboost',
  model_type: 'severity',
  status: 'COMPLETED',
  started_at: '2026-06-25T10:00:43',
  completed_at: '2026-06-25T10:01:20',
  duration_seconds: 37.5,
  train_rows: 84,
  val_rows: 18,
  test_rows: 18,
  hyperparameters: { iterations: 500 },
  frequency_metrics: null,
  severity_metrics: {
    rmse: 12500.5,
    mae: 9800.2,
    r2: 0.623,
    mape_pct: 22.4,
    n_samples: 18,
    mean_actual: 45000,
    mean_predicted: 44200,
  },
  feature_importance: { claim_amount: 0.22 },
  artifact_path: 'models/challengers/run_20260625_100000_abc123/catboost/severity',
  error_message: null,
}

const failedCandidate: CandidateModel = {
  model_id: 'cand-fail-001',
  run_id: 'run-002',
  algorithm: 'lightgbm',
  model_type: 'frequency',
  status: 'FAILED',
  started_at: '2026-06-25T09:00:00',
  completed_at: '2026-06-25T09:00:05',
  duration_seconds: 5.0,
  train_rows: null,
  val_rows: null,
  test_rows: null,
  hyperparameters: {},
  frequency_metrics: null,
  severity_metrics: null,
  feature_importance: null,
  artifact_path: null,
  error_message: 'Insufficient positive samples for stratified split',
}

const completedRun: TrainingRun = {
  run_id: 'run-001',
  run_tag: 'v5_run_20260625_100000_abc123',
  triggered_by: 'admin',
  triggered_at: '2026-06-25T10:00:00',
  completed_at: '2026-06-25T10:01:30',
  status: 'COMPLETED',
  dataset_version: 'v4_dataset_20260625',
  dataset_rows: 600,
  feature_count: 59,
  maturity_days: 90,
  train_rows: 420,
  val_rows: 90,
  test_rows: 90,
  error_message: null,
  summary: {
    best_frequency_model: { algorithm: 'xgboost', roc_auc: 0.7812 },
    best_severity_model: { algorithm: 'catboost', r2: 0.623 },
    governance_note: 'Promotion disabled — Phase 5B required',
    total_candidates: 2,
    completed_candidates: 2,
  },
  candidates: [freqCandidate, sevCandidate],
}

const failedRun: TrainingRun = {
  run_id: 'run-002',
  run_tag: 'v5_run_20260625_090000_def456',
  triggered_by: 'admin',
  triggered_at: '2026-06-25T09:00:00',
  completed_at: '2026-06-25T09:00:05',
  status: 'FAILED',
  dataset_version: null,
  dataset_rows: 600,
  feature_count: 59,
  maturity_days: 90,
  train_rows: null,
  val_rows: null,
  test_rows: null,
  error_message: 'Insufficient positive samples for stratified split',
  summary: null,
  candidates: [failedCandidate],
}

const runningRun: TrainingRun = {
  run_id: 'run-003',
  run_tag: 'v5_run_20260625_110000_ghi789',
  triggered_by: 'admin',
  triggered_at: '2026-06-25T11:00:00',
  completed_at: null,
  status: 'RUNNING',
  dataset_version: null,
  dataset_rows: null,
  feature_count: null,
  maturity_days: 90,
  train_rows: null,
  val_rows: null,
  test_rows: null,
  error_message: null,
  summary: null,
  candidates: [],
}

const emptyList: TrainingRunListResponse = {
  items: [],
  total: 0,
  page: 1,
  page_size: 20,
  total_pages: 0,
}

const listWithRuns: TrainingRunListResponse = {
  items: [completedRun, failedRun, runningRun],
  total: 3,
  page: 1,
  page_size: 20,
  total_pages: 1,
}

const triggerResponse: TrainingTriggerResponse = {
  run_id: 'run-new-001',
  run_tag: 'v5_run_20260625_120000_jkl000',
  status: 'RUNNING',
  message: 'V5 training run started in background',
  triggered_by: 'admin',
  triggered_at: '2026-06-25T12:00:00',
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

describe('TrainingPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // ─── Loading state ────────────────────────────────────────────────────────────

  it('shows loading spinner while fetching runs', () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockReturnValue(new Promise(() => {}))
    render(<TrainingPanel />)
    expect(screen.getByTestId('training-loading')).toBeInTheDocument()
  })

  // ─── Empty state ──────────────────────────────────────────────────────────────

  it('shows empty state when no training runs exist', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    render(<TrainingPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('training-empty')).toBeInTheDocument()
    })
    expect(screen.getByTestId('training-empty')).toHaveTextContent('No training runs yet')
  })

  // ─── Governance notice ────────────────────────────────────────────────────────

  it('displays governance notice on load', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    render(<TrainingPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('training-governance-notice')).toBeInTheDocument()
    })
    expect(screen.getByTestId('training-governance-notice')).toHaveTextContent(
      'challenger models only',
    )
  })

  it('governance notice contains no promote/activate buttons', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('training-panel')).toBeInTheDocument())
    const buttons = screen.getAllByRole('button')
    const buttonTexts = buttons.map(b => b.textContent?.toLowerCase() ?? '')
    expect(buttonTexts.some(t => /^promot|^activat/.test(t))).toBe(false)
  })

  // ─── Run list table ───────────────────────────────────────────────────────────

  it('renders all run rows from the API', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    render(<TrainingPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('training-runs-table')).toBeInTheDocument()
    })
    const rows = screen.getAllByTestId('training-run-row')
    expect(rows).toHaveLength(3)
  })

  it('shows run tag in mono font', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('training-run-row')).toHaveLength(3))
    expect(screen.getByText('v5_run_20260625_100000_abc123')).toBeInTheDocument()
  })

  it('shows correct status chips for COMPLETED, FAILED, RUNNING', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('training-run-row')).toHaveLength(3))
    expect(screen.getByText('COMPLETED')).toBeInTheDocument()
    expect(screen.getByText('FAILED')).toBeInTheDocument()
    expect(screen.getByText('RUNNING')).toBeInTheDocument()
  })

  it('shows running duration for in-progress runs', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('training-run-row')).toHaveLength(3))
    expect(screen.getByText('running…')).toBeInTheDocument()
  })

  // ─── Error state ──────────────────────────────────────────────────────────────

  it('shows error state when listTrainingRuns rejects', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockRejectedValue(new Error('Network error'))
    render(<TrainingPanel />)
    await waitFor(() => {
      expect(screen.getByTestId('training-error')).toBeInTheDocument()
    })
    expect(screen.getByTestId('training-error')).toHaveTextContent('Network error')
  })

  it('retry button calls listTrainingRuns again', async () => {
    vi.mocked(insuranceApi.listTrainingRuns)
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(emptyList)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('training-error')).toBeInTheDocument())
    fireEvent.click(screen.getByText('Retry'))
    await waitFor(() => expect(screen.getByTestId('training-empty')).toBeInTheDocument())
    expect(insuranceApi.listTrainingRuns).toHaveBeenCalledTimes(2)
  })

  // ─── Trigger training ─────────────────────────────────────────────────────────

  it('trigger button calls triggerTrainingRun with "admin"', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.triggerTrainingRun).mockResolvedValue(triggerResponse)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('trigger-training-btn')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('trigger-training-btn'))
    await waitFor(() => {
      expect(insuranceApi.triggerTrainingRun).toHaveBeenCalledWith('admin')
    })
  })

  it('shows success message after trigger', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.triggerTrainingRun).mockResolvedValue(triggerResponse)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('trigger-training-btn')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('trigger-training-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('trigger-message')).toBeInTheDocument()
    })
    expect(screen.getByTestId('trigger-message')).toHaveTextContent('v5_run_20260625_120000_jkl000')
  })

  it('shows error message when trigger fails', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    vi.mocked(insuranceApi.triggerTrainingRun).mockRejectedValue(new Error('Dataset not ready'))
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('trigger-training-btn')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('trigger-training-btn'))
    await waitFor(() => expect(screen.getByTestId('trigger-message')).toBeInTheDocument())
    expect(screen.getByTestId('trigger-message')).toHaveTextContent('Dataset not ready')
  })

  // ─── Run detail view ──────────────────────────────────────────────────────────

  it('clicking View loads and shows run detail panel', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => {
      expect(screen.getByTestId('run-detail-panel')).toBeInTheDocument()
    })
    expect(insuranceApi.getTrainingRun).toHaveBeenCalledWith('run-001')
  })

  it('run detail shows run tag, status, and dataset info', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getByTestId('run-detail-panel')).toBeInTheDocument())
    expect(screen.getByText('v5_run_20260625_100000_abc123')).toBeInTheDocument()
    expect(screen.getByText('600')).toBeInTheDocument()
    expect(screen.getByText('59')).toBeInTheDocument()
  })

  it('run detail shows governance summary note', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getByTestId('run-detail-panel')).toBeInTheDocument())
    expect(screen.getByText('Promotion disabled — Phase 5B required')).toBeInTheDocument()
  })

  it('run detail shows all candidate rows', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-row')).toHaveLength(2))
  })

  it('back button returns to run list', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getByTestId('run-detail-panel')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('run-detail-back'))
    await waitFor(() => expect(screen.getByTestId('training-runs-table')).toBeInTheDocument())
  })

  // ─── Candidate detail drawer ──────────────────────────────────────────────────

  it('clicking candidate Details opens drawer', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-detail-btn')).toHaveLength(2))
    fireEvent.click(screen.getAllByTestId('candidate-detail-btn')[0])
    expect(screen.getByTestId('candidate-detail-drawer')).toBeInTheDocument()
  })

  it('frequency candidate drawer shows freq metrics', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-detail-btn')).toHaveLength(2))
    fireEvent.click(screen.getAllByTestId('candidate-detail-btn')[0])
    expect(screen.getByTestId('freq-metrics')).toBeInTheDocument()
    expect(screen.getByTestId('freq-metrics')).toHaveTextContent('0.7812')
  })

  it('severity candidate drawer shows severity metrics', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-detail-btn')).toHaveLength(2))
    fireEvent.click(screen.getAllByTestId('candidate-detail-btn')[1])
    expect(screen.getByTestId('sev-metrics')).toBeInTheDocument()
    expect(screen.getByTestId('sev-metrics')).toHaveTextContent('0.6230')
  })

  it('drawer close button dismisses drawer', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-detail-btn')).toHaveLength(2))
    fireEvent.click(screen.getAllByTestId('candidate-detail-btn')[0])
    expect(screen.getByTestId('candidate-detail-drawer')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId('candidate-drawer-close'))
    expect(screen.queryByTestId('candidate-detail-drawer')).not.toBeInTheDocument()
  })

  it('feature importance list renders top features', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-detail-btn')).toHaveLength(2))
    fireEvent.click(screen.getAllByTestId('candidate-detail-btn')[0])
    expect(screen.getByTestId('feature-importance-list')).toBeInTheDocument()
    expect(screen.getByText('age')).toBeInTheDocument()
  })

  // ─── No promotion/activate buttons anywhere ────────────────────────────────────

  it('does NOT render any Promote button at any level', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('training-panel')).toBeInTheDocument())
    const buttons = screen.getAllByRole('button')
    const hasPromoteBtn = buttons.some(b => /^promot/i.test(b.textContent ?? ''))
    expect(hasPromoteBtn).toBe(false)
  })

  it('does NOT render any Activate button at any level', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('training-panel')).toBeInTheDocument())
    const buttons = screen.getAllByRole('button')
    const hasActivateBtn = buttons.some(b => /^activat/i.test(b.textContent ?? ''))
    expect(hasActivateBtn).toBe(false)
  })

  // ─── Algorithm badges ─────────────────────────────────────────────────────────

  it('renders algorithm badges for each algorithm type', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(listWithRuns)
    vi.mocked(insuranceApi.getTrainingRun).mockResolvedValue(completedRun)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getAllByTestId('view-run-btn')).toHaveLength(3))
    fireEvent.click(screen.getAllByTestId('view-run-btn')[0])
    await waitFor(() => expect(screen.getAllByTestId('candidate-row')).toHaveLength(2))
    expect(screen.getAllByText('xgboost').length).toBeGreaterThan(0)
    expect(screen.getAllByText('catboost').length).toBeGreaterThan(0)
  })

  // ─── Refresh button ───────────────────────────────────────────────────────────

  it('refresh button reloads the run list', async () => {
    vi.mocked(insuranceApi.listTrainingRuns).mockResolvedValue(emptyList)
    render(<TrainingPanel />)
    await waitFor(() => expect(screen.getByTestId('training-refresh-btn')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('training-refresh-btn'))
    await waitFor(() => {
      expect(insuranceApi.listTrainingRuns).toHaveBeenCalledTimes(2)
    })
  })
})
