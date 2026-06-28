import React, { useEffect, useState, useCallback } from 'react'
import { insuranceApi } from '../../api/insurance'
import type {
  TrainingRun,
  TrainingRunListResponse,
  CandidateModel,
  FrequencyTrainingMetrics,
  SeverityTrainingMetrics,
} from '../../types'

// ─── Status chips ───────────────────────────────────────────────────────────────

function RunStatusChip({ status }: { status: string }) {
  const map: Record<string, string> = {
    RUNNING:           'bg-blue-100 text-blue-800',
    COMPLETED:         'bg-green-100 text-green-800',
    FAILED:            'bg-red-100 text-red-800',
    VALIDATION_FAILED: 'bg-orange-100 text-orange-800',
    CANCELLED:         'bg-gray-100 text-gray-600',
  }
  const cls = map[status] ?? 'bg-gray-100 text-gray-600'
  const label = status.replace(/_/g, ' ')
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-semibold rounded ${cls}`}>
      {label}
    </span>
  )
}

function CandidateStatusChip({ status }: { status: string }) {
  const map: Record<string, string> = {
    TRAINING:  'bg-blue-100 text-blue-800',
    COMPLETED: 'bg-green-100 text-green-800',
    FAILED:    'bg-red-100 text-red-800',
  }
  return (
    <span className={`inline-block px-1.5 py-0.5 text-xs font-medium rounded ${map[status] ?? 'bg-gray-100 text-gray-600'}`}>
      {status}
    </span>
  )
}

function AlgorithmBadge({ alg }: { alg: string }) {
  const map: Record<string, string> = {
    xgboost:  'bg-purple-100 text-purple-800',
    catboost: 'bg-indigo-100 text-indigo-800',
    lightgbm: 'bg-teal-100 text-teal-800',
  }
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-semibold rounded-full ${map[alg] ?? 'bg-gray-100 text-gray-700'}`}>
      {alg}
    </span>
  )
}

// ─── Metrics display ────────────────────────────────────────────────────────────

function FrequencyMetricsCard({ m }: { m: FrequencyTrainingMetrics }) {
  return (
    <div data-testid="freq-metrics" className="grid grid-cols-3 gap-2 text-xs">
      {[
        ['ROC-AUC', m.roc_auc?.toFixed(4)],
        ['PR-AUC',  m.pr_auc?.toFixed(4)],
        ['F1',      m.f1?.toFixed(4)],
        ['Precision', m.precision?.toFixed(4)],
        ['Recall',    m.recall?.toFixed(4)],
        ['Brier',     m.brier_score?.toFixed(4)],
      ].map(([label, val]) => (
        <div key={String(label)} className="bg-gray-50 rounded p-1.5">
          <p className="text-gray-400">{label}</p>
          <p className="font-bold text-gray-700">{val ?? '—'}</p>
        </div>
      ))}
    </div>
  )
}

function SeverityMetricsCard({ m }: { m: SeverityTrainingMetrics }) {
  return (
    <div data-testid="sev-metrics" className="grid grid-cols-2 gap-2 text-xs">
      {[
        ['R²',    m.r2?.toFixed(4)],
        ['RMSE',  m.rmse?.toLocaleString()],
        ['MAE',   m.mae?.toLocaleString()],
        ['MAPE%', m.mape_pct != null ? m.mape_pct.toFixed(1) + '%' : '—'],
      ].map(([label, val]) => (
        <div key={String(label)} className="bg-gray-50 rounded p-1.5">
          <p className="text-gray-400">{label}</p>
          <p className="font-bold text-gray-700">{val ?? '—'}</p>
        </div>
      ))}
    </div>
  )
}

// ─── Candidate row ──────────────────────────────────────────────────────────────

function CandidateRow({
  cand,
  onClick,
}: {
  cand: CandidateModel
  onClick: () => void
}) {
  const metric = cand.model_type === 'frequency'
    ? cand.frequency_metrics ? `AUC ${cand.frequency_metrics.roc_auc?.toFixed(3)}` : '—'
    : cand.severity_metrics ? `R² ${cand.severity_metrics.r2?.toFixed(3)}` : '—'

  return (
    <tr
      data-testid="candidate-row"
      className="hover:bg-gray-50 cursor-pointer"
      onClick={onClick}
    >
      <td className="px-3 py-2"><AlgorithmBadge alg={cand.algorithm} /></td>
      <td className="px-3 py-2 text-xs text-gray-600 capitalize">{cand.model_type}</td>
      <td className="px-3 py-2"><CandidateStatusChip status={cand.status} /></td>
      <td className="px-3 py-2 text-xs text-gray-700 font-mono">{metric}</td>
      <td className="px-3 py-2 text-xs text-gray-500">
        {cand.duration_seconds != null ? `${cand.duration_seconds.toFixed(1)}s` : '—'}
      </td>
      <td className="px-3 py-2 text-right">
        <button
          data-testid="candidate-detail-btn"
          className="text-xs text-blue-600 hover:underline"
          onClick={e => { e.stopPropagation(); onClick() }}
        >
          Details
        </button>
      </td>
    </tr>
  )
}

// ─── Candidate detail drawer ────────────────────────────────────────────────────

function CandidateDetailDrawer({
  cand,
  onClose,
}: {
  cand: CandidateModel
  onClose: () => void
}) {
  return (
    <div data-testid="candidate-detail-drawer" className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black bg-opacity-20" onClick={onClose} />
      <div className="relative bg-white w-full max-w-lg shadow-xl overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-800">
            Candidate: <AlgorithmBadge alg={cand.algorithm} /> — {cand.model_type}
          </h3>
          <button
            data-testid="candidate-drawer-close"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 text-xl"
          >
            ✕
          </button>
        </div>

        <dl className="space-y-3 text-sm">
          <div>
            <dt className="text-xs text-gray-500">Status</dt>
            <dd><CandidateStatusChip status={cand.status} /></dd>
          </div>
          {cand.duration_seconds != null && (
            <div>
              <dt className="text-xs text-gray-500">Training Duration</dt>
              <dd>{cand.duration_seconds.toFixed(2)}s</dd>
            </div>
          )}
          {cand.train_rows != null && (
            <div>
              <dt className="text-xs text-gray-500">Dataset Split</dt>
              <dd className="text-xs">
                Train: {cand.train_rows} · Val: {cand.val_rows} · Test: {cand.test_rows}
              </dd>
            </div>
          )}
          {cand.frequency_metrics && (
            <div>
              <dt className="text-xs text-gray-500 mb-1">Frequency Metrics</dt>
              <dd><FrequencyMetricsCard m={cand.frequency_metrics} /></dd>
            </div>
          )}
          {cand.severity_metrics && (
            <div>
              <dt className="text-xs text-gray-500 mb-1">Severity Metrics</dt>
              <dd><SeverityMetricsCard m={cand.severity_metrics} /></dd>
            </div>
          )}
          {cand.feature_importance && Object.keys(cand.feature_importance).length > 0 && (
            <div>
              <dt className="text-xs text-gray-500 mb-1">Top Feature Importances</dt>
              <dd data-testid="feature-importance-list">
                <div className="space-y-1">
                  {Object.entries(cand.feature_importance)
                    .slice(0, 10)
                    .map(([feat, val]) => (
                      <div key={feat} className="flex items-center gap-2 text-xs">
                        <span className="text-gray-600 w-40 truncate">{feat}</span>
                        <div className="flex-1 bg-gray-100 rounded h-1.5 overflow-hidden">
                          <div
                            className="bg-blue-500 h-full"
                            style={{ width: `${Math.min(val * 100 * 5, 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-gray-500 w-14 text-right">
                          {Number(val).toFixed(4)}
                        </span>
                      </div>
                    ))}
                </div>
              </dd>
            </div>
          )}
          {cand.artifact_path && (
            <div>
              <dt className="text-xs text-gray-500">Artifact Path</dt>
              <dd className="font-mono text-xs text-gray-600 break-all">{cand.artifact_path}</dd>
            </div>
          )}
          {cand.error_message && (
            <div>
              <dt className="text-xs text-gray-500">Error</dt>
              <dd className="text-red-600 text-xs">{cand.error_message}</dd>
            </div>
          )}
        </dl>
      </div>
    </div>
  )
}

// ─── Run detail panel ──────────────────────────────────────────────────────────

function RunDetailPanel({
  run,
  onBack,
}: {
  run: TrainingRun
  onBack: () => void
}) {
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateModel | null>(null)

  const summary = run.summary as Record<string, unknown> | null

  return (
    <div data-testid="run-detail-panel" className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          data-testid="run-detail-back"
          onClick={onBack}
          className="text-sm text-blue-600 hover:underline"
        >
          ← Back
        </button>
        <h3 className="font-semibold text-gray-800 font-mono text-sm">{run.run_tag}</h3>
        <RunStatusChip status={run.status} />
      </div>

      {/* Run info */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          ['Dataset Rows', run.dataset_rows ?? '—'],
          ['Train / Val / Test', run.train_rows != null ? `${run.train_rows} / ${run.val_rows} / ${run.test_rows}` : '—'],
          ['Features', run.feature_count ?? '—'],
          ['Duration',
            run.completed_at && run.triggered_at
              ? `${((new Date(run.completed_at).getTime() - new Date(run.triggered_at).getTime()) / 1000).toFixed(0)}s`
              : '—',
          ],
        ].map(([label, val]) => (
          <div key={String(label)} className="bg-white border border-gray-200 rounded p-3">
            <p className="text-xs text-gray-500">{label}</p>
            <p className="font-semibold text-gray-800 text-sm">{val}</p>
          </div>
        ))}
      </div>

      {/* Summary best models */}
      {summary && (
        <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm">
          <p className="font-semibold text-blue-800 mb-2">Best Challenger Models</p>
          {Boolean(summary.best_frequency_model) && (
            <p className="text-blue-700 text-xs">
              Frequency: <strong>{String((summary.best_frequency_model as Record<string, unknown>).algorithm)}</strong>
              {' — AUC '}
              {String((summary.best_frequency_model as Record<string, unknown>).roc_auc)}
            </p>
          )}
          {Boolean(summary.best_severity_model) && (
            <p className="text-blue-700 text-xs mt-1">
              Severity: <strong>{String((summary.best_severity_model as Record<string, unknown>).algorithm)}</strong>
              {' — R² '}
              {String((summary.best_severity_model as Record<string, unknown>).r2)}
            </p>
          )}
          {Boolean(summary.governance_note) && (
            <p className="text-gray-500 text-xs mt-2 italic">{String(summary.governance_note)}</p>
          )}
        </div>
      )}

      {run.error_message && (
        <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">
          <strong>Error:</strong> {run.error_message}
        </div>
      )}

      {/* Candidate table */}
      {run.candidates.length > 0 && (
        <div className="bg-white border border-gray-200 rounded">
          <p className="text-xs font-semibold text-gray-600 px-4 py-3 border-b border-gray-100">
            Candidate Models ({run.candidates.length})
          </p>
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
              <tr>
                <th className="px-3 py-2 text-left">Algorithm</th>
                <th className="px-3 py-2 text-left">Type</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Key Metric</th>
                <th className="px-3 py-2 text-left">Duration</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {run.candidates.map(cand => (
                <CandidateRow
                  key={cand.model_id}
                  cand={cand}
                  onClick={() => setSelectedCandidate(cand)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedCandidate && (
        <CandidateDetailDrawer
          cand={selectedCandidate}
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </div>
  )
}

// ─── Governance notice ──────────────────────────────────────────────────────────

function GovernanceNotice() {
  return (
    <div
      data-testid="training-governance-notice"
      className="bg-amber-50 border border-amber-200 rounded p-3 text-sm mb-4"
    >
      <strong>Governance:</strong> Training produces challenger models only.
      Champions are not modified. Promotion requires Phase 5B approval workflow.
      Scheduled retraining remains disabled.
    </div>
  )
}

// ─── Main TrainingPanel ─────────────────────────────────────────────────────────

export default function TrainingPanel() {
  const [list, setList] = useState<TrainingRunListResponse | null>(null)
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [triggering, setTriggering] = useState(false)
  const [triggerMsg, setTriggerMsg] = useState<string | null>(null)
  const [page, setPage] = useState(1)

  const loadList = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await insuranceApi.listTrainingRuns({ page, page_size: 20 })
      setList(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load training runs')
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => {
    void loadList()
  }, [loadList])

  const handleTrigger = async () => {
    setTriggering(true)
    setTriggerMsg(null)
    try {
      const resp = await insuranceApi.triggerTrainingRun('admin')
      setTriggerMsg(`Run started: ${resp.run_tag} (${resp.run_id})`)
      await loadList()
    } catch (err: unknown) {
      setTriggerMsg(err instanceof Error ? `Error: ${err.message}` : 'Failed to trigger run')
    } finally {
      setTriggering(false)
    }
  }

  const handleSelectRun = async (runId: string) => {
    try {
      const run = await insuranceApi.getTrainingRun(runId)
      setSelectedRun(run)
    } catch {
      setError('Failed to load run details')
    }
  }

  if (selectedRun) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <RunDetailPanel run={selectedRun} onBack={() => setSelectedRun(null)} />
      </div>
    )
  }

  if (loading) {
    return (
      <div data-testid="training-loading" className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        <span className="ml-3 text-gray-500">Loading training runs…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div data-testid="training-error" className="text-red-600 text-sm p-6">
        Error: {error}
        <button onClick={() => void loadList()} className="ml-4 text-blue-600 underline">
          Retry
        </button>
      </div>
    )
  }

  return (
    <div data-testid="training-panel" className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900">V5 Training Engine</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Phase 5A — Challenger model training. No champion modification.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            data-testid="training-refresh-btn"
            onClick={() => void loadList()}
            className="text-xs px-3 py-1.5 border border-gray-200 rounded hover:bg-gray-50"
          >
            Refresh
          </button>
          <button
            data-testid="trigger-training-btn"
            onClick={() => void handleTrigger()}
            disabled={triggering}
            className="text-xs px-4 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {triggering ? 'Starting…' : 'Run Training'}
          </button>
        </div>
      </div>

      <GovernanceNotice />

      {triggerMsg && (
        <div
          data-testid="trigger-message"
          className={`mb-4 p-3 rounded text-sm border ${
            triggerMsg.startsWith('Error')
              ? 'bg-red-50 border-red-200 text-red-700'
              : 'bg-green-50 border-green-200 text-green-700'
          }`}
        >
          {triggerMsg}
        </div>
      )}

      {!list || list.items.length === 0 ? (
        <div data-testid="training-empty" className="text-center text-gray-400 text-sm py-12">
          No training runs yet. Click <strong>Run Training</strong> to start a V5 challenger run.
        </div>
      ) : (
        <div data-testid="training-runs-table" className="bg-white border border-gray-200 rounded">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
              <tr>
                <th className="px-4 py-2 text-left">Run Tag</th>
                <th className="px-4 py-2 text-left">Status</th>
                <th className="px-4 py-2 text-left">Triggered</th>
                <th className="px-4 py-2 text-right">Rows</th>
                <th className="px-4 py-2 text-right">Candidates</th>
                <th className="px-4 py-2 text-right">Duration</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {list.items.map(run => {
                const duration =
                  run.completed_at && run.triggered_at
                    ? `${((new Date(run.completed_at).getTime() - new Date(run.triggered_at).getTime()) / 1000).toFixed(0)}s`
                    : run.status === 'RUNNING' ? 'running…' : '—'

                return (
                  <tr
                    key={run.run_id}
                    data-testid="training-run-row"
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => void handleSelectRun(run.run_id)}
                  >
                    <td className="px-4 py-2 font-mono text-xs text-gray-700">{run.run_tag}</td>
                    <td className="px-4 py-2"><RunStatusChip status={run.status} /></td>
                    <td className="px-4 py-2 text-xs text-gray-500">
                      {run.triggered_at ? new Date(run.triggered_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-2 text-right text-xs">{run.dataset_rows ?? '—'}</td>
                    <td className="px-4 py-2 text-right text-xs">
                      {run.candidates?.length ?? '—'}
                    </td>
                    <td className="px-4 py-2 text-right text-xs">{duration}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        data-testid="view-run-btn"
                        className="text-xs text-blue-600 hover:underline"
                        onClick={e => { e.stopPropagation(); void handleSelectRun(run.run_id) }}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {list.total_pages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100">
              <p className="text-xs text-gray-500">
                Page {list.page} of {list.total_pages} ({list.total} runs)
              </p>
              <div className="flex gap-2">
                <button
                  data-testid="training-prev-page"
                  disabled={page <= 1}
                  onClick={() => setPage(p => p - 1)}
                  className="text-xs px-2 py-1 border rounded disabled:opacity-40"
                >
                  ← Prev
                </button>
                <button
                  data-testid="training-next-page"
                  disabled={page >= list.total_pages}
                  onClick={() => setPage(p => p + 1)}
                  className="text-xs px-2 py-1 border rounded disabled:opacity-40"
                >
                  Next →
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
