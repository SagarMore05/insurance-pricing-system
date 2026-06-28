import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type {
  ApprovalRequest,
  ApprovalRequestDetail,
  ApprovalAuditLogEntry,
  ModelCard,
} from '../../types'

// ─── Status badge ──────────────────────────────────────────────────────────────

const STATUS_STYLES: Record<string, string> = {
  pending:  'bg-amber-100 text-amber-700 border border-amber-200',
  approved: 'bg-green-100 text-green-700 border border-green-200',
  rejected: 'bg-red-100 text-red-700 border border-red-200',
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold capitalize ${STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-600'}`}>
      {status}
    </span>
  )
}

// ─── Model Card detail ─────────────────────────────────────────────────────────

function ModelCardView({ card }: { card: ModelCard }) {
  const [showImportance, setShowImportance] = useState(false)

  return (
    <div className="space-y-4 text-sm" data-testid="model-card-view">
      {/* Header row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <Metric label="Model Version" value={card.model_version} mono />
        <Metric label="Algorithm" value={card.algorithm} />
        <Metric label="Training Date" value={card.training_date} />
        <Metric label="Dataset Size" value={card.dataset_size.toLocaleString()} sub="rows" />
        <Metric label="Preprocessor" value={`V${card.preprocessor_version}`} />
        <Metric label="Champion Since" value={card.champion_since} />
      </div>

      {/* Frequency metrics */}
      {card.frequency_metrics && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Frequency Model Metrics</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {card.frequency_metrics.roc_auc != null && (
              <MetricPill label="ROC-AUC" value={card.frequency_metrics.roc_auc.toFixed(4)} color="blue" />
            )}
            {card.frequency_metrics.pr_auc != null && (
              <MetricPill label="PR-AUC" value={card.frequency_metrics.pr_auc.toFixed(4)} color="blue" />
            )}
            {card.frequency_metrics.f1 != null && (
              <MetricPill label="F1" value={card.frequency_metrics.f1.toFixed(4)} color="blue" />
            )}
            {card.frequency_metrics.brier_score != null && (
              <MetricPill label="Brier" value={card.frequency_metrics.brier_score.toFixed(4)} color="slate" />
            )}
          </div>
        </div>
      )}

      {/* Severity metrics */}
      {card.severity_metrics && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Severity Model Metrics</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {card.severity_metrics.r2 != null && (
              <MetricPill label="R²" value={card.severity_metrics.r2.toFixed(4)} color="purple" />
            )}
            {card.severity_metrics.rmse != null && (
              <MetricPill label="RMSE" value={`₹${Math.round(card.severity_metrics.rmse).toLocaleString('en-IN')}`} color="purple" />
            )}
            {card.severity_metrics.mape_pct != null && (
              <MetricPill label="MAPE" value={`${card.severity_metrics.mape_pct.toFixed(1)}%`} color="slate" />
            )}
          </div>
        </div>
      )}

      {/* Drift summary */}
      {card.drift_summary && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Latest Drift Summary</p>
          {card.drift_summary.overall_psi != null ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-500">Overall PSI:</span>{' '}
                <span className="font-mono font-semibold">{card.drift_summary.overall_psi.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-gray-500">Severity:</span>{' '}
                <span className={`font-semibold ${
                  card.drift_summary.overall_severity === 'high' ? 'text-red-600' :
                  card.drift_summary.overall_severity === 'medium' ? 'text-amber-600' :
                  'text-green-600'
                }`}>{(card.drift_summary.overall_severity ?? 'N/A').toUpperCase()}</span>
              </div>
              {card.drift_summary.high_drift_features.length > 0 && (
                <div>
                  <span className="text-gray-500">High Drift:</span>{' '}
                  <span className="font-semibold text-red-600">{card.drift_summary.high_drift_features.join(', ')}</span>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-gray-400">No drift snapshot available</p>
          )}
        </div>
      )}

      {/* Feature importance */}
      {card.feature_importance_summary && (
        <div>
          <button
            onClick={() => setShowImportance(v => !v)}
            className="text-xs text-blue-600 hover:text-blue-700 underline"
          >
            {showImportance ? 'Hide' : 'Show'} Feature Importance Summary
          </button>
          {showImportance && (
            <div className="mt-2 space-y-1" data-testid="feature-importance">
              {Object.entries(card.feature_importance_summary)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)
                .map(([feat, score]) => (
                  <div key={feat} className="flex items-center gap-2 text-xs">
                    <span className="w-36 text-gray-600 truncate">{feat}</span>
                    <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full"
                        style={{ width: `${Math.min(100, score * 600)}%` }}
                      />
                    </div>
                    <span className="font-mono text-gray-500 w-12 text-right">{score.toFixed(4)}</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* Recommendation banner */}
      <div className={`flex items-center gap-2 p-3 rounded-lg text-xs font-medium ${
        card.recommendation.includes('Retraining') ? 'bg-red-50 text-red-700 border border-red-200' :
        card.recommendation.includes('Investigate') ? 'bg-amber-50 text-amber-700 border border-amber-200' :
        'bg-green-50 text-green-700 border border-green-200'
      }`}>
        <span className="font-semibold">Recommendation:</span>
        <span>{card.recommendation}</span>
      </div>
    </div>
  )
}

// ─── Audit log ─────────────────────────────────────────────────────────────────

const EVENT_ICONS: Record<string, string> = {
  created: '📋',
  approved: '✅',
  rejected: '❌',
  email_notification: '📧',
}

function AuditLogView({ logs }: { logs: ApprovalAuditLogEntry[] }) {
  return (
    <div className="space-y-2" data-testid="audit-log">
      {logs.map((log) => (
        <div key={log.log_id} className="flex gap-3 text-sm border-b border-gray-100 pb-2">
          <span className="text-base flex-shrink-0 mt-0.5">{EVENT_ICONS[log.event_type] ?? '•'}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold capitalize text-gray-800">{log.event_type.replace(/_/g, ' ')}</span>
              {log.actor && <span className="text-gray-400 text-xs">by {log.actor}</span>}
              <span className="text-gray-300 text-xs ml-auto">{new Date(log.created_at).toLocaleString('en-IN')}</span>
            </div>
            {log.note && <p className="text-gray-500 text-xs mt-0.5">{log.note}</p>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Request row ───────────────────────────────────────────────────────────────

function RequestRow({
  req,
  isExpanded,
  onToggle,
  onApprove,
  onReject,
  isActing,
}: {
  req: ApprovalRequest
  isExpanded: boolean
  onToggle: () => void
  onApprove: (id: string) => void
  onReject: (id: string) => void
  isActing: boolean
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
        data-testid="approval-row"
      >
        <td className="py-3 px-3">
          <span className="font-mono text-xs text-gray-500">#{req.request_id.slice(-8).toUpperCase()}</span>
        </td>
        <td className="py-3 px-3">
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${
            req.model_type === 'frequency'
              ? 'bg-blue-50 text-blue-700 border border-blue-200'
              : 'bg-purple-50 text-purple-700 border border-purple-200'
          }`}>
            {req.model_type === 'frequency' ? 'Frequency' : 'Severity'}
          </span>
        </td>
        <td className="py-3 px-3 font-mono text-xs text-gray-700">{req.model_version}</td>
        <td className="py-3 px-3"><StatusBadge status={req.status} /></td>
        <td className="py-3 px-3 text-xs text-gray-500">
          {new Date(req.submitted_at).toLocaleDateString('en-IN')}
        </td>
        <td className="py-3 px-3 text-xs text-gray-500 max-w-[160px] truncate">
          {req.recommendation ?? '—'}
        </td>
        <td className="py-3 px-3">
          {req.status === 'pending' && (
            <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
              <button
                onClick={() => onApprove(req.request_id)}
                disabled={isActing}
                className="text-xs px-2.5 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 font-medium"
                data-testid="approve-btn"
              >
                Approve
              </button>
              <button
                onClick={() => onReject(req.request_id)}
                disabled={isActing}
                className="text-xs px-2.5 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 font-medium"
                data-testid="reject-btn"
              >
                Reject
              </button>
            </div>
          )}
        </td>
      </tr>
      {isExpanded && <ExpandedDetail requestId={req.request_id} />}
    </>
  )
}

// ─── Expanded detail row ───────────────────────────────────────────────────────

function ExpandedDetail({ requestId }: { requestId: string }) {
  const { data, isLoading } = useQuery<ApprovalRequestDetail>({
    queryKey: ['approval-detail', requestId],
    queryFn: () => insuranceApi.getApprovalRequest(requestId),
  })

  return (
    <tr>
      <td colSpan={7} className="p-0">
        <div className="bg-gray-50 border-b border-gray-200 px-4 py-4">
          {isLoading ? (
            <div className="animate-pulse h-24 bg-gray-200 rounded" />
          ) : data ? (
            <div className="space-y-4">
              {/* Reviewer info (if reviewed) */}
              {data.status !== 'pending' && (
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-gray-500">Reviewed by:</span>
                  <span className="font-semibold">{data.reviewed_by ?? '—'}</span>
                  {data.reviewed_at && (
                    <span className="text-gray-400 text-xs">{new Date(data.reviewed_at).toLocaleString('en-IN')}</span>
                  )}
                  {data.reviewer_note && (
                    <span className="italic text-gray-600 text-xs">"{data.reviewer_note}"</span>
                  )}
                </div>
              )}

              {/* Model card */}
              {data.model_card && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Model Card</p>
                  <ModelCardView card={data.model_card as ModelCard} />
                </div>
              )}

              {/* Audit log */}
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Audit Log</p>
                <AuditLogView logs={data.audit_logs} />
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-400">Could not load details.</p>
          )}
        </div>
      </td>
    </tr>
  )
}

// ─── Review confirmation dialog ────────────────────────────────────────────────

function ReviewDialog({
  action,
  requestId,
  onConfirm,
  onCancel,
  isPending,
}: {
  action: 'approve' | 'reject'
  requestId: string
  onConfirm: (note: string) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [note, setNote] = useState('')
  const isApprove = action === 'approve'

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50" data-testid="review-dialog">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
        <h3 className={`font-semibold text-lg mb-2 ${isApprove ? 'text-green-700' : 'text-red-700'}`}>
          {isApprove ? 'Approve Request' : 'Reject Request'}
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          {isApprove
            ? 'Approving records your governance decision. No model changes will occur — champion promotion requires a separate V5 pipeline run.'
            : 'Rejecting records your governance decision. No model changes will occur.'}
        </p>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder={isApprove ? 'Reviewer note (optional)…' : 'Rejection reason (optional)…'}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none h-20 focus:outline-none focus:ring-2 focus:ring-blue-300"
          data-testid="reviewer-note-input"
        />
        <div className="flex gap-3 mt-4 justify-end">
          <button
            onClick={onCancel}
            disabled={isPending}
            className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(note)}
            disabled={isPending}
            className={`px-4 py-2 text-sm text-white rounded-lg font-medium disabled:opacity-50 ${
              isApprove ? 'bg-green-600 hover:bg-green-700' : 'bg-red-500 hover:bg-red-600'
            }`}
            data-testid="confirm-review-btn"
          >
            {isPending ? 'Submitting…' : isApprove ? 'Confirm Approve' : 'Confirm Reject'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Create request dialog ─────────────────────────────────────────────────────

function CreateRequestDialog({
  onConfirm,
  onCancel,
  isPending,
}: {
  onConfirm: (modelType: 'frequency' | 'severity', notes: string) => void
  onCancel: () => void
  isPending: boolean
}) {
  const [modelType, setModelType] = useState<'frequency' | 'severity'>('frequency')
  const [notes, setNotes] = useState('')

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50" data-testid="create-dialog">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
        <h3 className="font-semibold text-lg mb-2 text-gray-900">Submit Approval Request</h3>
        <p className="text-sm text-gray-500 mb-4">
          Generates a model card from the current champion registry and latest drift snapshot.
          Does not modify production models.
        </p>
        <div className="mb-4">
          <label className="block text-xs font-semibold text-gray-500 mb-1">Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value as 'frequency' | 'severity')}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
            data-testid="model-type-select"
          >
            <option value="frequency">Frequency (XGBoost_V4 — Claim Probability)</option>
            <option value="severity">Severity (CatBoost_V4 — Claim Amount)</option>
          </select>
        </div>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Notes for reviewer (optional)…"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none h-20 focus:outline-none focus:ring-2 focus:ring-blue-300"
          data-testid="create-notes-input"
        />
        <div className="flex gap-3 mt-4 justify-end">
          <button
            onClick={onCancel}
            disabled={isPending}
            className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(modelType, notes)}
            disabled={isPending}
            className="px-4 py-2 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-lg font-medium disabled:opacity-50"
            data-testid="submit-request-btn"
          >
            {isPending ? 'Submitting…' : 'Submit Request'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Main panel ────────────────────────────────────────────────────────────────

export default function ApprovalPanel() {
  const qc = useQueryClient()
  const [activeView, setActiveView] = useState<'pending' | 'history'>('pending')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [reviewTarget, setReviewTarget] = useState<{
    action: 'approve' | 'reject'
    id: string
  } | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [page, setPage] = useState(1)

  const statusParam = activeView === 'pending' ? 'pending' : undefined

  const { data: listData, isLoading } = useQuery({
    queryKey: ['approvals-list', activeView, page],
    queryFn: () =>
      insuranceApi.listApprovalRequests({
        status: statusParam,
        page,
        page_size: 15,
      }),
    refetchInterval: 60_000,
  })

  const createMutation = useMutation({
    mutationFn: ({ modelType, notes }: { modelType: 'frequency' | 'severity'; notes: string }) =>
      insuranceApi.createApprovalRequest(modelType, notes || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['approvals-list'] })
      setShowCreate(false)
    },
  })

  const approveMutation = useMutation({
    mutationFn: ({ id, note }: { id: string; note: string }) =>
      insuranceApi.approveApprovalRequest(id, note || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['approvals-list'] })
      qc.invalidateQueries({ queryKey: ['approval-detail'] })
      setReviewTarget(null)
    },
  })

  const rejectMutation = useMutation({
    mutationFn: ({ id, note }: { id: string; note: string }) =>
      insuranceApi.rejectApprovalRequest(id, note || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['approvals-list'] })
      qc.invalidateQueries({ queryKey: ['approval-detail'] })
      setReviewTarget(null)
    },
  })

  const isActing = approveMutation.isPending || rejectMutation.isPending || createMutation.isPending

  const handleApprove = (id: string) => setReviewTarget({ action: 'approve', id })
  const handleReject = (id: string) => setReviewTarget({ action: 'reject', id })

  const handleReviewConfirm = (note: string) => {
    if (!reviewTarget) return
    if (reviewTarget.action === 'approve') {
      approveMutation.mutate({ id: reviewTarget.id, note })
    } else {
      rejectMutation.mutate({ id: reviewTarget.id, note })
    }
  }

  const pendingCount = activeView === 'pending' ? (listData?.total ?? 0) : undefined

  return (
    <div data-testid="approval-panel">
      {/* Dialogs */}
      {showCreate && (
        <CreateRequestDialog
          onConfirm={(modelType, notes) => createMutation.mutate({ modelType, notes })}
          onCancel={() => setShowCreate(false)}
          isPending={createMutation.isPending}
        />
      )}
      {reviewTarget && (
        <ReviewDialog
          action={reviewTarget.action}
          requestId={reviewTarget.id}
          onConfirm={handleReviewConfirm}
          onCancel={() => setReviewTarget(null)}
          isPending={approveMutation.isPending || rejectMutation.isPending}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Model Approval Workflow</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Human-in-the-loop governance for champion model decisions. Read-only — approvals do not promote models.
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          data-testid="new-request-btn"
        >
          + New Request
        </button>
      </div>

      {/* Safety notice */}
      <div className="mb-4 p-3 rounded-lg bg-blue-50 border border-blue-200 flex items-start gap-2 text-xs">
        <svg className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
        </svg>
        <span className="text-blue-700">
          <strong>Governance only.</strong> Approvals record your decision for audit purposes.
          No model promotion, no champion modification, no pricing changes occur automatically.
          V5 pipeline activation requires a separate release process.
        </span>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-4 border-b border-gray-200">
        {(['pending', 'history'] as const).map((view) => (
          <button
            key={view}
            onClick={() => { setActiveView(view); setPage(1) }}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors relative ${
              activeView === view
                ? 'bg-white text-blue-600 border-l border-r border-t border-gray-200 -mb-px'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {view === 'pending' ? 'Pending' : 'History'}
            {view === 'pending' && pendingCount != null && pendingCount > 0 && (
              <span className="ml-1.5 inline-flex items-center justify-center w-4 h-4 text-xs font-bold rounded-full bg-amber-500 text-white">
                {pendingCount > 9 ? '9+' : pendingCount}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Error banner */}
      {(createMutation.isError || approveMutation.isError || rejectMutation.isError) && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700" data-testid="error-banner">
          {((createMutation.error || approveMutation.error || rejectMutation.error) as Error)?.message ?? 'An error occurred.'}
        </div>
      )}

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {isLoading ? (
          <div data-testid="approval-loading" className="p-6 space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse h-10 bg-gray-100 rounded" />
            ))}
          </div>
        ) : !listData || listData.items.length === 0 ? (
          <div className="py-16 text-center text-gray-400 text-sm" data-testid="approval-empty">
            {activeView === 'pending'
              ? 'No pending approval requests. Use "+ New Request" to submit one.'
              : 'No approval history yet.'}
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                {['ID', 'Model', 'Version', 'Status', 'Submitted', 'Recommendation', 'Actions'].map((h) => (
                  <th key={h} className="text-left py-2.5 px-3 text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {listData.items.map((req) => (
                <RequestRow
                  key={req.request_id}
                  req={req}
                  isExpanded={expandedId === req.request_id}
                  onToggle={() => setExpandedId(expandedId === req.request_id ? null : req.request_id)}
                  onApprove={handleApprove}
                  onReject={handleReject}
                  isActing={isActing}
                />
              ))}
            </tbody>
          </table>
        )}

        {/* Pagination */}
        {listData && listData.total_pages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100 text-sm text-gray-500">
            <span>{listData.total} request{listData.total !== 1 ? 's' : ''}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
              >
                ←
              </button>
              <span className="px-2 py-1">{page} / {listData.total_pages}</span>
              <button
                onClick={() => setPage((p) => Math.min(listData.total_pages, p + 1))}
                disabled={page === listData.total_pages}
                className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
              >
                →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Micro helpers ─────────────────────────────────────────────────────────────

function Metric({ label, value, sub, mono }: { label: string; value: string; sub?: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs text-gray-400">{label}</p>
      <p className={`font-semibold text-gray-800 ${mono ? 'font-mono' : ''}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400">{sub}</p>}
    </div>
  )
}

function MetricPill({ label, value, color }: { label: string; value: string; color: 'blue' | 'purple' | 'slate' }) {
  const cls = color === 'blue' ? 'bg-blue-50 border-blue-200 text-blue-700'
    : color === 'purple' ? 'bg-purple-50 border-purple-200 text-purple-700'
    : 'bg-gray-50 border-gray-200 text-gray-700'
  return (
    <div className={`px-2 py-1.5 rounded border text-xs ${cls}`}>
      <p className="text-gray-400 text-xs">{label}</p>
      <p className="font-mono font-semibold">{value}</p>
    </div>
  )
}
