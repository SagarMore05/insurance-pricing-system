import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type { ReviewQueueItem, ReviewDetail, ReviewDecisionResponse } from '../../types'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number) {
  return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`
}

function formatRelative(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const s = Math.floor(diff / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return new Date(iso).toLocaleDateString('en-IN')
}

const TRIGGER_LABEL: Record<string, string> = {
  low_confidence:         'Low confidence',
  fraud_signals_present:  'Fraud signals',
  high_claim_probability: 'High claim probability',
}

const DECISION_BADGE: Record<string, string> = {
  approve: 'bg-green-100 text-green-700',
  flag:    'bg-amber-100 text-amber-700',
  refer:   'bg-red-100 text-red-700',
}

const SEVERITY_STYLE: Record<string, string> = {
  high:   'bg-red-100 text-red-700 border border-red-200',
  medium: 'bg-amber-100 text-amber-700 border border-amber-200',
  low:    'bg-gray-100 text-gray-600 border border-gray-200',
}

// ─── Review Detail Drawer ─────────────────────────────────────────────────────

function ReviewDrawer({
  runId,
  onClose,
  onActioned,
}: {
  runId: string
  onClose: () => void
  onActioned: (outcome: 'approved' | 'rejected') => void
}) {
  const qc = useQueryClient()
  const [note, setNote] = useState('')
  const [actionResult, setActionResult] = useState<ReviewDecisionResponse | null>(null)

  const { data: detail, isLoading } = useQuery<ReviewDetail>({
    queryKey: ['review-detail', runId],
    queryFn: () => insuranceApi.getReviewDetail(runId),
    staleTime: 30_000,
  })

  const approveMutation = useMutation({
    mutationFn: () => insuranceApi.approveReview(runId, { reviewer_note: note || undefined }),
    onSuccess: (res) => {
      setActionResult(res)
      qc.invalidateQueries({ queryKey: ['review-queue'] })
      qc.invalidateQueries({ queryKey: ['review-queue-count'] })
      qc.invalidateQueries({ queryKey: ['agent-runs'] })
      onActioned('approved')
    },
  })

  const rejectMutation = useMutation({
    mutationFn: () => insuranceApi.rejectReview(runId, { reviewer_note: note || undefined }),
    onSuccess: (res) => {
      setActionResult(res)
      qc.invalidateQueries({ queryKey: ['review-queue'] })
      qc.invalidateQueries({ queryKey: ['review-queue-count'] })
      qc.invalidateQueries({ queryKey: ['agent-runs'] })
      onActioned('rejected')
    },
  })

  const isPending = approveMutation.isPending || rejectMutation.isPending

  if (actionResult) {
    const approved = actionResult.review_outcome === 'approved'
    return (
      <div className={`rounded-xl border-2 ${approved ? 'border-green-400 bg-green-50' : 'border-red-400 bg-red-50'} p-5 space-y-2`}>
        <p className={`text-lg font-bold ${approved ? 'text-green-900' : 'text-red-900'}`}>
          {approved ? '✓ Application Approved' : '✕ Application Rejected'}
        </p>
        <p className="text-sm text-gray-500">
          Agent decision was: <span className="font-medium capitalize">{actionResult.agent_decision}</span>
        </p>
        {actionResult.reviewer_note && (
          <p className="text-sm text-gray-600 italic">"{actionResult.reviewer_note}"</p>
        )}
        <button onClick={onClose} className="text-sm text-gray-400 hover:text-gray-600 underline mt-2">
          Close
        </button>
      </div>
    )
  }

  return (
    <div className="border border-gray-200 rounded-xl bg-gray-50 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-gray-800">Review Detail</p>
        <button onClick={onClose} className="text-xs text-gray-400 hover:text-gray-600 px-2 py-1 rounded hover:bg-gray-100">
          Close ✕
        </button>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {[1, 2, 3].map(i => <div key={i} className="h-8 bg-gray-200 rounded animate-pulse" />)}
        </div>
      )}

      {detail && (
        <>
          {/* Agent decision + confidence */}
          <div className="grid grid-cols-4 gap-3 text-sm">
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-500">Agent Decision</p>
              <span className={`inline-block mt-1 px-2 py-0.5 rounded-full text-xs font-semibold capitalize ${DECISION_BADGE[detail.agent_decision ?? ''] ?? 'bg-gray-100 text-gray-600'}`}>
                {detail.agent_decision ?? '—'}
              </span>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-500">Confidence</p>
              <p className="font-semibold text-gray-900 mt-1">{pct(detail.confidence_score)}</p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-500">Claim Prob.</p>
              <p className="font-semibold text-gray-900 mt-1">{pct(detail.claim_probability)}</p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-500">Est. Premium</p>
              <p className="font-semibold text-gray-900 mt-1">{fmt(detail.premium_amount_inr)}</p>
            </div>
          </div>

          {/* Fraud signals */}
          {detail.fraud_signals.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-700 mb-2">
                Fraud Signals ({detail.fraud_signals.length})
              </p>
              <div className="space-y-1.5">
                {detail.fraud_signals.map((s, i) => (
                  <div key={i} className="flex items-start gap-2 bg-white rounded-lg px-3 py-2 border border-gray-100">
                    <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full shrink-0 mt-0.5 ${SEVERITY_STYLE[s.severity]}`}>
                      {s.severity.toUpperCase()}
                    </span>
                    <div>
                      <p className="text-sm font-medium text-gray-800">{s.signal.replace(/_/g, ' ')}</p>
                      <p className="text-xs text-gray-500">{s.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reasoning trace */}
          {detail.reasoning_trace && (
            <div>
              <p className="text-xs font-medium text-gray-700 mb-1">Underwriter Reasoning</p>
              <p className="text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2 leading-relaxed line-clamp-4">
                {detail.reasoning_trace}
              </p>
            </div>
          )}

          {/* Review triggers */}
          <div>
            <p className="text-xs font-medium text-gray-700 mb-1">Review Triggers</p>
            <div className="flex flex-wrap gap-1.5">
              {detail.review_triggers.map(t => (
                <span key={t} className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
                  {TRIGGER_LABEL[t] ?? t}
                </span>
              ))}
            </div>
          </div>

          {/* Reviewer note */}
          <div>
            <label className="text-xs font-medium text-gray-700">Reviewer Note (optional)</label>
            <textarea
              value={note}
              onChange={e => setNote(e.target.value)}
              placeholder="Add a note for the audit trail…"
              maxLength={500}
              rows={2}
              className="mt-1 w-full input-field text-sm"
            />
          </div>

          {/* Action buttons */}
          {(approveMutation.isError || rejectMutation.isError) && (
            <p className="text-red-600 text-xs">
              {((approveMutation.error || rejectMutation.error) as Error).message}
            </p>
          )}

          <div className="flex gap-3">
            <button
              onClick={() => approveMutation.mutate()}
              disabled={isPending}
              className="flex-1 py-2.5 bg-green-600 hover:bg-green-700 text-white font-semibold text-sm rounded-lg disabled:opacity-50 transition-colors"
            >
              {approveMutation.isPending ? 'Approving…' : '✓ Approve Application'}
            </button>
            <button
              onClick={() => rejectMutation.mutate()}
              disabled={isPending}
              className="flex-1 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold text-sm rounded-lg disabled:opacity-50 transition-colors"
            >
              {rejectMutation.isPending ? 'Rejecting…' : '✕ Reject Application'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ReviewQueuePanel() {
  const [page, setPage] = useState(1)
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)

  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['review-queue', page],
    queryFn: () => insuranceApi.getReviewQueue({ page, page_size: 15 }),
    staleTime: 15_000,
    refetchInterval: 30_000,
  })

  const items = (data?.items ?? []) as ReviewQueueItem[]

  const handleActioned = () => {
    setExpandedRunId(null)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <h3 className="font-semibold text-gray-900">Review Queue</h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Underwriting applications awaiting human review before a final decision
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50"
        >
          {isFetching ? '↻' : 'Refresh'}
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map(i => <div key={i} className="h-12 bg-gray-100 rounded animate-pulse" />)}
        </div>
      ) : items.length === 0 ? (
        <div className="text-center py-14 text-gray-400">
          <p className="text-2xl mb-2">✓</p>
          <p className="text-sm font-medium">No applications pending review</p>
          <p className="text-xs mt-1">
            Runs with confidence &lt; 85%, fraud signals, or claim probability &gt; 40%
            will appear here.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {items.map(item => (
            <div key={item.run_id} className="border border-gray-200 rounded-xl overflow-hidden">
              {/* Summary row */}
              <button
                onClick={() => setExpandedRunId(expandedRunId === item.run_id ? null : item.run_id)}
                className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-3 flex-wrap">
                  {/* Status pulse */}
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                    <span className="text-xs text-purple-700 font-medium">Pending</span>
                  </span>

                  {/* Agent decision */}
                  {item.agent_decision && (
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full capitalize ${DECISION_BADGE[item.agent_decision] ?? 'bg-gray-100 text-gray-600'}`}>
                      {item.agent_decision}
                    </span>
                  )}

                  {/* Key metrics */}
                  <span className="text-xs text-gray-500">
                    Confidence: <span className="font-medium text-gray-700">{pct(item.confidence_score)}</span>
                  </span>
                  <span className="text-xs text-gray-500">
                    Claim prob: <span className="font-medium text-gray-700">{pct(item.claim_probability)}</span>
                  </span>
                  {item.fraud_signal_count > 0 && (
                    <span className="text-xs text-red-600 font-medium">
                      {item.fraud_signal_count} signal{item.fraud_signal_count !== 1 ? 's' : ''}
                      {item.high_severity_count > 0 && ` (${item.high_severity_count} HIGH)`}
                    </span>
                  )}

                  {/* Trigger chips */}
                  <div className="flex gap-1 flex-wrap">
                    {item.review_triggers.map(t => (
                      <span key={t} className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full">
                        {TRIGGER_LABEL[t] ?? t}
                      </span>
                    ))}
                  </div>

                  <span className="ml-auto text-xs text-gray-400 whitespace-nowrap">
                    {formatRelative(item.started_at)} · {item.minutes_pending}min pending
                  </span>
                  <span className="text-gray-300 text-xs">{expandedRunId === item.run_id ? '▾' : '▸'}</span>
                </div>

                {/* Input summary */}
                <p className="text-xs text-gray-400 mt-1.5 truncate text-left">{item.input_summary}</p>
              </button>

              {/* Expanded review drawer */}
              {expandedRunId === item.run_id && (
                <div className="border-t border-gray-200 p-4">
                  <ReviewDrawer
                    runId={item.run_id}
                    onClose={() => setExpandedRunId(null)}
                    onActioned={handleActioned}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between mt-4 text-sm text-gray-500">
          <span>{data.total} application{data.total !== 1 ? 's' : ''} pending</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >←</button>
            <span className="px-2">{page} / {data.total_pages}</span>
            <button
              onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
              disabled={page === data.total_pages}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >→</button>
          </div>
        </div>
      )}
    </div>
  )
}
