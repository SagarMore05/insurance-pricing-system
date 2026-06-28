import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type { PromotionDetail, RollbackRecord, GateResult } from '../../types'

// ─── Helpers ──────────────────────────────────────────────────────────────────

type PromotionView = 'queue' | 'history' | 'rollbacks'

const STATUS_BADGE: Record<string, string> = {
  PENDING:    'bg-gray-100 text-gray-600',
  EVALUATING: 'bg-blue-100 text-blue-700',
  APPROVED:   'bg-green-100 text-green-700',
  REJECTED:   'bg-red-100 text-red-700',
  PROMOTING:  'bg-purple-100 text-purple-700',
  ACTIVE:     'bg-emerald-100 text-emerald-700',
  ROLLED_BACK:'bg-amber-100 text-amber-700',
  FAILED:     'bg-red-200 text-red-800',
}

const RB_STATUS_BADGE: Record<string, string> = {
  SUCCESS: 'bg-green-100 text-green-700',
  FAILED:  'bg-red-100 text-red-700',
  PARTIAL: 'bg-amber-100 text-amber-700',
}

function fmtDate(s: string | null | undefined) {
  if (!s) return '—'
  try { return new Date(s).toLocaleString('en-IN') } catch { return s }
}

function shortId(id: string | null | undefined) {
  if (!id) return '—'
  return `#${id.slice(-8).toUpperCase()}`
}

// ─── Gate Row ─────────────────────────────────────────────────────────────────

function GateRow({ name, gate }: { name: string; gate: GateResult }) {
  return (
    <div className="flex items-start gap-3 py-2 border-b border-gray-50 last:border-0">
      <span className={`mt-0.5 w-4 h-4 flex-shrink-0 rounded-full flex items-center justify-center text-xs font-bold
        ${gate.passed ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}`}>
        {gate.passed ? '✓' : '✗'}
      </span>
      <div className="min-w-0">
        <p className="text-sm font-medium text-gray-800 capitalize">
          {name.replace(/_/g, ' ')}
        </p>
        <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{gate.reason}</p>
      </div>
    </div>
  )
}

// ─── Promotion Detail Drawer ──────────────────────────────────────────────────

function PromotionDrawer({
  promotion,
  onClose,
  onPromote,
  onRollback,
  isPromoting,
  isRollingBack,
}: {
  promotion: PromotionDetail
  onClose: () => void
  onPromote: (id: string) => void
  onRollback: (id: string, reason: string) => void
  isPromoting: boolean
  isRollingBack: boolean
}) {
  const [rollbackReason, setRollbackReason] = useState('')
  const [showRollbackForm, setShowRollbackForm] = useState(false)

  const gates = promotion.evaluation_report
    ? (promotion.evaluation_report as Record<string, { gates?: Record<string, GateResult> }>)
        .gates_summary?.gates ?? {}
    : {}

  const hasGates = Object.keys(gates).length > 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        data-testid="promotion-drawer"
        className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto m-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-100 px-6 py-4 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-900 text-lg">Promotion Detail</h3>
            <p className="text-xs text-gray-400 font-mono mt-0.5">{promotion.promotion_id}</p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${STATUS_BADGE[promotion.status] ?? STATUS_BADGE.PENDING}`}>
              {promotion.status}
            </span>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
          </div>
        </div>

        <div className="px-6 py-5 space-y-5">
          {/* IDs */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-400 text-xs block mb-0.5">Run</span>
              <span className="font-mono text-gray-700">{shortId(promotion.run_id)}</span>
            </div>
            <div>
              <span className="text-gray-400 text-xs block mb-0.5">Created</span>
              <span className="text-gray-700">{fmtDate(promotion.created_at)}</span>
            </div>
            {promotion.promoted_by && (
              <div>
                <span className="text-gray-400 text-xs block mb-0.5">Promoted by</span>
                <span className="text-gray-700">{promotion.promoted_by}</span>
              </div>
            )}
            {promotion.promoted_at && (
              <div>
                <span className="text-gray-400 text-xs block mb-0.5">Promoted at</span>
                <span className="text-gray-700">{fmtDate(promotion.promoted_at)}</span>
              </div>
            )}
          </div>

          {/* Error */}
          {promotion.error_message && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3">
              <p className="text-xs font-semibold text-red-700 mb-1">Error</p>
              <p className="text-xs text-red-600 font-mono leading-relaxed">{promotion.error_message}</p>
            </div>
          )}

          {/* Gates */}
          {hasGates && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Governance Gates</h4>
              <div className="bg-gray-50 rounded-lg px-4 py-2">
                {Object.entries(gates as Record<string, GateResult>).map(([name, gate]) => (
                  <GateRow key={name} name={name} gate={gate} />
                ))}
              </div>
            </div>
          )}

          {/* Champion diff */}
          {(promotion.old_frequency_champion || promotion.new_frequency_champion) && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Champion Comparison</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 rounded-lg px-4 py-3">
                  <p className="text-xs font-semibold text-gray-500 mb-1">Before</p>
                  <p className="text-xs font-mono text-gray-700 break-all">
                    {JSON.stringify(promotion.old_frequency_champion, null, 2).slice(0, 200)}
                  </p>
                </div>
                <div className="bg-emerald-50 rounded-lg px-4 py-3">
                  <p className="text-xs font-semibold text-emerald-600 mb-1">After</p>
                  <p className="text-xs font-mono text-gray-700 break-all">
                    {JSON.stringify(promotion.new_frequency_champion, null, 2).slice(0, 200)}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Rollback history */}
          {promotion.rollback_records.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Rollback History</h4>
              <div className="space-y-2">
                {promotion.rollback_records.map(rb => (
                  <div key={rb.rollback_id}
                    className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-amber-800">{rb.rollback_trigger}</span>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${RB_STATUS_BADGE[rb.rollback_status] ?? ''}`}>
                        {rb.rollback_status}
                      </span>
                    </div>
                    <p className="text-amber-700">{rb.rollback_reason ?? '—'}</p>
                    <p className="text-amber-500 mt-0.5">{fmtDate(rb.rolled_back_at)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2 border-t border-gray-100 flex-wrap">
            {promotion.status === 'APPROVED' && (
              <button
                onClick={() => onPromote(promotion.promotion_id)}
                disabled={isPromoting}
                className="px-5 py-2 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
              >
                {isPromoting ? 'Promoting…' : 'Execute Promotion'}
              </button>
            )}
            {promotion.status === 'ACTIVE' && (
              <>
                {showRollbackForm ? (
                  <div className="flex items-center gap-2 flex-1">
                    <input
                      type="text"
                      value={rollbackReason}
                      onChange={e => setRollbackReason(e.target.value)}
                      placeholder="Rollback reason…"
                      className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                    />
                    <button
                      onClick={() => {
                        if (rollbackReason.trim()) {
                          onRollback(promotion.promotion_id, rollbackReason.trim())
                        }
                      }}
                      disabled={isRollingBack || !rollbackReason.trim()}
                      className="px-4 py-1.5 text-sm rounded-lg bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-50"
                    >
                      {isRollingBack ? 'Rolling back…' : 'Confirm Rollback'}
                    </button>
                    <button
                      onClick={() => { setShowRollbackForm(false); setRollbackReason('') }}
                      className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setShowRollbackForm(true)}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-amber-500 text-white hover:bg-amber-600"
                  >
                    Rollback Champion
                  </button>
                )}
              </>
            )}
            <button onClick={onClose}
              className="px-4 py-2 text-sm border border-gray-200 rounded-lg hover:bg-gray-50 ml-auto">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Promotions Table ─────────────────────────────────────────────────────────

function PromotionsTable({
  items,
  onSelect,
}: {
  items: PromotionDetail[]
  onSelect: (p: PromotionDetail) => void
}) {
  if (items.length === 0) {
    return (
      <p className="text-center text-gray-400 py-12 text-sm">
        No promotions found.
      </p>
    )
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            {['ID', 'Run', 'Status', 'All Gates', 'Promoted By', 'Promoted At', 'Created'].map(h => (
              <th key={h} className="text-left py-2 px-3 text-gray-500 font-medium whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map(p => (
            <tr
              key={p.promotion_id}
              data-testid="promotion-row"
              onClick={() => onSelect(p)}
              className="border-b border-gray-50 hover:bg-blue-50 cursor-pointer"
            >
              <td className="py-2.5 px-3 font-mono text-xs text-gray-600">{shortId(p.promotion_id)}</td>
              <td className="py-2.5 px-3 font-mono text-xs text-blue-600">{shortId(p.run_id)}</td>
              <td className="py-2.5 px-3">
                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${STATUS_BADGE[p.status] ?? ''}`}>
                  {p.status}
                </span>
              </td>
              <td className="py-2.5 px-3">
                {p.all_gates_passed === null ? '—'
                  : p.all_gates_passed
                    ? <span className="text-green-600 font-semibold">✓ All</span>
                    : <span className="text-red-500 font-semibold">✗ Failed</span>}
              </td>
              <td className="py-2.5 px-3 text-gray-600">{p.promoted_by ?? '—'}</td>
              <td className="py-2.5 px-3 text-gray-500 whitespace-nowrap">{fmtDate(p.promoted_at)}</td>
              <td className="py-2.5 px-3 text-gray-400 whitespace-nowrap">{fmtDate(p.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Rollbacks Table ──────────────────────────────────────────────────────────

function RollbacksTable({ items }: { items: RollbackRecord[] }) {
  if (items.length === 0) {
    return (
      <p className="text-center text-gray-400 py-12 text-sm">No rollback events recorded.</p>
    )
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            {['Rollback ID', 'Promotion', 'Trigger', 'Status', 'Reason', 'Rolled Back At'].map(h => (
              <th key={h} className="text-left py-2 px-3 text-gray-500 font-medium whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map(rb => (
            <tr key={rb.rollback_id} className="border-b border-gray-50 hover:bg-gray-50">
              <td className="py-2.5 px-3 font-mono text-xs text-gray-600">{shortId(rb.rollback_id)}</td>
              <td className="py-2.5 px-3 font-mono text-xs text-blue-600">{shortId(rb.promotion_id)}</td>
              <td className="py-2.5 px-3 text-gray-700 capitalize">{rb.rollback_trigger}</td>
              <td className="py-2.5 px-3">
                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${RB_STATUS_BADGE[rb.rollback_status] ?? ''}`}>
                  {rb.rollback_status}
                </span>
              </td>
              <td className="py-2.5 px-3 text-gray-500 max-w-xs truncate">{rb.rollback_reason ?? '—'}</td>
              <td className="py-2.5 px-3 text-gray-400 whitespace-nowrap">{fmtDate(rb.rolled_back_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Main Panel ───────────────────────────────────────────────────────────────

export default function PromotionPanel() {
  const [view, setView] = useState<PromotionView>('queue')
  const [statusFilter, setStatusFilter] = useState('')
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<PromotionDetail | null>(null)
  const qc = useQueryClient()

  const { data: listData, isLoading: listLoading } = useQuery({
    queryKey: ['promotions', page, statusFilter],
    queryFn: () => insuranceApi.listPromotions({
      page,
      page_size: 15,
      status: statusFilter || undefined,
    }),
  })

  const { data: rollbacks, isLoading: rbLoading } = useQuery({
    queryKey: ['promotion-rollbacks'],
    queryFn: () => insuranceApi.listRollbacks({ page: 1, page_size: 50 }),
    enabled: view === 'rollbacks',
  })

  const promoteMutation = useMutation({
    mutationFn: (promotionId: string) => insuranceApi.executePromotion(promotionId, 'admin'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['promotions'] })
      qc.invalidateQueries({ queryKey: ['promotion-rollbacks'] })
      setSelected(null)
    },
  })

  const rollbackMutation = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      insuranceApi.rollbackPromotion(id, reason, 'admin'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['promotions'] })
      qc.invalidateQueries({ queryKey: ['promotion-rollbacks'] })
      setSelected(null)
    },
  })

  const queueItems  = (listData?.items ?? []).filter(p => p.status === 'APPROVED')
  const historyItems = listData?.items ?? []

  return (
    <div className="space-y-4">
      {/* Sub-nav */}
      <div className="flex items-center gap-2 flex-wrap">
        {(['queue', 'history', 'rollbacks'] as PromotionView[]).map(v => (
          <button
            key={v}
            onClick={() => { setView(v); setPage(1); setStatusFilter('') }}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-colors
              ${view === v
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'}`}
          >
            {v === 'queue' ? `Promotion Queue${queueItems.length > 0 ? ` (${queueItems.length})` : ''}`
              : v === 'history' ? 'History'
              : 'Rollback History'}
          </button>
        ))}
      </div>

      {/* Promotion Queue — APPROVED promotions awaiting execution */}
      {view === 'queue' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-800">Promotion Queue</h3>
            <p className="text-xs text-gray-400">APPROVED evaluations awaiting execution</p>
          </div>
          {listLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map(i => <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />)}
            </div>
          ) : (
            <PromotionsTable
              items={queueItems}
              onSelect={setSelected}
            />
          )}
          {promoteMutation.isError && (
            <p className="text-red-600 text-xs mt-3">{(promoteMutation.error as Error).message}</p>
          )}
        </div>
      )}

      {/* History — all promotions */}
      {view === 'history' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <h3 className="font-semibold text-gray-800">Promotion History</h3>
            <select
              value={statusFilter}
              onChange={e => { setStatusFilter(e.target.value); setPage(1) }}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            >
              <option value="">All statuses</option>
              {['APPROVED', 'REJECTED', 'ACTIVE', 'ROLLED_BACK', 'FAILED', 'PROMOTING', 'EVALUATING'].map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          {listLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map(i => <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />)}
            </div>
          ) : (
            <>
              <PromotionsTable items={historyItems} onSelect={setSelected} />
              {listData && listData.total_pages > 1 && (
                <div className="flex items-center justify-between mt-4 text-sm text-gray-500">
                  <span>{listData.total} promotion{listData.total !== 1 ? 's' : ''}</span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      disabled={page === 1}
                      className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
                    >←</button>
                    <span className="px-2 py-1">{page} / {listData.total_pages}</span>
                    <button
                      onClick={() => setPage(p => Math.min(listData.total_pages, p + 1))}
                      disabled={page === listData.total_pages}
                      className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
                    >→</button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Rollback History */}
      {view === 'rollbacks' && (
        <div className="card">
          <h3 className="font-semibold text-gray-800 mb-4">Rollback History</h3>
          {rbLoading ? (
            <div className="space-y-2">
              {[1, 2].map(i => <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />)}
            </div>
          ) : (
            <RollbacksTable items={rollbacks ?? []} />
          )}
        </div>
      )}

      {/* Detail drawer */}
      {selected && (
        <PromotionDrawer
          promotion={selected}
          onClose={() => setSelected(null)}
          onPromote={id => promoteMutation.mutate(id)}
          onRollback={(id, reason) => rollbackMutation.mutate({ id, reason })}
          isPromoting={promoteMutation.isPending}
          isRollingBack={rollbackMutation.isPending}
        />
      )}
    </div>
  )
}
