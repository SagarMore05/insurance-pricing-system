import React, { useEffect, useState, useCallback } from 'react'
import { insuranceApi } from '../../api/insurance'
import type {
  ShadowStatusResponse,
  ShadowStatisticsResponse,
  ShadowHistoryResponse,
  ShadowPredictionRecord,
} from '../../types'

// ─── Status helpers ────────────────────────────────────────────────────────────

function StatusChip({ status }: { status: string }) {
  const map: Record<string, string> = {
    WAITING_FOR_CHALLENGER: 'bg-yellow-100 text-yellow-800',
    PENDING:     'bg-blue-100 text-blue-800',
    COMPLETED:   'bg-green-100 text-green-800',
    FAILED:      'bg-red-100 text-red-800',
    OPERATIONAL: 'bg-green-100 text-green-800',
  }
  const cls = map[status] ?? 'bg-gray-100 text-gray-700'
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-semibold rounded ${cls}`}>
      {status.replace(/_/g, ' ')}
    </span>
  )
}

function ComparisonChip({ result }: { result: string | null }) {
  if (!result) return <span className="text-gray-400 text-xs">—</span>
  const map: Record<string, string> = {
    MATCH:             'bg-green-100 text-green-800',
    DIVERGENT:         'bg-orange-100 text-orange-800',
    CHALLENGER_HIGHER: 'bg-red-100 text-red-800',
    CHALLENGER_LOWER:  'bg-blue-100 text-blue-800',
  }
  const cls = map[result] ?? 'bg-gray-100 text-gray-700'
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-semibold rounded ${cls}`}>
      {result.replace(/_/g, ' ')}
    </span>
  )
}

// ─── Status overview cards ──────────────────────────────────────────────────────

function StatusOverview({ status }: { status: ShadowStatusResponse }) {
  const cards = [
    { label: 'Total Predictions', value: status.total_shadow_predictions, color: 'blue' },
    { label: 'Waiting for Challenger', value: status.waiting_for_challenger, color: 'yellow' },
    { label: 'Completed Comparisons', value: status.completed_comparisons, color: 'green' },
    { label: 'Failed', value: status.failed_comparisons, color: 'red' },
  ]
  return (
    <div data-testid="shadow-status-overview" className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {cards.map(c => (
        <div key={c.label} className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide">{c.label}</p>
          <p className="text-2xl font-bold text-gray-800 mt-1">{c.value}</p>
        </div>
      ))}
    </div>
  )
}

// ─── Framework info banner ──────────────────────────────────────────────────────

function FrameworkBanner({ status }: { status: ShadowStatusResponse }) {
  const isActive = status.challenger_available
  return (
    <div
      data-testid="shadow-framework-banner"
      className={`rounded-lg border p-4 mb-6 ${
        isActive
          ? 'bg-green-50 border-green-200'
          : 'bg-yellow-50 border-yellow-200'
      }`}
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{isActive ? '✅' : '⏳'}</span>
        <div>
          <p className="font-semibold text-gray-800">
            Shadow Framework: <StatusChip status={status.framework_status} />
          </p>
          <p className="text-sm text-gray-600 mt-1">{status.message}</p>
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-500">
            <span>Champion: <strong>{status.champion_model}</strong></span>
            <span>Version: <strong>{status.champion_version}</strong></span>
            <span>Challenger: <StatusChip status={status.challenger_status} /></span>
            {status.last_recorded_at && (
              <span>Last recorded: <strong>{new Date(status.last_recorded_at).toLocaleString()}</strong></span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Statistics panel ───────────────────────────────────────────────────────────

function StatisticsPanel({ stats }: { stats: ShadowStatisticsResponse }) {
  return (
    <div data-testid="shadow-statistics" className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        Statistics — Last {stats.window_days} Days
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-gray-500 text-xs">Total Predictions</p>
          <p className="font-bold text-gray-800">{stats.total_shadow_predictions}</p>
        </div>
        <div>
          <p className="text-gray-500 text-xs">Completion Rate</p>
          <p className="font-bold text-gray-800">{stats.completion_rate_pct}%</p>
        </div>
        <div>
          <p className="text-gray-500 text-xs">Avg Premium Δ (INR)</p>
          <p className="font-bold text-gray-800">
            {stats.avg_premium_difference_inr != null
              ? `₹${stats.avg_premium_difference_inr.toFixed(0)}`
              : '—'}
          </p>
        </div>
        <div>
          <p className="text-gray-500 text-xs">Avg Latency (ms)</p>
          <p className="font-bold text-gray-800">
            {stats.avg_prediction_latency_ms != null
              ? `${stats.avg_prediction_latency_ms.toFixed(0)} ms`
              : '—'}
          </p>
        </div>
      </div>
    </div>
  )
}

// ─── History table ──────────────────────────────────────────────────────────────

interface HistoryTableProps {
  history: ShadowHistoryResponse
  page: number
  onPageChange: (p: number) => void
  onSelect: (rec: ShadowPredictionRecord) => void
  statusFilter: string
  onStatusFilter: (s: string) => void
}

function HistoryTable({
  history, page, onPageChange, onSelect, statusFilter, onStatusFilter,
}: HistoryTableProps) {
  const STATUS_OPTIONS = [
    '', 'WAITING_FOR_CHALLENGER', 'PENDING', 'COMPLETED', 'FAILED',
  ]

  return (
    <div data-testid="shadow-history-table" className="bg-white rounded-lg border border-gray-200">
      <div className="flex items-center justify-between p-4 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">Prediction History</h3>
        <select
          data-testid="shadow-status-filter"
          value={statusFilter}
          onChange={e => onStatusFilter(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1"
        >
          {STATUS_OPTIONS.map(s => (
            <option key={s} value={s}>{s || 'All statuses'}</option>
          ))}
        </select>
      </div>

      {history.items.length === 0 ? (
        <p data-testid="shadow-history-empty" className="text-center text-gray-400 text-sm py-8">
          No predictions recorded yet. Submit a V2 quote to begin accumulating baselines.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-xs text-gray-500 uppercase">
              <tr>
                <th className="px-4 py-2 text-left">Time</th>
                <th className="px-4 py-2 text-left">Champion</th>
                <th className="px-4 py-2 text-left">Status</th>
                <th className="px-4 py-2 text-left">Comparison</th>
                <th className="px-4 py-2 text-right">Premium Δ</th>
                <th className="px-4 py-2 text-right">Latency</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {history.items.map(item => (
                <tr
                  key={item.id}
                  data-testid="shadow-history-row"
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => onSelect(item)}
                >
                  <td className="px-4 py-2 text-gray-600 text-xs whitespace-nowrap">
                    {new Date(item.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-gray-700 text-xs">{item.champion_version}</td>
                  <td className="px-4 py-2"><StatusChip status={item.status} /></td>
                  <td className="px-4 py-2"><ComparisonChip result={item.comparison_result} /></td>
                  <td className="px-4 py-2 text-right text-xs">
                    {item.premium_difference != null
                      ? `₹${item.premium_difference.toFixed(0)}`
                      : '—'}
                  </td>
                  <td className="px-4 py-2 text-right text-xs">
                    {item.prediction_latency_ms != null ? `${item.prediction_latency_ms}ms` : '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      data-testid="shadow-detail-btn"
                      className="text-xs text-blue-600 hover:underline"
                      onClick={e => { e.stopPropagation(); onSelect(item) }}
                    >
                      Detail
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {history.total_pages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            Page {history.page} of {history.total_pages} ({history.total} records)
          </p>
          <div className="flex gap-2">
            <button
              data-testid="shadow-prev-page"
              disabled={page <= 1}
              onClick={() => onPageChange(page - 1)}
              className="text-xs px-2 py-1 border rounded disabled:opacity-40"
            >
              ← Prev
            </button>
            <button
              data-testid="shadow-next-page"
              disabled={page >= history.total_pages}
              onClick={() => onPageChange(page + 1)}
              className="text-xs px-2 py-1 border rounded disabled:opacity-40"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Detail drawer ──────────────────────────────────────────────────────────────

function DetailDrawer({
  record,
  onClose,
}: {
  record: ShadowPredictionRecord
  onClose: () => void
}) {
  return (
    <div
      data-testid="shadow-detail-drawer"
      className="fixed inset-0 z-50 flex justify-end"
    >
      <div className="absolute inset-0 bg-black bg-opacity-20" onClick={onClose} />
      <div className="relative bg-white w-full max-w-lg shadow-xl overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-800">Shadow Prediction Detail</h3>
          <button
            data-testid="shadow-detail-close"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 text-xl"
          >
            ✕
          </button>
        </div>

        <dl className="space-y-3 text-sm">
          <div>
            <dt className="text-xs text-gray-500">ID</dt>
            <dd className="font-mono text-xs text-gray-700 break-all">{record.id}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Status</dt>
            <dd><StatusChip status={record.status} /></dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Champion</dt>
            <dd>{record.champion_model_name} @ {record.champion_version}</dd>
          </div>
          {record.challenger_model_name && (
            <div>
              <dt className="text-xs text-gray-500">Challenger</dt>
              <dd>{record.challenger_model_name} @ {record.challenger_version}</dd>
            </div>
          )}
          {record.comparison_result && (
            <div>
              <dt className="text-xs text-gray-500">Comparison Result</dt>
              <dd><ComparisonChip result={record.comparison_result} /></dd>
            </div>
          )}
          {record.premium_difference != null && (
            <div>
              <dt className="text-xs text-gray-500">Premium Difference</dt>
              <dd>₹{record.premium_difference.toFixed(2)}</dd>
            </div>
          )}
          {record.risk_difference && (
            <div>
              <dt className="text-xs text-gray-500">Risk Level Delta</dt>
              <dd>{record.risk_difference}</dd>
            </div>
          )}
          {record.prediction_latency_ms != null && (
            <div>
              <dt className="text-xs text-gray-500">Prediction Latency</dt>
              <dd>{record.prediction_latency_ms} ms</dd>
            </div>
          )}
          {record.notes && (
            <div>
              <dt className="text-xs text-gray-500">Notes</dt>
              <dd className="text-gray-600">{record.notes}</dd>
            </div>
          )}
          {record.champion_prediction_json && (
            <div>
              <dt className="text-xs text-gray-500 mb-1">Champion Prediction</dt>
              <dd>
                <pre
                  data-testid="shadow-champion-json"
                  className="bg-gray-50 rounded p-2 text-xs overflow-x-auto"
                >
                  {JSON.stringify(record.champion_prediction_json, null, 2)}
                </pre>
              </dd>
            </div>
          )}
          {record.challenger_prediction_json && (
            <div>
              <dt className="text-xs text-gray-500 mb-1">Challenger Prediction</dt>
              <dd>
                <pre
                  data-testid="shadow-challenger-json"
                  className="bg-gray-50 rounded p-2 text-xs overflow-x-auto"
                >
                  {JSON.stringify(record.challenger_prediction_json, null, 2)}
                </pre>
              </dd>
            </div>
          )}
          <div>
            <dt className="text-xs text-gray-500">Recorded At</dt>
            <dd>{new Date(record.created_at).toLocaleString()}</dd>
          </div>
          {record.quote_id && (
            <div>
              <dt className="text-xs text-gray-500">Quote ID</dt>
              <dd className="font-mono text-xs text-gray-700 break-all">{record.quote_id}</dd>
            </div>
          )}
        </dl>
      </div>
    </div>
  )
}

// ─── Governance alert ───────────────────────────────────────────────────────────

function GovernanceAlert({ status }: { status: ShadowStatusResponse }) {
  if (status.challenger_available) return null
  return (
    <div
      data-testid="shadow-governance-alert"
      className="rounded border border-blue-200 bg-blue-50 p-3 mb-6 text-sm"
    >
      <strong>Governance Notice:</strong> No V5 challenger model is registered.
      All {status.total_shadow_predictions} champion predictions are being logged as baselines
      and will be compared automatically when a challenger is promoted.
    </div>
  )
}

// ─── Main ShadowPanel ──────────────────────────────────────────────────────────

export default function ShadowPanel() {
  const [shadowStatus, setShadowStatus] = useState<ShadowStatusResponse | null>(null)
  const [stats, setStats] = useState<ShadowStatisticsResponse | null>(null)
  const [history, setHistory] = useState<ShadowHistoryResponse | null>(null)
  const [selectedRecord, setSelectedRecord] = useState<ShadowPredictionRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState('')
  const [windowDays, setWindowDays] = useState(30)

  const loadAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [s, st, h] = await Promise.all([
        insuranceApi.getShadowStatus(),
        insuranceApi.getShadowStatistics(windowDays),
        insuranceApi.getShadowHistory({
          page,
          page_size: 20,
          ...(statusFilter ? { status: statusFilter } : {}),
        }),
      ])
      setShadowStatus(s)
      setStats(st)
      setHistory(h)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load shadow data')
    } finally {
      setLoading(false)
    }
  }, [page, statusFilter, windowDays])

  useEffect(() => {
    void loadAll()
  }, [loadAll])

  if (loading) {
    return (
      <div data-testid="shadow-loading" className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        <span className="ml-3 text-gray-500">Loading shadow deployment data…</span>
      </div>
    )
  }

  if (error) {
    return (
      <div data-testid="shadow-error" className="text-red-600 text-sm p-6">
        Error: {error}
        <button onClick={() => void loadAll()} className="ml-4 text-blue-600 underline">
          Retry
        </button>
      </div>
    )
  }

  return (
    <div data-testid="shadow-panel" className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Shadow Deployment Framework</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Phase 4C — Read-only observation layer. No champion modifications.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            data-testid="shadow-window-select"
            value={windowDays}
            onChange={e => { setWindowDays(Number(e.target.value)); setPage(1) }}
            className="text-xs border border-gray-200 rounded px-2 py-1"
          >
            <option value={7}>7 days</option>
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
          </select>
          <button
            data-testid="shadow-refresh-btn"
            onClick={() => void loadAll()}
            className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {shadowStatus && (
        <>
          <GovernanceAlert status={shadowStatus} />
          <FrameworkBanner status={shadowStatus} />
          <StatusOverview status={shadowStatus} />
        </>
      )}

      {stats && <StatisticsPanel stats={stats} />}

      {history && (
        <HistoryTable
          history={history}
          page={page}
          onPageChange={p => setPage(p)}
          onSelect={rec => setSelectedRecord(rec)}
          statusFilter={statusFilter}
          onStatusFilter={s => { setStatusFilter(s); setPage(1) }}
        />
      )}

      {selectedRecord && (
        <DetailDrawer
          record={selectedRecord}
          onClose={() => setSelectedRecord(null)}
        />
      )}
    </div>
  )
}
