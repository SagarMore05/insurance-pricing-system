import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type {
  AlgorithmResult,
  AlgorithmFrequencyMetrics,
  AlgorithmSeverityMetrics,
  ModelSelectionRun,
} from '../../types'

// ── Utilities ──────────────────────────────────────────────────────────────────

const ALGO_LABELS: Record<string, string> = {
  xgboost:  'XGBoost',
  lightgbm: 'LightGBM',
  catboost: 'CatBoost',
}
const ALGO_COLORS: Record<string, string> = {
  xgboost:  'text-blue-700',
  lightgbm: 'text-green-700',
  catboost: 'text-purple-700',
}

function fmt(v: number | null | undefined, decimals = 4): string {
  if (v == null) return '—'
  return v.toFixed(decimals)
}
function fmtMs(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v.toFixed(3)} ms`
}
function fmtSec(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${v.toFixed(1)}s`
}
function fmtINR(v: number | null | undefined): string {
  if (v == null) return '—'
  return `₹${Math.round(v).toLocaleString('en-IN')}`
}

// ── Sub-components ────────────────────────────────────────────────────────────

function WinnerBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-bold rounded-full bg-amber-100 text-amber-800 border border-amber-300">
      Winner
    </span>
  )
}

function StatusChip({ status }: { status: string }) {
  return status === 'success' ? (
    <span className="text-green-600 text-xs font-medium">✓</span>
  ) : (
    <span className="text-red-500 text-xs font-medium">✗</span>
  )
}

function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-3">
      <p className="text-sm font-semibold text-gray-800">{title}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}

// ── Frequency comparison table ────────────────────────────────────────────────

function FrequencyTable({ results }: { results: AlgorithmResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100">
            <th className="text-left py-2 text-gray-400 font-medium pr-3">Algorithm</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">ROC-AUC</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">F1</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">Precision</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">Recall</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">Brier</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">Train Time</th>
            <th className="text-right py-2 text-gray-400 font-medium pl-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map(r => {
            const m = r.metrics as AlgorithmFrequencyMetrics
            const isWinner = r.is_winner
            const rowCls = isWinner
              ? 'bg-amber-50 border-b border-amber-100'
              : 'border-b border-gray-50 hover:bg-gray-50'
            return (
              <tr key={r.algorithm} className={rowCls}>
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-2">
                    <span className={`font-semibold ${ALGO_COLORS[r.algorithm] ?? 'text-gray-800'}`}>
                      {ALGO_LABELS[r.algorithm] ?? r.algorithm}
                    </span>
                    {isWinner && <WinnerBadge />}
                  </div>
                </td>
                <td className={`text-right py-2 px-2 font-mono tabular-nums ${isWinner ? 'font-bold text-amber-800' : 'text-gray-700'}`}>
                  {fmt(m?.roc_auc)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmt(m?.f1)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmt(m?.precision)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmt(m?.recall)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmt(m?.brier_score)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-500">
                  {fmtSec(r.training_time_sec)}
                </td>
                <td className="text-right py-2 pl-2">
                  <StatusChip status={r.status} />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Severity comparison table ─────────────────────────────────────────────────
// Winner is selected by CV-R² (3-fold cross-validation), not Test R².
// Both metrics are shown so the selection rationale is unambiguous.

function SeverityTable({ results }: { results: AlgorithmResult[] }) {
  const hasCvR2 = results.some(r => (r.metrics as AlgorithmSeverityMetrics)?.cv_r2_mean != null)
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100">
            <th className="text-left py-2 text-gray-400 font-medium pr-3">Algorithm</th>
            {hasCvR2 && (
              <th className="text-right py-2 text-gray-400 font-medium px-2">
                CV-R²
                <span className="block text-gray-300 font-normal" style={{ fontSize: '0.65rem' }}>
                  selection metric
                </span>
              </th>
            )}
            <th className="text-right py-2 text-gray-400 font-medium px-2">Test R²</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">RMSE</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">MAE</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">MAPE%</th>
            <th className="text-right py-2 text-gray-400 font-medium px-2">Train Time</th>
            <th className="text-right py-2 text-gray-400 font-medium pl-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map(r => {
            const m = r.metrics as AlgorithmSeverityMetrics
            const isWinner = r.is_winner
            const rowCls = isWinner
              ? 'bg-amber-50 border-b border-amber-100'
              : 'border-b border-gray-50 hover:bg-gray-50'
            return (
              <tr key={r.algorithm} className={rowCls}>
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-2">
                    <span className={`font-semibold ${ALGO_COLORS[r.algorithm] ?? 'text-gray-800'}`}>
                      {ALGO_LABELS[r.algorithm] ?? r.algorithm}
                    </span>
                    {isWinner && <WinnerBadge />}
                  </div>
                </td>
                {hasCvR2 && (
                  <td className={`text-right py-2 px-2 font-mono tabular-nums ${isWinner ? 'font-bold text-amber-800' : 'text-gray-700'}`}>
                    {m?.cv_r2_mean != null ? (
                      <>
                        {fmt(m.cv_r2_mean)}
                        {m.cv_r2_std != null && (
                          <span className="text-gray-400 font-normal"> ±{fmt(m.cv_r2_std)}</span>
                        )}
                      </>
                    ) : '—'}
                  </td>
                )}
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-600">
                  {fmt(m?.r2)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmtINR(m?.rmse)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {fmtINR(m?.mae)}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-700">
                  {m?.mape_pct != null ? `${m.mape_pct.toFixed(2)}%` : '—'}
                </td>
                <td className="text-right py-2 px-2 font-mono tabular-nums text-gray-500">
                  {fmtSec(r.training_time_sec)}
                </td>
                <td className="text-right py-2 pl-2">
                  <StatusChip status={r.status} />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {hasCvR2 && (
        <p className="text-xs text-gray-400 mt-2">
          Winner selected by CV-R² (3-fold cross-validation). Test R² shown for reference only.
        </p>
      )}
    </div>
  )
}

// ── History table ─────────────────────────────────────────────────────────────

function HistoryTable({ runs }: { runs: ModelSelectionRun[] }) {
  if (runs.length === 0) {
    return <p className="text-xs text-gray-400 text-center py-4">No previous runs.</p>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100">
            <th className="text-left py-1.5 text-gray-400 font-medium">#</th>
            <th className="text-left py-1.5 text-gray-400 font-medium">Date</th>
            <th className="text-left py-1.5 text-gray-400 font-medium">Frequency Winner</th>
            <th className="text-left py-1.5 text-gray-400 font-medium">Freq AUC</th>
            <th className="text-left py-1.5 text-gray-400 font-medium">Severity Winner</th>
            <th className="text-left py-1.5 text-gray-400 font-medium">Sev R²</th>
            <th className="text-right py-1.5 text-gray-400 font-medium">Elapsed</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(run => {
            const freqWin = run.frequency_results.find(r => r.is_winner)
            const sevWin  = run.severity_results.find(r => r.is_winner)
            const freqAUC = (freqWin?.metrics as AlgorithmFrequencyMetrics)?.roc_auc
            const sevR2   = (sevWin?.metrics  as AlgorithmSeverityMetrics)?.r2
            return (
              <tr key={run.run_id} className="border-b border-gray-50 hover:bg-gray-50">
                <td className="py-1.5 text-gray-500 font-mono">{run.run_number}</td>
                <td className="py-1.5 text-gray-600">
                  {run.training_timestamp.slice(0, 16).replace('T', ' ')}
                </td>
                <td className="py-1.5">
                  <span className={`font-semibold ${ALGO_COLORS[run.frequency_winner] ?? ''}`}>
                    {ALGO_LABELS[run.frequency_winner] ?? run.frequency_winner}
                  </span>
                </td>
                <td className="py-1.5 font-mono tabular-nums text-gray-700">
                  {fmt(freqAUC)}
                </td>
                <td className="py-1.5">
                  <span className={`font-semibold ${ALGO_COLORS[run.severity_winner] ?? ''}`}>
                    {ALGO_LABELS[run.severity_winner] ?? run.severity_winner}
                  </span>
                </td>
                <td className="py-1.5 font-mono tabular-nums text-gray-700">
                  {fmt(sevR2)}
                </td>
                <td className="py-1.5 text-right font-mono tabular-nums text-gray-500">
                  {fmtSec(run.total_elapsed_sec)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ModelSelectionPanel() {
  const qc = useQueryClient()
  const [showHistory, setShowHistory] = useState(false)

  const { data: latest, isLoading, error } = useQuery({
    queryKey: ['model-selection-latest'],
    queryFn: insuranceApi.getModelSelectionLatest,
    staleTime: 60_000,
  })

  const { data: history, isLoading: histLoading } = useQuery({
    queryKey: ['model-selection-history'],
    queryFn: () => insuranceApi.getModelSelectionHistory({ page: 1, page_size: 20 }),
    staleTime: 60_000,
    enabled: showHistory,
  })

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ['model-selection-status'],
    queryFn: insuranceApi.getModelSelectionRetrainStatus,
    refetchInterval: 5_000,
    staleTime: 3_000,
  })

  const retrainMutation = useMutation({
    mutationFn: insuranceApi.triggerMultiAlgorithmRetrain,
    onSuccess: () => {
      refetchStatus()
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['model-selection-latest'] })
        qc.invalidateQueries({ queryKey: ['model-selection-history'] })
        qc.invalidateQueries({ queryKey: ['model-selection-status'] })
      }, 2000)
    },
  })

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2].map(i => (
          <div key={i} className="rounded-xl border border-gray-200 bg-white p-4 animate-pulse h-32" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load model selection data. Check that the backend is running.
      </div>
    )
  }

  const run  = latest?.latest_run
  const isRunning = status?.running === true

  return (
    <div className="space-y-4">

      {/* Header */}
      <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 px-4 py-3 border-b border-indigo-100 flex items-center justify-between flex-wrap gap-3">
          <div>
            <p className="text-xs text-indigo-500 font-medium uppercase tracking-wide">
              Enterprise Automatic Model Selection
            </p>
            <p className="text-base font-bold text-gray-900">Multi-Algorithm Selection Engine</p>
            <p className="text-xs text-gray-400 mt-0.5">
              XGBoost · LightGBM · CatBoost — evaluated on every training run
            </p>
          </div>
          <div className="flex items-center gap-2">
            {isRunning && (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-700 bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-full animate-pulse">
                <span className="w-2 h-2 rounded-full bg-blue-500" />
                Training in progress…
              </span>
            )}
            <button
              onClick={() => retrainMutation.mutate()}
              disabled={isRunning || retrainMutation.isPending}
              className="px-4 py-2 text-xs font-semibold bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
            >
              {isRunning ? 'Running…' : 'Run Training'}
            </button>
          </div>
        </div>

        {retrainMutation.isError && (
          <div className="px-4 py-2 text-xs text-red-600 bg-red-50 border-b border-red-100">
            {(retrainMutation.error as Error).message}
          </div>
        )}
        {retrainMutation.isSuccess && (
          <div className="px-4 py-2 text-xs text-green-700 bg-green-50 border-b border-green-100">
            Training started in background. Results will appear here when complete.
          </div>
        )}
        {status?.error && (
          <div className="px-4 py-2 text-xs text-red-600 bg-red-50 border-b border-red-100">
            Last run error: {status.error}
          </div>
        )}

        {/* Run meta */}
        {run && (
          <div className="px-4 py-3 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div>
              <p className="text-gray-400">Run</p>
              <p className="font-semibold text-gray-800">#{run.run_number} · {run.version}</p>
            </div>
            <div>
              <p className="text-gray-400">Date</p>
              <p className="font-semibold text-gray-800">
                {run.training_timestamp.slice(0, 16).replace('T', ' ')}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Dataset</p>
              <p className="font-semibold text-gray-800 truncate" title={run.dataset}>
                {run.dataset}
              </p>
            </div>
            <div>
              <p className="text-gray-400">Total rows</p>
              <p className="font-semibold text-gray-800">
                {run.n_total_rows?.toLocaleString() ?? '—'}
                {' · '}
                <span className="text-gray-500">
                  {run.n_train_rows?.toLocaleString() ?? '—'} train
                </span>
              </p>
            </div>
          </div>
        )}
      </div>

      {!latest?.has_runs ? (
        <div className="rounded-xl border border-dashed border-gray-200 bg-gray-50 p-8 text-center">
          <p className="text-sm font-medium text-gray-500 mb-1">No training runs yet</p>
          <p className="text-xs text-gray-400">
            Click "Run Training" to execute the multi-algorithm pipeline on{' '}
            <span className="font-mono">motor_insurance_master_dataset_50000.csv</span>
          </p>
        </div>
      ) : run && (
        <>
          {/* Frequency model section */}
          <div className="rounded-xl border border-blue-100 bg-white shadow-sm overflow-hidden">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-4 py-3 border-b border-blue-100 flex items-center justify-between flex-wrap gap-2">
              <div>
                <p className="text-xs text-blue-500 font-medium uppercase tracking-wide">
                  Frequency Model
                </p>
                <p className="text-sm font-bold text-gray-900">Claim Occurrence Prediction</p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-gray-500">Winner:</span>
                <span className={`font-bold ${ALGO_COLORS[run.frequency_winner]}`}>
                  {ALGO_LABELS[run.frequency_winner] ?? run.frequency_winner}
                </span>
                {run.shap_generated_frequency && (
                  <span className="text-gray-400">· SHAP ✓</span>
                )}
              </div>
            </div>
            <div className="p-4">
              <FrequencyTable results={run.frequency_results} />
              {run.promotion_reason_frequency && (
                <div className="mt-3 px-3 py-2 bg-amber-50 border border-amber-100 rounded-lg">
                  <p className="text-xs text-amber-800">
                    <span className="font-semibold">Promotion: </span>
                    {run.promotion_reason_frequency}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Severity model section */}
          <div className="rounded-xl border border-purple-100 bg-white shadow-sm overflow-hidden">
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 px-4 py-3 border-b border-purple-100 flex items-center justify-between flex-wrap gap-2">
              <div>
                <p className="text-xs text-purple-500 font-medium uppercase tracking-wide">
                  Severity Model
                </p>
                <p className="text-sm font-bold text-gray-900">Claim Amount Prediction</p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-gray-500">Winner:</span>
                <span className={`font-bold ${ALGO_COLORS[run.severity_winner]}`}>
                  {ALGO_LABELS[run.severity_winner] ?? run.severity_winner}
                </span>
                {run.shap_generated_severity && (
                  <span className="text-gray-400">· SHAP ✓</span>
                )}
              </div>
            </div>
            <div className="p-4">
              <SeverityTable results={run.severity_results} />
              {run.promotion_reason_severity && (
                <div className="mt-3 px-3 py-2 bg-amber-50 border border-amber-100 rounded-lg">
                  <p className="text-xs text-amber-800">
                    <span className="font-semibold">Promotion: </span>
                    {run.promotion_reason_severity}
                  </p>
                </div>
              )}
              {/* Severity note */}
              <p className="text-xs text-gray-400 mt-3 italic">
                Severity trained on claimants only ({run.n_sev_train_rows?.toLocaleString() ?? '—'} rows).
                R² ceiling ≈ 0.01 for this synthetic dataset (claim amounts are independently sampled).
              </p>
            </div>
          </div>

          {/* Training History */}
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
            <div
              className="px-4 py-3 border-b border-gray-100 flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => setShowHistory(v => !v)}
            >
              <p className="text-sm font-semibold text-gray-800">
                Training History{' '}
                <span className="text-gray-400 font-normal text-xs">
                  ({latest.total_runs} run{latest.total_runs !== 1 ? 's' : ''})
                </span>
              </p>
              <span className="text-gray-400 text-xs">{showHistory ? '▲ Hide' : '▼ Show'}</span>
            </div>
            {showHistory && (
              <div className="p-4">
                {histLoading ? (
                  <div className="animate-pulse h-16 bg-gray-100 rounded" />
                ) : (
                  <HistoryTable runs={history?.runs ?? []} />
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
