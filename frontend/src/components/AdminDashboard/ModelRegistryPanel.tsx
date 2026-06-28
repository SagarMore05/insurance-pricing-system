import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { insuranceApi } from '../../api/insurance'
import type { FrequencyRegistryEntry, SeverityRegistryEntry, ComparisonCell } from '../../types'

function fmt(n: number | null) {
  if (n === null || n === undefined) return '—'
  return `₹${Math.round(n).toLocaleString('en-IN')}`
}

function fmtPct(n: number | null) {
  if (n === null || n === undefined) return '—'
  return `${n.toFixed(2)}%`
}

function fmtNum(n: number | null, digits = 4) {
  if (n === null || n === undefined) return '—'
  return n.toFixed(digits)
}

function StatusChip({ status }: { status: string }) {
  const colorMap: Record<string, string> = {
    production: 'bg-green-100 text-green-800 border-green-200',
    active: 'bg-green-100 text-green-800 border-green-200',
    candidate: 'bg-amber-100 text-amber-800 border-amber-200',
    validated_not_active: 'bg-amber-100 text-amber-800 border-amber-200',
    archived: 'bg-gray-100 text-gray-600 border-gray-200',
  }
  const labelMap: Record<string, string> = {
    production: 'Production',
    active: 'Active',
    candidate: 'Candidate',
    validated_not_active: 'Validated — Not Active',
    archived: 'Archived',
  }
  const dotClass =
    status === 'production' || status === 'active'
      ? 'bg-green-500'
      : status === 'candidate' || status === 'validated_not_active'
      ? 'bg-amber-400 animate-pulse'
      : 'bg-gray-400'
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-0.5 rounded-full border',
      colorMap[status] ?? 'bg-gray-100 text-gray-600 border-gray-200',
    )}>
      <span className={clsx('w-1.5 h-1.5 rounded-full', dotClass)} />
      {labelMap[status] ?? status}
    </span>
  )
}

function MetricRow({ label, value, highlight }: {
  label: string
  value: string
  highlight?: 'green' | 'red' | 'blue' | 'amber'
}) {
  const cls =
    highlight === 'green' ? 'text-green-700 font-semibold' :
    highlight === 'red'   ? 'text-red-700 font-semibold' :
    highlight === 'blue'  ? 'text-blue-700 font-semibold' :
    highlight === 'amber' ? 'text-amber-700 font-semibold' :
    'text-gray-800'
  return (
    <div className="flex justify-between items-center py-1 border-b border-gray-50 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={clsx('text-xs tabular-nums', cls)}>{value}</span>
    </div>
  )
}

function VerdictBadge({ verdict, delta }: { verdict: string; delta: number | null }) {
  if (verdict === 'unavailable' || delta === null) {
    return <span className="text-xs text-gray-400">—</span>
  }
  const sign = delta >= 0 ? '+' : ''
  const improved = verdict === 'improved'
  const declined = verdict === 'declined'
  return (
    <span className={clsx(
      'inline-flex items-center gap-1 text-xs font-medium px-1.5 py-0.5 rounded',
      improved ? 'bg-green-50 text-green-700' :
      declined ? 'bg-red-50 text-red-700' :
      'bg-gray-50 text-gray-500',
    )}>
      {improved ? '▲' : declined ? '▼' : '='} {sign}{delta.toFixed(4)}
    </span>
  )
}

function FrequencyProductionCard({ model }: { model: FrequencyRegistryEntry }) {
  const m = model.metrics
  const t = model.targets
  return (
    <div className="rounded-lg border border-green-200 bg-green-50 p-3 mb-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-green-700 uppercase tracking-wide">
          Production — Currently Serving
        </p>
        <StatusChip status="production" />
      </div>
      <p className="text-xs text-gray-600 mb-2">
        {model.algorithm} · {model.version} · {model.feature_count} features
      </p>
      <div className="space-y-0.5">
        <MetricRow label="ROC-AUC"     value={fmtNum(m.auc_roc)}     highlight="green" />
        <MetricRow label="F1 Score"    value={fmtNum(m.f1)} />
        <MetricRow label="Precision"   value={fmtNum(m.precision)} />
        <MetricRow label="Recall"      value={fmtNum(m.recall)} />
        <MetricRow label="Brier Score" value={fmtNum(m.brier_score)} />
        {m.threshold !== null && m.threshold !== undefined && (
          <MetricRow label="Threshold" value={fmtNum(m.threshold)} />
        )}
      </div>
      {t != null && (
        <p className="text-xs text-green-600 mt-2 font-medium">
          {t.targets_hit}/{t.targets_total} targets met
        </p>
      )}
      {model.dataset_ceiling?.theoretical_auc_ceiling != null && (
        <p className="text-xs text-amber-600 mt-1">
          AUC ceiling ≈{model.dataset_ceiling.theoretical_auc_ceiling} (dataset constraint)
        </p>
      )}
    </div>
  )
}

function FrequencyCandidateCard({ model }: { model: FrequencyRegistryEntry }) {
  const m = model.metrics
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 mb-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide">
          Candidate — Not Yet Active
        </p>
        <StatusChip status="candidate" />
      </div>
      <p className="text-xs text-gray-600 mb-2">
        {model.algorithm} · {model.version} · {model.feature_count} features
      </p>
      <div className="space-y-0.5">
        <MetricRow label="ROC-AUC"     value={fmtNum(m.auc_roc)}     highlight="amber" />
        <MetricRow label="F1 Score"    value={fmtNum(m.f1)} />
        <MetricRow label="Precision"   value={fmtNum(m.precision)} />
        <MetricRow label="Recall"      value={fmtNum(m.recall)} />
        <MetricRow label="Brier Score" value={fmtNum(m.brier_score)} />
      </div>
      {model.reason_not_promoted && (
        <p className="text-xs text-red-600 mt-2">{model.reason_not_promoted}</p>
      )}
    </div>
  )
}

function SeverityProductionCard({ model }: { model: SeverityRegistryEntry }) {
  const m = model.metrics
  return (
    <div className="rounded-lg border border-green-200 bg-green-50 p-3 mb-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-green-700 uppercase tracking-wide">
          Production — Currently Serving
        </p>
        <StatusChip status="production" />
      </div>
      <p className="text-xs text-gray-600 mb-2">
        {model.algorithm} · {model.version} · {model.feature_count} features
      </p>
      <div className="space-y-0.5">
        <MetricRow label="R²"  value={fmtNum(m.r2)} highlight="green" />
        <MetricRow label="RMSE" value={fmt(m.rmse)} />
        <MetricRow label="MAE"  value={fmt(m.mae)} />
        <MetricRow label="MAPE" value={fmtPct(m.mape_pct)} />
      </div>
      {model.dataset_ceiling?.theoretical_r2_ceiling != null && (
        <p className="text-xs text-amber-600 mt-2">
          R² ceiling ≈{model.dataset_ceiling.theoretical_r2_ceiling} (dataset constraint)
        </p>
      )}
    </div>
  )
}

function SeverityCandidateCard({ model }: { model: SeverityRegistryEntry }) {
  const m = model.metrics
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 mb-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide">
          Candidate — Not Yet Active
        </p>
        <StatusChip status="candidate" />
      </div>
      <p className="text-xs text-gray-600 mb-2">
        {model.algorithm} · {model.version} · {model.feature_count} features
      </p>
      <div className="space-y-0.5">
        <MetricRow label="R²"  value={fmtNum(m.r2)} highlight="amber" />
        <MetricRow label="RMSE" value={fmt(m.rmse)} />
        <MetricRow label="MAE"  value={fmt(m.mae)} />
        {m.mape_pct != null && (
          <MetricRow label="MAPE" value={fmtPct(m.mape_pct)} />
        )}
      </div>
      {model.reason_not_promoted && (
        <p className="text-xs text-red-600 mt-2">{model.reason_not_promoted}</p>
      )}
      {model.comparison_note && (
        <p className="text-xs text-gray-500 mt-1 italic">{model.comparison_note}</p>
      )}
    </div>
  )
}

function ComparisonTable({
  title,
  rows,
  metrics,
}: {
  title: string
  rows: Record<string, ComparisonCell>
  metrics: Array<{ key: string; label: string }>
}) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">{title}</p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-100">
            <th className="text-left py-1.5 text-gray-400 font-medium">Metric</th>
            <th className="text-right py-1.5 text-green-700 font-medium">Production</th>
            <th className="text-right py-1.5 text-amber-700 font-medium">Candidate</th>
            <th className="text-right py-1.5 text-gray-500 font-medium">Change</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map(({ key, label }) => {
            const cell = rows[key]
            if (!cell || typeof cell !== 'object' || !('verdict' in cell)) return null
            return (
              <tr key={key} className="border-b border-gray-50">
                <td className="py-1.5 text-gray-600">{label}</td>
                <td className="text-right py-1.5 text-green-700 font-medium tabular-nums">
                  {cell.production != null ? cell.production.toFixed(4) : '—'}
                </td>
                <td className="text-right py-1.5 text-amber-700 tabular-nums">
                  {cell.candidate != null ? cell.candidate.toFixed(4) : '—'}
                </td>
                <td className="text-right py-1.5">
                  <VerdictBadge verdict={cell.verdict} delta={cell.delta} />
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelRegistryPanel() {
  const { data: registry, isLoading, isError } = useQuery({
    queryKey: ['model-registry-v2'],
    queryFn: insuranceApi.getModelRegistry,
    staleTime: 2 * 60_000,
  })

  const { data: comparison } = useQuery({
    queryKey: ['model-comparison'],
    queryFn: insuranceApi.getModelComparison,
    staleTime: 2 * 60_000,
  })

  if (isLoading) {
    return (
      <div className="grid lg:grid-cols-2 gap-6">
        {[1, 2].map(i => (
          <div key={i} className="card animate-pulse h-96 bg-gray-100" />
        ))}
      </div>
    )
  }

  if (isError || !registry) {
    return (
      <div className="card text-center py-10 text-red-600 text-sm">
        Failed to load model registry. Ensure backend is running and models/registry/model_registry.json exists.
      </div>
    )
  }

  const freqProd = registry.frequency.production
  const freqCand = registry.frequency.candidate
  const sevProd  = registry.severity.production
  const sevCand  = registry.severity.candidate

  const freqRows = comparison?.frequency as Record<string, ComparisonCell> | null | undefined
  const sevRows  = comparison?.severity  as Record<string, ComparisonCell> | null | undefined

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-gray-900">Model Registry</h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Registry v{registry.registry_version} · Updated {registry.last_updated.slice(0, 10)}
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-green-700 bg-green-50 border border-green-200 px-3 py-1.5 rounded-full">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          2 models in production
        </span>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-blue-100 shadow-sm overflow-hidden">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-4 py-3 border-b border-blue-100">
            <p className="text-xs text-blue-500 font-medium uppercase tracking-wide">Frequency Model</p>
            <p className="text-base font-bold text-gray-900">{freqProd.model_name}</p>
            <p className="text-xs text-gray-400 mt-0.5">
              Trained {freqProd.training_timestamp.slice(0, 10)} · {freqProd.dataset}
            </p>
          </div>
          <div className="p-4">
            <FrequencyProductionCard model={freqProd} />
            {freqCand ? (
              <FrequencyCandidateCard model={freqCand} />
            ) : (
              <div className="rounded-lg border border-dashed border-gray-200 p-3 text-xs text-gray-400 text-center">
                No candidate model has been trained using the current master dataset (motor_insurance_master_dataset_50000.csv).
              </div>
            )}
          </div>
          <div className="px-4 pb-3 grid grid-cols-2 gap-2">
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-400">Training rows</p>
              <p className="text-sm font-semibold text-gray-800">
                {(freqProd.n_total_rows ?? 0).toLocaleString()}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-400">Artifact</p>
              <p className="text-xs font-medium text-gray-700 truncate" title={freqProd.artifact_path}>
                {freqProd.artifact_path.split('/').pop()}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-purple-100 shadow-sm overflow-hidden">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 px-4 py-3 border-b border-purple-100">
            <p className="text-xs text-purple-500 font-medium uppercase tracking-wide">Severity Model</p>
            <p className="text-base font-bold text-gray-900">{sevProd.model_name}</p>
            <p className="text-xs text-gray-400 mt-0.5">
              Trained {sevProd.training_timestamp.slice(0, 10)} · {sevProd.dataset}
            </p>
          </div>
          <div className="p-4">
            <SeverityProductionCard model={sevProd} />
            {sevCand ? (
              <SeverityCandidateCard model={sevCand} />
            ) : (
              <div className="rounded-lg border border-dashed border-gray-200 p-3 text-xs text-gray-400 text-center">
                No candidate model has been trained using the current master dataset (motor_insurance_master_dataset_50000.csv).
              </div>
            )}
          </div>
          <div className="px-4 pb-3 grid grid-cols-2 gap-2">
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-400">Training rows</p>
              <p className="text-sm font-semibold text-gray-800">
                {(sevProd.n_total_rows ?? 0).toLocaleString()}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-2">
              <p className="text-xs text-gray-400">Artifact</p>
              <p className="text-xs font-medium text-gray-700 truncate" title={sevProd.artifact_path}>
                {sevProd.artifact_path.split('/').pop()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {(freqRows || sevRows) && (
        <div className="card bg-slate-50 border-slate-200 space-y-6">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Production vs Candidate — Metric Comparison
          </p>
          {freqRows && (
            <ComparisonTable
              title="Frequency Model"
              rows={freqRows}
              metrics={[
                { key: 'auc_roc',     label: 'ROC-AUC'    },
                { key: 'f1',          label: 'F1 Score'    },
                { key: 'precision',   label: 'Precision'   },
                { key: 'recall',      label: 'Recall'      },
                { key: 'brier_score', label: 'Brier Score' },
              ]}
            />
          )}
          {sevRows && (
            <ComparisonTable
              title="Severity Model"
              rows={sevRows}
              metrics={[
                { key: 'r2',   label: 'R²'  },
                { key: 'rmse', label: 'RMSE' },
                { key: 'mae',  label: 'MAE'  },
              ]}
            />
          )}
          {sevCand?.comparison_note && (
            <p className="text-xs text-amber-600 italic">{sevCand.comparison_note}</p>
          )}
        </div>
      )}

      <div className="card bg-slate-50 border-slate-200">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Production Architecture
        </p>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-xs text-gray-400">Preprocessor</p>
            <p className="font-medium text-gray-800">{freqProd.preprocessor}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Features</p>
            <p className="font-medium text-gray-800">{freqProd.feature_count} inputs</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Training Dataset</p>
            <p className="font-medium text-gray-800 truncate" title={freqProd.dataset}>
              {freqProd.dataset}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
