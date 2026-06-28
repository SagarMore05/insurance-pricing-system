import clsx from 'clsx'
import type { ChampionMetadataV2 } from '../../types/v2'

interface Props {
  championMetadata: ChampionMetadataV2
}

function friendlyModelName(raw: string, role: 'frequency' | 'severity'): string {
  const freq: Record<string, string> = {
    'FrequencyModel':    'Risk Assessment',
    'XGBClassifier-V1': 'Risk Assessment (V1)',
  }
  const sev: Record<string, string> = {
    'CatBoostSeverityWrapper': 'Repair Cost Analysis',
    'XGBRegressor-V1':         'Repair Cost Analysis (V1)',
  }
  const map = role === 'frequency' ? freq : sev
  return map[raw] ?? (role === 'frequency' ? 'Risk Assessment' : 'Repair Cost Analysis')
}

function ModelChip({ label, friendlyName, color }: {
  label: string
  friendlyName: string
  color: 'blue' | 'purple'
}) {
  const dotColor  = color === 'blue' ? 'bg-blue-500'  : 'bg-purple-500'
  const bgColor   = color === 'blue' ? 'bg-blue-50'   : 'bg-purple-50'
  const textColor = color === 'blue' ? 'text-blue-900' : 'text-purple-900'

  return (
    <div className={clsx('rounded-lg p-3 border border-gray-100', bgColor)}>
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <div className="flex items-center gap-2">
        <div className={clsx('w-2 h-2 rounded-full flex-shrink-0', dotColor)} />
        <span className={clsx('text-xs font-semibold', textColor)}>{friendlyName}</span>
      </div>
    </div>
  )
}

export default function GovernancePanel({ championMetadata }: Props) {
  const { frequency_model, severity_model, v4_ready } = championMetadata

  const freqFriendly = friendlyModelName(frequency_model, 'frequency')
  const sevFriendly  = friendlyModelName(severity_model,  'severity')

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-slate-100 rounded-lg flex items-center justify-center">
            <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-gray-900">About This Quote</h4>
            <p className="text-xs text-gray-400">AI model transparency</p>
          </div>
        </div>

        {/* Status badge */}
        <span className={clsx(
          'text-xs font-semibold px-3 py-1 rounded-full border',
          v4_ready
            ? 'bg-green-50 text-green-700 border-green-200'
            : 'bg-amber-50 text-amber-700 border-amber-200',
        )}>
          {v4_ready ? 'Enhanced AI' : 'Standard AI'}
        </span>
      </div>

      {/* AI models used */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <ModelChip
          label="Accident likelihood"
          friendlyName={freqFriendly}
          color="blue"
        />
        <ModelChip
          label="Repair cost estimate"
          friendlyName={sevFriendly}
          color="purple"
        />
      </div>

      {/* What this means for the customer */}
      <div className="bg-slate-50 rounded-lg px-3 py-2.5 border border-slate-100">
        <div className={clsx(
          'flex items-center gap-1.5 text-xs font-medium',
          v4_ready ? 'text-green-600' : 'text-amber-600',
        )}>
          <div className={clsx(
            'w-1.5 h-1.5 rounded-full',
            v4_ready ? 'bg-green-500' : 'bg-amber-400 animate-pulse',
          )} />
          {v4_ready
            ? 'Enhanced AI — using your full profile with vehicle and area data'
            : 'Standard AI — core profile analysis'}
        </div>
      </div>

      <p className="text-xs text-gray-400 mt-2 leading-relaxed">
        {v4_ready
          ? 'Your premium was calculated using two independently tested AI models. '
          + 'All decisions are logged and audited.'
          : 'Your premium was calculated using our standard AI models. '
          + 'Switch to the enhanced quote for a more personalised price.'}
      </p>
    </div>
  )
}
