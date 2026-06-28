import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { insuranceApi } from '../../api/insurance'
import type { GateResults, GovernanceLog, GovernanceEvent } from '../../types'

const RISK_COLORS: Record<string, string> = {
  low:    'bg-green-100 text-green-700',
  medium: 'bg-amber-100 text-amber-700',
  high:   'bg-red-100 text-red-700',
}

const EVENT_COLORS: Record<string, string> = {
  registry_creation: 'bg-blue-100 text-blue-700',
  activation:        'bg-green-100 text-green-700',
  rollback:          'bg-red-100 text-red-700',
  validation:        'bg-amber-100 text-amber-700',
}

function formatTs(iso: string): string {
  try {
    return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
  } catch {
    return iso
  }
}

function GateResultsPanel({ data }: { data: GateResults }) {
  const gateEntries = Object.entries(data.gates)
  const passCount = gateEntries.filter(([, v]) => v === 'PASS').length

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="font-semibold text-gray-900">Deployment Gate Results</h4>
          <p className="text-xs text-gray-400 mt-0.5">
            Run at {formatTs(data.timestamp)}
          </p>
        </div>
        <span className={clsx(
          'text-sm font-bold px-4 py-1.5 rounded-full border',
          data.all_pass
            ? 'bg-green-50 text-green-700 border-green-200'
            : 'bg-red-50 text-red-700 border-red-200',
        )}>
          {passCount}/{gateEntries.length} Gates Passed
        </span>
      </div>

      {/* Gate grid */}
      <div className="grid grid-cols-2 gap-2 mb-5">
        {gateEntries.map(([name, result]) => (
          <div key={name} className={clsx(
            'flex items-center gap-2.5 rounded-lg px-3 py-2.5 border',
            result === 'PASS'
              ? 'bg-green-50 border-green-100'
              : 'bg-red-50 border-red-100',
          )}>
            <span className={clsx(
              'text-sm font-bold flex-shrink-0',
              result === 'PASS' ? 'text-green-600' : 'text-red-600',
            )}>
              {result === 'PASS' ? '✓' : '✕'}
            </span>
            <span className="text-xs text-gray-700 font-medium">{name}</span>
          </div>
        ))}
        {/* Frontend gate */}
        <div className={clsx(
          'flex items-center gap-2.5 rounded-lg px-3 py-2.5 border',
          data.frontend_pass ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100',
        )}>
          <span className={clsx('text-sm font-bold flex-shrink-0', data.frontend_pass ? 'text-green-600' : 'text-red-600')}>
            {data.frontend_pass ? '✓' : '✕'}
          </span>
          <span className="text-xs text-gray-700 font-medium">Frontend Integration</span>
        </div>
      </div>

      {/* Scenario results */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Test Scenarios</p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100">
                {['Scenario', 'Claim Prob', 'Exp. Claim', 'Premium', 'Risk', 'Result'].map(h => (
                  <th key={h} className="text-left py-2 px-2 text-gray-400 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.scenario_results.map((s, i) => (
                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2 px-2 font-medium text-gray-800">{s.name}</td>
                  <td className="py-2 px-2 text-gray-600">{(s.claim_prob * 100).toFixed(1)}%</td>
                  <td className="py-2 px-2 text-gray-600">
                    ₹{s.expected_claim.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="py-2 px-2 text-gray-600">
                    ₹{s.premium.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="py-2 px-2">
                    <span className={clsx('px-1.5 py-0.5 rounded-full text-xs font-medium capitalize', RISK_COLORS[s.risk] ?? '')}>
                      {s.risk}
                    </span>
                  </td>
                  <td className="py-2 px-2">
                    <span className={clsx('font-bold', s.passed ? 'text-green-600' : 'text-red-600')}>
                      {s.passed ? '✓ PASS' : '✕ FAIL'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function GovernanceEventRow({ event }: { event: GovernanceEvent }) {
  return (
    <div className="flex gap-4 pb-5 relative">
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        <div className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 border-2',
          event.decision.includes('ACTIVATED')
            ? 'bg-green-100 border-green-400 text-green-700'
            : event.decision.includes('VALIDATED')
            ? 'bg-blue-100 border-blue-400 text-blue-700'
            : 'bg-gray-100 border-gray-300 text-gray-600',
        )}>
          {event.model_type === 'frequency' ? 'F' : 'S'}
        </div>
        <div className="flex-1 w-px bg-gray-200 mt-1" />
      </div>

      {/* Event content */}
      <div className="flex-1 pb-2">
        <div className="flex items-center gap-2 flex-wrap mb-1">
          <span className={clsx(
            'text-xs font-semibold px-2 py-0.5 rounded-full',
            EVENT_COLORS[event.event_type] ?? 'bg-gray-100 text-gray-600',
          )}>
            {event.event_type.replace(/_/g, ' ')}
          </span>
          <span className="text-xs font-bold text-gray-800">{event.challenger}</span>
          {event.incumbent && (
            <span className="text-xs text-gray-400">vs {event.incumbent}</span>
          )}
          <span className="ml-auto text-xs text-gray-400">{formatTs(event.timestamp)}</span>
        </div>

        <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
          <div className="flex items-center gap-4 text-xs mb-2">
            <span className="text-gray-500">Decision: <span className="font-semibold text-gray-800">{event.decision}</span></span>
            <span className="text-gray-500">
              {event.metric}: <span className="font-semibold text-blue-700">{event.new_score.toFixed(4)}</span>
              {event.old_score !== null && (
                <span className="text-gray-400 ml-1">(was {event.old_score.toFixed(4)})</span>
              )}
            </span>
          </div>
          <p className="text-xs text-gray-600 leading-relaxed">{event.notes}</p>
        </div>
      </div>
    </div>
  )
}

export default function GovernanceAdminPanel() {
  const { data: gateResults, isLoading: gatesLoading } = useQuery({
    queryKey: ['admin-gate-results'],
    queryFn: insuranceApi.getGateResults,
    staleTime: 10 * 60_000,
  })

  const { data: govLog, isLoading: logLoading } = useQuery({
    queryKey: ['admin-governance-log'],
    queryFn: insuranceApi.getGovernanceLog,
    staleTime: 10 * 60_000,
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="font-semibold text-gray-900">Model Governance</h3>
        <p className="text-xs text-gray-400 mt-0.5">
          Deployment gates, activation history, and full audit log
        </p>
      </div>

      {/* Gate results */}
      {gatesLoading ? (
        <div className="card animate-pulse h-48 bg-gray-100" />
      ) : gateResults ? (
        <GateResultsPanel data={gateResults} />
      ) : (
        <div className="card text-sm text-red-600 text-center py-8">Failed to load gate results.</div>
      )}

      {/* Governance audit log */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="font-semibold text-gray-900">Model Lifecycle Audit Log</h4>
            <p className="text-xs text-gray-400 mt-0.5">
              Append-only — {govLog?.events.length ?? '…'} events recorded
            </p>
          </div>
          <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full">
            Schema v{govLog?.schema_version ?? '…'}
          </span>
        </div>

        {logLoading ? (
          <div className="space-y-3">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="animate-pulse h-16 bg-gray-100 rounded-lg" />
            ))}
          </div>
        ) : govLog ? (
          <div>
            {[...govLog.events].reverse().map((event, i) => (
              <GovernanceEventRow key={i} event={event} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-red-600 text-center py-6">Failed to load governance log.</p>
        )}
      </div>
    </div>
  )
}
