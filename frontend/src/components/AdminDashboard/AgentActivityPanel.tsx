import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type { AgentRun, AgentRunDetail, AgentToolCall } from '../../types'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatDuration(ms: number | null): string {
  if (ms == null) return '—'
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(2)} s`
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

function formatAbsolute(iso: string): string {
  return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

// ─── Badge styles ──────────────────────────────────────────────────────────────

const STATUS_BADGE: Record<string, string> = {
  completed:       'bg-green-100 text-green-700',
  failed:          'bg-red-100 text-red-700',
  running:         'bg-amber-100 text-amber-700',
  awaiting_review: 'bg-purple-100 text-purple-700',
}

const STATUS_LABEL: Record<string, string> = {
  awaiting_review: 'Pending Review',
}

const AGENT_BADGE: Record<string, string> = {
  underwriting: 'bg-blue-100 text-blue-700',
  analytics:    'bg-purple-100 text-purple-700',
}

const INTENT_LABEL: Record<string, string> = {
  approve: 'Approved',
  flag:    'Flagged',
  refer:   'Referred',
  anomaly: 'Anomaly',
  recommendation: 'Recommendation',
  comparison: 'Comparison',
  general: 'General',
}

const INTENT_BADGE: Record<string, string> = {
  approve: 'bg-green-100 text-green-700',
  flag:    'bg-amber-100 text-amber-700',
  refer:   'bg-red-100 text-red-700',
}

const NODE_LABEL: Record<string, string> = {
  fraud_signals: 'Fraud Analysis',
  reasoning:     'LLM Reasoning',
  decision:      'Underwriting Decision',
}

// ─── Tool Call Timeline ───────────────────────────────────────────────────────

function ToolCallTimeline({ toolCalls }: { toolCalls: AgentToolCall[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)

  if (toolCalls.length === 0) {
    return (
      <p className="text-xs text-gray-400 italic py-2">
        No tool call data logged for this run.
      </p>
    )
  }

  const maxDuration = Math.max(...toolCalls.map(t => t.duration_ms ?? 0), 1)

  return (
    <div className="space-y-2">
      {toolCalls.map((tc, i) => (
        <div key={tc.call_id} className="border border-gray-100 rounded-lg overflow-hidden">
          <button
            onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
            className="w-full flex items-center gap-3 px-3 py-2 hover:bg-gray-50 text-left"
          >
            <span className="text-green-500 text-sm">✓</span>
            <span className="text-sm font-medium text-gray-800 flex-1">
              {NODE_LABEL[tc.tool_name] ?? tc.tool_name}
            </span>
            {tc.duration_ms != null && (
              <div className="flex items-center gap-2 shrink-0">
                <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-400 rounded-full"
                    style={{ width: `${Math.max(4, (tc.duration_ms / maxDuration) * 100)}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-16 text-right">
                  {formatDuration(tc.duration_ms)}
                </span>
              </div>
            )}
            <span className="text-gray-300 text-xs ml-1">{expandedIdx === i ? '▾' : '▸'}</span>
          </button>

          {expandedIdx === i && (
            <div className="px-3 pb-3 space-y-2 border-t border-gray-100 bg-gray-50">
              {tc.tool_input && (
                <div className="pt-2">
                  <p className="text-xs font-medium text-gray-500 mb-1">Input</p>
                  <p className="text-xs text-gray-600 leading-relaxed">
                    {typeof tc.tool_input === 'object'
                      ? (tc.tool_input as Record<string, string>).summary ?? JSON.stringify(tc.tool_input)
                      : String(tc.tool_input)}
                  </p>
                </div>
              )}
              {tc.tool_output && (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Output</p>
                  <p className="text-xs text-gray-600 leading-relaxed">{tc.tool_output}</p>
                </div>
              )}
              <p className="text-xs text-gray-300 pt-1">{formatAbsolute(tc.called_at)}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ─── Run Detail Drawer ────────────────────────────────────────────────────────

function RunDetailDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { data: detail, isLoading } = useQuery<AgentRunDetail>({
    queryKey: ['agent-run', runId],
    queryFn: () => insuranceApi.getAgentRun(runId),
    staleTime: 30_000,
  })

  return (
    <tr>
      <td colSpan={7} className="bg-gray-50 border-b border-gray-200 px-4 py-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-semibold text-gray-800">Run Detail</p>
          <button onClick={onClose} className="text-xs text-gray-400 hover:text-gray-600 px-2 py-1 rounded hover:bg-gray-100">
            Close ✕
          </button>
        </div>

        {isLoading && (
          <div className="space-y-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-8 bg-gray-200 rounded animate-pulse" />
            ))}
          </div>
        )}

        {detail && (
          <div className="grid lg:grid-cols-2 gap-4">
            {/* Left: input / output */}
            <div className="space-y-3">
              <div>
                <p className="text-xs font-medium text-gray-500 mb-1">Input</p>
                <p className="text-xs text-gray-700 bg-white border border-gray-200 rounded-lg px-3 py-2 leading-relaxed">
                  {detail.input_text}
                </p>
              </div>
              {detail.output_text && (
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Reasoning Trace</p>
                  <ReasoningTracePreview text={detail.output_text} />
                </div>
              )}
              <div className="text-xs text-gray-400 font-mono break-all">
                Run ID: {detail.run_id}
              </div>
            </div>

            {/* Right: tool call timeline */}
            <div>
              <p className="text-xs font-medium text-gray-500 mb-2">
                Tool Call Timeline ({detail.tool_calls.length} node{detail.tool_calls.length !== 1 ? 's' : ''})
              </p>
              <ToolCallTimeline toolCalls={detail.tool_calls} />
            </div>
          </div>
        )}
      </td>
    </tr>
  )
}

function ReasoningTracePreview({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false)
  const preview = text.slice(0, 300)
  const hasMore = text.length > 300

  return (
    <div>
      <div className="text-xs text-gray-700 bg-white border border-gray-200 rounded-lg px-3 py-2 leading-relaxed">
        {expanded ? text : preview}
        {hasMore && !expanded && '…'}
      </div>
      {hasMore && (
        <button onClick={() => setExpanded(!expanded)} className="mt-1 text-xs text-blue-500 hover:text-blue-600">
          {expanded ? 'Show less' : 'Show full trace'}
        </button>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function AgentActivityPanel() {
  const [page, setPage] = useState(1)
  const [agentTypeFilter, setAgentTypeFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)

  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ['agent-runs', page, agentTypeFilter, statusFilter],
    queryFn: () => insuranceApi.getAgentRuns({
      page,
      page_size: 15,
      agent_type: agentTypeFilter || undefined,
      status: statusFilter || undefined,
    }),
    staleTime: 15_000,
  })

  const runs = (data?.items ?? []) as AgentRun[]

  const toggleRow = (runId: string) => {
    setExpandedRunId(expandedRunId === runId ? null : runId)
  }

  return (
    <div className="card">
      {/* Header + filters */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <h3 className="font-semibold text-gray-900">Agent Activity</h3>
          <p className="text-xs text-gray-400 mt-0.5">All agent_runs records — underwriting and analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={agentTypeFilter}
            onChange={e => { setAgentTypeFilter(e.target.value); setPage(1); setExpandedRunId(null) }}
            className="input-field w-36 py-1.5 text-sm"
          >
            <option value="">All agents</option>
            <option value="underwriting">Underwriting</option>
            <option value="analytics">Analytics</option>
          </select>
          <select
            value={statusFilter}
            onChange={e => { setStatusFilter(e.target.value); setPage(1); setExpandedRunId(null) }}
            className="input-field w-32 py-1.5 text-sm"
          >
            <option value="">All statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="running">Running</option>
            <option value="awaiting_review">Pending Review</option>
          </select>
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            {isFetching ? '↻' : 'Refresh'}
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />
          ))}
        </div>
      ) : runs.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <p className="text-sm">No agent runs recorded yet.</p>
          <p className="text-xs mt-1">Submit an underwriting assessment or ask the Analytics Agent a question.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                {['Agent', 'Intent / Decision', 'Status', 'Duration', 'Input Summary', 'Started', 'Detail'].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-gray-500 font-medium text-xs whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <>
                  <tr
                    key={run.run_id}
                    className={`border-b border-gray-50 hover:bg-gray-50 cursor-pointer transition-colors ${expandedRunId === run.run_id ? 'bg-blue-50' : ''}`}
                    onClick={() => toggleRow(run.run_id)}
                  >
                    {/* Agent type */}
                    <td className="py-2.5 px-3">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${AGENT_BADGE[run.agent_type] ?? 'bg-gray-100 text-gray-600'}`}>
                        {run.agent_type}
                      </span>
                    </td>

                    {/* Intent / decision */}
                    <td className="py-2.5 px-3">
                      {run.intent ? (
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${INTENT_BADGE[run.intent] ?? 'bg-gray-100 text-gray-600'}`}>
                          {INTENT_LABEL[run.intent] ?? run.intent}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-300">—</span>
                      )}
                    </td>

                    {/* Status */}
                    <td className="py-2.5 px-3">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${STATUS_BADGE[run.status] ?? 'bg-gray-100 text-gray-600'}`}>
                        {(run.status === 'running' || run.status === 'awaiting_review') && (
                          <span className={`inline-block w-1.5 h-1.5 rounded-full animate-pulse mr-1 ${run.status === 'running' ? 'bg-amber-500' : 'bg-purple-500'}`} />
                        )}
                        {STATUS_LABEL[run.status] ?? run.status}
                      </span>
                    </td>

                    {/* Duration */}
                    <td className="py-2.5 px-3 text-gray-600 text-xs whitespace-nowrap">
                      {formatDuration(run.duration_ms)}
                    </td>

                    {/* Input summary */}
                    <td className="py-2.5 px-3 text-gray-500 text-xs max-w-xs">
                      <span className="truncate block max-w-[200px]" title={run.input_text}>
                        {run.input_text}
                      </span>
                    </td>

                    {/* Started */}
                    <td className="py-2.5 px-3 text-gray-400 text-xs whitespace-nowrap" title={formatAbsolute(run.started_at)}>
                      {formatRelative(run.started_at)}
                    </td>

                    {/* Expand */}
                    <td className="py-2.5 px-3 text-gray-400 text-xs">
                      <span className="hover:text-blue-600 transition-colors">
                        {expandedRunId === run.run_id ? '▾ Hide' : '▸ View'}
                      </span>
                    </td>
                  </tr>

                  {expandedRunId === run.run_id && (
                    <RunDetailDrawer
                      key={`detail-${run.run_id}`}
                      runId={run.run_id}
                      onClose={() => setExpandedRunId(null)}
                    />
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between mt-4 text-sm text-gray-500">
          <span>{data.total} run{data.total !== 1 ? 's' : ''}</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setPage(p => Math.max(1, p - 1)); setExpandedRunId(null) }}
              disabled={page === 1}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >←</button>
            <span className="px-2">{page} / {data.total_pages}</span>
            <button
              onClick={() => { setPage(p => Math.min(data.total_pages, p + 1)); setExpandedRunId(null) }}
              disabled={page === data.total_pages}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >→</button>
          </div>
        </div>
      )}
    </div>
  )
}
