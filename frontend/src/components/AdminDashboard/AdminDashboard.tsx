import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { insuranceApi } from '../../api/insurance'
import type { PremiumTrendPoint, RetrainResponse, ClaimAdminItem, ChampionRegistry } from '../../types'
import AIAssistant from './AIAssistant'
import UnderwritingPanel from './UnderwritingPanel'
import AgentActivityPanel from './AgentActivityPanel'
import ReviewQueuePanel from './ReviewQueuePanel'
import ModelRegistryPanel from './ModelRegistryPanel'
import ModelSelectionPanel from './ModelSelectionPanel'

interface TrendDataPoint {
  month: string
  [key: string]: number | string
}

const RISK_COLORS = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444' }

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'customers' | 'claims' | 'assistant' | 'underwriting' | 'agent activity' | 'review queue' | 'model registry' | 'model selection'>('overview')
  const [retrainResult, setRetrainResult] = useState<RetrainResponse | null>(null)
  const navigate = useNavigate()

  const { data: registry } = useQuery<ChampionRegistry>({
    queryKey: ['admin-registry'],
    queryFn: insuranceApi.getRegistry,
    staleTime: 5 * 60_000,
  })

  const logout = () => {
    localStorage.removeItem('admin_token')
    navigate('/admin/login')
  }

  const { data: reviewQueueData } = useQuery({
    queryKey: ['review-queue-count'],
    queryFn: () => insuranceApi.getReviewQueue({ page: 1, page_size: 1 }),
    refetchInterval: 30_000,
  })
  const pendingReviewCount = reviewQueueData?.total ?? 0

  const retrainMutation = useMutation({
    mutationFn: insuranceApi.triggerRetrain,
    onSuccess: (data) => setRetrainResult(data),
  })

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: insuranceApi.getDashboard,
  })

  const { data: trends } = useQuery({
    queryKey: ['premium-trends'],
    queryFn: insuranceApi.getPremiumTrends,
  })

  const { data: customers } = useQuery({
    queryKey: ['customers', 1],
    queryFn: () => insuranceApi.getCustomers({ page: 1, page_size: 20 }),
  })

  const trendData = buildTrendData(trends ?? [])
  const pieData = Object.entries(stats?.policies_by_risk_level ?? {}).map(([name, value]) => ({ name, value }))

  return (
    <div>
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
        <div className="flex items-center gap-3 flex-wrap">
        <div className="flex gap-2 flex-wrap">
          {(['overview', 'customers', 'claims', 'assistant', 'underwriting', 'agent activity', 'review queue', 'model registry', 'model selection'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-colors relative
                ${activeTab === tab ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'}`}
            >
              {tab === 'assistant' ? 'AI Assistant'
                : tab === 'underwriting' ? 'Underwriting'
                : tab === 'agent activity' ? 'Agent Activity'
                : tab === 'model registry' ? 'Model Registry'
                : tab === 'model selection' ? 'Model Selection'
                : tab === 'review queue'
                  ? (pendingReviewCount > 0
                    ? <span className="flex items-center gap-1.5">
                        Review Queue
                        <span className="inline-flex items-center justify-center w-5 h-5 text-xs font-bold rounded-full bg-purple-500 text-white">
                          {pendingReviewCount > 9 ? '9+' : pendingReviewCount}
                        </span>
                      </span>
                    : 'Review Queue')
                  : tab}
            </button>
          ))}
        </div>
        <button
          onClick={logout}
          className="px-4 py-2 rounded-lg text-sm font-medium text-red-600 border border-red-200 hover:bg-red-50 transition-colors whitespace-nowrap"
        >
          Logout
        </button>
        </div>
      </div>

      {activeTab === 'overview' && (
        <>
          {/* KPI Cards */}
          {statsLoading ? (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="card animate-pulse h-24 bg-gray-100" />
              ))}
            </div>
          ) : stats && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <KpiCard label="Total Policies" value={stats.total_policies.toLocaleString()} />
              <KpiCard label="Avg Premium" value={`₹${Math.round(stats.avg_premium_inr).toLocaleString('en-IN')}`} />
              <KpiCard label="Claims Ratio" value={`${(stats.claims_ratio * 100).toFixed(1)}%`} />
              <KpiCard label="Premium Collected" value={`₹${(stats.total_premium_collected / 1_000_000).toFixed(1)}M`} />
            </div>
          )}

          {/* Model health KPIs */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <KpiCard
              label="Frequency AUC"
              value={registry ? registry.frequency.eval_metrics.roc_auc.toFixed(4) : '—'}
              sub={registry ? `${registry.frequency.champion} · Active` : undefined}
              color="blue"
            />
            <KpiCard
              label="Severity R²"
              value={registry ? registry.severity.eval_metrics.r2.toFixed(4) : '—'}
              sub={registry ? `${registry.severity.champion} · Active` : undefined}
              color="purple"
            />
            <KpiCard
              label="Feature Pipeline"
              value={registry ? `${registry.frequency.feature_count} features` : '—'}
              sub={registry ? `InsurancePreprocessor${registry.frequency.preprocessor_version}` : undefined}
              color="slate"
            />
            <KpiCard
              label="Champions Active Since"
              value={registry ? registry.last_updated : '—'}
              sub="Last promotion date"
              color="green"
            />
          </div>

          {/* Champion status widget */}
          {registry && (
            <div className="card bg-gradient-to-r from-slate-50 to-blue-50 border-slate-200 mb-6">
              <div className="flex items-center gap-3 flex-wrap">
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-green-700 bg-green-100 border border-green-200 px-3 py-1.5 rounded-full">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  {[registry.frequency.deployment_status, registry.severity.deployment_status].filter(s => s === 'active').length} Champion Model{[registry.frequency.deployment_status, registry.severity.deployment_status].filter(s => s === 'active').length !== 1 ? 's' : ''} Active
                </span>
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-gray-500">Frequency:</span>
                  <span className="font-semibold text-gray-900">{registry.frequency.champion}</span>
                  <span className="text-blue-600 font-mono text-xs">ROC-AUC {registry.frequency.eval_metrics.roc_auc.toFixed(4)}</span>
                </div>
                <span className="text-gray-300">·</span>
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-gray-500">Severity:</span>
                  <span className="font-semibold text-gray-900">{registry.severity.champion}</span>
                  <span className="text-purple-600 font-mono text-xs">R² {registry.severity.eval_metrics.r2.toFixed(4)}</span>
                </div>
                <button
                  onClick={() => setActiveTab('model registry')}
                  className="ml-auto text-xs text-blue-600 hover:text-blue-700 border border-blue-200 px-3 py-1 rounded-lg hover:bg-blue-50 transition-colors"
                >
                  View Registry →
                </button>
              </div>
            </div>
          )}

          {/* Charts row */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">

            {/* Line chart — premium trends */}
            <div className="card">
              <h3 className="font-semibold text-gray-800 mb-4">Monthly Premium Trends</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                  <Tooltip formatter={(v: number) => `₹${v.toLocaleString('en-IN')}`} />
                  <Legend />
                  {(['low', 'medium', 'high'] as const).map(r => (
                    <Line
                      key={r}
                      type="monotone"
                      dataKey={r}
                      stroke={RISK_COLORS[r]}
                      strokeWidth={2}
                      dot={false}
                      name={`${r.charAt(0).toUpperCase() + r.slice(1)} Risk`}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Pie chart — policy distribution */}
            <div className="card">
              <h3 className="font-semibold text-gray-800 mb-4">Policy Distribution by Risk</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={RISK_COLORS[entry.name as keyof typeof RISK_COLORS] ?? '#94a3b8'} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Retrain Model */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-semibold text-gray-900">Retrain Model</h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  Triggers the retraining pipeline. Model is promoted only if AUC &ge; 0.72.
                </p>
              </div>
              <button
                className="btn-primary px-5 py-2 text-sm disabled:opacity-60"
                onClick={() => retrainMutation.mutate()}
                disabled={retrainMutation.isPending}
              >
                {retrainMutation.isPending ? 'Running…' : 'Retrain Now'}
              </button>
            </div>

            {retrainMutation.isError && (
              <p className="text-red-600 text-sm mt-2">{(retrainMutation.error as Error).message}</p>
            )}

            {retrainResult && (
              <div className="border-t border-gray-100 pt-4 mt-2 space-y-3">
                <p className={`text-sm font-medium ${retrainResult.triggered ? 'text-green-700' : 'text-amber-700'}`}>
                  {retrainResult.message}
                </p>
                {retrainResult.performance_report && (
                  <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                    {retrainResult.performance_report.evaluation_date && (
                      <MetricItem label="Evaluated" value={new Date(retrainResult.performance_report.evaluation_date).toLocaleDateString('en-IN')} />
                    )}
                    {retrainResult.performance_report.n_samples != null && (
                      <MetricItem label="Samples" value={retrainResult.performance_report.n_samples.toLocaleString()} />
                    )}
                    {retrainResult.performance_report.frequency_auc != null && (
                      <MetricItem
                        label="Frequency AUC"
                        value={retrainResult.performance_report.frequency_auc.toFixed(4)}
                        highlight={retrainResult.performance_report.frequency_auc >= 0.72 ? 'green' : 'red'}
                      />
                    )}
                    {retrainResult.performance_report.severity_rmse != null && (
                      <MetricItem label="Severity RMSE" value={`₹${Math.round(retrainResult.performance_report.severity_rmse).toLocaleString('en-IN')}`} />
                    )}
                    <MetricItem
                      label="Drift Detected"
                      value={retrainResult.performance_report.drift_detected ? 'Yes' : 'No'}
                      highlight={retrainResult.performance_report.drift_detected ? 'red' : 'green'}
                    />
                    <MetricItem
                      label="Retrain Recommended"
                      value={retrainResult.performance_report.retrain_recommended ? 'Yes' : 'No'}
                    />
                    {retrainResult.performance_report.promoted_version && (
                      <MetricItem label="Promoted Version" value={retrainResult.performance_report.promoted_version} />
                    )}
                    {(retrainResult.performance_report.hitl_reviewed_count ?? 0) > 0 && (
                      <MetricItem
                        label="HITL Reviewed"
                        value={retrainResult.performance_report.hitl_reviewed_count!.toLocaleString()}
                      />
                    )}
                    {retrainResult.performance_report.human_override_rate != null && (
                      <MetricItem
                        label="Human Override Rate"
                        value={`${(retrainResult.performance_report.human_override_rate * 100).toFixed(1)}%`}
                        highlight={retrainResult.performance_report.human_override_rate > 0.3 ? 'red' : 'green'}
                      />
                    )}
                    {retrainResult.performance_report.ml_human_agreement_rate != null && (
                      <MetricItem
                        label="ML–Human Agreement"
                        value={`${(retrainResult.performance_report.ml_human_agreement_rate * 100).toFixed(1)}%`}
                        highlight={retrainResult.performance_report.ml_human_agreement_rate >= 0.7 ? 'green' : 'red'}
                      />
                    )}
                    {retrainResult.performance_report.reason && (
                      <div className="col-span-2 sm:col-span-3">
                        <dt className="text-gray-500 text-xs">Reason</dt>
                        <dd className="font-medium text-gray-800">{retrainResult.performance_report.reason}</dd>
                      </div>
                    )}
                  </dl>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {activeTab === 'customers' && (
        <div className="card">
          <h3 className="font-semibold text-gray-800 mb-4">Customers</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  {['Age', 'Gender', 'City', 'Policies', 'Risk Level', 'Latest Premium', 'Joined'].map(h => (
                    <th key={h} className="text-left py-2 px-3 text-gray-500 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(customers?.items ?? []).map(c => (
                  <tr key={c.customer_id} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-2.5 px-3">{c.age}</td>
                    <td className="py-2.5 px-3 capitalize">{c.gender}</td>
                    <td className="py-2.5 px-3 capitalize">{c.city}</td>
                    <td className="py-2.5 px-3">{c.policy_count}</td>
                    <td className="py-2.5 px-3">
                      {c.latest_risk_level && (
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          c.latest_risk_level === 'low' ? 'bg-green-100 text-green-700' :
                          c.latest_risk_level === 'medium' ? 'bg-amber-100 text-amber-700' :
                          'bg-red-100 text-red-700'
                        }`}>{c.latest_risk_level}</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">{c.latest_premium ? `₹${Math.round(c.latest_premium).toLocaleString('en-IN')}` : '—'}</td>
                    <td className="py-2.5 px-3 text-gray-400">{new Date(c.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'claims' && <AdminClaimsTab />}

      {activeTab === 'assistant' && <AIAssistant />}

      {activeTab === 'underwriting' && <UnderwritingPanel />}

      {activeTab === 'agent activity' && <AgentActivityPanel />}

      {activeTab === 'review queue' && <ReviewQueuePanel />}

      {activeTab === 'model registry' && <ModelRegistryPanel />}

      {activeTab === 'model selection' && <ModelSelectionPanel />}
    </div>
  )
}

function KpiCard({ label, value, sub, color }: {
  label: string
  value: string
  sub?: string
  color?: 'blue' | 'purple' | 'green' | 'slate'
}) {
  const colorCls: Record<string, string> = {
    blue: 'text-blue-700', purple: 'text-purple-700', green: 'text-green-700', slate: 'text-slate-700',
  }
  return (
    <div className="card">
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color ? colorCls[color] : 'text-gray-900'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function MetricItem({ label, value, highlight }: { label: string; value: string; highlight?: 'green' | 'red' }) {
  const cls = highlight === 'green' ? 'text-green-700' : highlight === 'red' ? 'text-red-700' : 'text-gray-800'
  return (
    <div>
      <dt className="text-gray-500 text-xs">{label}</dt>
      <dd className={`font-medium ${cls}`}>{value}</dd>
    </div>
  )
}

function buildTrendData(trends: PremiumTrendPoint[]): TrendDataPoint[] {
  const byMonth: Record<string, TrendDataPoint> = {}
  trends.forEach(({ month, risk_level, avg_premium }) => {
    if (!byMonth[month]) byMonth[month] = { month }
    byMonth[month][risk_level] = avg_premium
  })
  return Object.values(byMonth).sort((a, b) => (a.month as string).localeCompare(b.month as string))
}

// ── Admin Claims Tab ──────────────────────────────────────────────────────────

const RISK_BADGE: Record<string, string> = {
  low:    'bg-green-100 text-green-700',
  medium: 'bg-amber-100 text-amber-700',
  high:   'bg-red-100 text-red-700',
}
const STATUS_BADGE: Record<string, string> = {
  pending:  'bg-amber-100 text-amber-700',
  approved: 'bg-green-100 text-green-700',
  rejected: 'bg-red-100 text-red-700',
}

function AdminClaimsTab() {
  const qc = useQueryClient()
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState('')
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [approveAmount, setApproveAmount] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['admin-claims', page, statusFilter],
    queryFn: () => insuranceApi.getAdminClaims({ page, page_size: 15, status: statusFilter || undefined }),
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, status, amount }: { id: string; status: 'approved' | 'rejected'; amount?: number }) =>
      insuranceApi.updateClaim(id, status, amount),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-claims'] })
      setApprovingId(null)
      setApproveAmount('')
    },
  })

  const handleApprove = (claim: ClaimAdminItem) => {
    if (approvingId === claim.claim_id) {
      const amt = parseFloat(approveAmount)
      if (!amt || amt <= 0) return
      updateMutation.mutate({ id: claim.claim_id, status: 'approved', amount: amt })
    } else {
      setApprovingId(claim.claim_id)
      setApproveAmount(String(claim.claimed_amount_inr))
    }
  }

  const handleReject = (claimId: string) => {
    updateMutation.mutate({ id: claimId, status: 'rejected' })
  }

  const claims = (data?.items ?? []) as ClaimAdminItem[]

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h3 className="font-semibold text-gray-800">Claims Management</h3>
        <select
          value={statusFilter}
          onChange={e => { setStatusFilter(e.target.value); setPage(1) }}
          className="input-field w-36 py-1.5 text-sm"
        >
          <option value="">All statuses</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>

      {updateMutation.isError && (
        <p className="text-red-600 text-sm mb-3">{(updateMutation.error as Error).message}</p>
      )}

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map(i => <div key={i} className="animate-pulse h-10 bg-gray-100 rounded" />)}
        </div>
      ) : claims.length === 0 ? (
        <p className="text-gray-400 text-sm text-center py-10">No claims found.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                {['Claim ID', 'Policy', 'Risk', 'Claimed (₹)', 'Approved (₹)', 'Date', 'Status', 'Actions'].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-gray-500 font-medium whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {claims.map(claim => (
                <tr key={claim.claim_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2.5 px-3 font-mono text-xs text-gray-600">
                    #{claim.claim_id.slice(-8).toUpperCase()}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-xs text-blue-600">
                    #{claim.policy_id.slice(-8).toUpperCase()}
                  </td>
                  <td className="py-2.5 px-3">
                    {claim.risk_level && (
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${RISK_BADGE[claim.risk_level] ?? ''}`}>
                        {claim.risk_level}
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 px-3">
                    {claim.claimed_amount_inr.toLocaleString('en-IN')}
                  </td>
                  <td className="py-2.5 px-3 text-gray-400">
                    {claim.approved_amount_inr != null
                      ? claim.approved_amount_inr.toLocaleString('en-IN')
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-gray-500 whitespace-nowrap">
                    {new Date(claim.claim_date).toLocaleDateString('en-IN')}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${STATUS_BADGE[claim.claim_status] ?? STATUS_BADGE.pending}`}>
                      {claim.claim_status}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    {claim.claim_status === 'pending' && (
                      <div className="flex items-center gap-2">
                        {approvingId === claim.claim_id ? (
                          <>
                            <input
                              type="number"
                              value={approveAmount}
                              onChange={e => setApproveAmount(e.target.value)}
                              className="input-field w-24 py-1 text-xs"
                              placeholder="Amount"
                              min={1}
                            />
                            <button
                              onClick={() => handleApprove(claim)}
                              disabled={updateMutation.isPending}
                              className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                            >
                              Confirm
                            </button>
                            <button
                              onClick={() => setApprovingId(null)}
                              className="text-xs px-2 py-1 border border-gray-300 rounded hover:bg-gray-50"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              onClick={() => handleApprove(claim)}
                              disabled={updateMutation.isPending}
                              className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                            >
                              Approve
                            </button>
                            <button
                              onClick={() => handleReject(claim.claim_id)}
                              disabled={updateMutation.isPending}
                              className="text-xs px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
                            >
                              Reject
                            </button>
                          </>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between mt-4 text-sm text-gray-500">
          <span>{data.total} claim{data.total !== 1 ? 's' : ''}</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >
              ←
            </button>
            <span className="px-2 py-1">{page} / {data.total_pages}</span>
            <button
              onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
              disabled={page === data.total_pages}
              className="px-3 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40"
            >
              →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
