import { useQueries } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { insuranceApi } from '../api/insurance'
import type { PolicyResponse } from '../types'

const STORAGE_KEY = 'insureai_policies'

const RISK_STYLES: Record<string, string> = {
  low:    'bg-green-100 text-green-700',
  medium: 'bg-amber-100 text-amber-700',
  high:   'bg-red-100 text-red-700',
}

export default function MyPoliciesPage() {
  const policyIds: string[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')

  const results = useQueries({
    queries: policyIds.map(id => ({
      queryKey: ['policy', id] as const,
      queryFn: () => insuranceApi.getPolicy(id),
    })),
  })

  const isLoading = results.some(r => r.isPending)
  const policies = results
    .filter(r => r.isSuccess && r.data)
    .map(r => r.data as PolicyResponse)

  if (policyIds.length === 0) {
    return (
      <div className="max-w-2xl mx-auto text-center py-20">
        <h1 className="text-2xl font-bold text-gray-900 mb-3">My Policies</h1>
        <p className="text-gray-500 mb-6">No policies found in this browser.</p>
        <Link to="/quote" className="btn-primary inline-block px-8 py-3">
          Get a Quote
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-2">
        <h1 className="text-2xl font-bold text-gray-900">My Policies</h1>
        <Link to="/quote" className="btn-primary text-sm px-4 py-2">
          New Quote
        </Link>
      </div>
      <p className="text-xs text-gray-400 mb-6">
        Showing {policyIds.length} {policyIds.length === 1 ? 'policy' : 'policies'} from this browser session.
      </p>

      {isLoading ? (
        <div className="space-y-3">
          {policyIds.map(id => (
            <div key={id} className="card animate-pulse h-20 bg-gray-100" />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          {policies.map(policy => (
            <PolicyCard key={policy.policy_id} policy={policy} />
          ))}
        </div>
      )}
    </div>
  )
}

function PolicyCard({ policy }: { policy: PolicyResponse }) {
  const riskCls = RISK_STYLES[policy.risk_level] ?? RISK_STYLES.medium
  const shortId = policy.policy_id.slice(-8).toUpperCase()

  return (
    <div className="card hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="font-mono text-sm font-semibold text-gray-900">#{shortId}</span>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${riskCls}`}>
              {policy.risk_level.charAt(0).toUpperCase() + policy.risk_level.slice(1)} Risk
            </span>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
              policy.is_active ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'
            }`}>
              {policy.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span>₹{Math.round(policy.premium_amount_inr).toLocaleString('en-IN')} / year</span>
            <span>{new Date(policy.created_at).toLocaleDateString('en-IN')}</span>
          </div>
        </div>
        <Link
          to={`/policy/${policy.policy_id}`}
          className="text-blue-600 hover:text-blue-700 text-sm font-medium ml-4 shrink-0"
        >
          View Details →
        </Link>
      </div>
    </div>
  )
}
