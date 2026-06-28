import { Link } from 'react-router-dom'
import { useQueries } from '@tanstack/react-query'
import { insuranceApi } from '../api/insurance'
import type { ClaimResponse } from '../types'

const POLICIES_KEY = 'insureai_policies'

const STATUS_STYLES: Record<string, string> = {
  pending:  'bg-amber-100 text-amber-700',
  approved: 'bg-green-100 text-green-700',
  rejected: 'bg-red-100 text-red-700',
}

export default function ClaimsPage() {
  const policyIds: string[] = JSON.parse(localStorage.getItem(POLICIES_KEY) ?? '[]')

  const results = useQueries({
    queries: policyIds.map(id => ({
      queryKey: ['claims', id] as const,
      queryFn: () => insuranceApi.getClaims(id),
      staleTime: 30_000,
    })),
  })

  const isLoading = results.some(r => r.isPending)

  const claims: ClaimResponse[] = results
    .flatMap(r => r.data ?? [])
    .sort((a, b) => new Date(b.claim_date).getTime() - new Date(a.claim_date).getTime())

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-2">
        <h1 className="text-2xl font-bold text-gray-900">Claims History</h1>
        <Link to="/policies" className="text-sm text-blue-600 hover:text-blue-700 font-medium">
          ← My Policies
        </Link>
      </div>
      <p className="text-xs text-gray-400 mb-6">
        Live from database — reflects current approval status.
      </p>

      {policyIds.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-gray-500 mb-4">No policies found in this browser.</p>
          <Link to="/quote" className="text-blue-600 hover:text-blue-700 text-sm font-medium">
            Get a Quote
          </Link>
        </div>
      ) : isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="card animate-pulse h-12 bg-gray-100" />
          ))}
        </div>
      ) : claims.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-gray-500 mb-4">No claims filed yet.</p>
          <Link to="/policies" className="text-blue-600 hover:text-blue-700 text-sm font-medium">
            View My Policies
          </Link>
        </div>
      ) : (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                {['Claim ID', 'Policy', 'Claimed (₹)', 'Approved (₹)', 'Date', 'Status'].map(h => (
                  <th key={h} className="text-left py-3 px-4 text-gray-500 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {claims.map(claim => (
                <tr key={claim.claim_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-3 px-4 font-mono text-xs text-gray-600">
                    #{claim.claim_id.slice(-8).toUpperCase()}
                  </td>
                  <td className="py-3 px-4">
                    <Link
                      to={`/policy/${claim.policy_id}`}
                      className="text-blue-600 hover:text-blue-700 font-mono text-xs"
                    >
                      #{claim.policy_id.slice(-8).toUpperCase()}
                    </Link>
                  </td>
                  <td className="py-3 px-4">
                    {claim.claimed_amount_inr.toLocaleString('en-IN')}
                  </td>
                  <td className="py-3 px-4 text-gray-400">
                    {claim.approved_amount_inr != null
                      ? claim.approved_amount_inr.toLocaleString('en-IN')
                      : '—'}
                  </td>
                  <td className="py-3 px-4 text-gray-500">
                    {new Date(claim.claim_date).toLocaleDateString('en-IN')}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${
                      STATUS_STYLES[claim.claim_status] ?? STATUS_STYLES.pending
                    }`}>
                      {claim.claim_status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
