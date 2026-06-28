import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { insuranceApi } from '../api/insurance'
import FileClaim from '../components/FileClaim/FileClaim'
import FeedbackForm from '../components/FeedbackForm/FeedbackForm'

const RISK_LABELS: Record<string, { cls: string; text: string }> = {
  low:    { cls: 'bg-green-100 text-green-700',  text: 'Low Risk' },
  medium: { cls: 'bg-amber-100 text-amber-700',  text: 'Medium Risk' },
  high:   { cls: 'bg-red-100 text-red-700',      text: 'High Risk' },
}

const fmt = (n: number) => `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
const pct = (n: number) => `${(n * 100).toFixed(1)}%`

export default function PolicyPage() {
  const { id } = useParams<{ id: string }>()
  const [showBreakdown, setShowBreakdown] = useState(false)

  const { data: policy, isLoading, isError } = useQuery({
    queryKey: ['policy', id],
    queryFn: () => insuranceApi.getPolicy(id!),
    enabled: !!id,
  })

  if (isLoading) return <div className="text-center py-16 text-gray-400">Loading policy…</div>
  if (isError || !policy) return <div className="text-center py-16 text-red-500">Policy not found.</div>

  const riskStyle = RISK_LABELS[policy.risk_level] ?? RISK_LABELS.medium
  const pred = policy.prediction

  return (
    <div className="max-w-2xl mx-auto space-y-6">

      {/* ── Header ── */}
      <div className="card text-center">
        <div className="flex items-center justify-center gap-3 mb-2">
          <h2 className="text-xl font-bold text-gray-900">
            {policy.is_active ? 'Policy Active' : 'Policy Inactive'}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${riskStyle.cls}`}>
            {riskStyle.text}
          </span>
        </div>
        <p className="text-4xl font-bold text-gray-900 mt-4">
          {fmt(policy.premium_amount_inr)}
        </p>
        <p className="text-gray-400 text-sm mt-1">Annual Premium</p>
        <p className="text-gray-400 text-xs mt-3 font-mono">
          Policy ID: {policy.policy_id}
        </p>
      </div>

      {/* ── Policy metadata ── */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">Policy Details</h3>
        <dl className="space-y-2 text-sm">
          <Row label="Status"   value={policy.is_active ? 'Active' : 'Inactive'} />
          <Row label="Risk Level" value={riskStyle.text} />
          <Row label="Pricing"  value="AI-Powered Risk-Based Pricing" />
          <Row label="Issued"   value={new Date(policy.created_at).toLocaleDateString('en-IN', {
            year: 'numeric', month: 'long', day: 'numeric',
          })} />
        </dl>
      </div>

      {/* ── Prediction details (only when backend returned a prediction) ── */}
      {pred && (
        <>
          {/* AI summary */}
          <div className="card bg-blue-50 border-blue-100">
            <p className="text-blue-800 text-sm leading-relaxed">{pred.explanation.summary}</p>
          </div>

          {/* Risk assessment */}
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Risk Assessment</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Accident Risk</span>
                  <span className="font-medium">{pct(pred.claim_probability)}</span>
                </div>
                <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      pred.claim_probability < 0.1  ? 'bg-green-500' :
                      pred.claim_probability < 0.25 ? 'bg-amber-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(100, pred.claim_probability * 100)}%` }}
                  />
                </div>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Estimated Repair Cost</span>
                <span className="font-medium text-gray-900">{fmt(pred.expected_claim_amount_inr)}</span>
              </div>
            </div>
          </div>

          {/* Premium breakdown (expandable) */}
          <div className="card">
            <button
              className="w-full flex items-center justify-between text-left"
              onClick={() => setShowBreakdown(v => !v)}
            >
              <span className="font-semibold text-gray-900">Premium Breakdown</span>
              <span className="text-gray-400 text-xl">{showBreakdown ? '−' : '+'}</span>
            </button>

            {showBreakdown && (
              <div className="mt-4 space-y-3 border-t border-gray-100 pt-4">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Starting premium</span>
                  <span className="font-medium">{fmt(pred.explanation.base_premium)}</span>
                </div>

                {pred.explanation.adjustments.map((adj, i) => (
                  <div key={i} className="flex justify-between text-sm">
                    <span className="text-gray-700 capitalize">
                      {adj.reason.replace(/_/g, ' ')}
                    </span>
                    <span className={`font-medium ${adj.impact_inr < 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {adj.impact_inr < 0 ? '−' : '+'}{fmt(Math.abs(adj.impact_inr))}
                    </span>
                  </div>
                ))}

                <div className="flex justify-between text-sm font-bold border-t border-gray-200 pt-3">
                  <span>Final Premium</span>
                  <span>{fmt(pred.explanation.final_premium)}</span>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* ── File a claim (active policies only) ── */}
      {policy.is_active && <FileClaim policyId={policy.policy_id} />}

      {/* ── Outcome feedback (only when prediction exists) ── */}
      {pred && <FeedbackForm predictionId={pred.prediction_id} />}

      {/* ── Footer links ── */}
      <div className="text-center text-sm space-x-4 text-gray-400">
        <Link to="/policies" className="hover:text-gray-600">← My Policies</Link>
        <Link to="/claims"   className="hover:text-gray-600">Claims History</Link>
      </div>

    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-medium text-gray-900">{value}</dd>
    </div>
  )
}
