import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import type { PremiumQuoteResponse, QuoteFormData } from '../../types'
import { insuranceApi } from '../../api/insurance'

const POLICIES_KEY = 'insureai_policies'

function savePolicy(policyId: string): void {
  const existing: string[] = JSON.parse(localStorage.getItem(POLICIES_KEY) ?? '[]')
  if (!existing.includes(policyId)) {
    localStorage.setItem(POLICIES_KEY, JSON.stringify([policyId, ...existing]))
  }
}

// Map raw backend adjustment reason keys → customer-readable labels
const REASON_LABELS: Record<string, string> = {
  high_claims_count:      'Multiple past claims',
  young_driver:           'Young driver surcharge',
  elderly_driver:         'Mature driver adjustment',
  low_mileage:            'Low annual mileage discount',
  high_mileage:           'High annual mileage adjustment',
  ncb_discount:           'No-claim bonus discount',
  low_driving_score:      'Lower driving safety score',
  high_driving_score:     'Excellent driving score discount',
  high_vehicle_value:     'High-value vehicle adjustment',
  vehicle_age_discount:   'Older vehicle discount',
}
function readableReason(raw: string): string {
  return REASON_LABELS[raw] ?? raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

interface Props {
  quote: PremiumQuoteResponse
  formData: QuoteFormData
  onBack: () => void
}

const RISK_STYLES = {
  low:    { badge: 'bg-green-100 text-green-700',  border: 'border-green-300',  text: 'Low Risk' },
  medium: { badge: 'bg-amber-100 text-amber-700',  border: 'border-amber-300',  text: 'Medium Risk' },
  high:   { badge: 'bg-red-100 text-red-700',      border: 'border-red-300',    text: 'High Risk' },
}

const TIPS: Record<string, string[]> = {
  high: [
    'Maintain a claim-free year to earn a no-claim bonus next renewal.',
    'Consider a higher voluntary excess to lower your premium.',
    'Install an approved immobiliser to reduce theft risk loading.',
  ],
  medium: [
    'A claim-free year typically earns a 5–15% no-claim bonus.',
    'Parking in a garage overnight can reduce your area risk factor.',
  ],
  low: [
    'Excellent profile — your low-risk rating earns you our best rates.',
    'Keep your no-claim record intact to maintain these savings.',
  ],
}

export default function QuoteResult({ quote, formData, onBack }: Props) {
  const [showAdjustments, setShowAdjustments] = useState(false)
  const navigate = useNavigate()
  const riskStyle = RISK_STYLES[quote.risk_level]
  const tips = TIPS[quote.risk_level] ?? []

  const buyMutation = useMutation({
    mutationFn: () => insuranceApi.createPolicy(quote.prediction_id, formData),
    onSuccess: (policy) => {
      savePolicy(policy.policy_id)
      navigate(`/policy/${policy.policy_id}`)
    },
  })

  const fmt = (n: number) => `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Main premium card */}
      <div className={`card border-2 ${riskStyle.border} text-center`}>
        <p className="text-gray-500 text-sm mb-2">Your Annual Premium</p>
        <div className="text-5xl font-bold text-gray-900 mb-3">
          {fmt(quote.premium_amount_inr)}
        </div>
        <span className={`inline-block px-4 py-1.5 rounded-full text-sm font-semibold ${riskStyle.badge}`}>
          {riskStyle.text}
        </span>

        {/* Likelihood of claiming bar */}
        <div className="mt-6">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Likelihood of Making a Claim</span>
            <span className="font-medium">{pct(quote.claim_probability)}</span>
          </div>
          <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                quote.claim_probability < 0.1  ? 'bg-green-500' :
                quote.claim_probability < 0.25 ? 'bg-amber-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.min(100, quote.claim_probability * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="card bg-blue-50 border-blue-100">
        <p className="text-blue-800 text-sm leading-relaxed">{quote.explanation.summary}</p>
      </div>

      {/* Driving Safety Score */}
      {quote.driving_score_info && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-3">Your Driving Safety Score</h3>
          <div className="flex items-center gap-3 mb-4">
            <span className="text-4xl font-bold text-gray-900">{quote.driving_score_info.score}</span>
            <span className="text-xl text-gray-400">/100</span>
            <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
              quote.driving_score_info.score >= 85 ? 'bg-green-100 text-green-700' :
              quote.driving_score_info.score >= 60 ? 'bg-amber-100 text-amber-700' :
                                                     'bg-red-100 text-red-700'
            }`}>
              {quote.driving_score_info.score >= 85 ? 'Excellent' :
               quote.driving_score_info.score >= 60 ? 'Good' : 'Needs Improvement'}
            </span>
          </div>
          {quote.driving_score_info.factors.length > 0 && (
            <ul className="space-y-1.5">
              {quote.driving_score_info.factors.map((factor, i) => {
                const isPositive = factor.includes('(+')
                return (
                  <li key={i} className={`flex items-center gap-2 text-sm ${isPositive ? 'text-green-700' : 'text-red-600'}`}>
                    <span className="w-4 text-center font-bold">{isPositive ? '+' : '−'}</span>
                    <span>{factor}</span>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}

      {/* Premium breakdown — expandable */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-left"
          onClick={() => setShowAdjustments(v => !v)}
        >
          <span className="font-semibold text-gray-900">Why this premium?</span>
          <span className="text-gray-400 text-xl">{showAdjustments ? '−' : '+'}</span>
        </button>

        {showAdjustments && (
          <div className="mt-4 space-y-3 border-t border-gray-100 pt-4">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Starting premium</span>
              <span className="font-medium">{fmt(quote.explanation.base_premium)}</span>
            </div>

            {quote.explanation.adjustments.map((adj, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span className="text-gray-700">{readableReason(adj.reason)}</span>
                <span className={`font-medium ${adj.impact_inr < 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {adj.impact_inr < 0 ? '−' : '+'}{fmt(Math.abs(adj.impact_inr))}
                </span>
              </div>
            ))}

            <div className="flex justify-between text-sm font-bold border-t border-gray-200 pt-3 mt-2">
              <span>Your Premium</span>
              <span>{fmt(quote.explanation.final_premium)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Tips to reduce premium */}
      {tips.length > 0 && (
        <div className="card bg-gray-50 border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Ways to improve your premium</h3>
          <ul className="space-y-2">
            {tips.map((tip, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                <span className="text-blue-500 mt-0.5 flex-shrink-0">•</span>
                {tip}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* CTA */}
      <div className="flex gap-4">
        <button className="btn-secondary flex-1" onClick={onBack}>
          Recalculate
        </button>
        <button
          className="btn-primary flex-1"
          onClick={() => buyMutation.mutate()}
          disabled={buyMutation.isPending}
        >
          {buyMutation.isPending ? 'Processing...' : 'Buy Policy'}
        </button>
      </div>

      {buyMutation.isError && (
        <p className="text-red-600 text-sm text-center">{buyMutation.error.message}</p>
      )}

      <p className="text-xs text-gray-400 text-center">Quote valid for 30 days</p>
    </div>
  )
}
