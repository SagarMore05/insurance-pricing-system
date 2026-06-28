import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import type { QuoteResponseV2, CustomerQuoteRequestV2 } from '../../types/v2'
import type { QuoteFormData } from '../../types'
import { insuranceApi } from '../../api/insurance'
import DerivedFeaturesPanel from '../DerivedFeaturesPanel/DerivedFeaturesPanel'
import clsx from 'clsx'

// Map raw backend adjustment reason keys → customer-readable labels
const REASON_LABELS: Record<string, string> = {
  young_driver_surcharge:   'Young Driver Loading Applied',
  senior_driver_surcharge:  'Senior Driver Adjustment',
  safe_driver_discount:     'Safe Driver Discount',
  poor_driver_surcharge:    'Driving Profile Adjustment',
  no_claims_bonus:          'No-Claims Bonus Applied',
  high_claims_surcharge:    'Multiple Prior Claims Loading',
  high_mileage_surcharge:   'High Annual Mileage Adjustment',
  low_mileage_discount:     'Low Annual Mileage Discount',
  luxury_vehicle_surcharge: 'High-Value Vehicle Loading',
}
function readableReason(raw: string): string {
  return REASON_LABELS[raw] ?? raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

const TIPS: Record<string, string[]> = {
  high: [
    'Maintain a claim-free year to earn a no-claim bonus at renewal.',
    'Consider a higher voluntary excess to lower your premium.',
    'Install an approved immobiliser to reduce theft risk loading.',
  ],
  medium: [
    'A claim-free year earns 20% NCB — rising to 50% after 5 continuous claim-free years (IRDAI schedule).',
    'Parking in a garage overnight can reduce your area risk factor.',
  ],
  low: [
    'Excellent profile — your low-risk rating earns you our best rates.',
    'Keep your no-claim record intact to maintain these savings.',
  ],
}

const POLICIES_KEY = 'insureai_policies'

function savePolicy(policyId: string) {
  const existing: string[] = JSON.parse(localStorage.getItem(POLICIES_KEY) ?? '[]')
  if (!existing.includes(policyId)) {
    localStorage.setItem(POLICIES_KEY, JSON.stringify([policyId, ...existing]))
  }
}

const RISK_STYLES = {
  low:    { badge: 'bg-green-100 text-green-700 border-green-200', border: 'border-green-300', label: 'Low Risk' },
  medium: { badge: 'bg-amber-100 text-amber-700 border-amber-200', border: 'border-amber-300', label: 'Medium Risk' },
  high:   { badge: 'bg-red-100 text-red-700 border-red-200',       border: 'border-red-300',   label: 'High Risk' },
}

interface Props {
  quote: QuoteResponseV2
  requestData: CustomerQuoteRequestV2
  onBack: () => void
}

export default function QuoteResultV2({ quote, requestData, onBack }: Props) {
  const [activeTab, setActiveTab] = useState<'summary' | 'features'>('summary')
  const navigate = useNavigate()
  const riskStyle = RISK_STYLES[quote.risk_level] ?? RISK_STYLES.medium
  const tips = TIPS[quote.risk_level] ?? []

  const fmt   = (n: number) => `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
  const pct   = (n: number) => `${(n * 100).toFixed(1)}%`

  // Buy policy reuses V1 createPolicy — maps V2 fields to V1 QuoteFormData
  const v1FormData: QuoteFormData = {
    age:              requestData.age,
    gender:           requestData.gender.toLowerCase(),
    city:             requestData.city.toLowerCase(),
    vehicle_brand:    requestData.vehicle_make.toLowerCase(),
    vehicle_segment:  'Sedan',
    fuel_type:        'Petrol',
    vehicle_age_years: requestData.vehicle_age_years,
    vehicle_value_inr: requestData.vehicle_value_inr,
    annual_mileage_km: requestData.annual_mileage_km,
    previous_claims:   requestData.previous_claims_count,
    years_licensed:    requestData.years_licensed,
  }

  const buyMutation = useMutation({
    mutationFn: () => insuranceApi.createPolicy(quote.prediction_id, v1FormData),
    onSuccess: (policy) => {
      savePolicy(policy.policy_id)
      navigate(`/policy/${policy.policy_id}`)
    },
  })

  return (
    <div className="max-w-2xl mx-auto space-y-5">

      {/* Premium hero card */}
      <div className={clsx('card border-2 text-center', riskStyle.border)}>
        <p className="text-gray-500 text-sm mb-2">Your Annual Premium</p>
        <div className="text-5xl font-bold text-gray-900 mb-3">
          {fmt(quote.premium_amount_inr)}
        </div>
        <span className={clsx('inline-block px-4 py-1.5 rounded-full text-sm font-semibold border', riskStyle.badge)}>
          {riskStyle.label}
        </span>

        {/* Accident risk bar */}
        <div className="mt-5">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Accident Risk</span>
            <span className="font-medium">{pct(quote.claim_probability)}</span>
          </div>
          <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={clsx('h-full rounded-full transition-all duration-700',
                quote.claim_probability < 0.1 ? 'bg-green-500' :
                quote.claim_probability < 0.25 ? 'bg-amber-500' : 'bg-red-500',
              )}
              style={{ width: `${Math.min(100, quote.claim_probability * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Summary text */}
      <div className="card bg-blue-50 border-blue-100">
        <p className="text-blue-800 text-sm leading-relaxed">{quote.explanation.summary}</p>
      </div>

      {/* Tabs */}
      <div>
        <div className="flex gap-1 bg-gray-100 rounded-xl p-1">
          {(['summary', 'features'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                'flex-1 text-xs font-medium py-2 rounded-lg transition-all capitalize',
                activeTab === tab
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700',
              )}
            >
              {tab === 'features' ? 'Your Profile' : 'Breakdown'}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {/* Summary tab — premium breakdown */}
          {activeTab === 'summary' && (
            <div className="space-y-4">
              <div className="card space-y-4">
                <h3 className="font-semibold text-gray-900">Why this premium?</h3>

                {/* Starting point */}
                <div className="flex justify-between text-sm text-gray-500">
                  <span>Starting premium</span>
                  <span className="font-medium text-gray-700">{fmt(quote.explanation.base_premium)}</span>
                </div>

                {/* Factors increasing premium */}
                {quote.explanation.adjustments.filter(a => a.impact_inr > 0).length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">
                      Increasing your premium
                    </p>
                    <div className="space-y-2">
                      {quote.explanation.adjustments.filter(a => a.impact_inr > 0).map((adj, i) => (
                        <div key={i} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <span className="text-red-400 text-base leading-none">↑</span>
                            <span className="text-gray-700">{readableReason(adj.reason)}</span>
                          </div>
                          <span className="font-medium text-red-600">+{fmt(adj.impact_inr)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Factors reducing premium */}
                {quote.explanation.adjustments.filter(a => a.impact_inr < 0).length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-green-500 uppercase tracking-wide mb-2">
                      Reducing your premium
                    </p>
                    <div className="space-y-2">
                      {quote.explanation.adjustments.filter(a => a.impact_inr < 0).map((adj, i) => (
                        <div key={i} className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-2">
                            <span className="text-green-500 text-base leading-none">↓</span>
                            <span className="text-gray-700">{readableReason(adj.reason)}</span>
                          </div>
                          <span className="font-medium text-green-600">−{fmt(Math.abs(adj.impact_inr))}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Final total */}
                <div className="border-t border-gray-100 pt-3 space-y-1">
                  <div className="flex justify-between text-sm font-bold text-gray-900">
                    <span>Your Annual Premium</span>
                    <span>{fmt(quote.explanation.final_premium)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Estimated repair cost for your profile</span>
                    <span className="font-medium text-gray-700">{fmt(quote.expected_claim_amount_inr)}</span>
                  </div>
                </div>
              </div>

              {/* Tips */}
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
            </div>
          )}

          {/* AI Features tab */}
          {activeTab === 'features' && (
            <DerivedFeaturesPanel enrichedFeatures={quote.enriched_features} />
          )}

        </div>
      </div>

      {/* CTA buttons */}
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
