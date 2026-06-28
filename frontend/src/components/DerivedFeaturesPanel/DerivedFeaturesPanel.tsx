import clsx from 'clsx'
import type { EnrichedFeaturesV2 } from '../../types/v2'

interface Props {
  enrichedFeatures: EnrichedFeaturesV2
}

const MONTHS = [
  '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

const REPAIR_BAND_STYLES: Record<string, string> = {
  Budget:   'bg-green-100 text-green-800',
  Standard: 'bg-blue-100 text-blue-800',
  Premium:  'bg-amber-100 text-amber-800',
  Luxury:   'bg-purple-100 text-purple-800',
}

function riskTier(value: number): { label: string; color: string } {
  if (value <= 3.5) return { label: 'Low',      color: 'text-green-600' }
  if (value <= 6.5) return { label: 'Moderate', color: 'text-amber-600' }
  return              { label: 'High',     color: 'text-red-600'   }
}

function RiskBar({ label, value, low, mod, high }: {
  label: string
  value: number
  low: string
  mod: string
  high: string
}) {
  const pct = ((value - 1) / 9) * 100  // 1-10 scale → 0-100%
  const tier = riskTier(value)
  const barColor =
    value <= 3.5 ? 'bg-green-500' :
    value <= 6.5 ? 'bg-amber-500' :
                   'bg-red-500'
  const description = value <= 3.5 ? low : value <= 6.5 ? mod : high
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-600 mb-1">
        <span>{label}</span>
        <span className={clsx('font-semibold', tier.color)}>{tier.label}</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden mb-1.5">
        <div
          className={clsx('h-full rounded-full transition-all duration-500', barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">{description}</p>
    </div>
  )
}

function StarRating({ rating, max = 5 }: { rating: number; max?: number }) {
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: max }).map((_, i) => (
        <svg
          key={i}
          className={clsx('w-4 h-4', i < rating ? 'text-amber-400' : 'text-gray-200')}
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      ))}
    </div>
  )
}

export default function DerivedFeaturesPanel({ enrichedFeatures }: Props) {
  const { vehicle, city, driving_score, policy_inception_month, no_claim_bonus_pct } = enrichedFeatures

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-blue-500" />
        <h3 className="font-semibold text-gray-900 text-sm uppercase tracking-wide">
          Your Profile Analysis
        </h3>
        <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
          AI-calculated · Not editable
        </span>
      </div>

      {/* Driving Score */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
        <p className="text-xs font-medium text-blue-600 uppercase tracking-wide mb-2">
          Driving Safety Score
        </p>
        <div className="flex items-baseline gap-2 mb-1">
          <span className="text-4xl font-bold text-gray-900">{driving_score.score}</span>
          <span className="text-gray-400">/100</span>
          <span className={clsx(
            'text-sm font-semibold px-2 py-0.5 rounded-full ml-2',
            driving_score.score >= 85 ? 'bg-green-100 text-green-700' :
            driving_score.score >= 60 ? 'bg-amber-100 text-amber-700' :
                                        'bg-red-100 text-red-700',
          )}>
            {driving_score.score >= 85 ? 'Excellent Driver' : driving_score.score >= 60 ? 'Good Driver' : 'Needs Improvement'}
          </span>
        </div>
        <p className="text-xs text-gray-500 mb-3">
          {driving_score.score >= 85
            ? 'Your clean record and driving experience earn you preferential rates'
            : driving_score.score >= 60
            ? 'Good driving profile — a claim-free year can improve this further'
            : 'Higher-risk profile — contact us for ways to lower your premium'}
        </p>
        {driving_score.factors.length > 0 && (
          <ul className="space-y-1">
            {driving_score.factors.map((f, i) => {
              const isPos = f.includes('(+')
              return (
                <li key={i} className={clsx('text-xs flex items-center gap-1.5', isPos ? 'text-green-700' : 'text-red-600')}>
                  <span className="font-bold">{isPos ? '+' : '−'}</span>
                  {f}
                </li>
              )
            })}
          </ul>
        )}

        {/* No Claim Bonus — auto-derived from claim history + tenure */}
        <div className="mt-3 pt-3 border-t border-blue-100 flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-blue-700">No Claim Bonus</p>
            <p className="text-xs text-blue-400 mt-0.5">derived from claim history &amp; tenure</p>
          </div>
          <div className="text-right">
            <span className={clsx(
              'text-2xl font-bold',
              no_claim_bonus_pct > 0 ? 'text-green-700' : 'text-gray-500',
            )}>
              {no_claim_bonus_pct}%
            </span>
            <p className="text-xs text-gray-400 mt-0.5">
              {no_claim_bonus_pct > 0 ? 'IRDAI NCB applied to your premium' : 'no bonus yet'}
            </p>
          </div>
        </div>
      </div>

      {/* Vehicle Specs */}
      <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
            Your Vehicle
          </p>
          <span className={clsx(
            'text-xs px-2 py-0.5 rounded-full font-medium',
            vehicle.lookup_hit ? 'bg-green-50 text-green-700' : 'bg-gray-100 text-gray-500',
          )}>
            {vehicle.lookup_hit ? 'Vehicle recognised' : 'Standard values applied'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-gray-500 mb-1">Safety Rating</p>
            <StarRating rating={vehicle.vehicle_safety_rating} />
            <p className="text-xs text-gray-400 mt-0.5">{vehicle.vehicle_safety_rating}/5 stars</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Airbags</p>
            <p className="text-lg font-bold text-gray-900">{vehicle.airbags_count}</p>
            <p className="text-xs text-gray-400">factory fitted</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Body Style</p>
            <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-md font-medium">
              {vehicle.vehicle_body_style}
            </span>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Repair Tier</p>
            <span className={clsx(
              'text-xs px-2 py-1 rounded-md font-medium',
              REPAIR_BAND_STYLES[vehicle.repair_cost_band] ?? 'bg-gray-100 text-gray-700',
            )}>
              {vehicle.repair_cost_band}
            </span>
          </div>
        </div>
      </div>

      {/* City Area Risk */}
      <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">
            Your Area
          </p>
          <span className={clsx(
            'text-xs px-2 py-0.5 rounded-full font-semibold',
            city.region_risk_category === 'High'
              ? 'bg-red-100 text-red-700'
              : 'bg-amber-100 text-amber-700',
          )}>
            {city.region_risk_category} Risk Area
          </span>
        </div>

        <div className="space-y-3.5">
          <RiskBar
            label="Flood exposure"
            value={city.flood_risk_index}
            low="Minimal flood risk — this area is rarely affected by flooding"
            mod="Moderate flood exposure — seasonal flooding may occur"
            high="High flood risk — this area is prone to seasonal flooding"
          />
          <RiskBar
            label="Theft risk"
            value={city.theft_risk_index}
            low="Low theft incidence — vehicle theft is uncommon in your area"
            mod="Moderate theft risk — consider additional security measures"
            high="High theft exposure — enhanced vehicle security is strongly advisable"
          />
          <RiskBar
            label="Monsoon impact"
            value={city.monsoon_exposure_index}
            low="Low monsoon impact — light seasonal rainfall in your area"
            mod="Moderate monsoon exposure — periodic heavy rainfall expected"
            high="High monsoon impact — this area receives heavy seasonal rainfall"
          />
        </div>

        <div className="mt-3 pt-3 border-t border-gray-50 flex justify-between items-center">
          <p className="text-xs text-gray-500">Overall area risk</p>
          <div className={clsx('text-sm font-semibold', riskTier(city.pincode_risk_score).color)}>
            {riskTier(city.pincode_risk_score).label}
            <span className="text-gray-400 font-normal ml-1 text-xs">
              ({city.pincode_risk_score.toFixed(1)} / 10)
            </span>
          </div>
        </div>
      </div>

      {/* Policy start month — simplified */}
      <div className="flex items-center gap-3 bg-gray-50 rounded-lg px-4 py-3 border border-gray-100">
        <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
          <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <div>
          <p className="text-xs text-gray-500">Policy starts</p>
          <p className="text-sm font-semibold text-gray-900">{MONTHS[policy_inception_month]} 2026</p>
        </div>
      </div>
    </div>
  )
}
