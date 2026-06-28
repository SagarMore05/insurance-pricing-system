import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { insuranceApi } from '../../api/insurance'
import type {
  QuoteFormData, UnderwritingResult, FraudSignal,
  CustomerHistorySummary, UnderwritingDecision, NodeLogEntry, ReasoningStep,
  AgentQuoteAwaitingResponse, AgentRunDetail,
} from '../../types'
import type {
  EnrichedFeaturesV2, Occupation, MaritalStatus,
  ParkingType, VehicleUsageType, FuelType, AnnualIncomeBand,
} from '../../types/v2'
import DerivedFeaturesPanel from '../DerivedFeaturesPanel/DerivedFeaturesPanel'

function isAwaitingReview(data: unknown): data is AgentQuoteAwaitingResponse {
  return (
    typeof data === 'object' &&
    data !== null &&
    'status' in data &&
    (data as AgentQuoteAwaitingResponse).status === 'awaiting_review'
  )
}

// ─── Constants ────────────────────────────────────────────────────────────────

const CITIES = [
  'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune',
  'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Indore', 'Bhopal', 'Nagpur',
  'Coimbatore', 'Patna', 'Vadodara', 'Agra', 'Visakhapatnam', 'Kanpur',
]

const BRANDS = [
  'Maruti', 'Hyundai', 'Tata', 'Mahindra', 'Honda', 'Toyota', 'Ford',
  'Volkswagen', 'Kia', 'Renault', 'Skoda', 'MG', 'Nissan', 'Jeep',
  'BMW', 'Mercedes', 'Audi',
]

const VEHICLE_SEGMENTS = ['Hatchback', 'Sedan', 'SUV', 'Luxury']
const FUEL_TYPES       = ['Petrol', 'Diesel', 'CNG', 'EV']
const OCCUPATIONS: Occupation[] = ['Salaried', 'Self-Employed', 'Professional', 'Retired', 'Student', 'Manual']
const MARITAL_STATUSES = ['Single', 'Married', 'Other'] as const
const PARKING_TYPES: { value: ParkingType; label: string }[] = [
  { value: 'Garage',        label: 'Garage' },
  { value: 'Open_Compound', label: 'Open Compound' },
  { value: 'Street',        label: 'Street' },
]
const USAGE_TYPES: { value: VehicleUsageType; label: string }[] = [
  { value: 'Personal',   label: 'Personal' },
  { value: 'Commercial', label: 'Commercial' },
  { value: 'Rideshare',  label: 'Rideshare / Taxi' },
]

// ─── Schema (extended with enrichment fields) ─────────────────────────────────
// Core fields match QuoteFormData exactly; enrichment-only fields are optional
// and used exclusively to populate the V2 feature enrichment preview.
//
// optNum: NaN-safe preprocessor for optional number inputs. React Hook Form's
// valueAsNumber returns NaN for empty inputs; this coerces NaN → undefined so
// Zod's .optional() accepts it rather than rejecting the whole form.

function optNum(inner: z.ZodNumber) {
  return z.preprocess(
    v => (v === '' || v === undefined || v === null || (typeof v === 'number' && isNaN(v)) ? undefined : Number(v)),
    inner.optional(),
  )
}

const schema = z.object({
  // Section 1 — Applicant
  age:              z.number({ invalid_type_error: 'Required' }).int().min(18, 'Min 18').max(70, 'Max 70'),
  gender:           z.enum(['male', 'female'], { required_error: 'Required' }),
  city:             z.string().min(1, 'Required'),
  occupation:       z.enum(['Salaried', 'Self-Employed', 'Professional', 'Retired', 'Student', 'Manual']).default('Salaried'),
  marital_status:   z.enum(['Single', 'Married', 'Other']).default('Single'),
  annual_income_inr: optNum(z.number().min(0)),

  // Section 2 — Vehicle
  vehicle_brand:    z.string().min(1, 'Required'),
  vehicle_model:    z.string().optional(),
  vehicle_segment:  z.enum(['Hatchback', 'Sedan', 'SUV', 'Luxury'], { required_error: 'Required' }),
  fuel_type:        z.enum(['Petrol', 'Diesel', 'CNG', 'EV'], { required_error: 'Required' }),
  engine_cc:        optNum(z.number().int().min(0).max(10000)),
  vehicle_age_years: z.number({ invalid_type_error: 'Required' }).int().min(0).max(30),
  vehicle_value_inr: z.number({ invalid_type_error: 'Required' }).min(1, 'Required').max(50000000, 'Max ₹5 Cr'),

  // Section 3 — Driving & Insurance
  years_licensed:   z.number({ invalid_type_error: 'Required' }).int().min(0).max(67),
  annual_mileage_km: z.number({ invalid_type_error: 'Required' }).int().min(0).max(200000),
  previous_claims:  z.number({ invalid_type_error: 'Required' }).int().min(0).max(20),
  policy_tenure_years: optNum(z.number().min(0).max(30)),
  driving_score:    optNum(z.number().min(0).max(100)),

  // Section 4 — Risk & Verification
  parking_type:       z.enum(['Garage', 'Open_Compound', 'Street']).default('Street'),
  vehicle_usage_type: z.enum(['Personal', 'Commercial', 'Rideshare']).default('Personal'),
})

type UWFormData = z.infer<typeof schema>

// ─── Decision config ──────────────────────────────────────────────────────────

const DECISION_CONFIG = {
  approve: {
    label: 'Approved',
    bg: 'bg-green-50',
    border: 'border-green-400',
    text: 'text-green-800',
    badge: 'bg-green-100 text-green-800',
    icon: '✓',
  },
  flag: {
    label: 'Flagged for Review',
    bg: 'bg-amber-50',
    border: 'border-amber-400',
    text: 'text-amber-800',
    badge: 'bg-amber-100 text-amber-800',
    icon: '⚠',
  },
  refer: {
    label: 'Referred — Manual Review Required',
    bg: 'bg-red-50',
    border: 'border-red-400',
    text: 'text-red-800',
    badge: 'bg-red-100 text-red-800',
    icon: '✕',
  },
} as const

const SEVERITY_STYLE: Record<string, string> = {
  high:   'bg-red-100 text-red-700 border border-red-200',
  medium: 'bg-amber-100 text-amber-700 border border-amber-200',
  low:    'bg-gray-100 text-gray-600 border border-gray-200',
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function confidenceBadge(confidence: number): string {
  if (confidence >= 0.85) return 'bg-green-100 text-green-700'
  if (confidence >= 0.70) return 'bg-amber-100 text-amber-700'
  return 'bg-red-100 text-red-700'
}

function confidenceBar(confidence: number): string {
  if (confidence >= 0.85) return 'bg-green-500'
  if (confidence >= 0.70) return 'bg-amber-500'
  return 'bg-red-500'
}

function fmt(n: number) {
  return `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`
}

function toTitleCase(s: string): string {
  return s.replace(/\b\w/g, c => c.toUpperCase())
}

function mapFuelTypeToV2(fuel: string): FuelType {
  return fuel as FuelType
}

function deriveIncomeBand(income?: number): AnnualIncomeBand {
  if (!income)        return '7-15 LPA'
  if (income < 300_000)   return '<3 LPA'
  if (income < 700_000)   return '3-7 LPA'
  if (income < 1_500_000) return '7-15 LPA'
  if (income < 3_000_000) return '15-30 LPA'
  return '>30 LPA'
}

function deriveNcb(claims: number, tenure?: number): number {
  if (claims > 0 || !tenure) return 0
  const t = Math.floor(tenure)
  if (t >= 5) return 50
  if (t === 4) return 45
  if (t === 3) return 35
  if (t === 2) return 25
  if (t === 1) return 20
  return 0
}

function buildTimeline(result: UnderwritingResult): ReasoningStep[] {
  const nodeMap = new Map(result.node_execution_log.map((n: NodeLogEntry) => [n.node, n]))
  const fraudNode     = nodeMap.get('fraud_signals')
  const reasoningNode = nodeMap.get('reasoning')
  const decisionNode  = nodeMap.get('decision')
  const decisionDuration =
    ((reasoningNode?.duration_ms ?? 0) + (decisionNode?.duration_ms ?? 0)) || undefined

  return [
    { label: 'Validate Input',        status: 'complete', detail: 'ML model readiness verified' },
    { label: 'Fraud Analysis',        status: 'complete', duration_ms: fraudNode?.duration_ms,
      detail: `${result.fraud_signals.length} signal(s) evaluated` },
    { label: 'Customer History Lookup', status: 'complete',
      detail: `${result.customer_history.similar_policies_count} comparable policies found` },
    { label: 'Frequency Prediction',  status: 'complete', detail: `Claim probability ${pct(result.claim_probability)}` },
    { label: 'Severity Prediction',   status: 'complete', detail: `Expected loss ${fmt(result.expected_claim_amount_inr)}` },
    { label: 'Pricing Calculation',   status: 'complete', detail: `Premium ${fmt(result.premium_amount_inr)}` },
    { label: 'Underwriting Decision', status: 'complete', duration_ms: decisionDuration,
      detail: result.underwriting_decision.decision.toUpperCase() },
  ]
}

// ─── Result sub-components (unchanged) ───────────────────────────────────────

function DecisionBanner({ decision }: { decision: UnderwritingDecision }) {
  const cfg = DECISION_CONFIG[decision.decision]
  return (
    <div className={`rounded-xl border-2 ${cfg.border} ${cfg.bg} p-5`}>
      <div className="flex items-center gap-3 mb-4">
        <span className={`text-2xl font-bold ${cfg.text}`}>{cfg.icon}</span>
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wide font-medium">Underwriting Decision</p>
          <p className={`text-xl font-bold ${cfg.text}`}>{cfg.label}</p>
        </div>
        <span className={`ml-auto px-3 py-1 rounded-full text-sm font-semibold ${cfg.badge}`}>
          {decision.decision.toUpperCase()}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">Confidence</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${confidenceBar(decision.confidence)}`}
                style={{ width: `${decision.confidence * 100}%` }}
              />
            </div>
            <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${confidenceBadge(decision.confidence)}`}>
              {pct(decision.confidence)}
            </span>
          </div>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Risk Adjustment</p>
          <p className="font-semibold text-gray-900">
            {decision.risk_adjustment_factor === 1.0
              ? 'None (1.0×)'
              : `+${((decision.risk_adjustment_factor - 1) * 100).toFixed(0)}% (${decision.risk_adjustment_factor}×)`}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Agent-Adjusted Premium</p>
          <p className="font-semibold text-gray-900">{fmt(decision.agent_adjusted_premium)}</p>
        </div>
      </div>
    </div>
  )
}

function FraudSignalList({ signals }: { signals: FraudSignal[] }) {
  if (signals.length === 0) {
    return (
      <div className="card">
        <h4 className="font-semibold text-gray-800 mb-2">Fraud Signals</h4>
        <p className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2">
          No risk signals detected — profile aligns with standard criteria.
        </p>
      </div>
    )
  }
  const highCount = signals.filter(s => s.severity === 'high').length
  const medCount  = signals.filter(s => s.severity === 'medium').length
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-800">Fraud Signals</h4>
        <span className="text-xs text-gray-500">
          {signals.length} signal{signals.length !== 1 ? 's' : ''}
          {highCount > 0 && <span className="text-red-600 font-medium"> · {highCount} HIGH</span>}
          {medCount  > 0 && <span className="text-amber-600 font-medium"> · {medCount} MED</span>}
        </span>
      </div>
      <div className="space-y-2">
        {signals.map((s, i) => (
          <div key={i} className="rounded-lg p-3 bg-gray-50 border border-gray-100">
            <div className="flex items-start gap-2">
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full shrink-0 mt-0.5 ${SEVERITY_STYLE[s.severity]}`}>
                {s.severity.toUpperCase()}
              </span>
              <div>
                <p className="text-sm font-medium text-gray-800">{s.signal.replace(/_/g, ' ')}</p>
                <p className="text-xs text-gray-500 mt-0.5">{s.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function CustomerHistoryCard({ history, thisPremium }: { history: CustomerHistorySummary; thisPremium: number }) {
  const delta     = thisPremium - history.avg_premium_in_segment
  const deltaSign = delta >= 0 ? '+' : ''
  return (
    <div className="card">
      <h4 className="font-semibold text-gray-800 mb-3">Customer History &amp; Segment</h4>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500">Similar Policies</p>
          <p className="text-lg font-bold text-gray-900">{history.similar_policies_count.toLocaleString()}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500">Segment Avg Premium</p>
          <p className="text-lg font-bold text-gray-900">{fmt(history.avg_premium_in_segment)}</p>
          {history.avg_premium_in_segment > 0 && (
            <p className={`text-xs font-medium mt-0.5 ${delta >= 0 ? 'text-red-600' : 'text-green-600'}`}>
              {deltaSign}{fmt(Math.abs(delta))} vs this quote
            </p>
          )}
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500">Segment Claim Freq.</p>
          <p className="text-lg font-bold text-gray-900">{pct(history.segment_claim_frequency)}</p>
        </div>
        <div className="bg-gray-50 rounded-lg p-3">
          <p className="text-xs text-gray-500">Total Claims in Segment</p>
          <p className="text-lg font-bold text-gray-900">{history.total_claims_in_segment.toLocaleString()}</p>
        </div>
      </div>
    </div>
  )
}

function ReasoningTraceCard({ trace }: { trace: string }) {
  const [expanded, setExpanded] = useState(false)
  const preview = trace.slice(0, 220)
  const hasMore = trace.length > 220
  return (
    <div className="card">
      <h4 className="font-semibold text-gray-800 mb-2">Underwriter Reasoning</h4>
      <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm text-gray-700 leading-relaxed">
        {expanded ? trace : preview}
        {hasMore && !expanded && '…'}
      </div>
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-xs text-blue-600 hover:text-blue-700"
        >
          {expanded ? 'Collapse' : 'Read full assessment'}
        </button>
      )}
    </div>
  )
}

function ExecutionTimeline({ steps, runId }: { steps: ReasoningStep[]; runId: string }) {
  const maxDuration = Math.max(...steps.map(s => s.duration_ms ?? 0), 1)
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-gray-800">Execution Timeline</h4>
        <span className="text-xs text-gray-400 font-mono">Run {runId.slice(-8).toUpperCase()}</span>
      </div>
      <div className="space-y-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="text-green-500 text-sm w-4 shrink-0">✓</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-gray-800 font-medium truncate">{step.label}</span>
                {step.duration_ms != null && (
                  <span className="text-xs text-gray-400 shrink-0">{step.duration_ms.toLocaleString()} ms</span>
                )}
              </div>
              {step.duration_ms != null && (
                <div className="mt-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-400 rounded-full"
                    style={{ width: `${Math.max(2, (step.duration_ms / maxDuration) * 100)}%` }}
                  />
                </div>
              )}
              {step.detail && <p className="text-xs text-gray-400 mt-0.5">{step.detail}</p>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function RunIdFooter({ runId, sessionId }: { runId: string; sessionId: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(runId).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div className="flex items-center gap-3 text-xs text-gray-400 border-t border-gray-100 pt-3 mt-1">
      <span>Agent Run ID:</span>
      <span className="font-mono text-gray-600">{runId}</span>
      <button onClick={copy} className="px-2 py-0.5 rounded border border-gray-200 hover:bg-gray-50 text-gray-500">
        {copied ? 'Copied!' : 'Copy'}
      </button>
      <span className="ml-auto">Session: {sessionId.slice(-8).toUpperCase()}</span>
    </div>
  )
}

const TRIGGER_LABEL: Record<string, string> = {
  low_confidence:         'Low decision confidence (< 85%)',
  fraud_signals_present:  'Fraud signals detected',
  high_claim_probability: 'High claim probability (> 40%)',
}

function PendingReviewCard({ pending, runStatus }: { pending: AgentQuoteAwaitingResponse; runStatus: AgentRunDetail | undefined }) {
  const isPolling = !runStatus || runStatus.status === 'awaiting_review'
  return (
    <div className="rounded-xl border-2 border-purple-400 bg-purple-50 p-5 space-y-4">
      <div className="flex items-center gap-3">
        <span className="inline-flex h-3 w-3 shrink-0">
          <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-purple-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500" />
        </span>
        <div className="flex-1">
          <p className="text-xs text-purple-600 uppercase tracking-wide font-medium">Awaiting Human Review</p>
          <p className="text-base font-bold text-purple-900">Application queued for reviewer decision</p>
        </div>
        {isPolling && <span className="text-xs text-purple-500 animate-pulse">Checking status…</span>}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-white rounded-lg p-3 border border-purple-100">
          <p className="text-xs text-gray-500">Agent Decision</p>
          <p className="font-semibold text-gray-900 capitalize">{pending.agent_decision}</p>
        </div>
        <div className="bg-white rounded-lg p-3 border border-purple-100">
          <p className="text-xs text-gray-500">Confidence</p>
          <p className="font-semibold text-gray-900">{(pending.confidence * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-white rounded-lg p-3 border border-purple-100">
          <p className="text-xs text-gray-500">Est. Premium</p>
          <p className="font-semibold text-gray-900">
            ₹{pending.premium_amount_inr.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </p>
        </div>
      </div>
      <div>
        <p className="text-xs font-medium text-purple-700 mb-1">Review Triggers</p>
        <div className="space-y-1">
          {pending.review_triggers.map(t => (
            <div key={t} className="flex items-center gap-2 text-xs text-purple-800 bg-purple-100 px-3 py-1.5 rounded-lg">
              <span className="text-purple-500">▸</span>
              {TRIGGER_LABEL[t] ?? t}
            </div>
          ))}
        </div>
      </div>
      <div className="border-t border-purple-200 pt-3 flex items-center gap-2">
        <p className="text-xs text-purple-600 flex-1">{pending.message}</p>
        <span className="text-xs font-mono text-purple-400">{pending.run_id.slice(-8).toUpperCase()}</span>
      </div>
    </div>
  )
}

function ReviewCompleteCard({ runStatus, onReset }: { runStatus: AgentRunDetail; onReset: () => void }) {
  const approved = runStatus.review_outcome === 'approved'
  const cfg = approved
    ? { border: 'border-green-400', bg: 'bg-green-50', text: 'text-green-900', icon: '✓', label: 'Application Approved' }
    : { border: 'border-red-400',   bg: 'bg-red-50',   text: 'text-red-900',   icon: '✕', label: 'Application Rejected' }
  return (
    <div className={`rounded-xl border-2 ${cfg.border} ${cfg.bg} p-5 space-y-3`}>
      <div className="flex items-center gap-3">
        <span className={`text-2xl font-bold ${cfg.text}`}>{cfg.icon}</span>
        <div className="flex-1">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-medium">Review Decision</p>
          <p className={`text-xl font-bold ${cfg.text}`}>{cfg.label}</p>
        </div>
        <button onClick={onReset} className="text-sm text-gray-400 hover:text-gray-600 px-3 py-1.5 border border-gray-200 rounded-lg hover:bg-white bg-white/50">
          New Assessment
        </button>
      </div>
      {runStatus.completed_at && (
        <p className="text-xs text-gray-500">
          Reviewed at {new Date(runStatus.completed_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}
        </p>
      )}
      <p className="text-xs text-gray-500">View the Agent Activity tab for the full execution detail and reviewer notes.</p>
    </div>
  )
}

// ─── Form section header ──────────────────────────────────────────────────────

function SectionHeader({ icon, title, subtitle }: { icon: string; title: string; subtitle?: string }) {
  return (
    <div className="flex items-center gap-2.5 px-4 py-2.5 bg-slate-50 border-b border-gray-200">
      <span className="text-base">{icon}</span>
      <div>
        <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide">{title}</p>
        {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
      </div>
    </div>
  )
}

// ─── Verification toggle ──────────────────────────────────────────────────────

function VerificationToggle({
  label, description, checked, onChange,
}: { label: string; description: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={clsx(
        'w-full text-left rounded-lg border px-3 py-2.5 transition-colors',
        checked
          ? 'bg-green-50 border-green-300'
          : 'bg-gray-50 border-gray-200 hover:bg-gray-100',
      )}
    >
      <div className="flex items-center gap-2">
        <div className={clsx(
          'w-4 h-4 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors',
          checked ? 'bg-green-500 border-green-500' : 'border-gray-400',
        )}>
          {checked && <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>}
        </div>
        <div>
          <p className={clsx('text-xs font-semibold', checked ? 'text-green-800' : 'text-gray-700')}>{label}</p>
          <p className="text-xs text-gray-400">{description}</p>
        </div>
      </div>
    </button>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function UnderwritingPanel() {
  const [result, setResult]               = useState<UnderwritingResult | null>(null)
  const [pendingReview, setPendingReview] = useState<AgentQuoteAwaitingResponse | null>(null)
  const [reviewComplete, setReviewComplete] = useState<AgentRunDetail | null>(null)
  const [enrichedFeatures, setEnrichedFeatures] = useState<EnrichedFeaturesV2 | null>(null)
  const [lastSubmitted, setLastSubmitted] = useState<UWFormData | null>(null)

  // Local verification flags — workflow tracking only, not sent to any API
  const [inspectionRequired, setInspectionRequired] = useState(false)
  const [documentVerified,   setDocumentVerified]   = useState(false)

  const { register, handleSubmit, watch, formState: { errors }, reset } = useForm<UWFormData>({
    resolver: zodResolver(schema),
    defaultValues: {
      previous_claims:    0,
      occupation:         'Salaried',
      marital_status:     'Single',
      parking_type:       'Street',
      vehicle_usage_type: 'Personal',
    },
  })

  // Live NCB derivation for display
  const watchedClaims = watch('previous_claims')
  const watchedTenure = watch('policy_tenure_years')
  const displayNcb    = deriveNcb(watchedClaims ?? 0, watchedTenure)

  // ── Enrichment mutation (V2 enrichment preview with full profile) ───────────
  // marital_status 'Other' is a UI-only value; map it to 'Single' for the V2 API
  // which only accepts: Single | Married | Divorced | Widowed
  const enrichmentMutation = useMutation({
    mutationFn: (formData: UWFormData) => insuranceApi.previewQuoteV2({
      age:                     formData.age,
      gender:                  toTitleCase(formData.gender) as 'Male' | 'Female' | 'Other',
      occupation:              formData.occupation as Occupation,
      marital_status:          (formData.marital_status === 'Other' ? 'Single' : formData.marital_status) as MaritalStatus,
      annual_income_band:      deriveIncomeBand(formData.annual_income_inr),
      city:                    toTitleCase(formData.city),
      years_licensed:          formData.years_licensed,
      previous_claims_count:   formData.previous_claims,
      months_since_last_claim: 0,
      policy_tenure_years:     formData.policy_tenure_years ?? 1,
      vehicle_make:            toTitleCase(formData.vehicle_brand),
      vehicle_model:           formData.vehicle_model || 'N/A',
      vehicle_age_years:       formData.vehicle_age_years,
      vehicle_value_inr:       formData.vehicle_value_inr,
      engine_cc:               formData.engine_cc ?? 0,
      fuel_type:               mapFuelTypeToV2(formData.fuel_type),
      vehicle_usage_type:      formData.vehicle_usage_type as VehicleUsageType,
      annual_mileage_km:       formData.annual_mileage_km,
      parking_type:            formData.parking_type as ParkingType,
    }),
    onSuccess: (data) => setEnrichedFeatures(data.enriched_features),
  })

  // ── Underwriting agent mutation ────────────────────────────────────────────
  const mutation = useMutation({
    mutationFn: (data: QuoteFormData) => insuranceApi.agentQuote(data),
    onSuccess: (data) => {
      if (lastSubmitted) enrichmentMutation.mutate(lastSubmitted)
      if (isAwaitingReview(data)) {
        setPendingReview(data)
      } else {
        setResult(data as UnderwritingResult)
      }
    },
  })

  // ── Poll the AgentRun status every 5 seconds while awaiting review ─────────
  const { data: runStatus } = useQuery({
    queryKey: ['poll-run', pendingReview?.run_id],
    queryFn:  () => insuranceApi.getAgentRun(pendingReview!.run_id),
    enabled:  !!pendingReview,
    refetchInterval: pendingReview ? 5000 : false,
    staleTime: 0,
  })

  useEffect(() => {
    if (runStatus && runStatus.status === 'completed') {
      setReviewComplete(runStatus)
      setPendingReview(null)
    }
  }, [runStatus?.status])

  const onSubmit = (data: UWFormData) => {
    setLastSubmitted(data)
    // Build the V1 agent payload. engine_cc MUST be an integer (never null/undefined)
    // because tools.py calls int(request_data.get("engine_cc", 0)) — if the key exists
    // with None (Pydantic default), int(None) raises TypeError and crashes the graph.
    const quoteData: QuoteFormData = {
      age:               data.age,
      gender:            data.gender,
      city:              data.city,
      annual_income_inr: data.annual_income_inr,
      vehicle_brand:     data.vehicle_brand,
      vehicle_segment:   data.vehicle_segment,
      fuel_type:         data.fuel_type,
      vehicle_age_years: data.vehicle_age_years,
      vehicle_value_inr: data.vehicle_value_inr,
      annual_mileage_km: data.annual_mileage_km,
      previous_claims:   data.previous_claims,
      years_licensed:    data.years_licensed,
      driving_score:     data.driving_score,
      car_model:         data.vehicle_model || undefined,
      engine_cc:         data.engine_cc ?? 0,
    }
    mutation.mutate(quoteData)
  }

  const resetPanel = () => {
    setResult(null)
    setPendingReview(null)
    setReviewComplete(null)
    setEnrichedFeatures(null)
    setLastSubmitted(null)
    setInspectionRequired(false)
    setDocumentVerified(false)
    reset()
    mutation.reset()
  }

  // ── Review complete screen ─────────────────────────────────────────────────
  if (reviewComplete) {
    return (
      <div className="space-y-4">
        <ReviewCompleteCard runStatus={reviewComplete} onReset={resetPanel} />
      </div>
    )
  }

  // ── Awaiting review screen ─────────────────────────────────────────────────
  if (pendingReview) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">Underwriting Assessment</h3>
            <p className="text-xs text-gray-400 mt-0.5">Pending human review</p>
          </div>
          <button onClick={resetPanel} className="text-sm text-gray-400 hover:text-gray-600 px-3 py-1.5 border border-gray-200 rounded-lg hover:bg-gray-50">
            New Assessment
          </button>
        </div>
        <PendingReviewCard pending={pendingReview} runStatus={runStatus} />
      </div>
    )
  }

  // ── Results screen ─────────────────────────────────────────────────────────
  if (result) {
    const timeline = buildTimeline(result)
    const cfg      = DECISION_CONFIG[result.underwriting_decision.decision]
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">Underwriting Assessment</h3>
            <p className="text-xs text-gray-400 mt-0.5">
              {result.risk_level.toUpperCase()} risk · Model {result.model_version}
            </p>
          </div>
          <button
            onClick={() => { setResult(null); reset(); mutation.reset() }}
            className="text-sm text-gray-400 hover:text-gray-600 px-3 py-1.5 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            New Assessment
          </button>
        </div>

        <DecisionBanner decision={result.underwriting_decision} />

        <div className="grid lg:grid-cols-2 gap-4">
          <FraudSignalList signals={result.fraud_signals} />
          <CustomerHistoryCard history={result.customer_history} thisPremium={result.premium_amount_inr} />
        </div>

        <div className="card">
          <h4 className="font-semibold text-gray-800 mb-3">Premium &amp; Risk Summary</h4>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-gray-500">Calculated Premium</p>
              <p className="text-lg font-bold text-gray-900">{fmt(result.premium_amount_inr)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Claim Probability</p>
              <p className="text-lg font-bold text-gray-900">{pct(result.claim_probability)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Expected Severity</p>
              <p className="text-lg font-bold text-gray-900">{fmt(result.expected_claim_amount_inr)}</p>
            </div>
          </div>
          {result.driving_score_info && (
            <div className="mt-3 pt-3 border-t border-gray-100 flex items-center gap-2 text-sm">
              <span className="text-gray-500">Driver Safety Score:</span>
              <span className="font-bold text-gray-900">{result.driving_score_info.score}/100</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                result.driving_score_info.score >= 85 ? 'bg-green-100 text-green-700' :
                result.driving_score_info.score >= 60 ? 'bg-amber-100 text-amber-700' :
                                                         'bg-red-100 text-red-700'
              }`}>
                {result.driving_score_info.score >= 85 ? 'Excellent' : result.driving_score_info.score >= 60 ? 'Good' : 'Poor'}
              </span>
              <span className="text-xs text-gray-400 ml-1">auto-generated</span>
            </div>
          )}
          {result.underwriting_decision.risk_adjustment_factor !== 1.0 && (
            <div className={`mt-3 rounded-lg px-3 py-2 text-sm font-medium ${cfg.badge}`}>
              Agent recommends adjusted premium: {fmt(result.underwriting_decision.agent_adjusted_premium)}
              {' '}({result.underwriting_decision.risk_adjustment_factor}× factor — advisory only)
            </div>
          )}
        </div>

        <ReasoningTraceCard trace={result.reasoning_trace} />
        <ExecutionTimeline steps={timeline} runId={result.session_id} />

        {enrichedFeatures ? (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">V4 Risk Enrichment</span>
              <span className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-700 border border-blue-100 px-2 py-0.5 rounded-full font-medium">
                InsurancePreprocessorV4 · 59 features
              </span>
            </div>
            <DerivedFeaturesPanel enrichedFeatures={enrichedFeatures} />
          </div>
        ) : enrichmentMutation.isPending ? (
          <div className="card animate-pulse flex items-center gap-3 py-4">
            <span className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-gray-400">Fetching V4 risk enrichment…</span>
          </div>
        ) : null}

        <RunIdFooter runId={result.session_id} sessionId={result.session_id} />
      </div>
    )
  }

  // ── Underwriting intake form ───────────────────────────────────────────────
  return (
    <div className="space-y-5">

      {/* Page header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-bold text-gray-900 text-base">Underwriting Workstation</h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Fraud detection · Segment analysis · LLM reasoning · HITL review
          </p>
        </div>
        <span className="text-xs bg-blue-50 text-blue-700 border border-blue-200 px-2.5 py-1 rounded-full font-medium">
          New Application
        </span>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">

        {/* ── Section 1: Applicant Information ──────────────────────────────── */}
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <SectionHeader icon="👤" title="Applicant Information" subtitle="Personal and demographic details" />
          <div className="p-4 space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="label">Age</label>
                <input
                  type="number" min="18" max="70" placeholder="18–70"
                  className="input-field"
                  {...register('age', { valueAsNumber: true })}
                />
                {errors.age && <p className="error-msg">{errors.age.message}</p>}
              </div>
              <div>
                <label className="label">Gender</label>
                <select className="input-field" {...register('gender')}>
                  <option value="">Select…</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
                {errors.gender && <p className="error-msg">{errors.gender.message}</p>}
              </div>
              <div>
                <label className="label">City</label>
                <select className="input-field" {...register('city')}>
                  <option value="">Select…</option>
                  {CITIES.map(c => <option key={c} value={c.toLowerCase()}>{c}</option>)}
                </select>
                {errors.city && <p className="error-msg">{errors.city.message}</p>}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="label">Occupation</label>
                <select className="input-field" {...register('occupation')}>
                  {OCCUPATIONS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className="label">Marital Status</label>
                <select className="input-field" {...register('marital_status')}>
                  {MARITAL_STATUSES.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="label">Annual Income (₹) <span className="text-gray-400 font-normal">optional</span></label>
                <input
                  type="number" min="0" placeholder="e.g. 600000"
                  className="input-field"
                  {...register('annual_income_inr', { valueAsNumber: true, setValueAs: v => v === '' || v === undefined ? undefined : Number(v) })}
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── Section 2: Vehicle Information ────────────────────────────────── */}
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <SectionHeader icon="🚗" title="Vehicle Information" subtitle="Insured vehicle details" />
          <div className="p-4 space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="label">Brand</label>
                <select className="input-field" {...register('vehicle_brand')}>
                  <option value="">Select…</option>
                  {BRANDS.map(b => <option key={b} value={b.toLowerCase()}>{b}</option>)}
                </select>
                {errors.vehicle_brand && <p className="error-msg">{errors.vehicle_brand.message}</p>}
              </div>
              <div className="col-span-2">
                <label className="label">Model <span className="text-gray-400 font-normal">optional — used for feature enrichment</span></label>
                <input
                  type="text" placeholder="e.g. Swift, Creta, Nexon…"
                  className="input-field"
                  {...register('vehicle_model')}
                />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-3">
              <div>
                <label className="label">Segment</label>
                <select className="input-field" {...register('vehicle_segment')}>
                  <option value="">Select…</option>
                  {VEHICLE_SEGMENTS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
                {errors.vehicle_segment && <p className="error-msg">{errors.vehicle_segment.message}</p>}
              </div>
              <div>
                <label className="label">Fuel Type</label>
                <select className="input-field" {...register('fuel_type')}>
                  <option value="">Select…</option>
                  {FUEL_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
                {errors.fuel_type && <p className="error-msg">{errors.fuel_type.message}</p>}
              </div>
              <div>
                <label className="label">Engine CC <span className="text-gray-400 font-normal">opt.</span></label>
                <input
                  type="number" min="0" max="10000" placeholder="e.g. 1197"
                  className="input-field"
                  {...register('engine_cc', { valueAsNumber: true, setValueAs: v => v === '' || v === undefined ? undefined : Number(v) })}
                />
              </div>
              <div>
                <label className="label">Vehicle Age (yrs)</label>
                <input
                  type="number" min="0" max="30" placeholder="0–30"
                  className="input-field"
                  {...register('vehicle_age_years', { valueAsNumber: true })}
                />
                {errors.vehicle_age_years && <p className="error-msg">{errors.vehicle_age_years.message}</p>}
              </div>
            </div>
            <div>
              <label className="label">Market Value (₹)</label>
              <input
                type="number" min="1" placeholder="e.g. 800000"
                className="input-field"
                {...register('vehicle_value_inr', { valueAsNumber: true })}
              />
              {errors.vehicle_value_inr && <p className="error-msg">{errors.vehicle_value_inr.message}</p>}
            </div>
          </div>
        </div>

        {/* ── Section 3: Driving & Insurance History ────────────────────────── */}
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <SectionHeader icon="📋" title="Driving & Insurance History" subtitle="Licence, claims, and policy tenure" />
          <div className="p-4 space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="label">Years Licensed</label>
                <input
                  type="number" min="0" max="67" placeholder="0–67"
                  className="input-field"
                  {...register('years_licensed', { valueAsNumber: true })}
                />
                {errors.years_licensed && <p className="error-msg">{errors.years_licensed.message}</p>}
              </div>
              <div>
                <label className="label">Annual Mileage (km)</label>
                <input
                  type="number" min="0" max="200000" placeholder="e.g. 15000"
                  className="input-field"
                  {...register('annual_mileage_km', { valueAsNumber: true })}
                />
                {errors.annual_mileage_km && <p className="error-msg">{errors.annual_mileage_km.message}</p>}
              </div>
              <div>
                <label className="label">Previous Claims</label>
                <input
                  type="number" min="0" max="20" placeholder="0–20"
                  className="input-field"
                  {...register('previous_claims', { valueAsNumber: true })}
                />
                {errors.previous_claims && <p className="error-msg">{errors.previous_claims.message}</p>}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3 items-start">
              <div>
                <label className="label">Policy Tenure (yrs) <span className="text-gray-400 font-normal">opt.</span></label>
                <input
                  type="number" min="0" max="30" step="0.5" placeholder="e.g. 3"
                  className="input-field"
                  {...register('policy_tenure_years', { valueAsNumber: true, setValueAs: v => v === '' || v === undefined ? undefined : Number(v) })}
                />
              </div>
              <div>
                <label className="label">No Claim Bonus</label>
                <div className="input-field bg-gray-50 flex items-center gap-2 cursor-default select-none">
                  <span className={clsx(
                    'text-sm font-bold',
                    displayNcb > 0 ? 'text-green-700' : 'text-gray-400',
                  )}>
                    {displayNcb}%
                  </span>
                  <span className="text-xs text-gray-400">
                    {displayNcb > 0 ? 'IRDAI NCB slab — auto-derived' : 'No bonus (claims present or tenure < 1 yr)'}
                  </span>
                </div>
              </div>
              <div>
                <label className="label">Override Driving Score <span className="text-gray-400 font-normal">opt.</span></label>
                <input
                  type="number" min="0" max="100" placeholder="0–100 (auto-computed if blank)"
                  className="input-field"
                  {...register('driving_score', { valueAsNumber: true, setValueAs: v => v === '' || v === undefined ? undefined : Number(v) })}
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── Section 4: Risk & Verification ────────────────────────────────── */}
        <div className="rounded-xl border border-gray-200 overflow-hidden">
          <SectionHeader icon="⚠️" title="Risk & Verification" subtitle="Usage, parking, and inspection checklist" />
          <div className="p-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Parking Type</label>
                <select className="input-field" {...register('parking_type')}>
                  {PARKING_TYPES.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                </select>
              </div>
              <div>
                <label className="label">Vehicle Usage</label>
                <select className="input-field" {...register('vehicle_usage_type')}>
                  {USAGE_TYPES.map(u => <option key={u.value} value={u.value}>{u.label}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <VerificationToggle
                label="Inspection Required"
                description="Physical inspection of vehicle before policy issuance"
                checked={inspectionRequired}
                onChange={setInspectionRequired}
              />
              <VerificationToggle
                label="Documents Verified"
                description="RC book, driving licence, and ID confirmed by underwriter"
                checked={documentVerified}
                onChange={setDocumentVerified}
              />
            </div>

            <p className="text-xs text-gray-400 bg-slate-50 border border-slate-100 rounded-lg px-3 py-2">
              Area risk (flood exposure, theft index, monsoon impact) is automatically derived from the city
              selection during assessment and shown in the V4 Risk Enrichment section of results.
            </p>
          </div>
        </div>

        {/* ── Submit ────────────────────────────────────────────────────────── */}
        {mutation.isError && (
          <p className="text-red-600 text-sm">{(mutation.error as Error).message}</p>
        )}

        <button
          type="submit"
          disabled={mutation.isPending}
          className="btn-primary w-full py-3 flex items-center justify-center gap-2"
        >
          {mutation.isPending ? (
            <>
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Running assessment — fraud detection · reasoning · decision…</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Underwriting Assessment
            </>
          )}
        </button>
      </form>
    </div>
  )
}
