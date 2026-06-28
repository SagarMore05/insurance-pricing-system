import { useState, useRef, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import { insuranceApiV2 } from '../../api/insuranceV2'
import type {
  CustomerQuoteRequestV2,
  QuoteResponseV2,
  QuotePreviewResponseV2,
  Occupation,
  MaritalStatus,
} from '../../types/v2'
import { V4_MAKES_MODELS, V4_CITIES } from '../../types/v2'
import DerivedFeaturesPanel from '../DerivedFeaturesPanel/DerivedFeaturesPanel'

import QuoteResultV2 from '../QuoteResultV2/QuoteResultV2'
import {
  OCCUPATION_BACKEND_MAP,
  estimateVehicleValue,
  deriveNcbPct,
} from '../../lib/insuranceUtils'

// ─── Vehicle Model Searchable Combobox ───────────────────────────────────────

export function VehicleModelCombobox({
  make, value, onChange, isCustom, onCustomToggle, error,
}: {
  make: string
  value: string
  onChange: (v: string) => void
  isCustom: boolean
  onCustomToggle: (c: boolean) => void
  error?: string
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const models = make ? (V4_MAKES_MODELS[make] ?? []) : []
  const filtered = query.trim()
    ? models.filter(m => m.toLowerCase().includes(query.toLowerCase()))
    : models

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onOutside)
    return () => document.removeEventListener('mousedown', onOutside)
  }, [])

  useEffect(() => { setQuery(''); setOpen(false) }, [make])
  useEffect(() => { if (!value && !isCustom) setQuery('') }, [value, isCustom])

  function selectModel(model: string) {
    onChange(model)
    setQuery(model)
    setOpen(false)
  }

  function selectOther() {
    onChange('')
    setQuery('')
    setOpen(false)
    onCustomToggle(true)
  }

  if (isCustom) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs bg-amber-50 border border-amber-100 rounded-lg px-3 py-2">
          <span className="text-amber-700">
            Custom model — standard vehicle data will be used for your quote
          </span>
          <button
            type="button"
            onClick={() => { onCustomToggle(false); onChange('') }}
            className="text-blue-600 hover:text-blue-800 font-medium ml-3 flex-shrink-0"
          >
            Back to list
          </button>
        </div>
        <input
          type="text"
          className="input-field"
          placeholder="Enter your vehicle model (e.g. Baleno Sigma 2023)"
          value={value}
          onChange={e => onChange(e.target.value)}
          aria-label="Custom vehicle model"
        />
        {error && <p className="error-msg">{error}</p>}
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative">
      <input
        type="text"
        className={clsx('input-field', !make && 'bg-gray-50 text-gray-400')}
        placeholder={make ? 'Search or select model...' : 'Select a make first'}
        value={open ? query : (value || '')}
        disabled={!make}
        aria-label="Vehicle model search"
        aria-expanded={open}
        onFocus={() => { if (make) { setQuery(value || ''); setOpen(true) } }}
        onChange={e => {
          setQuery(e.target.value)
          if (!open) setOpen(true)
          if (value) onChange('')
        }}
      />

      {open && (
        <div className="absolute z-20 top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden max-h-52 flex flex-col">
          <div className="overflow-y-auto flex-1">
            {filtered.length > 0 ? (
              filtered.map(model => (
                <button
                  key={model}
                  type="button"
                  onMouseDown={e => e.preventDefault()}
                  onClick={() => selectModel(model)}
                  className={clsx(
                    'w-full text-left px-4 py-2.5 text-sm transition-colors',
                    value === model ? 'bg-blue-50 text-blue-700 font-semibold' : 'text-gray-700 hover:bg-gray-50',
                  )}
                >
                  {model}
                </button>
              ))
            ) : (
              <p className="px-4 py-3 text-sm text-gray-400 italic">No models match "{query}"</p>
            )}
          </div>
          <button
            type="button"
            onMouseDown={e => e.preventDefault()}
            onClick={selectOther}
            className="w-full text-left px-4 py-2.5 text-sm text-gray-500 hover:bg-gray-50 border-t border-gray-100 flex-shrink-0"
          >
            Other / Model Not Listed
          </button>
        </div>
      )}

      {error && <p className="error-msg">{error}</p>}
    </div>
  )
}

// ─── Zod schema ───────────────────────────────────────────────────────────────

const schema = z.object({
  age:               z.number({ invalid_type_error: 'Required' }).int().min(18, 'Minimum 18').max(70, 'Maximum 70'),
  gender:            z.enum(['Male', 'Female', 'Other'], { required_error: 'Select a gender' }),
  occupation:        z.enum(
    ['Salaried', 'Self-Employed', 'Business Owner', 'Government Employee', 'Professional', 'Student', 'Retired', 'Other'],
    { required_error: 'Select occupation' },
  ),
  marital_status:    z.enum(['Single', 'Married', 'Other'], { required_error: 'Select marital status' }),
  annual_income_band: z.enum(['<3 LPA', '3-7 LPA', '7-15 LPA', '15-30 LPA', '>30 LPA'], { required_error: 'Select income band' }),

  city:              z.string().min(1, 'Select a city'),

  years_licensed:          z.number({ invalid_type_error: 'Required' }).int().min(0).max(80, 'Maximum 80'),
  previous_claims_count:   z.number({ invalid_type_error: 'Required' }).int().min(0).max(20, 'Maximum 20'),
  months_since_last_claim: z.number({ invalid_type_error: 'Required' }).int().min(0).max(240, 'Maximum 240'),
  policy_tenure_years:     z.number({ invalid_type_error: 'Required' }).min(0).max(30, 'Maximum 30'),

  vehicle_make:      z.string().min(1, 'Select a make'),
  vehicle_model:     z.string().min(1, 'Select or enter a model'),
  vehicle_age_years: z.number({ invalid_type_error: 'Required' }).int().min(0).max(30, 'Maximum 30'),
  vehicle_value_inr: z.number({ invalid_type_error: 'Required' }).gt(0, 'Must be > 0').max(50_000_000),
  engine_cc:         z.number({ invalid_type_error: 'Required' }).int().min(0, 'Must be 0 or more').max(10_000),

  fuel_type:          z.enum(['Petrol', 'Diesel', 'EV', 'CNG'], { required_error: 'Select fuel type' }),
  vehicle_usage_type: z.enum(['Personal', 'Commercial', 'Rideshare'], { required_error: 'Select usage type' }),
  annual_mileage_km:  z.number({ invalid_type_error: 'Required' }).int().min(0).max(200_000),
  parking_type:       z.enum(['Garage', 'Street', 'Open_Compound'], { required_error: 'Select parking type' }),
})

type FormValues = z.infer<typeof schema>

// ─── Step definitions ─────────────────────────────────────────────────────────

const STEP_LABELS = ['Customer Info', 'Driving History', 'Vehicle Details', 'Review', 'Your Quote']

const STEP_FIELDS: (keyof FormValues)[][] = [
  ['age', 'gender', 'occupation', 'marital_status', 'annual_income_band'],
  ['city', 'years_licensed', 'previous_claims_count', 'months_since_last_claim', 'policy_tenure_years'],
  ['vehicle_make', 'vehicle_model', 'vehicle_age_years', 'vehicle_value_inr', 'engine_cc', 'fuel_type', 'vehicle_usage_type', 'annual_mileage_km', 'parking_type'],
]

// ─── Progress indicator ───────────────────────────────────────────────────────

function ProgressBar({ current, total }: { current: number; total: number }) {
  return (
    <div className="mb-8">
      <div className="flex justify-between mb-2">
        {STEP_LABELS.map((label, i) => (
          <div key={i} className="flex flex-col items-center" style={{ width: `${100 / total}%` }}>
            <div className={clsx(
              'w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-all',
              i < current   ? 'bg-blue-600 border-blue-600 text-white' :
              i === current ? 'bg-white border-blue-600 text-blue-600' :
                              'bg-white border-gray-200 text-gray-400',
            )}>
              {i < current ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              ) : i + 1}
            </div>
            <span className={clsx(
              'text-xs mt-1 text-center hidden sm:block',
              i === current ? 'text-blue-600 font-semibold' :
              i < current   ? 'text-gray-500' : 'text-gray-400',
            )}>
              {label}
            </span>
          </div>
        ))}
      </div>
      <div className="relative h-1.5 bg-gray-100 rounded-full mt-1">
        <div
          className="absolute inset-y-0 left-0 bg-blue-600 rounded-full transition-all duration-500"
          style={{ width: `${(current / (total - 1)) * 100}%` }}
        />
      </div>
    </div>
  )
}

function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null
  return <p className="error-msg">{msg}</p>
}

// ─── Main wizard component ────────────────────────────────────────────────────

export default function QuoteWizardV2() {
  const [step, setStep] = useState(0)
  const [preview, setPreview] = useState<QuotePreviewResponseV2 | null>(null)
  const [quoteResult, setQuoteResult] = useState<QuoteResponseV2 | null>(null)
  const [submittedData, setSubmittedData] = useState<CustomerQuoteRequestV2 | null>(null)
  const [isCustomVehicleModel, setIsCustomVehicleModel] = useState(false)

  const {
    register,
    handleSubmit,
    watch,
    trigger,
    getValues,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      previous_claims_count:   0,
      months_since_last_claim: 0,
      policy_tenure_years:     1,
      vehicle_age_years:       0,
    },
  })

  const selectedMake       = watch('vehicle_make')
  const currentModel       = watch('vehicle_model')
  const vehicleAge         = watch('vehicle_age_years')
  const previousClaims     = watch('previous_claims_count')
  const policyTenure       = watch('policy_tenure_years')

  // Auto-clear months_since_last_claim when no previous claims
  useEffect(() => {
    if ((previousClaims ?? 0) === 0) {
      setValue('months_since_last_claim', 0)
    }
  }, [previousClaims, setValue])

  // Vehicle value estimate for current selection
  const vehicleValueEstimate = !isCustomVehicleModel && selectedMake && currentModel
    ? estimateVehicleValue(selectedMake, currentModel, vehicleAge ?? 0)
    : null

  // ── Payload builder (maps frontend form values to backend types) ──

  function buildPayload(data: FormValues): CustomerQuoteRequestV2 {
    return {
      age:                    data.age,
      gender:                 data.gender,
      occupation:             OCCUPATION_BACKEND_MAP[data.occupation] ?? 'Salaried',
      marital_status:         (data.marital_status === 'Other' ? 'Single' : data.marital_status) as MaritalStatus,
      annual_income_band:     data.annual_income_band,
      city:                   data.city,
      years_licensed:         data.years_licensed,
      previous_claims_count:  data.previous_claims_count,
      months_since_last_claim: data.months_since_last_claim,
      policy_tenure_years:    data.policy_tenure_years,
      vehicle_make:           data.vehicle_make,
      vehicle_model:          data.vehicle_model,
      vehicle_age_years:      data.vehicle_age_years,
      vehicle_value_inr:      data.vehicle_value_inr,
      engine_cc:              data.engine_cc,
      fuel_type:              data.fuel_type,
      vehicle_usage_type:     data.vehicle_usage_type,
      annual_mileage_km:      data.annual_mileage_km,
      parking_type:           data.parking_type,
    }
  }

  // ── Mutations ──

  const previewMutation = useMutation({
    mutationFn: (data: CustomerQuoteRequestV2) => insuranceApiV2.previewQuoteV2(data),
    onSuccess: (result) => setPreview(result),
  })

  const quoteMutation = useMutation({
    mutationFn: (data: CustomerQuoteRequestV2) => insuranceApiV2.getQuoteV2(data),
    onSuccess: (result) => { setQuoteResult(result); setStep(4) },
  })

  const goNext = async () => {
    if (step < 3) {
      const valid = await trigger(STEP_FIELDS[step])
      if (valid) setStep(s => s + 1)
    }
  }

  const goBack = () => { if (step > 0) setStep(s => s - 1) }

  const onSubmitFullQuote = handleSubmit((data) => {
    const payload = buildPayload(data)
    setSubmittedData(payload)
    quoteMutation.mutate(payload)
  })

  const handlePreview = async () => {
    const valid = await trigger()
    if (!valid) return
    previewMutation.mutate(buildPayload(getValues()))
  }

  if (step === 4 && quoteResult && submittedData) {
    return (
      <QuoteResultV2
        quote={quoteResult}
        requestData={submittedData}
        onBack={() => { setStep(0); setQuoteResult(null); setPreview(null) }}
      />
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <ProgressBar current={step} total={5} />

      <form onSubmit={onSubmitFullQuote}>
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-1">{STEP_LABELS[step]}</h2>
          <p className="text-sm text-gray-500 mb-6">
            {step === 0 && 'Tell us about yourself'}
            {step === 1 && 'Your driving experience and claims history'}
            {step === 2 && 'Details about the vehicle to be insured'}
            {step === 3 && 'Review your details before we calculate your premium'}
          </p>

          {/* ── Step 0: Customer Information ── */}
          {step === 0 && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Age</label>
                  <input
                    type="number"
                    className="input-field"
                    placeholder="18–70"
                    aria-label="Age"
                    {...register('age', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.age?.message} />
                </div>
                <div>
                  <label className="label">Gender</label>
                  <select className="input-field" aria-label="Gender" {...register('gender')}>
                    <option value="">Select...</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other / Prefer not to say</option>
                  </select>
                  <FieldError msg={errors.gender?.message} />
                </div>
              </div>

              <div>
                <label className="label">Occupation</label>
                <select className="input-field" aria-label="Occupation" {...register('occupation')}>
                  <option value="">Select your occupation...</option>
                  <option value="Salaried">Salaried Employee</option>
                  <option value="Government Employee">Government / Public Sector Employee</option>
                  <option value="Self-Employed">Self-Employed</option>
                  <option value="Business Owner">Business Owner</option>
                  <option value="Professional">Professional (Doctor, Lawyer, CA etc.)</option>
                  <option value="Student">Student</option>
                  <option value="Retired">Retired</option>
                  <option value="Other">Other</option>
                </select>
                <FieldError msg={errors.occupation?.message} />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Marital Status</label>
                  <select className="input-field" aria-label="Marital status" {...register('marital_status')}>
                    <option value="">Select...</option>
                    <option value="Single">Single</option>
                    <option value="Married">Married</option>
                    <option value="Other">Other</option>
                  </select>
                  <FieldError msg={errors.marital_status?.message} />
                </div>
                <div>
                  <label className="label">Annual Income</label>
                  <select className="input-field" aria-label="Annual income band" {...register('annual_income_band')}>
                    <option value="">Select band...</option>
                    <option value="<3 LPA">Below ₹3 LPA</option>
                    <option value="3-7 LPA">₹3–7 LPA</option>
                    <option value="7-15 LPA">₹7–15 LPA</option>
                    <option value="15-30 LPA">₹15–30 LPA</option>
                    <option value=">30 LPA">Above ₹30 LPA</option>
                  </select>
                  <FieldError msg={errors.annual_income_band?.message} />
                </div>
              </div>
            </div>
          )}

          {/* ── Step 1: Driving History ── */}
          {step === 1 && (
            <div className="space-y-4">
              <div>
                <label className="label">City</label>
                <select className="input-field" aria-label="City" {...register('city')}>
                  <option value="">Select city...</option>
                  {V4_CITIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
                <p className="text-xs text-gray-400 mt-1">
                  We support 20 cities with full area risk analysis.
                </p>
                <FieldError msg={errors.city?.message} />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Years Licensed</label>
                  <input
                    type="number"
                    className="input-field"
                    min="0" max="80"
                    placeholder="0–80"
                    aria-label="Years licensed"
                    {...register('years_licensed', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.years_licensed?.message} />
                </div>
                <div>
                  <label className="label">Previous Claims</label>
                  <input
                    type="number"
                    className="input-field"
                    min="0" max="20"
                    placeholder="0–20"
                    aria-label="Previous claims count"
                    {...register('previous_claims_count', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.previous_claims_count?.message} />
                </div>
              </div>

              {/* Only show months since last claim when previous claims > 0 */}
              {(previousClaims ?? 0) > 0 ? (
                <div>
                  <label className="label">Months Since Last Claim</label>
                  <input
                    type="number"
                    className="input-field"
                    min="1" max="240"
                    placeholder="e.g. 18"
                    aria-label="Months since last claim"
                    {...register('months_since_last_claim', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.months_since_last_claim?.message} />
                </div>
              ) : (
                <div className="flex items-center gap-2 bg-green-50 border border-green-100 rounded-lg px-3 py-2.5">
                  <svg className="w-4 h-4 text-green-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-xs text-green-700 font-medium">No previous claims recorded</span>
                </div>
              )}

              <div>
                <label className="label">Years Insured</label>
                <input
                  type="number"
                  className="input-field"
                  min="0" max="30" step="0.5"
                  placeholder="1, 2, 3..."
                  aria-label="Years insured"
                  {...register('policy_tenure_years', { valueAsNumber: true })}
                />
                <p className="text-xs text-gray-400 mt-1">
                  How many years have you held motor insurance? Your No Claim Bonus is calculated automatically.
                </p>
                <FieldError msg={errors.policy_tenure_years?.message} />
              </div>
            </div>
          )}

          {/* ── Step 2: Vehicle Information ── */}
          {step === 2 && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Vehicle Make</label>
                  <select
                    className="input-field"
                    aria-label="Vehicle make"
                    {...register('vehicle_make')}
                    onChange={e => {
                      setValue('vehicle_make', e.target.value)
                      setValue('vehicle_model', '')
                      setIsCustomVehicleModel(false)
                    }}
                  >
                    <option value="">Select make...</option>
                    {Object.keys(V4_MAKES_MODELS).map(make => (
                      <option key={make} value={make}>{make}</option>
                    ))}
                  </select>
                  <FieldError msg={errors.vehicle_make?.message} />
                </div>

                <div>
                  <label className="label">Vehicle Model</label>
                  <input type="hidden" {...register('vehicle_model')} />
                  <VehicleModelCombobox
                    make={selectedMake ?? ''}
                    value={currentModel ?? ''}
                    onChange={(v) => setValue('vehicle_model', v, { shouldValidate: true })}
                    isCustom={isCustomVehicleModel}
                    onCustomToggle={setIsCustomVehicleModel}
                    error={errors.vehicle_model?.message}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="label">Vehicle Age (yrs)</label>
                  <input
                    type="number"
                    className="input-field"
                    min="0" max="30"
                    aria-label="Vehicle age in years"
                    {...register('vehicle_age_years', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.vehicle_age_years?.message} />
                </div>
                <div>
                  <label className="label">Engine CC</label>
                  <input
                    type="number"
                    className="input-field"
                    min="0" max="10000"
                    placeholder="e.g. 1498 · 0 for Electric"
                    aria-label="Engine displacement in cc"
                    {...register('engine_cc', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.engine_cc?.message} />
                </div>
                <div>
                  <label className="label">Fuel Type</label>
                  <select className="input-field" aria-label="Fuel type" {...register('fuel_type')}>
                    <option value="">Select...</option>
                    <option value="Petrol">Petrol</option>
                    <option value="Diesel">Diesel</option>
                    <option value="EV">EV</option>
                    <option value="CNG">CNG</option>
                  </select>
                  <FieldError msg={errors.fuel_type?.message} />
                </div>
              </div>

              <div>
                <label className="label">Vehicle Value (₹)</label>
                <input
                  type="number"
                  className="input-field"
                  min="0" max="50000000"
                  placeholder="Current market value in INR"
                  aria-label="Vehicle value in rupees"
                  {...register('vehicle_value_inr', { valueAsNumber: true })}
                />
                {vehicleValueEstimate !== null && (
                  <button
                    type="button"
                    onClick={() => setValue('vehicle_value_inr', vehicleValueEstimate, { shouldValidate: true })}
                    className="mt-1.5 text-xs text-blue-600 hover:text-blue-800 font-medium"
                  >
                    Use estimated value: ₹{(vehicleValueEstimate / 100_000).toFixed(1)}L
                    <span className="font-normal text-gray-400 ml-1">(based on model age)</span>
                  </button>
                )}
                <FieldError msg={errors.vehicle_value_inr?.message} />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="label">Usage Type</label>
                  <select className="input-field" aria-label="Vehicle usage type" {...register('vehicle_usage_type')}>
                    <option value="">Select...</option>
                    <option value="Personal">Personal</option>
                    <option value="Commercial">Commercial</option>
                    <option value="Rideshare">Rideshare / Taxi</option>
                  </select>
                  <FieldError msg={errors.vehicle_usage_type?.message} />
                </div>
                <div>
                  <label className="label">Annual Mileage (km)</label>
                  <input
                    type="number"
                    className="input-field"
                    min="0" max="200000"
                    placeholder="e.g. 15000"
                    aria-label="Annual mileage in km"
                    {...register('annual_mileage_km', { valueAsNumber: true })}
                  />
                  <FieldError msg={errors.annual_mileage_km?.message} />
                </div>
                <div>
                  <label className="label">Parking Type</label>
                  <select className="input-field" aria-label="Parking type" {...register('parking_type')}>
                    <option value="">Select...</option>
                    <option value="Garage">Garage</option>
                    <option value="Street">Street Parking</option>
                    <option value="Open_Compound">Open Compound</option>
                  </select>
                  <FieldError msg={errors.parking_type?.message} />
                </div>
              </div>
            </div>
          )}

          {/* ── Step 3: Review ── */}
          {step === 3 && (
            <div className="space-y-4">
              {/* Customer Profile */}
              <SummarySection title="Customer Profile">
                <SummaryGrid>
                  <SummaryItem label="Age" value={`${getValues('age')} years`} />
                  <SummaryItem label="Gender" value={getValues('gender')} />
                  <SummaryItem label="Occupation" value={getValues('occupation')} />
                  <SummaryItem label="Annual Income" value={getValues('annual_income_band')} />
                  <SummaryItem label="Marital Status" value={getValues('marital_status')} />
                  <SummaryItem label="City" value={getValues('city')} />
                </SummaryGrid>
              </SummarySection>

              {/* Vehicle Profile */}
              <SummarySection title="Vehicle Profile">
                <SummaryGrid>
                  <SummaryItem label="Make & Model" value={`${getValues('vehicle_make')} ${getValues('vehicle_model')}`} />
                  <SummaryItem label="Vehicle Age" value={`${getValues('vehicle_age_years')} years`} />
                  <SummaryItem label="Engine" value={`${getValues('engine_cc')} cc · ${getValues('fuel_type')}`} />
                  <SummaryItem
                    label="Market Value"
                    value={`₹${(getValues('vehicle_value_inr') || 0).toLocaleString('en-IN')}`}
                  />
                  <SummaryItem label="Usage" value={getValues('vehicle_usage_type')} />
                  <SummaryItem label="Parking" value={(getValues('parking_type') ?? '').replace('_', ' ')} />
                </SummaryGrid>
              </SummarySection>

              {/* Driving Profile */}
              <SummarySection title="Driving Profile">
                <SummaryGrid>
                  <SummaryItem label="Years Licensed" value={`${getValues('years_licensed')} years`} />
                  <SummaryItem
                    label="Annual Mileage"
                    value={`${(getValues('annual_mileage_km') || 0).toLocaleString('en-IN')} km`}
                  />
                  <SummaryItem label="Previous Claims" value={String(getValues('previous_claims_count'))} />
                  {(getValues('previous_claims_count') ?? 0) > 0 && (
                    <SummaryItem
                      label="Last Claim"
                      value={`${getValues('months_since_last_claim')} months ago`}
                    />
                  )}
                </SummaryGrid>
              </SummarySection>

              {/* Benefits & Discounts */}
              <SummarySection title="Benefits & Discounts">
                <div className="space-y-2">
                  <BenefitItem
                    label="No Claim Bonus"
                    value={`${deriveNcbPct(getValues('previous_claims_count') ?? 0, policyTenure ?? 1)}%`}
                    description="IRDAI slab: 20% (1yr) → 25% → 35% → 45% → 50% (5+ yrs), claim-free only"
                  />
                  <BenefitItem
                    label="Years Insured"
                    value={`${getValues('policy_tenure_years')} yrs`}
                    description="Your policy tenure with continuous coverage"
                  />
                </div>
              </SummarySection>

              {/* Enrichment Preview */}
              <div className="border-t border-gray-100 pt-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">Risk Profile Preview</p>
                    <p className="text-xs text-gray-500">
                      See your AI-calculated risk factors before getting your premium
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handlePreview}
                    disabled={previewMutation.isPending}
                    className={clsx(
                      'text-sm px-4 py-2 rounded-lg border font-medium transition-colors',
                      previewMutation.isPending
                        ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                        : 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100',
                    )}
                  >
                    {previewMutation.isPending ? 'Loading...' : 'Preview Risk Profile'}
                  </button>
                </div>

                {previewMutation.isError && (
                  <p className="text-red-500 text-xs mb-3">{previewMutation.error.message}</p>
                )}

                {preview && (
                  <div className="mt-3">
                    <DerivedFeaturesPanel enrichedFeatures={preview.enriched_features} />
                  </div>
                )}
              </div>

              {quoteMutation.isError && (
                <p className="text-red-600 text-sm mt-2">{quoteMutation.error.message}</p>
              )}
            </div>
          )}

          {/* ── Navigation ── */}
          <div className="flex justify-between mt-8">
            {step > 0 ? (
              <button type="button" className="btn-secondary" onClick={goBack}>
                Back
              </button>
            ) : (
              <div />
            )}

            {step < 3 && (
              <button type="button" className="btn-primary ml-auto" onClick={goNext}>
                Next
              </button>
            )}

            {step === 3 && (
              <button
                type="submit"
                className="btn-primary ml-auto"
                disabled={quoteMutation.isPending}
              >
                {quoteMutation.isPending ? (
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Calculating...
                  </span>
                ) : 'Get My Quote'}
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  )
}

// ─── Review section components ────────────────────────────────────────────────

function SummarySection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">{title}</p>
      {children}
    </div>
  )
}

function SummaryGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">{children}</div>
}

function SummaryItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-xs font-semibold text-gray-900 truncate">{String(value) || '—'}</p>
    </div>
  )
}

function BenefitItem({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="flex items-center justify-between bg-white rounded-lg px-3 py-2.5 border border-gray-100">
      <div>
        <p className="text-xs font-medium text-gray-800">{label}</p>
        <p className="text-xs text-gray-400">{description}</p>
      </div>
      <div className="flex items-center gap-1.5 ml-3">
        <span className="text-sm font-bold text-blue-700">{value}</span>
        <span className="text-xs text-blue-500 bg-blue-50 border border-blue-100 px-1.5 py-0.5 rounded-full leading-none">
          auto
        </span>
      </div>
    </div>
  )
}

function DerivedRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-gray-500">{label}</span>
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-semibold text-blue-700">{value}</span>
        <span className="text-xs text-blue-500 bg-blue-50 border border-blue-100 px-1.5 py-0.5 rounded-full leading-none">
          auto
        </span>
      </div>
    </div>
  )
}

// Keep Row for potential reuse in DerivedRow context
function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-xs font-medium text-gray-900 text-right max-w-[60%] truncate">
        {String(value || '—')}
      </span>
    </div>
  )
}

// Suppress unused warning — Row is kept for downstream compatibility
void Row
void DerivedRow
