import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'
import type { PremiumQuoteResponse, QuoteFormData } from '../../types'
import QuoteResult from '../QuoteResult/QuoteResult'

const CITIES = [
  'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune',
  'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Indore', 'Bhopal', 'Nagpur',
  'Coimbatore', 'Patna', 'Vadodara', 'Agra', 'Visakhapatnam', 'Kanpur',
]

// Master dataset vehicle brands (17 total)
const VEHICLE_BRANDS = [
  'Maruti', 'Hyundai', 'Tata', 'Mahindra', 'Honda', 'Toyota', 'Ford',
  'Volkswagen', 'Kia', 'Renault', 'Skoda', 'MG', 'Nissan', 'Jeep',
  'BMW', 'Mercedes', 'Audi',
]

const VEHICLE_SEGMENTS = ['Hatchback', 'Sedan', 'SUV', 'Luxury']
const FUEL_TYPES = ['Petrol', 'Diesel', 'CNG', 'EV']

const schema = z.object({
  age: z.number().int().min(18).max(70),
  gender: z.enum(['male', 'female']),
  city: z.string().min(1),
  annual_income_inr: z.number().min(0).optional(),
  vehicle_brand: z.string().min(1),
  vehicle_segment: z.enum(['Hatchback', 'Sedan', 'SUV', 'Luxury']),
  fuel_type: z.enum(['Petrol', 'Diesel', 'CNG', 'EV']),
  vehicle_age_years: z.number().int().min(0).max(15),
  vehicle_value_inr: z.number().min(300000).max(4000000),
  annual_mileage_km: z.number().int().min(0).max(200000),
  previous_claims: z.number().int().min(0).max(5),
  years_licensed: z.number().int().min(0).max(60),
  driving_score: z.number().min(20).max(100).optional(),
})

const STEPS = ['Personal Details', 'Vehicle Details', 'Driving Profile', 'Your Quote']

export default function QuoteForm() {
  const [step, setStep] = useState(0)
  const [quoteResult, setQuoteResult] = useState<PremiumQuoteResponse | null>(null)
  const [formData, setFormData] = useState<Partial<QuoteFormData>>({})

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
    trigger,
    getValues,
  } = useForm<QuoteFormData>({
    resolver: zodResolver(schema),
    defaultValues: { previous_claims: 0 },
  })

  const mutation = useMutation({
    mutationFn: (data: QuoteFormData) => insuranceApi.getQuote(data),
    onSuccess: (result) => {
      setQuoteResult(result)
      setStep(3)
    },
  })

  const nextStep = async () => {
    const fields: (keyof QuoteFormData)[][] = [
      ['age', 'gender', 'city'],
      ['vehicle_brand', 'vehicle_segment', 'fuel_type', 'vehicle_age_years', 'vehicle_value_inr'],
      ['annual_mileage_km', 'previous_claims', 'years_licensed'],
    ]
    const valid = await trigger(fields[step])
    if (valid) setStep(s => s + 1)
  }

  const onSubmit = (data: QuoteFormData) => {
    setFormData(data)
    mutation.mutate(data)
  }

  if (step === 3 && quoteResult) {
    return <QuoteResult quote={quoteResult} formData={formData as QuoteFormData} onBack={() => setStep(0)} />
  }

  return (
    <div className="max-w-2xl mx-auto">
      {/* Progress */}
      <div className="flex items-center justify-between mb-8">
        {STEPS.slice(0, 3).map((label, i) => (
          <div key={i} className="flex items-center">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
              ${i <= step ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'}`}>
              {i + 1}
            </div>
            <span className={`ml-2 text-sm hidden sm:block ${i === step ? 'font-semibold text-blue-600' : 'text-gray-500'}`}>
              {label}
            </span>
            {i < 2 && <div className={`h-px w-8 mx-3 ${i < step ? 'bg-blue-600' : 'bg-gray-200'}`} />}
          </div>
        ))}
      </div>

      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-6">{STEPS[step]}</h2>

        <form onSubmit={handleSubmit(onSubmit)}>
          {/* Step 0 — Personal Details */}
          {step === 0 && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Age</label>
                  <input
                    type="number"
                    className="input-field"
                    min="18" max="70"
                    {...register('age', { valueAsNumber: true })}
                  />
                  {errors.age && <p className="error-msg">{errors.age.message}</p>}
                </div>
                <div>
                  <label className="label">Gender</label>
                  <select className="input-field" {...register('gender')}>
                    <option value="">Select...</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                  {errors.gender && <p className="error-msg">{errors.gender.message}</p>}
                </div>
              </div>
              <div>
                <label className="label">City</label>
                <select className="input-field" {...register('city')}>
                  <option value="">Select city...</option>
                  {CITIES.map(c => <option key={c} value={c.toLowerCase()}>{c}</option>)}
                </select>
                {errors.city && <p className="error-msg">{errors.city.message}</p>}
              </div>
              <div>
                <label className="label">Annual Income (₹) <span className="text-gray-400 font-normal text-xs">— optional</span></label>
                <input
                  type="number"
                  className="input-field"
                  placeholder="e.g. 600000"
                  {...register('annual_income_inr', { valueAsNumber: true })}
                />
                {errors.annual_income_inr && <p className="error-msg">{errors.annual_income_inr.message}</p>}
              </div>
            </div>
          )}

          {/* Step 1 — Vehicle Details */}
          {step === 1 && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Vehicle Brand</label>
                  <select className="input-field" {...register('vehicle_brand')}>
                    <option value="">Select brand...</option>
                    {VEHICLE_BRANDS.map(b => (
                      <option key={b} value={b.toLowerCase()}>{b}</option>
                    ))}
                  </select>
                  {errors.vehicle_brand && <p className="error-msg">{errors.vehicle_brand.message}</p>}
                </div>
                <div>
                  <label className="label">Vehicle Segment</label>
                  <select className="input-field" {...register('vehicle_segment')}>
                    <option value="">Select segment...</option>
                    {VEHICLE_SEGMENTS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                  {errors.vehicle_segment && <p className="error-msg">{errors.vehicle_segment.message}</p>}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Fuel Type</label>
                  <select className="input-field" {...register('fuel_type')}>
                    <option value="">Select fuel...</option>
                    {FUEL_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                  {errors.fuel_type && <p className="error-msg">{errors.fuel_type.message}</p>}
                </div>
                <div>
                  <label className="label">Vehicle Age (years)</label>
                  <input type="number" className="input-field" min="0" max="15" {...register('vehicle_age_years', { valueAsNumber: true })} />
                  {errors.vehicle_age_years && <p className="error-msg">{errors.vehicle_age_years.message}</p>}
                </div>
              </div>
              <div>
                <label className="label">Vehicle Value (₹)</label>
                <input
                  type="number"
                  className="input-field"
                  placeholder="e.g. 800000"
                  {...register('vehicle_value_inr', { valueAsNumber: true })}
                />
                {errors.vehicle_value_inr && <p className="error-msg">{errors.vehicle_value_inr.message}</p>}
              </div>
            </div>
          )}

          {/* Step 2 — Driving Profile */}
          {step === 2 && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Annual Mileage (km)</label>
                  <input type="number" className="input-field" {...register('annual_mileage_km', { valueAsNumber: true })} />
                  {errors.annual_mileage_km && <p className="error-msg">{errors.annual_mileage_km.message}</p>}
                </div>
                <div>
                  <label className="label">Previous Claims</label>
                  <div className="flex items-center gap-3 mt-1">
                    <button
                      type="button"
                      onClick={() => {
                        const curr = getValues('previous_claims') || 0
                        if (curr > 0) {
                          const el = document.getElementById('claims-input') as HTMLInputElement
                          if (el) { el.value = String(curr - 1); el.dispatchEvent(new Event('input', { bubbles: true })) }
                        }
                      }}
                      className="w-9 h-9 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:bg-gray-100"
                    >−</button>
                    <input
                      id="claims-input"
                      type="number"
                      className="input-field text-center w-16"
                      min="0" max="5"
                      {...register('previous_claims', { valueAsNumber: true })}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const curr = getValues('previous_claims') || 0
                        if (curr < 5) {
                          const el = document.getElementById('claims-input') as HTMLInputElement
                          if (el) { el.value = String(curr + 1); el.dispatchEvent(new Event('input', { bubbles: true })) }
                        }
                      }}
                      className="w-9 h-9 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:bg-gray-100"
                    >+</button>
                  </div>
                  {errors.previous_claims && <p className="error-msg">{errors.previous_claims.message}</p>}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Years Licensed</label>
                  <input type="number" className="input-field" min="0" max="60" {...register('years_licensed', { valueAsNumber: true })} />
                  {errors.years_licensed && <p className="error-msg">{errors.years_licensed.message}</p>}
                </div>
                <div>
                  <label className="label">
                    Driving Score (20–100) <span className="text-gray-400 font-normal text-xs">— optional</span>
                  </label>
                  <input
                    type="number"
                    className="input-field"
                    min="20" max="100"
                    placeholder="Auto-calculated"
                    {...register('driving_score', { valueAsNumber: true })}
                  />
                  {errors.driving_score && <p className="error-msg">{errors.driving_score.message}</p>}
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-between mt-8">
            {step > 0 && (
              <button type="button" className="btn-secondary" onClick={() => setStep(s => s - 1)}>
                Back
              </button>
            )}
            {step < 2 ? (
              <button type="button" className="btn-primary ml-auto" onClick={nextStep}>
                Next
              </button>
            ) : (
              <button
                type="submit"
                className="btn-primary ml-auto"
                disabled={mutation.isPending}
              >
                {mutation.isPending ? 'Calculating...' : 'Get My Quote'}
              </button>
            )}
          </div>

          {mutation.isError && (
            <p className="text-red-600 text-sm mt-4 text-center">{mutation.error.message}</p>
          )}
        </form>
      </div>
    </div>
  )
}
