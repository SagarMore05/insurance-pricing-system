import { describe, it, expect } from 'vitest'
import {
  estimateVehicleValue,
  deriveNcbPct,
  OCCUPATION_BACKEND_MAP,
  VEHICLE_BASE_INR,
} from '../insuranceUtils'

// ─── estimateVehicleValue ────────────────────────────────────────────────────

describe('estimateVehicleValue', () => {
  describe('returns null for unknown make/model', () => {
    it('unknown make', () => {
      expect(estimateVehicleValue('Lamborghini', 'Huracan', 1)).toBeNull()
    })
    it('known make but unknown model', () => {
      expect(estimateVehicleValue('Maruti', 'Wagon R', 1)).toBeNull()
    })
    it('empty strings', () => {
      expect(estimateVehicleValue('', '', 0)).toBeNull()
    })
  })

  describe('IRDAI depreciation schedule — Maruti Alto (base ₹3,50,000)', () => {
    const base = 350_000

    it('age = 0 → 95% → ₹3,32,500 rounded to ₹3,33,000', () => {
      // 350000 * 0.95 = 332500 → round to nearest 1000 = 333000
      expect(estimateVehicleValue('Maruti', 'Alto', 0)).toBe(Math.round(base * 0.95 / 1000) * 1000)
    })
    it('age 0.5 (< 1) → 85%', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 0.5)).toBe(Math.round(base * 0.85 / 1000) * 1000)
    })
    it('age 1 (1–2 bracket) → 80%', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 1)).toBe(Math.round(base * 0.80 / 1000) * 1000)
    })
    it('age 1.9 (still 1–2 bracket) → 80%', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 1.9)).toBe(Math.round(base * 0.80 / 1000) * 1000)
    })
    it('age 2 (2–3 bracket) → 70%', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 2)).toBe(Math.round(base * 0.70 / 1000) * 1000)
    })
    it('age 3 (3–4 bracket) → 60%', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 3)).toBe(Math.round(base * 0.60 / 1000) * 1000)
    })
    it('age 4 (4+ bracket) → 50% IRDAI cap', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 4)).toBe(Math.round(base * 0.50 / 1000) * 1000)
    })
    it('age 10 (4+ bracket) → 50% IRDAI cap', () => {
      expect(estimateVehicleValue('Maruti', 'Alto', 10)).toBe(Math.round(base * 0.50 / 1000) * 1000)
    })
  })

  describe('high-value vehicle — Porsche Cayenne (base ₹1,34,00,000)', () => {
    const base = 13_400_000
    it('new Cayenne → 95%', () => {
      expect(estimateVehicleValue('Porsche', 'Cayenne', 0)).toBe(Math.round(base * 0.95 / 1000) * 1000)
    })
    it('3-year Cayenne → 60%', () => {
      expect(estimateVehicleValue('Porsche', 'Cayenne', 3)).toBe(Math.round(base * 0.60 / 1000) * 1000)
    })
  })

  describe('result is rounded to nearest ₹1000', () => {
    it('always divisible by 1000', () => {
      const makes = Object.keys(VEHICLE_BASE_INR)
      for (const make of makes) {
        for (const model of Object.keys(VEHICLE_BASE_INR[make])) {
          const result = estimateVehicleValue(make, model, 2)
          expect(result! % 1000).toBe(0)
        }
      }
    })
    it('result is always positive', () => {
      const result = estimateVehicleValue('Tata', 'Nexon', 5)
      expect(result).toBeGreaterThan(0)
    })
  })
})

// ─── deriveNcbPct ────────────────────────────────────────────────────────────

describe('deriveNcbPct', () => {
  describe('with prior claims → always 0%', () => {
    it('1 claim, any tenure → 0%', () => {
      expect(deriveNcbPct(1, 5)).toBe(0)
    })
    it('3 claims, 10 years tenure → 0%', () => {
      expect(deriveNcbPct(3, 10)).toBe(0)
    })
  })

  describe('0 claims — IRDAI NCB slabs', () => {
    it('tenure 0 → 0%', () => {
      expect(deriveNcbPct(0, 0)).toBe(0)
    })
    it('tenure 0.9 (floor=0) → 0%', () => {
      expect(deriveNcbPct(0, 0.9)).toBe(0)
    })
    it('tenure 1 → 20%', () => {
      expect(deriveNcbPct(0, 1)).toBe(20)
    })
    it('tenure 1.8 (floor=1) → 20%', () => {
      expect(deriveNcbPct(0, 1.8)).toBe(20)
    })
    it('tenure 2 → 25%', () => {
      expect(deriveNcbPct(0, 2)).toBe(25)
    })
    it('tenure 3 → 35%', () => {
      expect(deriveNcbPct(0, 3)).toBe(35)
    })
    it('tenure 4 → 45%', () => {
      expect(deriveNcbPct(0, 4)).toBe(45)
    })
    it('tenure 5 → 50% (max slab)', () => {
      expect(deriveNcbPct(0, 5)).toBe(50)
    })
    it('tenure 15 → 50% (max slab)', () => {
      expect(deriveNcbPct(0, 15)).toBe(50)
    })
  })
})

// ─── OCCUPATION_BACKEND_MAP ───────────────────────────────────────────────────

describe('OCCUPATION_BACKEND_MAP', () => {
  it('has 8 customer-facing options', () => {
    expect(Object.keys(OCCUPATION_BACKEND_MAP)).toHaveLength(8)
  })

  it('Business Owner maps to Self-Employed', () => {
    expect(OCCUPATION_BACKEND_MAP['Business Owner']).toBe('Self-Employed')
  })
  it('Government Employee maps to Salaried', () => {
    expect(OCCUPATION_BACKEND_MAP['Government Employee']).toBe('Salaried')
  })
  it('Other maps to Manual', () => {
    expect(OCCUPATION_BACKEND_MAP['Other']).toBe('Manual')
  })
  it('Salaried maps to Salaried', () => {
    expect(OCCUPATION_BACKEND_MAP['Salaried']).toBe('Salaried')
  })
  it('Self-Employed maps to Self-Employed', () => {
    expect(OCCUPATION_BACKEND_MAP['Self-Employed']).toBe('Self-Employed')
  })
  it('Professional maps to Professional', () => {
    expect(OCCUPATION_BACKEND_MAP['Professional']).toBe('Professional')
  })
  it('Student maps to Student', () => {
    expect(OCCUPATION_BACKEND_MAP['Student']).toBe('Student')
  })
  it('Retired maps to Retired', () => {
    expect(OCCUPATION_BACKEND_MAP['Retired']).toBe('Retired')
  })

  it('all backend values are valid Occupation enum members', () => {
    const validValues = ['Salaried', 'Self-Employed', 'Professional', 'Retired', 'Student', 'Manual']
    for (const val of Object.values(OCCUPATION_BACKEND_MAP)) {
      expect(validValues).toContain(val)
    }
  })
})

// ─── VEHICLE_BASE_INR coverage ───────────────────────────────────────────────

describe('VEHICLE_BASE_INR', () => {
  it('covers 17 makes', () => {
    expect(Object.keys(VEHICLE_BASE_INR)).toHaveLength(17)
  })
  it('every base price is a positive integer', () => {
    for (const models of Object.values(VEHICLE_BASE_INR)) {
      for (const price of Object.values(models)) {
        expect(price).toBeGreaterThan(0)
        expect(Number.isInteger(price)).toBe(true)
      }
    }
  })
  it('includes Maruti Alto (most affordable)', () => {
    expect(VEHICLE_BASE_INR['Maruti']['Alto']).toBe(350_000)
  })
  it('includes Porsche Cayenne (most expensive)', () => {
    expect(VEHICLE_BASE_INR['Porsche']['Cayenne']).toBe(13_400_000)
  })
})
