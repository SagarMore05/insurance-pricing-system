import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { insuranceApi } from '../../api/insurance'
import type { ClaimResponse } from '../../types'

const STORAGE_KEY = 'insureai_claims'

function saveClaim(claim: ClaimResponse): void {
  const existing: ClaimResponse[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
  localStorage.setItem(STORAGE_KEY, JSON.stringify([claim, ...existing]))
}

interface Props {
  policyId: string
}

export default function FileClaim({ policyId }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [amount, setAmount] = useState('')
  const [filed, setFiled] = useState<ClaimResponse | null>(null)

  const mutation = useMutation({
    mutationFn: () => insuranceApi.fileClaim(policyId, parseFloat(amount)),
    onSuccess: (claim) => {
      saveClaim(claim)
      setFiled(claim)
    },
  })

  if (filed) {
    return (
      <div className="card border border-green-200 bg-green-50">
        <h3 className="font-semibold text-green-800 mb-3">Claim Filed Successfully</h3>
        <dl className="space-y-2 text-sm">
          <Row label="Claim ID" value={`#${filed.claim_id.slice(-8).toUpperCase()}`} />
          <Row label="Amount Claimed" value={`₹${filed.claimed_amount_inr.toLocaleString('en-IN')}`} />
          <Row label="Status" value="Pending review" />
          <Row label="Date" value={new Date(filed.claim_date).toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' })} />
        </dl>
        <Link to="/claims" className="text-sm text-blue-600 hover:text-blue-700 mt-4 inline-block font-medium">
          View Claims History →
        </Link>
      </div>
    )
  }

  if (!expanded) {
    return (
      <button className="btn-secondary w-full" onClick={() => setExpanded(true)}>
        File a Claim
      </button>
    )
  }

  return (
    <div className="card border border-amber-200">
      <h3 className="font-semibold text-gray-900 mb-1">File a Claim</h3>
      <p className="text-xs text-gray-400 mb-4">Claims can only be filed against active policies.</p>
      <label className="label">Claimed Amount (₹)</label>
      <input
        type="number"
        min="1"
        step="1"
        className="input-field mb-4"
        placeholder="Enter amount in ₹"
        value={amount}
        onChange={e => setAmount(e.target.value)}
      />
      <div className="flex gap-3">
        <button
          className="btn-secondary flex-1"
          onClick={() => { setExpanded(false); setAmount('') }}
          disabled={mutation.isPending}
        >
          Cancel
        </button>
        <button
          className="btn-primary flex-1"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending || !amount || parseFloat(amount) <= 0}
        >
          {mutation.isPending ? 'Submitting…' : 'Submit Claim'}
        </button>
      </div>
      {mutation.isError && (
        <p className="text-red-600 text-sm mt-3">{mutation.error.message}</p>
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-medium text-gray-800">{value}</dd>
    </div>
  )
}
