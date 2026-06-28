import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { insuranceApi } from '../../api/insurance'

const STORAGE_KEY = 'insureai_feedback_sent'

function isFeedbackSent(predictionId: string): boolean {
  const sent: string[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
  return sent.includes(predictionId)
}

function markFeedbackSent(predictionId: string): void {
  const sent: string[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...sent, predictionId]))
}

interface FeedbackVars {
  occurred: boolean
  amount?: number
}

interface Props {
  predictionId: string
}

export default function FeedbackForm({ predictionId }: Props) {
  const [showAmountInput, setShowAmountInput] = useState(false)
  const [amount, setAmount] = useState('')
  const [done, setDone] = useState(() => isFeedbackSent(predictionId))

  const mutation = useMutation({
    mutationFn: ({ occurred, amount }: FeedbackVars) =>
      insuranceApi.submitFeedback(predictionId, occurred, amount),
    onSuccess: () => {
      markFeedbackSent(predictionId)
      setDone(true)
    },
  })

  if (done) {
    return (
      <div className="card bg-blue-50 border-blue-100">
        <p className="text-blue-800 text-sm">
          Thank you — your feedback helps improve future pricing accuracy.
        </p>
      </div>
    )
  }

  return (
    <div className="card border border-blue-100 bg-blue-50">
      <p className="text-sm font-medium text-blue-900 mb-3">
        Help us improve — did this policy result in a claim?
      </p>

      {!showAmountInput ? (
        <div className="flex gap-3">
          <button
            className="flex-1 px-4 py-2 rounded-lg text-sm font-medium border border-green-300 bg-white text-green-700 hover:bg-green-50 transition-colors disabled:opacity-50"
            onClick={() => mutation.mutate({ occurred: false })}
            disabled={mutation.isPending}
          >
            No claim occurred
          </button>
          <button
            className="flex-1 px-4 py-2 rounded-lg text-sm font-medium border border-red-300 bg-white text-red-700 hover:bg-red-50 transition-colors disabled:opacity-50"
            onClick={() => setShowAmountInput(true)}
            disabled={mutation.isPending}
          >
            Yes — claim occurred
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="label">Claim amount (₹) — optional</label>
            <input
              type="number"
              min="0"
              className="input-field"
              placeholder="Leave blank if unknown"
              value={amount}
              onChange={e => setAmount(e.target.value)}
            />
          </div>
          <div className="flex gap-3">
            <button
              className="btn-secondary flex-1"
              onClick={() => setShowAmountInput(false)}
              disabled={mutation.isPending}
            >
              Back
            </button>
            <button
              className="btn-primary flex-1"
              onClick={() =>
                mutation.mutate({
                  occurred: true,
                  amount: amount ? parseFloat(amount) : undefined,
                })
              }
              disabled={mutation.isPending}
            >
              {mutation.isPending ? 'Submitting…' : 'Submit Feedback'}
            </button>
          </div>
        </div>
      )}

      {mutation.isError && (
        <p className="text-red-600 text-sm mt-2">{mutation.error.message}</p>
      )}
      <p className="text-xs text-blue-500 mt-3">
        Your feedback helps our models learn from real outcomes.
      </p>
    </div>
  )
}
