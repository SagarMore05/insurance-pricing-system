import QuoteForm from '../components/QuoteForm/QuoteForm'

export default function QuotePage() {
  return (
    <div>
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Get Your Quote</h1>
        <p className="text-gray-500 mt-2">Complete 3 quick steps — we'll calculate your premium instantly.</p>
      </div>
      <QuoteForm />
    </div>
  )
}
