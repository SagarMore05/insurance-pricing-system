import QuoteWizardV2 from '../components/QuoteWizardV2/QuoteWizardV2'

export default function QuotePageV2() {
  return (
    <div>
      {/* Page header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-4 py-1.5 rounded-full text-sm font-medium mb-4">
          <span className="w-2 h-2 bg-blue-500 rounded-full" />
          AI-Powered Quote · Personalised Profile
        </div>
        <h1 className="text-3xl font-bold text-gray-900">Get Your Quote</h1>
        <p className="text-gray-500 mt-2 max-w-lg mx-auto">
          Our 5-step wizard captures your details and automatically analyses your vehicle,
          local area risk, and driving profile for a personalised premium.
        </p>
      </div>

      <QuoteWizardV2 />
    </div>
  )
}
