import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="max-w-4xl mx-auto text-center py-16">
      {/* V4 status badge */}
      <div className="inline-flex items-center gap-2 bg-green-50 text-green-700 px-4 py-1.5 rounded-full text-sm font-medium mb-6 border border-green-200">
        <span className="w-2 h-2 bg-green-500 rounded-full" />
        AI-Powered Pricing
      </div>

      <h1 className="text-5xl font-bold text-gray-900 mb-4 leading-tight">
        Car Insurance Priced<br />
        <span className="text-blue-600">Right for You</span>
      </h1>
      <p className="text-gray-500 text-lg mb-4 max-w-2xl mx-auto">
        Two champion AI models analyse your driving profile, vehicle data, and geographic risk
        to calculate a fair, transparent premium — 59 enriched features, no hidden fees.
      </p>

      {/* Customer benefit pills */}
      <div className="inline-flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-xl px-5 py-2.5 text-xs mb-8">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-blue-500 rounded-full" />
          <span className="text-slate-600">Personalised to your profile</span>
        </div>
        <span className="text-slate-300">|</span>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-green-500 rounded-full" />
          <span className="text-slate-600">See exactly why</span>
        </div>
        <span className="text-slate-300">|</span>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-purple-500 rounded-full" />
          <span className="text-slate-600">Quote in 2 minutes</span>
        </div>
      </div>

      {/* Primary CTA */}
      <div className="flex items-center justify-center mb-16">
        <Link to="/quote-v2" className="btn-primary text-base px-8 py-3 inline-block">
          Get Your Quote
        </Link>
      </div>

      <div className="grid grid-cols-3 gap-6 text-left">
        {[
          {
            title: 'Priced Around You',
            desc: 'Your vehicle safety record, driving history, and local area risk are all '
                + 'factored in — so you only pay for the risk you actually carry.',
          },
          {
            title: 'Transparent Breakdown',
            desc: 'See every factor that changed your premium — discounts for safe driving, '
                + 'adjustments for your location, and your no-claim bonus.',
          },
          {
            title: 'Instant & Trustworthy',
            desc: 'Your quote is calculated by AI models that are continuously tested and '
                + 'audited, so you can trust the number you see.',
          },
        ].map(f => (
          <div key={f.title} className="card">
            <h3 className="font-semibold text-gray-900 mb-2">{f.title}</h3>
            <p className="text-gray-500 text-sm">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
