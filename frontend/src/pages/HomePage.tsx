import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'

// ─── Icon primitives (Heroicons outline, 24 px grid) ─────────────────────────

function IcoArrow() {
  return (
    <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
    </svg>
  )
}

function IcoCheck() {
  return (
    <svg className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
    </svg>
  )
}

function IcoUser() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
    </svg>
  )
}

function IcoChart() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
    </svg>
  )
}

function IcoClock() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function IcoLock() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
    </svg>
  )
}

function IcoDoc() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )
}

function IcoBrain() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
    </svg>
  )
}

function IcoShield() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
    </svg>
  )
}

// ─── Reusable components ──────────────────────────────────────────────────────

/** Uppercase section eyebrow label */
function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <p className="text-blue-700 text-xs font-semibold uppercase tracking-widest mb-3">
      {children}
    </p>
  )
}

/** 2-line section heading */
function SectionHeading({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 leading-snug">
      {children}
    </h2>
  )
}

/** Trust / feature card */
interface FeatureCardProps {
  icon: ReactNode
  title: string
  description: string
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm hover:shadow-md transition-shadow duration-200">
      <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-blue-50 text-blue-700 mb-4">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-gray-900 mb-1.5">{title}</h3>
      <p className="text-gray-500 text-sm leading-relaxed">{description}</p>
    </div>
  )
}

/** Numbered process step */
interface StepProps {
  number: number
  title: string
  description: string
}

function Step({ number, title, description }: StepProps) {
  return (
    <div className="flex flex-col items-center text-center px-2">
      <div className="w-10 h-10 rounded-full border-2 border-blue-700 text-blue-700 font-bold text-sm flex items-center justify-center mb-4 bg-white">
        {number}
      </div>
      <h4 className="text-sm font-semibold text-gray-900 mb-1.5">{title}</h4>
      <p className="text-gray-500 text-sm leading-relaxed">{description}</p>
    </div>
  )
}

/** Single benefit row with emerald check */
function BenefitRow({ text }: { text: string }) {
  return (
    <li className="flex items-start gap-3">
      <span className="mt-0.5 flex-shrink-0 inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-50 text-emerald-600">
        <IcoCheck />
      </span>
      <span className="text-gray-700 text-sm leading-snug">{text}</span>
    </li>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function HomePage() {
  return (
    /*
     * Negate the Layout's horizontal padding and top padding so each section
     * can span the full container width with its own background.
     */
    <div className="-mt-8 -mx-4 sm:-mx-6 lg:-mx-8">

      {/* ─────────────────────────────────── HERO ─── */}
      <section className="bg-white px-4 sm:px-6 lg:px-8 pt-16 pb-20 border-b border-gray-100">
        <div className="max-w-3xl mx-auto text-center">

          {/* Badge */}
          <div className="inline-flex items-center gap-2 border border-blue-100 bg-blue-50 text-blue-700 px-3.5 py-1 rounded-full text-xs font-semibold mb-8 tracking-wide">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
            AI-Powered Insurance Pricing
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 leading-tight tracking-tight mb-5">
            Get a Fair Car Insurance<br />
            <span className="text-blue-700">Premium Designed for You</span>
          </h1>

          {/* Subtitle */}
          <p className="text-gray-500 text-lg leading-relaxed mb-10 max-w-2xl mx-auto">
            Answer a few simple questions and receive a personalized insurance quote
            based on your driving profile, vehicle details and risk factors.
          </p>

          {/* CTA row */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mb-12">
            <Link
              to="/quote-v2"
              className="inline-flex items-center gap-2 bg-blue-700 text-white font-semibold px-7 py-3 rounded-lg hover:bg-blue-800 transition-colors duration-200 shadow-sm"
            >
              Get Your Quote
              <IcoArrow />
            </Link>
            <a
              href="#how-it-works"
              className="inline-flex items-center gap-2 border border-gray-200 bg-white text-gray-700 font-medium px-7 py-3 rounded-lg hover:bg-gray-50 transition-colors duration-200"
            >
              Learn More
            </a>
          </div>

          {/* Social-proof strip */}
          <div className="flex flex-wrap items-center justify-center gap-6 text-xs text-gray-400">
            <span className="flex items-center gap-1.5">
              <span className="text-emerald-500"><IcoCheck /></span>
              Personalized to your profile
            </span>
            <span className="text-gray-200 select-none">|</span>
            <span className="flex items-center gap-1.5">
              <span className="text-emerald-500"><IcoCheck /></span>
              Transparent breakdown included
            </span>
            <span className="text-gray-200 select-none">|</span>
            <span className="flex items-center gap-1.5">
              <span className="text-emerald-500"><IcoCheck /></span>
              Quote in under 2 minutes
            </span>
          </div>

        </div>
      </section>

      {/* ──────────────────────────── TRUST PILLARS ─── */}
      <section className="bg-gray-50 px-4 sm:px-6 lg:px-8 py-16 border-b border-gray-100">
        <div className="max-w-6xl mx-auto">

          <div className="text-center mb-10">
            <Eyebrow>Why Trust Us</Eyebrow>
            <SectionHeading>Insurance Pricing You Can Rely On</SectionHeading>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <FeatureCard
              icon={<IcoUser />}
              title="Personalized Premium"
              description="Pay according to your driving history and vehicle profile — not a one-size-fits-all rate."
            />
            <FeatureCard
              icon={<IcoChart />}
              title="Transparent Pricing"
              description="Every factor that affects your premium is shown clearly, with no hidden charges."
            />
            <FeatureCard
              icon={<IcoClock />}
              title="Fast Quote"
              description="Receive your personalized quote in just a few minutes, with no paperwork."
            />
            <FeatureCard
              icon={<IcoLock />}
              title="Secure Platform"
              description="Your information is handled securely and never shared without your consent."
            />
          </div>

        </div>
      </section>

      {/* ──────────────────────────── HOW IT WORKS ─── */}
      <section id="how-it-works" className="bg-white px-4 sm:px-6 lg:px-8 py-16 border-b border-gray-100">
        <div className="max-w-5xl mx-auto">

          <div className="text-center mb-12">
            <Eyebrow>Simple Process</Eyebrow>
            <SectionHeading>How It Works</SectionHeading>
            <p className="text-gray-500 text-sm mt-3 max-w-md mx-auto">
              Getting a personalized quote takes just a few minutes and no technical knowledge.
            </p>
          </div>

          {/* Step grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 relative">
            {/* Connector (desktop) */}
            <div
              aria-hidden="true"
              className="hidden lg:block absolute top-5 left-[calc(12.5%+20px)] right-[calc(12.5%+20px)] h-px border-t border-dashed border-gray-300"
            />
            <Step
              number={1}
              title="Enter Your Details"
              description="Provide your personal information, vehicle details and driving history."
            />
            <Step
              number={2}
              title="AI Evaluates Your Profile"
              description="Our platform assesses your driving record, vehicle characteristics and risk factors."
            />
            <Step
              number={3}
              title="Premium is Calculated"
              description="A personalized premium is computed using your actual risk profile and applicable discounts."
            />
            <Step
              number={4}
              title="Receive Your Quote"
              description="Get your quote with a clear breakdown of every factor that influenced your price."
            />
          </div>

          <div className="text-center mt-12">
            <Link
              to="/quote-v2"
              className="inline-flex items-center gap-2 bg-blue-700 text-white font-semibold px-7 py-3 rounded-lg hover:bg-blue-800 transition-colors duration-200 shadow-sm"
            >
              Get Your Quote Now
              <IcoArrow />
            </Link>
          </div>

        </div>
      </section>

      {/* ──────────────────────── WHY CHOOSE INSUREAI ─── */}
      <section className="bg-gray-50 px-4 sm:px-6 lg:px-8 py-16 border-b border-gray-100">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start">

            {/* Left column – copy */}
            <div>
              <Eyebrow>Our Advantages</Eyebrow>
              <SectionHeading>Why Choose InsureAI?</SectionHeading>
              <p className="text-gray-500 text-sm leading-relaxed mt-4 mb-5">
                Traditional insurance pricing applies the same rate to everyone.
                InsureAI evaluates your individual profile to deliver a premium
                that reflects your actual risk — nothing more, nothing less.
              </p>
              <p className="text-gray-500 text-sm leading-relaxed mb-8">
                Our platform examines real-world signals — your driving history,
                vehicle details and location — so you are never charged more than
                your risk profile warrants.
              </p>
              <Link
                to="/quote-v2"
                className="inline-flex items-center gap-1.5 text-blue-700 font-semibold text-sm hover:underline"
              >
                Get your personalized quote
                <IcoArrow />
              </Link>
            </div>

            {/* Right column – benefit list */}
            <ul className="space-y-4 pt-1">
              {[
                { text: 'Fair Risk-Based Pricing — your premium matches your actual risk profile' },
                { text: 'Personalized Premiums — based on your driving history and vehicle' },
                { text: 'Transparent Breakdown — see exactly what shaped your price' },
                { text: 'Fast Quote Generation — results delivered in under 2 minutes' },
                { text: 'AI-Assisted Risk Assessment — intelligent, consistent evaluation' },
                { text: 'Secure & Reliable Platform — your data is handled with care' },
              ].map(({ text }) => (
                <BenefitRow key={text} text={text} />
              ))}
            </ul>

          </div>
        </div>
      </section>

      {/* ────────────────────── WHAT WE EVALUATE strip ─── */}
      <section className="bg-white px-4 sm:px-6 lg:px-8 py-14 border-b border-gray-100">
        <div className="max-w-6xl mx-auto">

          <div className="text-center mb-10">
            <Eyebrow>What We Evaluate</Eyebrow>
            <SectionHeading>Factors That Shape Your Premium</SectionHeading>
            <p className="text-gray-500 text-sm mt-3 max-w-xl mx-auto">
              Your premium is calculated using a combination of driving, vehicle and risk
              factors — ensuring the price is fair and specific to you.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Card 1 */}
            <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6">
              <div className="flex items-center gap-2.5 mb-4">
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-100 text-blue-700">
                  <IcoUser />
                </span>
                <h3 className="text-sm font-semibold text-gray-900">Driving Profile</h3>
              </div>
              <ul className="space-y-2.5">
                {[
                  'Years of driving experience',
                  'Previous claims record',
                  'Annual mileage driven',
                  'No-claim bonus eligibility',
                ].map((item) => (
                  <li key={item} className="flex items-center gap-2 text-gray-500 text-sm">
                    <span className="w-1 h-1 rounded-full bg-gray-400 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            {/* Card 2 */}
            <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6">
              <div className="flex items-center gap-2.5 mb-4">
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-100 text-blue-700">
                  <IcoDoc />
                </span>
                <h3 className="text-sm font-semibold text-gray-900">Vehicle Details</h3>
              </div>
              <ul className="space-y-2.5">
                {[
                  'Make, model and brand',
                  'Vehicle age and condition',
                  'Fuel type',
                  'Segment and market value',
                ].map((item) => (
                  <li key={item} className="flex items-center gap-2 text-gray-500 text-sm">
                    <span className="w-1 h-1 rounded-full bg-gray-400 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            {/* Card 3 */}
            <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6">
              <div className="flex items-center gap-2.5 mb-4">
                <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-blue-100 text-blue-700">
                  <IcoBrain />
                </span>
                <h3 className="text-sm font-semibold text-gray-900">Risk Factors</h3>
              </div>
              <ul className="space-y-2.5">
                {[
                  'Geographic risk zone',
                  'Driver age profile',
                  'Income-to-vehicle ratio',
                  'Applicable regulatory discounts',
                ].map((item) => (
                  <li key={item} className="flex items-center gap-2 text-gray-500 text-sm">
                    <span className="w-1 h-1 rounded-full bg-gray-400 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

        </div>
      </section>

      {/* ─────────────────────────────── FINAL CTA ─── */}
      <section className="bg-slate-900 px-4 sm:px-6 lg:px-8 py-20">
        <div className="max-w-2xl mx-auto text-center">

          <div className="inline-flex items-center gap-2 border border-slate-700 text-slate-400 px-3.5 py-1 rounded-full text-xs font-medium mb-6">
            <span className="text-emerald-500"><IcoShield /></span>
            Secure &amp; Transparent Pricing
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4 leading-snug">
            Ready to Find Your<br />Personalized Premium?
          </h2>

          <p className="text-slate-400 text-sm leading-relaxed mb-9 max-w-md mx-auto">
            Get your quote today and experience fair, transparent insurance pricing
            powered by intelligent technology. No hidden fees, no surprises.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              to="/quote-v2"
              className="inline-flex items-center gap-2 bg-blue-600 text-white font-semibold px-8 py-3 rounded-lg hover:bg-blue-500 transition-colors duration-200"
            >
              Get Your Quote
              <IcoArrow />
            </Link>
            <Link
              to="/policies"
              className="inline-flex items-center gap-2 border border-slate-700 text-slate-300 font-medium px-7 py-3 rounded-lg hover:bg-slate-800 transition-colors duration-200"
            >
              View My Policies
            </Link>
          </div>

          <p className="text-slate-600 text-xs mt-7">
            Free · No commitment required · Results in under 2 minutes
          </p>

        </div>
      </section>

    </div>
  )
}
