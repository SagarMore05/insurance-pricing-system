import { Outlet, Link, useLocation } from 'react-router-dom'
import clsx from 'clsx'

export default function Layout() {
  const { pathname } = useLocation()

  return (
    <div className="min-h-screen flex flex-col">
      <nav className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <span className="font-bold text-gray-900 text-lg">InsureAI</span>
            </Link>
            <div className="flex items-center gap-6">
              <NavLink to="/" active={pathname === '/'}>Home</NavLink>
              <NavLink to="/quote-v2" active={pathname === '/quote-v2' || pathname === '/quote'}>Get Quote</NavLink>
              <NavLink to="/policies" active={
                pathname === '/policies' ||
                pathname.startsWith('/policy/') ||
                pathname === '/claims'
              }>My Policies</NavLink>
              <NavLink to="/admin" active={pathname.startsWith('/admin')}>Admin</NavLink>
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      <footer className="bg-white border-t border-gray-200 py-4 text-center text-sm text-gray-500">
        InsureAI — AI-Powered Car Insurance Pricing
      </footer>
    </div>
  )
}

function NavLink({ to, active, children }: { to: string; active: boolean; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className={clsx(
        'text-sm font-medium transition-colors',
        active ? 'text-blue-600' : 'text-gray-600 hover:text-gray-900'
      )}
    >
      {children}
    </Link>
  )
}
