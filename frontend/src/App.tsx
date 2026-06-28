import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout/Layout'
import HomePage from './pages/HomePage'
import QuotePage from './pages/QuotePage'
import QuotePageV2 from './pages/QuotePageV2'
import PolicyPage from './pages/PolicyPage'
import MyPoliciesPage from './pages/MyPoliciesPage'
import ClaimsPage from './pages/ClaimsPage'
import AdminLoginPage from './pages/AdminLoginPage'
import AdminProtectedRoute from './components/AdminProtectedRoute'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/quote" element={<QuotePage />} />
        <Route path="/quote-v2" element={<QuotePageV2 />} />
        <Route path="/policy/:id" element={<PolicyPage />} />
        <Route path="/policies" element={<MyPoliciesPage />} />
        <Route path="/claims" element={<ClaimsPage />} />
        <Route path="/admin/login" element={<AdminLoginPage />} />
        <Route path="/admin" element={<AdminProtectedRoute />} />
      </Route>
    </Routes>
  )
}
