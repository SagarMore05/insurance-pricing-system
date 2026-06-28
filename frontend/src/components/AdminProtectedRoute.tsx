import { Navigate } from 'react-router-dom'
import AdminPage from '../pages/AdminPage'

export default function AdminProtectedRoute() {
  const token = localStorage.getItem('admin_token')
  if (!token) return <Navigate to="/admin/login" replace />
  return <AdminPage />
}
