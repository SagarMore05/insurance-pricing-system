import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { insuranceApi } from '../api/insurance'

export default function AdminLoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const login = async () => {
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required')
      return
    }
    setLoading(true)
    setError('')
    try {
      const { access_token } = await insuranceApi.adminLogin(username, password)
      localStorage.setItem('admin_token', access_token)
      navigate('/admin')
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') login()
  }

  return (
    <div className="max-w-sm mx-auto mt-20">
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-1">Admin Login</h2>
        <p className="text-sm text-gray-500 mb-6">InsureAI Administration</p>

        <label className="label">Username</label>
        <input
          type="text"
          className="input-field mb-4"
          value={username}
          onChange={e => setUsername(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="admin"
          autoComplete="username"
          autoFocus
        />

        <label className="label">Password</label>
        <input
          type="password"
          className="input-field mb-4"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="••••••••"
          autoComplete="current-password"
        />

        {error && (
          <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2 mb-4">
            {error}
          </p>
        )}

        <button
          className="btn-primary w-full"
          onClick={login}
          disabled={loading}
        >
          {loading ? 'Logging in…' : 'Login'}
        </button>
      </div>
    </div>
  )
}
