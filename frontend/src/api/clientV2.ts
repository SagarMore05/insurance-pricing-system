import axios from 'axios'

// Separate axios instance for V2 API — does NOT modify the V1 client
export const apiClientV2 = axios.create({
  baseURL: '/api/v2',
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT token the same way V1 client does
apiClientV2.interceptors.request.use((config) => {
  const token = localStorage.getItem('admin_token')
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`
  }
  return config
})

apiClientV2.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      const url: string = err.config?.url ?? ''
      const hasToken = Boolean(localStorage.getItem('admin_token'))
      const isLoginEndpoint = url.includes('/admin/login')
      if (hasToken && !isLoginEndpoint) {
        localStorage.removeItem('admin_token')
        window.location.href = '/admin/login'
        return Promise.reject(new Error('Session expired. Please log in again.'))
      }
    }
    const detail = err.response?.data?.detail
    const message =
      Array.isArray(detail)
        ? detail.map((e: { msg: string }) => e.msg).join('; ')
        : detail || err.message || 'Request failed'
    return Promise.reject(new Error(String(message)))
  },
)
