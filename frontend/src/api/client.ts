import axios, { type InternalAxiosRequestConfig } from 'axios'

function attachToken(config: InternalAxiosRequestConfig) {
  const token = localStorage.getItem('admin_token')
  if (token) config.headers['Authorization'] = `Bearer ${token}`
  return config
}

function handleError(err: any): Promise<never> {
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
  const message = err.response?.data?.detail || err.message || 'Request failed'
  return Promise.reject(new Error(String(message)))
}

export const apiClient = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})
apiClient.interceptors.request.use(attachToken)
apiClient.interceptors.response.use((res) => res, handleError)

export const apiClientV2 = axios.create({
  baseURL: '/api/v2',
  headers: { 'Content-Type': 'application/json' },
})
apiClientV2.interceptors.request.use(attachToken)
apiClientV2.interceptors.response.use((res) => res, handleError)
