import { useState, useRef, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { insuranceApi } from '../../api/insurance'
import type { AgentAnalyticsResponse, AnomalyItem, RecommendationItem } from '../../types'

const SESSION_KEY = 'insureai_agent_session'

function getOrCreateSession(): string {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  response?: AgentAnalyticsResponse
}

const EXAMPLE_QUESTIONS = [
  'What is the average premium for customers under 30?',
  'Which city has the highest claims ratio?',
  'How many policies have high risk level?',
  'Show monthly premium trends for the last 6 months',
  'Find any anomalies in recent claim amounts',
  'Recommend actions for reducing high-risk policies',
]

const INTENT_LABEL: Record<string, string> = {
  anomaly: 'Anomaly Detection',
  recommendation: 'Recommendations',
  comparison: 'Comparison',
  general: 'Analytics',
}

const INTENT_COLOR: Record<string, string> = {
  anomaly: 'bg-red-100 text-red-700',
  recommendation: 'bg-purple-100 text-purple-700',
  comparison: 'bg-blue-100 text-blue-700',
  general: 'bg-gray-100 text-gray-600',
}

const PRIORITY_STYLE: Record<string, string> = {
  high: 'border-l-red-500 bg-red-50',
  medium: 'border-l-amber-500 bg-amber-50',
  low: 'border-l-green-500 bg-green-50',
}

const PRIORITY_BADGE: Record<string, string> = {
  high: 'bg-red-100 text-red-700',
  medium: 'bg-amber-100 text-amber-700',
  low: 'bg-green-100 text-green-700',
}

const CATEGORY_LABEL: Record<string, string> = {
  pricing: 'Pricing',
  risk_management: 'Risk Mgmt',
  claims: 'Claims',
  customer_segment: 'Segments',
  operations: 'Operations',
}

export default function AIAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sqlExpanded, setSqlExpanded] = useState<number | null>(null)
  const [sessionId] = useState(getOrCreateSession)
  const bottomRef = useRef<HTMLDivElement>(null)

  const mutation = useMutation({
    mutationFn: (question: string) => insuranceApi.agentAnalyticsQuery(question, sessionId),
    onSuccess: (data, question) => {
      setMessages(prev => [
        ...prev,
        { role: 'user', content: question },
        { role: 'assistant', content: data.answer, response: data },
      ])
    },
    onError: (err, question) => {
      setMessages(prev => [
        ...prev,
        { role: 'user', content: question },
        { role: 'assistant', content: `Error: ${err.message}` },
      ])
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = (q: string) => {
    if (!q.trim()) return
    setInput('')
    mutation.mutate(q)
  }

  const newConversation = () => {
    const newId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, newId)
    insuranceApi.clearAgentSession(sessionId).catch(() => {})
    setMessages([])
    setSqlExpanded(null)
    // Force page reload to reset sessionId state
    window.location.reload()
  }

  return (
    <div className="card flex flex-col h-[680px]">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center">
          <span className="text-white text-xs font-bold">AI</span>
        </div>
        <div>
          <h3 className="font-semibold text-gray-900 leading-tight">Analytics Agent</h3>
          <p className="text-xs text-gray-400">Groq · Llama 3.3 70B · Session memory · Anomaly detection</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={newConversation}
            className="ml-auto text-xs text-gray-400 hover:text-gray-600 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
          >
            New conversation
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
        {messages.length === 0 && (
          <div className="text-center py-6">
            <p className="text-gray-400 text-sm mb-4">
              Ask me anything about your insurance data. I can detect anomalies, compare segments, and generate recommendations.
            </p>
            <div className="flex flex-col gap-2">
              {EXAMPLE_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-sm text-blue-600 hover:text-blue-700 text-left px-3 py-2 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] ${msg.role === 'user'
              ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm'
              : 'space-y-2 w-full'
            }`}>
              {msg.role === 'user' ? (
                msg.content
              ) : (
                <>
                  {/* Intent badge */}
                  {msg.response?.intent && (
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${INTENT_COLOR[msg.response.intent] ?? INTENT_COLOR.general}`}>
                        {INTENT_LABEL[msg.response.intent] ?? 'Analytics'}
                      </span>
                    </div>
                  )}

                  {/* Answer bubble */}
                  <div className="bg-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-gray-800">
                    {msg.content}
                  </div>

                  {/* SQL toggle */}
                  {msg.response?.sql_used && msg.response.sql_used !== '[blocked]' && (
                    <div>
                      <button
                        onClick={() => setSqlExpanded(sqlExpanded === i ? null : i)}
                        className="text-xs text-gray-400 hover:text-gray-600"
                      >
                        {sqlExpanded === i ? '▾' : '▸'} View SQL
                      </button>
                      {sqlExpanded === i && (
                        <pre className="bg-gray-900 text-green-400 text-xs p-3 rounded-lg mt-1 overflow-x-auto">
                          {msg.response.sql_used}
                        </pre>
                      )}
                    </div>
                  )}

                  {/* Chart */}
                  {msg.response?.data && msg.response.chart_suggestion !== 'none' && (
                    <ChartPreview
                      data={msg.response.data}
                      chartType={msg.response.chart_suggestion}
                    />
                  )}

                  {/* Anomaly cards */}
                  {msg.response?.anomalies && msg.response.anomalies.length > 0 && (
                    <AnomalyCards anomalies={msg.response.anomalies} />
                  )}

                  {/* Recommendation cards */}
                  {msg.response?.recommendations && msg.response.recommendations.length > 0 && (
                    <RecommendationCards recommendations={msg.response.recommendations} />
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {mutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl px-4 py-3 text-sm text-gray-500 animate-pulse">
              Analyzing...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
          placeholder="Ask about your insurance data..."
          className="input-field flex-1"
          disabled={mutation.isPending}
        />
        <button
          onClick={() => send(input)}
          disabled={mutation.isPending || !input.trim()}
          className="btn-primary px-4"
        >
          Send
        </button>
      </div>
    </div>
  )
}

function AnomalyCards({ anomalies }: { anomalies: AnomalyItem[] }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="mt-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs font-medium text-red-600 hover:text-red-700"
      >
        <span className="w-4 h-4 bg-red-100 rounded-full flex items-center justify-center text-red-600 font-bold">!</span>
        {anomalies.length} Statistical {anomalies.length === 1 ? 'Anomaly' : 'Anomalies'} Detected
        <span className="text-red-400">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5">
          {anomalies.map((a, i) => (
            <div key={i} className="bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-red-700">{a.column.replace(/_/g, ' ')}</span>
                <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                  a.direction === 'high' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'
                }`}>
                  {a.direction === 'high' ? 'High outlier' : 'Low outlier'}
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-0.5">
                Value: <span className="font-medium">{a.value.toLocaleString()}</span>
                <span className="text-gray-400 ml-2">· deviation score: {a.deviation_score}×</span>
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function RecommendationCards({ recommendations }: { recommendations: RecommendationItem[] }) {
  const [expanded, setExpanded] = useState(true)
  return (
    <div className="mt-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs font-medium text-purple-600 hover:text-purple-700 mb-1.5"
      >
        <span className="w-4 h-4 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 text-xs">+</span>
        {recommendations.length} {recommendations.length === 1 ? 'Recommendation' : 'Recommendations'}
        <span className="text-purple-400">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="space-y-1.5">
          {recommendations.map((r, i) => (
            <div key={i} className={`border-l-4 rounded-r-lg px-3 py-2 ${PRIORITY_STYLE[r.priority] ?? PRIORITY_STYLE.medium}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-xs font-medium px-1.5 py-0.5 rounded-full ${PRIORITY_BADGE[r.priority] ?? PRIORITY_BADGE.medium}`}>
                  {r.priority.toUpperCase()}
                </span>
                <span className="text-xs text-gray-500">{CATEGORY_LABEL[r.category] ?? r.category}</span>
              </div>
              <p className="text-xs text-gray-700 leading-relaxed">{r.text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const PIE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

function ChartPreview({ data, chartType }: { data: Record<string, string>[]; chartType: string }) {
  if (!data.length) return null
  const keys = Object.keys(data[0])
  const xKey = keys[0]
  const yKey = keys[1]

  const numericData = data.map(row => ({
    ...row,
    [yKey]: Number.isNaN(Number(row[yKey])) ? row[yKey] : Number(row[yKey]),
  }))

  if (chartType === 'pie') {
    const pieData = numericData.map(row => ({
      name: String(row[xKey]),
      value: typeof row[yKey] === 'number' ? (row[yKey] as number) : 0,
    }))
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-3 mt-1">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={70}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={false}
            >
              {pieData.map((_, idx) => (
                <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 mt-1">
      <ResponsiveContainer width="100%" height={180}>
        {chartType === 'line' ? (
          <LineChart data={numericData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Line type="monotone" dataKey={yKey} stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        ) : (
          <BarChart data={numericData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey={yKey} fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}
