import { apiClient, apiClientV2 } from './client'
import type {
  QuoteFormData, PremiumQuoteResponse, PolicyResponse,
  DashboardStats, PremiumTrendPoint, CustomerListItem,
  PaginatedResponse, AssistantQueryResponse, ClaimResponse,
  ClaimAdminItem, FeedbackResponse, RetrainResponse,
  AgentAnalyticsResponse, UnderwritingResult, AgentRun, AgentRunDetail,
  AgentQuoteAwaitingResponse, ReviewQueueItem, ReviewDetail,
  ReviewDecisionRequest, ReviewDecisionResponse,
  ChampionRegistry, GovernanceLog, GateResults,
  ModelRegistryV2, ModelComparisonResponse,
  ApprovalRequest, ApprovalRequestDetail, ApprovalRequestListResponse,
  ApprovalAuditLogEntry,
  ShadowStatusResponse, ShadowStatisticsResponse, ShadowHistoryResponse, ShadowPredictionRecord,
  TrainingRun, TrainingRunListResponse, TrainingTriggerResponse, CandidateModel,
  PromotionEvaluationResponse, PromotionDetail, PromotionListResponse, RollbackRecord,
  ModelSelectionLatestResponse, ModelSelectionHistoryResponse,
  ModelSelectionComparisonResponse, ModelSelectionRetrainStatus,
} from '../types'
import type { CustomerQuoteRequestV2, QuotePreviewResponseV2 } from '../types/v2'

export const insuranceApi = {
  adminLogin: async (username: string, password: string): Promise<{ access_token: string; token_type: string }> => {
    const res = await apiClient.post('/admin/login', { username, password })
    return res.data
  },

  getQuote: async (data: QuoteFormData): Promise<PremiumQuoteResponse> => {
    const res = await apiClient.post('/quote', data)
    return res.data
  },

  createPolicy: async (predictionId: string, customerData: QuoteFormData): Promise<PolicyResponse> => {
    const res = await apiClient.post('/policies', {
      prediction_id: predictionId,
      customer_data: customerData,
    })
    return res.data
  },

  getPolicy: async (policyId: string): Promise<PolicyResponse> => {
    const res = await apiClient.get(`/policies/${policyId}`)
    return res.data
  },

  fileClaim: async (policyId: string, amount: number): Promise<ClaimResponse> => {
    const res = await apiClient.post('/claims', {
      policy_id: policyId,
      claimed_amount_inr: amount,
    })
    return res.data
  },

  getDashboard: async (): Promise<DashboardStats> => {
    const res = await apiClient.get('/admin/dashboard')
    return res.data
  },

  getCustomers: async (params: {
    page?: number
    page_size?: number
    risk_level?: string
    city?: string
  }): Promise<PaginatedResponse<CustomerListItem>> => {
    const res = await apiClient.get('/admin/customers', { params })
    return res.data
  },

  getPremiumTrends: async (): Promise<PremiumTrendPoint[]> => {
    const res = await apiClient.get('/admin/analytics/premium-trends')
    return res.data
  },

  submitFeedback: async (predictionId: string, actualOccurred: boolean, actualAmount?: number): Promise<FeedbackResponse> => {
    const res = await apiClient.post('/feedback', {
      prediction_id: predictionId,
      actual_claim_occurred: actualOccurred,
      actual_claim_amount_inr: actualAmount,
    })
    return res.data
  },

  assistantQuery: async (question: string): Promise<AssistantQueryResponse> => {
    const res = await apiClient.post('/assistant/query', { question })
    return res.data
  },

  getClaims: async (policyId: string): Promise<ClaimResponse[]> => {
    const res = await apiClient.get('/claims', { params: { policy_id: policyId } })
    return res.data
  },

  getAdminClaims: async (params: {
    page?: number
    page_size?: number
    status?: string
  }): Promise<PaginatedResponse<ClaimAdminItem>> => {
    const res = await apiClient.get('/admin/claims', { params })
    return res.data
  },

  updateClaim: async (
    claimId: string,
    claimStatus: 'approved' | 'rejected',
    approvedAmount?: number,
  ): Promise<ClaimResponse> => {
    const res = await apiClient.patch(`/claims/${claimId}`, {
      claim_status: claimStatus,
      approved_amount_inr: approvedAmount ?? null,
    })
    return res.data
  },

  triggerRetrain: async (): Promise<RetrainResponse> => {
    const res = await apiClient.post('/admin/retrain')
    return res.data
  },

  agentAnalyticsQuery: async (question: string, sessionId: string): Promise<AgentAnalyticsResponse> => {
    const res = await apiClient.post('/agent/analytics/query', { question, session_id: sessionId })
    return res.data
  },

  clearAgentSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/agent/analytics/session/${sessionId}`)
  },

  agentQuote: async (data: QuoteFormData): Promise<UnderwritingResult | AgentQuoteAwaitingResponse> => {
    const res = await apiClient.post('/agent/quote', data)
    return res.data
  },

  getAgentRuns: async (params: {
    page?: number
    page_size?: number
    agent_type?: string
    status?: string
  }): Promise<PaginatedResponse<AgentRun>> => {
    const res = await apiClient.get('/admin/agent-runs', { params })
    return res.data
  },

  getAgentRun: async (runId: string): Promise<AgentRunDetail> => {
    const res = await apiClient.get(`/admin/agent-runs/${runId}`)
    return res.data
  },

  getReviewQueue: async (params: {
    page?: number
    page_size?: number
  }): Promise<PaginatedResponse<ReviewQueueItem>> => {
    const res = await apiClient.get('/admin/review-queue', { params })
    return res.data
  },

  getReviewDetail: async (runId: string): Promise<ReviewDetail> => {
    const res = await apiClient.get(`/admin/review/${runId}`)
    return res.data
  },

  approveReview: async (runId: string, body: ReviewDecisionRequest): Promise<ReviewDecisionResponse> => {
    const res = await apiClient.post(`/admin/review/${runId}/approve`, body)
    return res.data
  },

  rejectReview: async (runId: string, body: ReviewDecisionRequest): Promise<ReviewDecisionResponse> => {
    const res = await apiClient.post(`/admin/review/${runId}/reject`, body)
    return res.data
  },

  // ─── Champion Registry & Governance ────────────────────────────────────────

  getRegistry: async (): Promise<ChampionRegistry> => {
    const res = await apiClient.get('/admin/registry')
    return res.data
  },

  getModelRegistry: async (): Promise<ModelRegistryV2> => {
    const res = await apiClient.get('/admin/model-registry')
    return res.data
  },

  getModelComparison: async (): Promise<ModelComparisonResponse> => {
    const res = await apiClient.get('/admin/model-registry/comparison')
    return res.data
  },

  getGovernanceLog: async (): Promise<GovernanceLog> => {
    const res = await apiClient.get('/admin/governance')
    return res.data
  },

  getGateResults: async (): Promise<GateResults> => {
    const res = await apiClient.get('/admin/gate-results')
    return res.data
  },

  previewQuoteV2: async (data: CustomerQuoteRequestV2): Promise<QuotePreviewResponseV2> => {
    const res = await apiClientV2.post('/quote/preview', data)
    return res.data
  },

  // ── Phase 4B — Human Approval Workflow ──────────────────────────────────
  createApprovalRequest: async (
    modelType: 'frequency' | 'severity',
    notes?: string,
  ): Promise<ApprovalRequestDetail> => {
    const res = await apiClient.post('/admin/approvals', { model_type: modelType, notes })
    return res.data
  },

  listApprovalRequests: async (params?: {
    status?: string
    model_type?: string
    page?: number
    page_size?: number
  }): Promise<ApprovalRequestListResponse> => {
    const res = await apiClient.get('/admin/approvals', { params })
    return res.data
  },

  getApprovalRequest: async (requestId: string): Promise<ApprovalRequestDetail> => {
    const res = await apiClient.get(`/admin/approvals/${requestId}`)
    return res.data
  },

  approveApprovalRequest: async (
    requestId: string,
    reviewerNote?: string,
  ): Promise<ApprovalRequestDetail> => {
    const res = await apiClient.post(`/admin/approvals/${requestId}/approve`, {
      reviewer_note: reviewerNote,
    })
    return res.data
  },

  rejectApprovalRequest: async (
    requestId: string,
    reviewerNote?: string,
  ): Promise<ApprovalRequestDetail> => {
    const res = await apiClient.post(`/admin/approvals/${requestId}/reject`, {
      reviewer_note: reviewerNote,
    })
    return res.data
  },

  getApprovalAuditLog: async (requestId: string): Promise<ApprovalAuditLogEntry[]> => {
    const res = await apiClient.get(`/admin/approvals/${requestId}/audit-log`)
    return res.data
  },

  // ── Phase 4C — Shadow Deployment Framework ───────────────────────────────
  getShadowStatus: async (): Promise<ShadowStatusResponse> => {
    const res = await apiClient.get('/admin/shadow/status')
    return res.data
  },

  getShadowStatistics: async (windowDays = 30): Promise<ShadowStatisticsResponse> => {
    const res = await apiClient.get('/admin/shadow/statistics', {
      params: { window_days: windowDays },
    })
    return res.data
  },

  getShadowHistory: async (params?: {
    page?: number
    page_size?: number
    status?: string
  }): Promise<ShadowHistoryResponse> => {
    const res = await apiClient.get('/admin/shadow/history', { params })
    return res.data
  },

  getShadowRecord: async (shadowId: string): Promise<ShadowPredictionRecord> => {
    const res = await apiClient.get(`/admin/shadow/${shadowId}`)
    return res.data
  },

  // ── Phase 5A — V5 Training Engine ─────────────────────────────────────────
  triggerTrainingRun: async (triggeredBy = 'admin'): Promise<TrainingTriggerResponse> => {
    const res = await apiClient.post('/admin/training/run', null, {
      params: { triggered_by: triggeredBy },
    })
    return res.data
  },

  listTrainingRuns: async (params?: {
    page?: number
    page_size?: number
  }): Promise<TrainingRunListResponse> => {
    const res = await apiClient.get('/admin/training/runs', { params })
    return res.data
  },

  getTrainingRun: async (runId: string): Promise<TrainingRun> => {
    const res = await apiClient.get(`/admin/training/runs/${runId}`)
    return res.data
  },

  getCandidateModel: async (modelId: string): Promise<CandidateModel> => {
    const res = await apiClient.get(`/admin/training/candidates/${modelId}`)
    return res.data
  },

  // ── Phase 5B — Champion Promotion Engine ────────────────────────────────────
  evaluatePromotion: async (body: {
    run_id: string
    frequency_candidate_id: string
    severity_candidate_id: string
    approval_id?: string
    bypass_shadow?: boolean
    bypass_drift?: boolean
    notes?: string
  }): Promise<PromotionEvaluationResponse> => {
    const res = await apiClient.post('/admin/promotions/evaluate', body)
    return res.data
  },

  executePromotion: async (
    promotionId: string,
    promotedBy = 'admin',
  ): Promise<PromotionDetail> => {
    const res = await apiClient.post(`/admin/promotions/${promotionId}/promote`, {
      promoted_by: promotedBy,
    })
    return res.data
  },

  rollbackPromotion: async (
    promotionId: string,
    reason: string,
    rolledBackBy = 'admin',
  ): Promise<PromotionDetail> => {
    const res = await apiClient.post(`/admin/promotions/${promotionId}/rollback`, {
      reason,
      rolled_back_by: rolledBackBy,
    })
    return res.data
  },

  listPromotions: async (params?: {
    status?: string
    page?: number
    page_size?: number
  }): Promise<PromotionListResponse> => {
    const res = await apiClient.get('/admin/promotions', { params })
    return res.data
  },

  getPromotion: async (promotionId: string): Promise<PromotionDetail> => {
    const res = await apiClient.get(`/admin/promotions/${promotionId}`)
    return res.data
  },

  listRollbacks: async (params?: {
    page?: number
    page_size?: number
  }): Promise<RollbackRecord[]> => {
    const res = await apiClient.get('/admin/promotions/rollbacks', { params })
    return res.data
  },

  // ── Multi-Algorithm Model Selection Engine ─────────────────────────────────
  getModelSelectionLatest: async (): Promise<ModelSelectionLatestResponse> => {
    const res = await apiClient.get('/admin/model-selection/latest')
    return res.data
  },

  getModelSelectionHistory: async (params?: {
    page?: number
    page_size?: number
  }): Promise<ModelSelectionHistoryResponse> => {
    const res = await apiClient.get('/admin/model-selection/history', { params })
    return res.data
  },

  getModelSelectionComparison: async (): Promise<ModelSelectionComparisonResponse> => {
    const res = await apiClient.get('/admin/model-selection/comparison')
    return res.data
  },

  triggerMultiAlgorithmRetrain: async (): Promise<{ triggered: boolean; message: string }> => {
    const res = await apiClient.post('/admin/model-selection/retrain')
    return res.data
  },

  getModelSelectionRetrainStatus: async (): Promise<ModelSelectionRetrainStatus> => {
    const res = await apiClient.get('/admin/model-selection/status')
    return res.data
  },
}
