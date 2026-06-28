export type RiskLevel = 'low' | 'medium' | 'high'
export type ClaimStatus = 'pending' | 'approved' | 'rejected'

export interface AdjustmentDetail {
  reason: string
  factor: number
  impact_inr: number
}

export interface ExplanationDetail {
  base_premium: number
  expected_loss: number
  adjustments: AdjustmentDetail[]
  final_premium: number
  risk_level: RiskLevel
  claim_probability: number
  summary: string
}

export interface DrivingScoreResult {
  score: number
  factors: string[]
}

export interface PremiumQuoteResponse {
  prediction_id: string
  premium_amount_inr: number
  risk_level: RiskLevel
  claim_probability: number
  expected_claim_amount_inr: number
  explanation: ExplanationDetail
  model_version: string
  created_at: string
  driving_score_info?: DrivingScoreResult
}

export interface PolicyResponse {
  policy_id: string
  customer_id: string
  vehicle_id: string
  premium_amount_inr: number
  risk_level: RiskLevel
  model_version: string
  created_at: string
  is_active: boolean
  prediction?: PremiumQuoteResponse
}

export interface ClaimResponse {
  claim_id: string
  policy_id: string
  claimed_amount_inr: number
  approved_amount_inr: number | null
  claim_date: string
  claim_status: ClaimStatus
}

export interface DashboardStats {
  total_policies: number
  active_policies: number
  avg_premium_inr: number
  total_claims: number
  claims_ratio: number
  total_premium_collected: number
  premium_by_risk_level: Record<string, number>
  policies_by_risk_level: Record<string, number>
}

export interface PremiumTrendPoint {
  month: string
  risk_level: string
  avg_premium: number
  policy_count: number
}

export interface CustomerListItem {
  customer_id: string
  age: number
  gender: string
  city: string
  created_at: string
  policy_count: number
  latest_risk_level: string | null
  latest_premium: number | null
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface AssistantQueryResponse {
  answer: string
  sql_used: string | null
  data: Record<string, string>[] | null
  chart_suggestion: 'bar' | 'line' | 'pie' | 'none'
}

export interface QuoteFormData {
  age: number
  gender: string
  city: string
  annual_income_inr?: number
  vehicle_brand: string
  vehicle_segment: string
  fuel_type: string
  vehicle_age_years: number
  vehicle_value_inr: number
  annual_mileage_km: number
  previous_claims: number
  years_licensed: number
  driving_score?: number
  car_model?: string   // maps to CustomerQuoteRequest.car_model
  engine_cc?: number   // maps to CustomerQuoteRequest.engine_cc
}

export interface ClaimAdminItem {
  claim_id: string
  policy_id: string
  claimed_amount_inr: number
  approved_amount_inr: number | null
  claim_date: string
  claim_status: ClaimStatus
  risk_level: string | null
}

export interface FeedbackResponse {
  feedback_id: string
  prediction_id: string
  actual_claim_occurred: boolean
  actual_claim_amount_inr: number | null
  feedback_date: string
}

export interface RetrainPerformanceReport {
  evaluation_date: string
  n_samples: number
  frequency_auc: number | null
  severity_rmse: number | null
  drift_detected: boolean
  retrain_recommended: boolean
  reason?: string
  promoted_version?: string
  retrain_metrics?: Record<string, unknown>
  hitl_reviewed_count?: number
  human_override_rate?: number | null
  ml_human_agreement_rate?: number | null
}

export interface RetrainResponse {
  message: string
  triggered: boolean
  performance_report: RetrainPerformanceReport | null
}

export interface AnomalyItem {
  column: string
  value: number
  row_data: Record<string, string>
  direction: 'high' | 'low'
  deviation_score: number
}

export interface RecommendationItem {
  category: 'pricing' | 'risk_management' | 'claims' | 'customer_segment' | 'operations'
  text: string
  priority: 'high' | 'medium' | 'low'
}

export interface AgentAnalyticsResponse {
  answer: string
  sql_used: string | null
  data: Record<string, string>[] | null
  chart_suggestion: 'bar' | 'line' | 'pie' | 'none'
  session_id: string
  intent: 'anomaly' | 'recommendation' | 'comparison' | 'general'
  anomalies: AnomalyItem[] | null
  recommendations: RecommendationItem[] | null
}

// ─── Underwriting Agent ───────────────────────────────────────────────────────

export interface FraudSignal {
  signal: string
  severity: 'high' | 'medium' | 'low'
  description: string
}

export interface CustomerHistorySummary {
  similar_policies_count: number
  avg_premium_in_segment: number
  segment_claim_frequency: number
  total_claims_in_segment: number
}

export interface UnderwritingDecision {
  decision: 'approve' | 'flag' | 'refer'
  confidence: number
  risk_adjustment_factor: number
  agent_adjusted_premium: number
}

export interface NodeLogEntry {
  node: string
  started_at: string
  duration_ms: number
  input_summary: string
  output_summary: string
}

export interface ReasoningStep {
  label: string
  status: 'complete' | 'skipped'
  duration_ms?: number
  detail?: string
}

export interface UnderwritingResult {
  prediction_id: string
  premium_amount_inr: number
  risk_level: RiskLevel
  claim_probability: number
  expected_claim_amount_inr: number
  explanation: ExplanationDetail
  model_version: string
  created_at: string
  session_id: string
  fraud_signals: FraudSignal[]
  customer_history: CustomerHistorySummary
  underwriting_decision: UnderwritingDecision
  reasoning_trace: string
  node_execution_log: NodeLogEntry[]
  driving_score_info?: DrivingScoreResult
}

// ─── Agent Observability ──────────────────────────────────────────────────────

export interface AgentToolCall {
  call_id: string
  tool_name: string
  tool_input: Record<string, string> | null
  tool_output: string | null
  called_at: string
  duration_ms: number | null
}

export interface AgentRun {
  run_id: string
  session_id: string
  agent_type: 'analytics' | 'underwriting'
  intent: string | null
  status: 'running' | 'completed' | 'failed' | 'awaiting_review'
  started_at: string
  completed_at: string | null
  duration_ms: number | null
  input_text: string
  review_outcome: 'approved' | 'rejected' | null
}

export interface AgentRunDetail extends AgentRun {
  output_text: string | null
  tool_calls: AgentToolCall[]
}

// ─── HITL Review ──────────────────────────────────────────────────────────────

export interface AgentQuoteAwaitingResponse {
  run_id: string
  prediction_id: string
  status: 'awaiting_review'
  premium_amount_inr: number
  risk_level: string
  claim_probability: number
  agent_decision: string
  confidence: number
  fraud_signal_count: number
  review_triggers: string[]
  message: string
}

export interface ReviewQueueItem {
  run_id: string
  prediction_id: string
  started_at: string
  minutes_pending: number
  agent_decision: string | null
  confidence_score: number
  claim_probability: number
  fraud_signal_count: number
  high_severity_count: number
  premium_amount_inr: number
  review_triggers: string[]
  input_summary: string
}

export interface ReviewDetail extends ReviewQueueItem {
  fraud_signals: FraudSignal[]
  customer_history: CustomerHistorySummary
  underwriting_decision: UnderwritingDecision
  reasoning_trace: string
  node_execution_log: NodeLogEntry[]
}

export interface ReviewDecisionRequest {
  reviewer_note?: string
}

export interface ReviewDecisionResponse {
  run_id: string
  review_outcome: 'approved' | 'rejected'
  reviewed_at: string
  reviewer_note: string | null
  agent_decision: string | null
}

// ─── Champion Registry ────────────────────────────────────────────────────────

export interface FrequencyEvalMetrics {
  roc_auc: number
  pr_auc: number
  f1: number
  precision: number
  recall: number
  brier_score: number
  log_loss: number
}

export interface SeverityEvalMetrics {
  r2: number
  rmse: number
  mae: number
  mape_pct: number
  median_abs_error: number
}

export interface FrequencyProductionMetrics {
  auc_roc: number
  f1: number
  precision: number
  recall: number
  brier: number
  n_features: number
  algorithm: string
}

export interface SeverityProductionMetrics {
  r2: number
  rmse: number
  mae: number
  mape_pct: number
  note?: string
}

export interface FrequencyChampion {
  champion: string
  algorithm: string
  metric: string
  score: number
  baseline_score: number
  feature_count: number
  preprocessor_version: string
  version: string
  trained_on_rows: number
  train_dataset: string
  promotion_date: string
  eval_metrics: FrequencyEvalMetrics
  deployment_status: string
  production_fallback_version: string
  production_fallback_score: number
  production_fallback_metrics?: FrequencyProductionMetrics
  production_fallback_updated?: string
  production_fallback_dataset?: string
}

export interface SeverityChampion {
  champion: string
  algorithm: string
  metric: string
  score: number
  baseline_score: number
  feature_count: number
  preprocessor_version: string
  version: string
  trained_on_rows: number
  train_dataset: string
  promotion_date: string
  eval_metrics: SeverityEvalMetrics
  deployment_status: string
  production_fallback_version: string
  production_fallback_score: number
  production_fallback_metrics?: SeverityProductionMetrics
  production_fallback_updated?: string
  production_fallback_dataset?: string
}

export interface ChampionRegistry {
  registry_version: string
  last_updated: string
  frequency: FrequencyChampion
  severity: SeverityChampion
}

// ─── Model Registry V2 ───────────────────────────────────────────────────────

export interface FrequencyModelMetrics {
  auc_roc: number | null
  pr_auc: number | null
  f1: number | null
  precision: number | null
  recall: number | null
  brier_score: number | null
  log_loss: number | null
  threshold: number | null
}

export interface SeverityModelMetrics {
  r2: number | null
  rmse: number | null
  mae: number | null
  mape_pct: number | null
  median_abs_error?: number | null
}

export interface ModelTargets {
  targets_hit: number
  targets_total: number
  [key: string]: boolean | number
}

export interface DatasetCeiling {
  theoretical_auc_ceiling?: number
  theoretical_r2_ceiling?: number
  note: string
}

export interface RegistryModelBase {
  model_id: string
  model_name: string
  model_type: 'frequency' | 'severity'
  algorithm: string
  version: string
  status: 'production' | 'candidate' | 'archived'
  serving_status: string
  training_timestamp: string
  promotion_date: string
  promoted_by: string
  dataset: string
  n_total_rows: number | null
  n_train_rows: number | null
  n_val_rows: number | null
  n_test_rows: number | null
  feature_count: number
  feature_names: string[] | null
  preprocessor: string
  preprocessor_path: string
  artifact_path: string
  artifact_size_mb: number | null
  notes: string
  reason_not_promoted?: string
  git_commit: string | null
}

export interface FrequencyRegistryEntry extends RegistryModelBase {
  model_type: 'frequency'
  metrics: FrequencyModelMetrics
  targets?: ModelTargets | null
  dataset_ceiling?: DatasetCeiling | null
}

export interface SeverityRegistryEntry extends RegistryModelBase {
  model_type: 'severity'
  metrics: SeverityModelMetrics
  dataset_ceiling?: DatasetCeiling | null
  comparison_note?: string
}

export interface ModelRegistryV2 {
  registry_version: string
  schema: string
  last_updated: string
  description: string
  frequency: {
    production: FrequencyRegistryEntry
    candidate: FrequencyRegistryEntry | null
    archived: FrequencyRegistryEntry[]
  }
  severity: {
    production: SeverityRegistryEntry
    candidate: SeverityRegistryEntry | null
    archived: SeverityRegistryEntry[]
  }
}

export interface ComparisonCell {
  production: number | null
  candidate: number | null
  delta: number | null
  verdict: 'improved' | 'declined' | 'unchanged' | 'unavailable'
}

export interface ModelComparisonResponse {
  frequency: Record<string, ComparisonCell> | null
  severity: Record<string, ComparisonCell | string | undefined> | null
  last_updated: string
}

// ─── Governance Log ───────────────────────────────────────────────────────────

export interface GovernanceEvent {
  timestamp: string
  event_type: string
  model_type: string
  action: string
  metric: string
  old_score: number | null
  new_score: number
  decision: string
  challenger: string
  incumbent: string | null
  notes: string
}

export interface GovernanceLog {
  schema_version: string
  events: GovernanceEvent[]
}

// ─── Gate Results ─────────────────────────────────────────────────────────────

export interface GateScenario {
  name: string
  claim_prob: number
  expected_claim: number
  premium: number
  risk: string
  passed: boolean
}

export interface GateResults {
  timestamp: string
  gates: Record<string, string>
  frontend_pass: boolean
  all_pass: boolean
  scenario_results: GateScenario[]
}

// ─── Phase 4B — Human Approval Workflow ──────────────────────────────────────

export interface ModelCardFrequencyMetrics {
  roc_auc: number
  pr_auc?: number | null
  f1?: number | null
  precision?: number | null
  recall?: number | null
  brier_score?: number | null
  log_loss?: number | null
}

export interface ModelCardSeverityMetrics {
  r2: number
  rmse?: number | null
  mae?: number | null
  mape_pct?: number | null
  median_abs_error?: number | null
}

export interface ModelCardDriftSummary {
  computed_at?: string | null
  overall_psi?: number | null
  overall_severity?: string | null
  high_drift_features: string[]
  medium_drift_features: string[]
  features_computed?: number | null
  recommendation?: string | null
}

export interface ModelCard {
  model_type: string
  model_version: string
  algorithm: string
  training_date: string
  dataset_size: number
  preprocessor_version: string
  champion_since: string
  frequency_metrics?: ModelCardFrequencyMetrics | null
  severity_metrics?: ModelCardSeverityMetrics | null
  feature_importance_summary?: Record<string, number> | null
  drift_summary?: ModelCardDriftSummary | null
  recommendation: string
}

export interface ApprovalAuditLogEntry {
  log_id: string
  request_id: string
  event_type: string
  actor?: string | null
  note?: string | null
  event_data?: Record<string, unknown> | null
  created_at: string
}

export interface ApprovalRequest {
  request_id: string
  model_type: string
  model_version: string
  status: 'pending' | 'approved' | 'rejected'
  submitted_by: string
  submitted_at: string
  reviewed_by?: string | null
  reviewed_at?: string | null
  reviewer_note?: string | null
  model_card?: ModelCard | null
  recommendation?: string | null
}

export interface ApprovalRequestDetail extends ApprovalRequest {
  audit_logs: ApprovalAuditLogEntry[]
}

export interface ApprovalRequestListResponse {
  items: ApprovalRequest[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// ─── Phase 4C — Shadow Deployment Framework ───────────────────────────────────

export interface ShadowValidationStatus {
  status: 'NOT_STARTED' | 'IN_PROGRESS' | 'COMPLETED'
  challenger_available: boolean
  note: string
}

export interface ShadowPredictionRecord {
  id: string
  request_id: string
  quote_id: string | null
  created_at: string
  champion_model_name: string
  champion_version: string
  challenger_model_name: string | null
  challenger_version: string | null
  champion_prediction_json: Record<string, unknown> | null
  challenger_prediction_json: Record<string, unknown> | null
  premium_difference: number | null
  risk_difference: string | null
  prediction_latency_ms: number | null
  status: 'WAITING_FOR_CHALLENGER' | 'PENDING' | 'COMPLETED' | 'FAILED'
  comparison_result: 'MATCH' | 'DIVERGENT' | 'CHALLENGER_HIGHER' | 'CHALLENGER_LOWER' | null
  notes: string | null
}

export interface ShadowStatusResponse {
  framework_status: string
  challenger_available: boolean
  challenger_status: 'AVAILABLE' | 'UNAVAILABLE'
  total_shadow_predictions: number
  waiting_for_challenger: number
  completed_comparisons: number
  failed_comparisons: number
  champion_model: string
  champion_version: string
  last_recorded_at: string | null
  message: string
}

export interface ShadowStatisticsResponse {
  window_days: number
  total_shadow_predictions: number
  completed_comparisons: number
  waiting_for_challenger: number
  completion_rate_pct: number
  avg_premium_difference_inr: number | null
  avg_prediction_latency_ms: number | null
  challenger_available: boolean
}

export interface ShadowHistoryResponse {
  items: ShadowPredictionRecord[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// ─── Phase 5A — V5 Training Engine ───────────────────────────────────────────

export interface FrequencyTrainingMetrics {
  roc_auc: number
  pr_auc: number
  precision: number
  recall: number
  f1: number
  brier_score: number
  threshold?: number
  n_samples?: number
  n_positive?: number
  positive_rate?: number
}

export interface SeverityTrainingMetrics {
  rmse: number
  mae: number
  r2: number
  mape_pct: number | null
  n_samples?: number
  mean_actual?: number
  mean_predicted?: number
}

export interface CandidateModel {
  model_id: string
  run_id: string
  algorithm: 'xgboost' | 'catboost' | 'lightgbm'
  model_type: 'frequency' | 'severity'
  status: 'TRAINING' | 'COMPLETED' | 'FAILED'
  started_at: string | null
  completed_at: string | null
  duration_seconds: number | null
  train_rows: number | null
  val_rows: number | null
  test_rows: number | null
  hyperparameters: Record<string, unknown> | null
  frequency_metrics: FrequencyTrainingMetrics | null
  severity_metrics: SeverityTrainingMetrics | null
  feature_importance: Record<string, number> | null
  artifact_path: string | null
  error_message: string | null
}

export interface TrainingRun {
  run_id: string
  run_tag: string
  triggered_by: string
  triggered_at: string | null
  completed_at: string | null
  status: 'RUNNING' | 'COMPLETED' | 'FAILED' | 'VALIDATION_FAILED' | 'CANCELLED'
  dataset_rows: number | null
  feature_count: number | null
  maturity_days: number
  train_rows: number | null
  val_rows: number | null
  test_rows: number | null
  error_message: string | null
  summary: Record<string, unknown> | null
  candidates: CandidateModel[]
}

export interface TrainingRunListResponse {
  items: TrainingRun[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface TrainingTriggerResponse {
  run_id: string
  run_tag: string
  status: string
  message: string
  triggered_by: string
  triggered_at: string
}

// ─── Phase 5B — Champion Promotion Engine ─────────────────────────────────────

export type PromotionStatus =
  | 'PENDING' | 'EVALUATING' | 'APPROVED' | 'REJECTED'
  | 'PROMOTING' | 'ACTIVE' | 'ROLLED_BACK' | 'FAILED'

export interface GateResult {
  gate: string
  passed: boolean
  reason: string
  details: Record<string, unknown> | null
}

export interface PromotionEvaluationResponse {
  promotion_id: string
  run_id: string
  status: PromotionStatus
  all_gates_passed: boolean
  gates: Record<string, GateResult>
  freq_comparison: Record<string, unknown> | null
  sev_comparison: Record<string, unknown> | null
  can_promote: boolean
  message: string
  created_at: string
}

export interface RollbackRecord {
  rollback_id: string
  promotion_id: string
  rollback_reason: string | null
  rollback_trigger: 'health_check' | 'load_failure' | 'prediction_failure' | 'registry_failure' | 'manual' | 'auto'
  rollback_status: 'SUCCESS' | 'FAILED' | 'PARTIAL'
  rollback_duration_seconds: number | null
  rolled_back_by: string | null
  rolled_back_at: string
  error_message: string | null
  restored_frequency_champion: Record<string, unknown> | null
  restored_severity_champion: Record<string, unknown> | null
}

export interface PromotionDetail {
  promotion_id: string
  run_id: string
  frequency_candidate_id: string | null
  severity_candidate_id: string | null
  status: PromotionStatus
  all_gates_passed: boolean | null
  evaluation_report: Record<string, unknown> | null
  gates_passed: Record<string, unknown> | null
  gates_failed: Record<string, unknown> | null
  old_frequency_champion: Record<string, unknown> | null
  old_severity_champion: Record<string, unknown> | null
  new_frequency_champion: Record<string, unknown> | null
  new_severity_champion: Record<string, unknown> | null
  promoted_by: string | null
  promoted_at: string | null
  promotion_duration_seconds: number | null
  backup_path: string | null
  error_message: string | null
  notes: string | null
  created_at: string
  updated_at: string | null
  rollback_records: RollbackRecord[]
}

export interface PromotionListResponse {
  items: PromotionDetail[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// ─── Multi-Algorithm Model Selection Engine ───────────────────────────────────

export interface AlgorithmFrequencyMetrics {
  roc_auc: number | null
  pr_auc: number | null
  f1: number | null
  precision: number | null
  recall: number | null
  brier_score: number | null
  threshold: number | null
  n_samples: number | null
  n_positive: number | null
  positive_rate: number | null
}

export interface AlgorithmSeverityMetrics {
  r2: number | null
  cv_r2_mean: number | null
  cv_r2_std: number | null
  rmse: number | null
  mae: number | null
  mape_pct: number | null
  n_samples: number | null
  mean_actual: number | null
  mean_predicted: number | null
}

export interface AlgorithmResult {
  algorithm: 'xgboost' | 'lightgbm' | 'catboost'
  model_type: 'frequency' | 'severity'
  status: 'success' | 'failed'
  is_winner: boolean
  metrics: AlgorithmFrequencyMetrics | AlgorithmSeverityMetrics
  training_time_sec: number | null
  inference_time_ms: number | null
  model_size_mb: number | null
  artifact_path: string | null
  error: string | null
}

export interface ModelSelectionRun {
  run_id: string
  run_number: number
  version: string
  training_timestamp: string
  dataset: string
  dataset_hash: string | null
  n_total_rows: number | null
  n_train_rows: number | null
  n_val_rows: number | null
  n_test_rows: number | null
  n_sev_train_rows: number | null
  n_sev_val_rows: number | null
  n_sev_test_rows: number | null
  frequency_results: AlgorithmResult[]
  severity_results: AlgorithmResult[]
  frequency_winner: 'xgboost' | 'lightgbm' | 'catboost'
  severity_winner: 'xgboost' | 'lightgbm' | 'catboost'
  promotion_reason_frequency: string
  promotion_reason_severity: string
  frequency_production_path: string | null
  severity_production_path: string | null
  shap_generated_frequency: boolean
  shap_generated_severity: boolean
  shap_path_frequency: string | null
  shap_path_severity: string | null
  total_elapsed_sec: number | null
}

export interface ModelSelectionLatestResponse {
  has_runs: boolean
  message?: string
  total_runs: number
  latest_run: ModelSelectionRun | null
}

export interface ModelSelectionHistoryResponse {
  total: number
  page: number
  page_size: number
  total_pages: number
  runs: ModelSelectionRun[]
  latest_run_id: string | null
}

export interface ModelSelectionComparisonResponse {
  has_runs: boolean
  message?: string
  run_id?: string
  run_number?: number
  training_timestamp?: string
  dataset?: string
  frequency_comparison: AlgorithmResult[]
  severity_comparison: AlgorithmResult[]
  frequency_winner: string | null
  severity_winner: string | null
  promotion_reason_frequency?: string | null
  promotion_reason_severity?: string | null
}

export interface ModelSelectionRetrainStatus {
  running: boolean
  last_run_id: string | null
  error: string | null
}
