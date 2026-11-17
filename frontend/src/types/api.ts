/** API response shapes verified against the current FastAPI/Pydantic schemas. */
export interface UserRead {
  id: string;
  email: string;
  is_active: boolean;
  push_subscription: Record<string, unknown> | null;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  user: UserRead;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
}

export interface RegisterResponse {
  message: string;
}

/** Compact item returned by GET /api/sessions and included in SessionRead. */
export interface SessionListItem {
  id: string;
  name: string;
  status: string;
  target_column: string | null;
  task_type: string | null;
  created_at: string;
}

/** Fields needed from GET /api/sessions/{session_id}; source_path is deliberately omitted. */
export interface SessionDetails extends SessionListItem {
  source_type: string;
  updated_at: string;
  dataset_meta: Record<string, unknown>;
}

/** A recent list item enriched when its full session metadata was fetched. */
export interface DashboardSession extends SessionListItem {
  updated_at?: string;
  dataset_meta?: Record<string, unknown>;
}

export interface AnalysisOptions {
  target_column?: string;
  task_type?: "classification" | "regression";
  feature_selection?: FeatureSelectionConfig;
}

export interface FeatureSelectionConfig {
  method:
    | "variance_filter"
    | "correlation_filter"
    | "mutual_information"
    | "univariate_test"
    | "extra_trees";
  threshold?: number;
  top_k?: number;
  min_features?: number;
  n_estimators?: number;
}

export interface ChatRequest {
  message: string;
  mode?: "simple" | "advanced";
  analysis_options?: AnalysisOptions;
}

export interface ChatAcceptedResponse {
  run_id: string;
  status: string;
  status_url: string;
}

/** Persisted conversation item returned by GET /api/chat/{session_id}/messages. */
export interface AgentMessageRead {
  id: string;
  role: string;
  content: string;
  tool_calls?: Record<string, unknown> | null;
  tool_results?: Record<string, unknown> | null;
  created_at: string;
}

/** Current run snapshot returned by GET /api/runs/{run_id}. */
export interface AnalysisRunStatus {
  id: string;
  session_id: string;
  status: string;
  current_step: string | null;
  progress_percent: number;
  message: string | null;
  error_message: string | null;
  error_code?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

/** Model registry response returned by GET /api/predict/models/{session_id}. */
export interface RegisteredModelVersion {
  id: string;
  model_name: string;
  version: number;
  metrics: Record<string, unknown>;
  status: string;
}

/** Safe fields used from GET /api/artifacts/{session_id}; file_path is intentionally omitted. */
export interface SessionArtifact {
  id: string;
  session_id: string;
  artifact_type: string;
}

/** Persisted event shape returned by the run history endpoint and embedded in SSE data. */
export interface RunEventRead {
  id: string;
  run_id: string;
  sequence: number;
  event_type: string;
  step: string | null;
  progress_percent: number;
  message: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

/** Cursor page returned by GET /api/runs/{run_id}/events/history. */
export interface RunEventHistoryPage {
  items: RunEventRead[];
  next_cursor: string | null;
  has_more: boolean;
}

/** Response returned by POST /api/runs/{run_id}/cancel. */
export interface RunCancellationResponse {
  run_id: string;
  status: string;
  cancellation_requested: boolean;
  message: string;
}

export interface PredictRequest {
  records: Array<Record<string, unknown>>;
}

export interface PredictResponse {
  predictions: Array<string | number>;
  probabilities: number[][] | null;
  model_version_id: string;
  model_name: string;
}

export interface ApiErrorBody {
  detail?: string | Record<string, unknown> | unknown[];
}
