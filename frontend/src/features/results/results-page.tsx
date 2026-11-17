"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, Download, FileBarChart, FileCode2, FileSpreadsheet } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert, EmptyState, Skeleton } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { SelectField, TextInput } from "@/components/ui/form-controls";
import { Badge, Card } from "@/components/ui/surfaces";
import { Modal } from "@/components/ui/modal";
import { useAuth } from "@/features/auth/auth-provider";
import { getAgentRun, getAgentSession, listAgentSessions } from "@/features/agent-workspace/agent-api";
import {
  downloadSessionArtifact,
  listSessionArtifacts,
  listSessionModelVersions,
} from "@/features/results/results-api";
import type { DownloadableArtifactType } from "@/features/results/results-api";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import type {
  AnalysisRunStatus,
  RegisteredModelVersion,
  SessionArtifact,
  SessionDetails,
  SessionListItem,
} from "@/types/api";

const SESSION_PAGE_SIZE = 50;
const RESULT_IMAGE = "/images/automl-agent-illustrations/04-results-analytics.webp";
const EMPTY_MODELS: RegisteredModelVersion[] = [];
const EMPTY_METRICS: Record<string, unknown> = {};
const TERMINAL_STATUSES = new Set(["succeeded", "partially_completed", "failed", "cancelled"]);
const RESULT_READY_STATUSES = new Set(["succeeded", "partially_completed"]);
const METRIC_DIRECTIONS: Record<string, "higher" | "lower"> = {
  Accuracy: "higher",
  F1_Weighted: "higher",
  F1_Macro: "higher",
  R2: "higher",
  MAE: "lower",
  MSE: "lower",
  RMSE: "lower",
};

interface ResultData {
  key: string;
  session: SessionDetails | null;
  sessionError: string;
  run: AnalysisRunStatus | null;
  runError: string;
  models: RegisteredModelVersion[];
  modelsError: string;
  artifacts: SessionArtifact[];
  artifactsError: string;
}

interface ComparisonRow {
  model: string;
  values: Record<string, number>;
  originalIndex: number;
}

interface EvaluationData {
  available: boolean;
  method: string | null;
  testRows: number | null;
  metrics: Record<string, number>;
}

const DOWNLOADABLE_ARTIFACTS: Record<string, {
  label: string;
  fileName: string;
  downloadType: DownloadableArtifactType;
  icon: typeof FileBarChart;
}> = {
  report: { label: "PDF analysis report", fileName: "analysis-report.pdf", downloadType: "report", icon: FileBarChart },
  notebook: { label: "Jupyter notebook", fileName: "analysis_pipeline.ipynb", downloadType: "notebook", icon: FileCode2 },
  cleaned_data: { label: "Cleaned dataset", fileName: "cleaned_data.csv", downloadType: "cleaned_data", icon: FileSpreadsheet },
};

function selectionKey(sessionId: string | null, runId: string | null, generation: number): string {
  return JSON.stringify([sessionId ?? "", runId ?? "", generation]);
}

function agentWorkspaceHref(sessionId: string, runId?: string | null): string {
  const query = new URLSearchParams({ session_id: sessionId });
  if (runId) query.set("run_id", runId);
  return `/workspace/agent?${query.toString()}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function recordOf(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function finiteNumbers(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter((entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
    ),
  );
}

function readComparisonRows(metrics: Record<string, unknown>): ComparisonRow[] {
  if (!Array.isArray(metrics.results)) return [];
  return metrics.results.flatMap((item, originalIndex) => {
    if (!isRecord(item) || typeof item.model !== "string") return [];
    return [{ model: item.model, values: finiteNumbers(item), originalIndex }];
  });
}

function readEvaluation(metrics: Record<string, unknown>): EvaluationData {
  const evaluation = recordOf(metrics.evaluation);
  const values = finiteNumbers(evaluation.metrics);
  const fallback = finiteNumbers(metrics.final_test_metrics);
  const resultMetrics = Object.keys(values).length > 0 ? values : fallback;
  return {
    available: evaluation.available === true && Object.keys(resultMetrics).length > 0,
    method: typeof evaluation.method === "string" ? evaluation.method : null,
    testRows: typeof evaluation.test_rows === "number" && Number.isSafeInteger(evaluation.test_rows) && evaluation.test_rows >= 0
      ? evaluation.test_rows
      : null,
    metrics: resultMetrics,
  };
}

function safeErrorMessage(error: unknown, resource: "session" | "run" | "models" | "artifacts"): string {
  if (error instanceof ApiTimeoutError) return "The request took too long. Check your connection and try again.";
  if (error instanceof ApiError) {
    if (error.status === 404) return resource === "run"
      ? "This run is unavailable to your account."
      : resource === "session"
        ? "This session is unavailable to your account."
        : resource === "artifacts"
          ? "Artifact information is unavailable for this session."
          : "Registered model results are unavailable for this session.";
    if (error.status >= 500) return "The service is temporarily unavailable. Please try again shortly.";
  }
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return resource === "run"
    ? "The selected run could not be verified."
    : resource === "session"
      ? "Session details could not be loaded."
      : resource === "artifacts"
        ? "Available artifacts could not be loaded."
        : "Model results could not be loaded.";
}

function isAuthorizationError(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

function dateLabel(value: string | null | undefined): string {
  if (!value) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function metricLabel(metric: string): string {
  const names: Record<string, string> = {
    F1_Weighted: "F1 weighted",
    F1_Macro: "F1 macro",
    R2: "R²",
  };
  return names[metric] ?? metric.replaceAll("_", " ");
}

function metricValueLabel(metric: string, value: number): string {
  if (["Accuracy", "F1_Weighted", "F1_Macro"].includes(metric) && value >= 0 && value <= 1) {
    return `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value);
}

function statusTone(status: string): "neutral" | "blue" | "green" | "amber" | "red" {
  if (["succeeded", "trained", "reported", "production"].includes(status)) return "green";
  if (["queued", "running", "candidate"].includes(status)) return "blue";
  if (["partially_completed", "awaiting_input", "awaiting_approval", "cancelling"].includes(status)) return "amber";
  if (["failed", "cancelled", "archived"].includes(status)) return status === "failed" ? "red" : "neutral";
  return "neutral";
}

function commonComparisonMetrics(rows: ComparisonRow[]): string[] {
  if (rows.length === 0) return [];
  return Object.keys(rows[0].values).filter((key) =>
    key !== "model"
    && key in METRIC_DIRECTIONS
    && rows.every((row) => typeof row.values[key] === "number" && Number.isFinite(row.values[key])),
  );
}

function sortComparisonRows(rows: ComparisonRow[], metric: string): ComparisonRow[] {
  const direction = METRIC_DIRECTIONS[metric];
  if (!direction) return rows;
  return [...rows].sort((left, right) => {
    const delta = left.values[metric] - right.values[metric];
    return (direction === "higher" ? -delta : delta) || left.originalIndex - right.originalIndex;
  });
}

function latestArtifacts(artifacts: SessionArtifact[]): SessionArtifact[] {
  const byType = new Map<string, SessionArtifact>();
  for (const artifact of artifacts) byType.set(artifact.artifact_type, artifact);
  return [...byType.values()];
}

function SessionSkeleton() {
  return <div className="results-loading-card"><Skeleton /><Skeleton /><Skeleton /></div>;
}

function RunStatusCard({ run, sessionId, runError }: {
  run: AnalysisRunStatus | null;
  sessionId: string;
  runError: string;
}) {
  if (runError) {
    return <Alert tone="warning" title="Run status unavailable">
      {runError} Session results remain available separately when returned by the model registry.
    </Alert>;
  }
  if (!run) return null;
  const isReady = RESULT_READY_STATUSES.has(run.status);
  const tone = run.status === "failed" ? "error" : isReady ? (run.status === "partially_completed" ? "warning" : "success") : "info";
  const message = run.message || (isReady
    ? "The backend marked this run as complete."
    : TERMINAL_STATUSES.has(run.status)
      ? "This run did not complete with final results."
      : "Final results are not available until this run reaches a terminal state.");

  return (
    <Card className="results-run-card">
      <div className="results-run-heading">
        <div><span className="section-overline">SELECTED RUN</span><h2>Run lifecycle</h2></div>
        <Badge tone={statusTone(run.status)}>{run.status}</Badge>
      </div>
      <dl className="results-run-meta">
        <div><dt>Run ID</dt><dd><code>{run.id}</code></dd></div>
        <div><dt>Created</dt><dd>{dateLabel(run.created_at)}</dd></div>
        {run.finished_at && <div><dt>Finished</dt><dd>{dateLabel(run.finished_at)}</dd></div>}
        {run.current_step && <div><dt>Current step</dt><dd>{run.current_step}</dd></div>}
      </dl>
      <Alert tone={tone} title={`Run status: ${run.status}`}>{message}</Alert>
      {!isReady && (
        <Link className="button button-outline results-return-link" href={agentWorkspaceHref(sessionId, run?.id)}>
          Return to AI Workspace <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
      )}
    </Card>
  );
}

function ModelDetails({ model, session }: { model: RegisteredModelVersion; session: SessionDetails }) {
  const metrics = model.metrics ?? {};
  const evaluation = readEvaluation(metrics);
  const rows = readComparisonRows(metrics);
  const selectionMetric = typeof metrics.selection_metric === "string" ? metrics.selection_metric : null;
  const score = typeof metrics.cv_selection_score === "number" && Number.isFinite(metrics.cv_selection_score)
    ? metrics.cv_selection_score
    : typeof metrics.best_score === "number" && Number.isFinite(metrics.best_score)
      ? metrics.best_score
      : null;

  return (
    <div className="results-model-detail">
      <dl className="results-detail-list">
        <div><dt>Model name</dt><dd>{model.model_name}</dd></div>
        <div><dt>Model version</dt><dd>{model.version}</dd></div>
        <div><dt>Model ID</dt><dd><code>{model.id}</code></dd></div>
        <div><dt>Registry status</dt><dd><Badge tone={statusTone(model.status)}>{model.status}</Badge></dd></div>
        {session.task_type && <div><dt>Task type</dt><dd>{session.task_type}</dd></div>}
        {session.target_column && <div><dt>Target column</dt><dd>{session.target_column}</dd></div>}
        {selectionMetric && <div><dt>Selection metric</dt><dd>{metricLabel(selectionMetric)}</dd></div>}
        {score !== null && selectionMetric && <div><dt>Cross-validation score</dt><dd>{metricValueLabel(selectionMetric, score)}</dd></div>}
        {evaluation.available && evaluation.testRows !== null && <div><dt>Holdout rows</dt><dd>{evaluation.testRows.toLocaleString()}</dd></div>}
        <div><dt>Compared candidates</dt><dd>{rows.length > 0 ? rows.length.toLocaleString() : "Not returned"}</dd></div>
      </dl>
      {evaluation.available && Object.keys(evaluation.metrics).length > 0 && (
        <div className="results-detail-metrics">
          <h3>Backend-reported holdout metrics</h3>
          <ul>{Object.entries(evaluation.metrics).map(([name, value]) => <li key={name}><span>{metricLabel(name)}</span><strong>{metricValueLabel(name, value)}</strong></li>)}</ul>
        </div>
      )}
      <p className="results-detail-note">The model registry response does not include hyperparameters, model file paths, or a run identifier.</p>
    </div>
  );
}

function EvaluationSection({ taskType, evaluation }: { taskType: string | null; evaluation: EvaluationData }) {
  if (!evaluation.available) {
    return (
      <Card className="results-section-card">
        <div className="results-section-heading"><div><span className="section-overline">GENERALIZATION</span><h2>Independent holdout evaluation</h2></div></div>
        <p className="results-unavailable-copy">No structured holdout metrics were returned for this model version.</p>
      </Card>
    );
  }

  const classification = taskType === "classification";
  const regression = taskType === "regression";
  return (
    <Card className="results-section-card">
      <div className="results-section-heading">
        <div><span className="section-overline">GENERALIZATION</span><h2>Independent holdout evaluation</h2></div>
        <span className="results-section-caption">
          {evaluation.testRows !== null ? `${evaluation.testRows.toLocaleString()} test rows` : "Holdout metrics"}
          {evaluation.method ? ` · ${evaluation.method.replaceAll("_", " ")}` : ""}
        </span>
      </div>
      <ul className="results-metric-grid">
        {Object.entries(evaluation.metrics).map(([name, value]) => {
          const showClassificationBar = classification
            && ["Accuracy", "F1_Weighted", "F1_Macro"].includes(name)
            && value >= 0 && value <= 1;
          return (
            <li className="results-metric-card" key={name}>
              <span>{metricLabel(name)}</span>
              <strong>{metricValueLabel(name, value)}</strong>
              {showClassificationBar && <progress max={1} value={value} aria-label={`${metricLabel(name)} ${metricValueLabel(name, value)}`} />}
              {regression && <small>{["MAE", "MSE", "RMSE"].includes(name) ? "Lower is better" : name === "R2" ? "Higher is better" : "Backend-reported metric"}</small>}
            </li>
          );
        })}
      </ul>
      {classification && <p className="results-evaluation-note">The backend returns aggregate classification metrics here. Confusion-matrix values and per-class reports are not included in this API response.</p>}
      {regression && <p className="results-evaluation-note">The backend returns aggregate regression metrics here. Actual-versus-predicted samples and residual values are not included in this API response.</p>}
      {!classification && !regression && <p className="results-evaluation-note">Task-specific charts are hidden because the session does not identify a classification or regression task.</p>}
    </Card>
  );
}

function FeatureSelectionSection({ metrics }: { metrics: Record<string, unknown> }) {
  const selection = recordOf(metrics.feature_selection);
  const enabled = selection.enabled === true;
  const selected = Array.isArray(selection.selected_features)
    ? selection.selected_features.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
  const excluded = Array.isArray(selection.excluded_features)
    ? selection.excluded_features.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
  const folds = Array.isArray(selection.cross_validation_folds)
    ? selection.cross_validation_folds.filter((item): item is Record<string, unknown> => isRecord(item))
    : [];
  const configuration = recordOf(selection.configuration);
  const foldSets = folds.map((fold) => typeof fold.selected_feature_signature === "string"
    ? fold.selected_feature_signature
    : Array.isArray(fold.selected_feature_names)
      ? fold.selected_feature_names.filter((name): name is string => typeof name === "string").sort().join("\u0000")
      : "");
  const foldVariability = new Set(foldSets).size > 1;

  return (
    <Card className="results-section-card">
      <div className="results-section-heading"><div><span className="section-overline">TRAINING FEATURE SELECTION</span><h2>Selected features</h2></div></div>
      {!enabled ? (
        <p className="results-unavailable-copy">{selection.enabled === false ? "Feature selection was disabled for this run." : "No feature-selection result was recorded for this model version."}</p>
      ) : (
        <>
          <p className="results-selection-summary">
            <strong>{typeof selection.method === "string" ? selection.method.replaceAll("_", " ") : "Configured method"}</strong>
            {typeof selection.original_source_features === "number" && ` · ${selection.original_source_features} original source features`}
            {typeof selection.input_features === "number" && ` · ${selection.input_features} transformed inputs`}
            {typeof selection.retained_features === "number" && ` · ${selection.retained_features} retained`}
            {typeof selection.fit_rows === "number" && ` · fitted on ${selection.fit_rows.toLocaleString()} training rows`}
          </p>
          <p className="results-selection-note">Selector scores and retained features describe training-only fitting. Holdout metrics are evaluated separately; permutation importance is a separate post-training analysis.</p>
          <p className="results-selection-note">Scores are specific to the selected method and should not be compared across different selector methods.</p>
          {Object.keys(configuration).length > 0 && (
            <dl className="results-selection-config">{Object.entries(configuration).filter(([, value]) => value !== null).map(([key, value]) => (
              <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{typeof value === "number" || typeof value === "string" ? String(value) : "Recorded"}</dd></div>
            ))}</dl>
          )}
          {folds.length > 0 && <p className="results-selection-note">Cross-validation recorded {folds.length} fold-specific selector fit{folds.length === 1 ? "" : "s"}{foldVariability ? "; retained feature sets differed between folds." : "; retained feature sets were consistent across folds."}</p>}
          {selected.length > 0 ? (
            <div className="results-table-scroll"><table className="results-table">
              <thead><tr><th scope="col">Transformed feature</th><th scope="col">Source feature</th><th scope="col">Training selector score</th><th scope="col">Rank</th></tr></thead>
              <tbody>{selected.map((item, index) => <tr key={`${String(item.name ?? "feature")}-${index}`}>
                <th scope="row">{typeof item.name === "string" ? item.name : "Unnamed feature"}</th>
                <td>{typeof item.source_feature === "string" ? item.source_feature : "Unavailable"}</td>
                <td>{typeof item.score === "number" && Number.isFinite(item.score) ? item.score.toPrecision(5) : "—"}</td>
                <td>{typeof item.rank === "number" && Number.isSafeInteger(item.rank) ? item.rank : "—"}</td>
              </tr>)}</tbody>
            </table></div>
          ) : <p className="results-unavailable-copy">The selector ran, but selected feature details were not returned.</p>}
          {typeof selection.selected_feature_names_truncated === "number" && selection.selected_feature_names_truncated > 0 && (
            <p className="results-selection-note">Showing the first 300 selected transformed features; {selection.selected_feature_names_truncated.toLocaleString()} more names were omitted from the metadata for size.</p>
          )}
          {excluded.length > 0 && (
            <details className="results-selection-excluded">
              <summary>Excluded features ({excluded.length + (typeof selection.excluded_features_truncated === "number" ? selection.excluded_features_truncated : 0)})</summary>
              <ul>{excluded.slice(0, 30).map((item, index) => <li key={`${String(item.name ?? "feature")}-${index}`}><strong>{typeof item.name === "string" ? item.name : "Unnamed feature"}</strong><span>{typeof item.reason === "string" ? item.reason : "Excluded by fitted preprocessing or selector."}</span></li>)}</ul>
              {(excluded.length > 30 || (typeof selection.excluded_features_truncated === "number" && selection.excluded_features_truncated > 0)) && <p className="results-unavailable-copy">Showing up to 30 exclusions from the recorded details.</p>}
            </details>
          )}
        </>
      )}
    </Card>
  );
}

export function ResultsPage({
  initialSessionId,
  initialRunId,
  selectionError,
}: {
  initialSessionId: string | null;
  initialRunId: string | null;
  selectionError: string;
}) {
  const { accessToken, logout } = useAuth();
  const router = useRouter();
  const [selectedSessionId, setSelectedSessionId] = useState(initialSessionId);
  const [selectedRunId, setSelectedRunId] = useState(initialRunId);
  const [selectionErrorMessage, setSelectionErrorMessage] = useState(selectionError);
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsLoadingMore, setSessionsLoadingMore] = useState(false);
  const [sessionsHasMore, setSessionsHasMore] = useState(false);
  const [sessionsOffset, setSessionsOffset] = useState(0);
  const [sessionsError, setSessionsError] = useState("");
  const [sessionRequest, setSessionRequest] = useState({ offset: 0, generation: 0 });
  const sessionRequestGeneration = useRef(0);
  const [resultData, setResultData] = useState<ResultData | null>(null);
  const [reloadGeneration, setReloadGeneration] = useState(0);
  const [modelSearch, setModelSearch] = useState("");
  const [modelStatusFilter, setModelStatusFilter] = useState("");
  const [manualSortMetric, setManualSortMetric] = useState<string | null>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [detailModelId, setDetailModelId] = useState<string | null>(null);
  const [downloadingArtifact, setDownloadingArtifact] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [downloadNotice, setDownloadNotice] = useState("");

  const replaceResultsUrl = useCallback((sessionId: string | null, runId: string | null) => {
    const query = new URLSearchParams();
    if (sessionId) query.set("session_id", sessionId);
    if (runId) query.set("run_id", runId);
    const suffix = query.toString();
    router.replace(`/workspace/results${suffix ? `?${suffix}` : ""}`, { scroll: false });
  }, [router]);

  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    const { offset } = sessionRequest;
    listAgentSessions(accessToken, SESSION_PAGE_SIZE, offset, controller.signal)
      .then((page) => {
        if (controller.signal.aborted) return;
        setSessions((current) => {
          if (offset === 0) return page;
          const ids = new Set(current.map((item) => item.id));
          return [...current, ...page.filter((item) => !ids.has(item.id))];
        });
        setSessionsOffset(offset + page.length);
        setSessionsHasMore(page.length === SESSION_PAGE_SIZE);
        setSessionsError("");
        setSessionsLoading(false);
        setSessionsLoadingMore(false);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (isAuthorizationError(error)) {
          logout();
          return;
        }
        setSessionsError(safeErrorMessage(error, "session"));
        setSessionsLoading(false);
        setSessionsLoadingMore(false);
      });
    return () => controller.abort();
  }, [accessToken, logout, sessionRequest]);

  const currentRequestKey = selectionKey(selectedSessionId, selectedRunId, reloadGeneration);
  useEffect(() => {
    if (!accessToken || selectionErrorMessage || (!selectedSessionId && !selectedRunId)) return;
    if (resultData?.key === currentRequestKey) return;

    const token = accessToken;
    const controller = new AbortController();
    let active = true;
    const requestedSessionId = selectedSessionId;
    const requestedRunId = selectedRunId;

    async function loadResults() {
      let run: AnalysisRunStatus | null = null;
      let runError = "";
      let resolvedSessionId = requestedSessionId;

      if (requestedRunId) {
        try {
          run = await getAgentRun(token, requestedRunId!, controller.signal);
          if (!resolvedSessionId) resolvedSessionId = run.session_id;
          else if (run.session_id !== resolvedSessionId) {
            runError = "The selected run does not belong to the selected session.";
          }
        } catch (error) {
          if (controller.signal.aborted) return;
          if (isAuthorizationError(error)) {
            logout();
            return;
          }
          runError = safeErrorMessage(error, "run");
        }
      }

      if (!resolvedSessionId || runError === "The selected run does not belong to the selected session.") {
        if (!active) return;
        setResultData({
          key: selectionKey(resolvedSessionId, requestedRunId, reloadGeneration),
          session: null,
          sessionError: "",
          run,
          runError,
          models: [],
          modelsError: "",
          artifacts: [],
          artifactsError: "",
        });
        return;
      }

      const sessionId = resolvedSessionId!;
      let session: SessionDetails;
      try {
        session = await getAgentSession(token, sessionId, controller.signal);
      } catch (error) {
        if (controller.signal.aborted) return;
        if (isAuthorizationError(error)) {
          logout();
          return;
        }
        if (!active) return;
        setResultData({
          key: selectionKey(sessionId, requestedRunId, reloadGeneration),
          session: null,
          sessionError: safeErrorMessage(error, "session"),
          run,
          runError,
          models: [],
          modelsError: "",
          artifacts: [],
          artifactsError: "",
        });
        return;
      }

      const [modelsResult, artifactsResult] = await Promise.allSettled([
        listSessionModelVersions(token, sessionId, controller.signal),
        listSessionArtifacts(token, sessionId, controller.signal),
      ]);
      if (!active || controller.signal.aborted) return;

      const authorizationFailure = [modelsResult, artifactsResult].find((result) =>
        result.status === "rejected" && isAuthorizationError(result.reason),
      );
      if (authorizationFailure?.status === "rejected") {
        logout();
        return;
      }

      const models = modelsResult.status === "fulfilled" && Array.isArray(modelsResult.value) ? modelsResult.value : [];
      const artifacts = artifactsResult.status === "fulfilled" && Array.isArray(artifactsResult.value) ? artifactsResult.value : [];
      const finalKey = selectionKey(sessionId, requestedRunId, reloadGeneration);
      setResultData({
        key: finalKey,
        session,
        sessionError: "",
        run,
        runError,
        models,
        modelsError: modelsResult.status === "rejected" ? safeErrorMessage(modelsResult.reason, "models") : "",
        artifacts,
        artifactsError: artifactsResult.status === "rejected" ? safeErrorMessage(artifactsResult.reason, "artifacts") : "",
      });
      if (!requestedSessionId) {
        setSelectedSessionId(sessionId);
        replaceResultsUrl(sessionId, requestedRunId);
      }
    }

    void loadResults();
    return () => {
      active = false;
      controller.abort();
    };
  }, [accessToken, currentRequestKey, logout, reloadGeneration, replaceResultsUrl, resultData?.key, selectedRunId, selectedSessionId, selectionErrorMessage]);

  const visibleData = resultData?.key === currentRequestKey ? resultData : null;
  const selectedSession = visibleData?.session ?? null;
  const models = visibleData?.models ?? EMPTY_MODELS;
  const selectedModel = models.find((model) => model.id === activeModelId) ?? models[0] ?? null;
  const modelMetrics = selectedModel?.metrics ?? EMPTY_METRICS;
  const comparisonRows = useMemo(() => readComparisonRows(modelMetrics), [modelMetrics]);
  const commonMetrics = useMemo(() => commonComparisonMetrics(comparisonRows), [comparisonRows]);
  const primaryMetric = typeof modelMetrics.selection_metric === "string"
    ? modelMetrics.selection_metric
    : typeof modelMetrics.metric === "string" ? modelMetrics.metric : "";
  const defaultSortMetric = commonMetrics.includes(primaryMetric) ? primaryMetric : "";
  const activeSortMetric = manualSortMetric ?? defaultSortMetric;
  const filteredComparison = useMemo(() => {
    const query = modelSearch.trim().toLocaleLowerCase();
    const filtered = comparisonRows.filter((row) => !query || row.model.toLocaleLowerCase().includes(query));
    return sortComparisonRows(filtered, activeSortMetric);
  }, [activeSortMetric, comparisonRows, modelSearch]);
  const modelStatuses = useMemo(() => [...new Set(models.map((model) => model.status))].sort(), [models]);
  const filteredModels = useMemo(() => {
    const query = modelSearch.trim().toLocaleLowerCase();
    return models.filter((model) =>
      (!query || model.model_name.toLocaleLowerCase().includes(query) || model.id.toLocaleLowerCase().includes(query))
      && (!modelStatusFilter || model.status === modelStatusFilter),
    );
  }, [modelSearch, modelStatusFilter, models]);
  const selectedEvaluation = useMemo(() => readEvaluation(modelMetrics), [modelMetrics]);
  const artifacts = useMemo(() => latestArtifacts(visibleData?.artifacts ?? []), [visibleData?.artifacts]);
  const detailModel = models.find((model) => model.id === detailModelId) ?? null;
  const isLoadingResults = Boolean(
    accessToken && !selectionErrorMessage && (selectedSessionId || selectedRunId) && !visibleData,
  );
  const canShowFinalResults = !selectedRunId
    || Boolean(visibleData?.run && !visibleData.runError && RESULT_READY_STATUSES.has(visibleData.run.status));

  function selectSession(value: string) {
    setSelectionErrorMessage("");
    setSelectedSessionId(value || null);
    setSelectedRunId(null);
    setActiveModelId(null);
    setDetailModelId(null);
    setModelSearch("");
    setModelStatusFilter("");
    setManualSortMetric(null);
    setResultData(null);
    setDownloadError("");
    setDownloadNotice("");
    replaceResultsUrl(value || null, null);
  }

  function clearRunSelection() {
    setSelectedRunId(null);
    setResultData(null);
    replaceResultsUrl(selectedSessionId, null);
  }

  function loadMoreSessions() {
    if (!sessionsHasMore || sessionsLoading || sessionsLoadingMore) return;
    setSessionsLoadingMore(true);
    setSessionRequest({ offset: sessionsOffset, generation: ++sessionRequestGeneration.current });
  }

  function retrySessions() {
    setSessionsError("");
    setSessionsLoading(true);
    setSessionsLoadingMore(false);
    setSessions([]);
    setSessionsOffset(0);
    setSessionsHasMore(false);
    setSessionRequest({ offset: 0, generation: ++sessionRequestGeneration.current });
  }

  function retryResults() {
    setResultData(null);
    setReloadGeneration((current) => current + 1);
  }

  async function handleDownload(artifact: SessionArtifact) {
    if (!accessToken || !selectedSession) return;
    const config = DOWNLOADABLE_ARTIFACTS[artifact.artifact_type];
    if (!config) return;
    setDownloadingArtifact(artifact.id);
    setDownloadError("");
    setDownloadNotice("");
    try {
      const blob = await downloadSessionArtifact(accessToken, selectedSession.id, config.downloadType);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = config.fileName;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
      setDownloadNotice(`${config.label} download started.`);
    } catch (error) {
      if (isAuthorizationError(error)) {
        logout();
        return;
      }
      setDownloadError(error instanceof ApiError && error.status === 404
        ? "This artifact file is no longer available."
        : safeErrorMessage(error, "artifacts"));
    } finally {
      setDownloadingArtifact("");
    }
  }

  const selectorSessions = selectedSession && !sessions.some((item) => item.id === selectedSession.id)
    ? [selectedSession, ...sessions]
    : sessions;
  const artifactsForDisplay = artifacts;
  const runIsPending = Boolean(visibleData?.run && !TERMINAL_STATUSES.has(visibleData.run.status));
  const runDidNotSucceed = Boolean(visibleData?.run && ["failed", "cancelled"].includes(visibleData.run.status));
  const reportsHref = visibleData?.session
    ? `/workspace/reports?session_id=${encodeURIComponent(visibleData.session.id)}${visibleData.run?.id ? `&run_id=${encodeURIComponent(visibleData.run.id)}` : ""}`
    : undefined;

  return (
    <AppShell
      title="Results & Model Insights"
      eyebrow="MODEL EVALUATION"
      description="Review model comparisons, holdout metrics, and available analysis files for your dataset sessions."
      breadcrumbs={[{ label: "Workspace", href: "/workspace" }, { label: "Results" }]}
      activeItem="Results"
      homeHref="/workspace"
      statusLabel="Private workspace"
      actions={reportsHref ? (
        <Link className="button button-outline" href={reportsHref}>
          View reports <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
      ) : undefined}
    >
      <div className="results-page">
        {selectionErrorMessage && <Alert tone="error" title="Invalid results link">{selectionErrorMessage}</Alert>}
        {sessionsError && <div className="results-error-row"><Alert tone="error" title="Sessions unavailable">{sessionsError}</Alert><Button variant="outline" onClick={retrySessions}>Retry</Button></div>}

        <Card className="results-selector-card">
          <div className="results-selector-copy"><span className="section-overline">ANALYSIS CONTEXT</span><h2>Select a dataset session</h2><p>Results are loaded only after the backend confirms access to the selected session.</p></div>
          <div className="results-session-select">
            <SelectField
              label="Dataset session"
              value={selectedSessionId ?? ""}
              disabled={sessionsLoading && selectorSessions.length === 0}
              options={[
                { label: sessionsLoading && selectorSessions.length === 0 ? "Loading sessions…" : "Choose a session", value: "" },
                ...selectorSessions.map((session) => ({ label: `${session.name} · ${session.status}`, value: session.id })),
              ]}
              onChange={(event) => selectSession(event.currentTarget.value)}
            />
            {sessionsHasMore && (
              <Button type="button" variant="ghost" disabled={sessionsLoadingMore} loading={sessionsLoadingMore} onClick={loadMoreSessions}>
                Load more sessions
              </Button>
            )}
            {!sessionsHasMore && sessions.length > 0 && <span className="results-session-count">{sessions.length.toLocaleString()} sessions loaded</span>}
          </div>
        </Card>

        {selectedRunId && (
          <div className="results-selected-run-line">
            <span>Run context · <code>{selectedRunId}</code></span>
            <Button type="button" variant="ghost" onClick={clearRunSelection}>Clear run</Button>
          </div>
        )}

        {visibleData?.sessionError && (
          <div className="results-error-row">
            <Alert tone="error" title="Session unavailable">{visibleData.sessionError}</Alert>
            <Button variant="outline" onClick={retryResults}>Retry</Button>
          </div>
        )}
        {visibleData?.runError && <Alert tone="warning" title="Run could not be verified">{visibleData.runError}</Alert>}

        {isLoadingResults ? (
          <div className="results-loading-grid" aria-label="Loading results" role="status"><SessionSkeleton /><SessionSkeleton /><SessionSkeleton /></div>
        ) : visibleData?.session && (
          <>
            {visibleData.run && <RunStatusCard run={visibleData.run} sessionId={visibleData.session.id} runError={visibleData.runError} />}
            {selectedRunId && visibleData.run && !visibleData.runError && (
              <Alert tone="info" title="Session-scoped model registry">
                The model registry endpoint returns versions for the session and does not include their run IDs. These records are not presented as belonging to the selected run.
              </Alert>
            )}

            {runIsPending && (
              <Card className="results-no-final-card">
                <div className="results-empty-state">
                  <h2>Final results are not ready</h2>
                  <p>The selected run is still in progress. Check its live timeline in the AI Workspace.</p>
                  <Link className="button button-outline" href={agentWorkspaceHref(visibleData.session.id, selectedRunId)}>
                    Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" />
                  </Link>
                </div>
              </Card>
            )}
            {runDidNotSucceed && (
              <Card className="results-no-final-card">
                <div className="results-empty-state">
                  <h2>This run has no final results</h2>
                  <p>Its exact backend status is shown above. Model versions returned for the session cannot be reliably associated with this run.</p>
                  <Link className="button button-outline" href={agentWorkspaceHref(visibleData.session.id, selectedRunId)}>
                    Return to AI Workspace <ArrowUpRight size={14} aria-hidden="true" />
                  </Link>
                </div>
              </Card>
            )}

            {canShowFinalResults && (
              <>
                <Card className="results-overview-card">
                  <div className="results-overview-heading">
                    <div><span className="section-overline">RESULTS OVERVIEW</span><h2>{visibleData.session.name}</h2></div>
                    <Badge tone={statusTone(visibleData.session.status)}>{visibleData.session.status}</Badge>
                  </div>
                  <dl className="results-overview-grid">
                    <div><dt>Session ID</dt><dd><code>{visibleData.session.id}</code></dd></div>
                    {visibleData.run && <div><dt>Run status</dt><dd><Badge tone={statusTone(visibleData.run.status)}>{visibleData.run.status}</Badge></dd></div>}
                    {visibleData.session.task_type && <div><dt>Task type</dt><dd>{visibleData.session.task_type}</dd></div>}
                    {visibleData.session.target_column && <div><dt>Target column</dt><dd>{visibleData.session.target_column}</dd></div>}
                    <div><dt>Session created</dt><dd>{dateLabel(visibleData.session.created_at)}</dd></div>
                    {selectedModel && <div><dt>Selected registry model</dt><dd>{selectedModel.model_name} · v{selectedModel.version}</dd></div>}
                    {selectedModel && primaryMetric && <div><dt>CV selection metric</dt><dd>{metricLabel(primaryMetric)}</dd></div>}
                    {selectedModel && primaryMetric && typeof modelMetrics.cv_selection_score === "number" && Number.isFinite(modelMetrics.cv_selection_score) && (
                      <div><dt>CV selection score</dt><dd>{metricValueLabel(primaryMetric, modelMetrics.cv_selection_score)}</dd></div>
                    )}
                    {selectedModel && comparisonRows.length > 0 && <div><dt>Evaluated candidates</dt><dd>{comparisonRows.length.toLocaleString()}</dd></div>}
                  </dl>
                  <div className="results-overview-actions">
                    <Link className="button button-ghost" href={agentWorkspaceHref(visibleData.session.id, selectedRunId)}>
                      Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" />
                    </Link>
                  </div>
                </Card>

                {visibleData.modelsError ? (
                  <div className="results-error-row"><Alert tone="error" title="Model results unavailable">{visibleData.modelsError}</Alert><Button variant="outline" onClick={retryResults}>Retry</Button></div>
                ) : visibleData.models.length === 0 ? (
                  <Card className="results-no-final-card results-empty-card">
                    <EmptyState
                      illustration={RESULT_IMAGE}
                      title="No registered model results"
                      description="This session has no model versions in the registry yet. Start or continue an analysis in the AI Workspace."
                      action={<Link className="button button-outline" href={agentWorkspaceHref(visibleData.session.id, selectedRunId)}>Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" /></Link>}
                    />
                  </Card>
                ) : (
                  <>
                    <Card className="results-section-card">
                      <div className="results-section-heading">
                        <div><span className="section-overline">MODEL COMPARISON</span><h2>Cross-validation candidates</h2></div>
                        <span className="results-section-caption">{comparisonRows.length.toLocaleString()} returned for the selected registry version</span>
                      </div>
                      <div className="results-control-grid">
                        <TextInput
                          label="Search models"
                          value={modelSearch}
                          onChange={(event) => setModelSearch(event.currentTarget.value)}
                          placeholder="Search model names"
                        />
                        {commonMetrics.length > 0 && (
                          <SelectField
                            label="Sort candidates by"
                            value={activeSortMetric}
                            options={[
                              { label: "Backend order (no ranking)", value: "" },
                              ...commonMetrics.map((metric) => ({
                                label: `${metricLabel(metric)} · ${METRIC_DIRECTIONS[metric]} is better`,
                                value: metric,
                              })),
                            ]}
                            onChange={(event) => setManualSortMetric(event.currentTarget.value)}
                          />
                        )}
                        {visibleData.models.length > 1 && (
                          <SelectField
                            label="Metrics record"
                            value={selectedModel?.id ?? ""}
                            options={visibleData.models.map((model) => ({ label: `${model.model_name} · v${model.version}`, value: model.id }))}
                            onChange={(event) => setActiveModelId(event.currentTarget.value)}
                          />
                        )}
                      </div>
                      {activeSortMetric
                        ? <p className="results-ranking-note">Sorted by {metricLabel(activeSortMetric)}; {METRIC_DIRECTIONS[activeSortMetric]} values rank higher. These are training-only cross-validation scores.</p>
                        : <p className="results-ranking-note">No shared supported ranking metric is available. Candidate rows retain the backend order; no winner is inferred.</p>}
                      {comparisonRows.length === 0 ? (
                        <p className="results-unavailable-copy">This registry record does not include candidate comparison rows.</p>
                      ) : filteredComparison.length === 0 ? (
                        <p className="results-unavailable-copy">No candidates match this search.</p>
                      ) : (
                        <div className="results-table-scroll">
                          <table className="results-table">
                            <thead><tr>
                              {activeSortMetric && <th scope="col">Rank</th>}
                              <th scope="col">Model</th>
                              {commonMetrics.map((metric) => <th scope="col" key={metric}>{metricLabel(metric)}</th>)}
                            </tr></thead>
                            <tbody>{filteredComparison.map((row, index) => (
                              <tr key={`${row.model}-${row.originalIndex}`}>
                                {activeSortMetric && <td>{index + 1}</td>}
                                <th scope="row">{row.model}</th>
                                {commonMetrics.map((metric) => <td key={metric}>{metricValueLabel(metric, row.values[metric])}</td>)}
                              </tr>
                            ))}</tbody>
                          </table>
                        </div>
                      )}
                    </Card>

                    <EvaluationSection taskType={visibleData.session.task_type} evaluation={selectedEvaluation} />

                    <Card className="results-section-card">
                      <div className="results-section-heading"><div><span className="section-overline">MODEL REGISTRY</span><h2>Registered versions</h2></div><span className="results-section-caption">{visibleData.models.length.toLocaleString()} returned by the session endpoint</span></div>
                      <p className="results-ranking-note">The backend returns versions newest-first but does not provide run IDs or pagination for this list.</p>
                      {modelStatuses.length > 1 && (
                        <div className="results-registry-toolbar">
                          <SelectField
                            label="Filter registered versions by status"
                            value={modelStatusFilter}
                            options={[{ label: "All statuses", value: "" }, ...modelStatuses.map((status) => ({ label: status, value: status }))]}
                            onChange={(event) => setModelStatusFilter(event.currentTarget.value)}
                          />
                        </div>
                      )}
                      {filteredModels.length > 0 ? (
                        <div className="results-table-scroll"><table className="results-table results-registry-table">
                          <thead><tr><th scope="col">Model</th><th scope="col">Version</th><th scope="col">Registry status</th><th scope="col">Selection metric</th><th scope="col">CV score</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
                          <tbody>{filteredModels.map((model) => {
                            const metricRecord = model.metrics ?? {};
                            const sortName = typeof metricRecord.selection_metric === "string" ? metricRecord.selection_metric : "";
                            const scoreValue = typeof metricRecord.cv_selection_score === "number" && Number.isFinite(metricRecord.cv_selection_score)
                              ? metricRecord.cv_selection_score
                              : null;
                            return <tr key={model.id}>
                              <th scope="row"><span className="results-model-name">{model.model_name}</span><code>{model.id}</code></th>
                              <td>v{model.version}</td>
                              <td><Badge tone={statusTone(model.status)}>{model.status}</Badge></td>
                              <td>{sortName ? metricLabel(sortName) : "Not returned"}</td>
                              <td>{scoreValue !== null && sortName ? metricValueLabel(sortName, scoreValue) : "—"}</td>
                              <td><Button type="button" variant="outline" onClick={() => { setActiveModelId(model.id); setDetailModelId(model.id); }}>Inspect</Button></td>
                            </tr>;
                          })}</tbody>
                        </table></div>
                      ) : <p className="results-unavailable-copy">No registered model versions match the current filters.</p>}
                    </Card>
                  </>
                )}

                <Card className="results-section-card">
                  <div className="results-section-heading"><div><span className="section-overline">FEATURE INSIGHTS</span><h2>Feature importance</h2></div></div>
                  <p className="results-unavailable-copy">Permutation importance is a separate post-training holdout analysis. Structured permutation-importance values are not included in the model registry response.</p>
                </Card>

                <FeatureSelectionSection metrics={modelMetrics} />

                <Card className="results-section-card">
                  <div className="results-section-heading"><div><span className="section-overline">ANALYSIS FILES</span><h2>Available artifacts</h2></div></div>
                  {visibleData.artifactsError ? (
                    <div className="results-error-row"><Alert tone="error" title="Artifact list unavailable">{visibleData.artifactsError}</Alert><Button variant="outline" onClick={retryResults}>Retry</Button></div>
                  ) : artifactsForDisplay.length === 0 ? (
                    <p className="results-unavailable-copy">No downloadable report, notebook, or cleaned-data artifact is registered for this session.</p>
                  ) : (
                    <ul className="results-artifact-list">{artifactsForDisplay.map((artifact) => {
                      const config = DOWNLOADABLE_ARTIFACTS[artifact.artifact_type];
                      const Icon = config?.icon ?? FileBarChart;
                      return <li key={artifact.id}>
                        <span className="results-artifact-icon"><Icon size={16} aria-hidden="true" /></span>
                        <div><strong>{config?.label ?? artifact.artifact_type}</strong><small>{config?.fileName ?? "No download route is available for this artifact type."}</small></div>
                        {config && <Button type="button" variant="outline" loading={downloadingArtifact === artifact.id} disabled={Boolean(downloadingArtifact)} leadingIcon={<Download size={14} aria-hidden="true" />} onClick={() => { void handleDownload(artifact); }}>Download</Button>}
                      </li>;
                    })}</ul>
                  )}
                  {downloadError && <Alert tone="error" title="Download failed">{downloadError}</Alert>}
                  {downloadNotice && <Alert tone="success" title="Download started">{downloadNotice}</Alert>}
                </Card>
              </>
            )}
          </>
        )}

        {!isLoadingResults && !selectedSessionId && !selectedRunId && !selectionErrorMessage && (
          <Card className="results-no-final-card results-empty-card">
            {sessions.length === 0 && !sessionsLoading && !sessionsError ? (
              <EmptyState
                illustration={RESULT_IMAGE}
                title="No dataset sessions yet"
                description="Create a dataset session in the AI Workspace before reviewing model results."
                action={<Link className="button button-outline" href="/workspace/agent">Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" /></Link>}
              />
            ) : (
              <EmptyState
                illustration={RESULT_IMAGE}
                title="Choose a session to view results"
                description="Select one of your accessible dataset sessions above. Results appear only when the backend has registered model data."
              />
            )}
          </Card>
        )}

        {visibleData?.session && visibleData.modelsError && <Button type="button" variant="outline" onClick={retryResults}>Retry loading results</Button>}
      </div>

      <Modal open={Boolean(detailModel)} onClose={() => setDetailModelId(null)} title={detailModel?.model_name ?? "Model details"}>
        {detailModel && selectedSession && <ModelDetails model={detailModel} session={selectedSession} />}
      </Modal>
    </AppShell>
  );
}
