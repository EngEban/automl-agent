"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Bot,
  CheckCircle2,
  Database,
  FileText,
  Plus,
  RotateCw,
  Send,
  Sparkles,
  Upload,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { SelectField, TextInput, Toggle } from "@/components/ui/form-controls";
import { Badge, Card } from "@/components/ui/surfaces";
import { ProtectedRoute } from "@/features/auth/protected-route";
import { useAuth } from "@/features/auth/auth-provider";
import { UploadDatasetDialog } from "@/features/dashboard/upload-dataset-dialog";
import {
  getAgentRun,
  getAgentSession,
  listAgentMessages,
  listAgentSessions,
  sendAgentMessage,
} from "@/features/agent-workspace/agent-api";
import { AgentMessageList } from "@/features/agent-workspace/agent-message-list";
import { RunMonitorPanel } from "@/features/agent-workspace/run-monitor-panel";
import { useRunMonitor } from "@/features/agent-workspace/use-run-monitor";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import type {
  AgentMessageRead,
  AnalysisOptions,
  FeatureSelectionConfig,
  SessionDetails,
  SessionListItem,
} from "@/types/api";

type LoadState = "loading" | "ready" | "error";
const SESSION_PAGE_SIZE = 50;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const SUGGESTED_PROMPTS = [
  "Analyze my dataset and summarize its main characteristics.",
  "Identify a suitable target column for a prediction task.",
  "Check the dataset quality and highlight missing values.",
  "Prepare an ML training plan for this dataset.",
  "Explain the dataset's main characteristics in plain language.",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeErrorMessage(error: unknown, area: "sessions" | "messages" | "chat" | "run"): string {
  if (error instanceof ApiTimeoutError) {
    return area === "chat"
      ? "We could not confirm whether the request was accepted. Check this conversation before trying again."
      : "The request took too long. Check your connection and try again.";
  }
  if (error instanceof ApiError) {
    if (error.status === 404) return area === "run"
      ? "This run is no longer available. Refresh the conversation to review saved messages."
      : "This dataset session is unavailable. Choose another session or upload a dataset.";
    if (error.status === 422) return area === "chat"
      ? "The message or selected analysis options were not accepted. Review them and try again."
      : "The request was not accepted. Please try again.";
    if (error.status >= 500) return "The service is temporarily unavailable. Please try again shortly.";
  }
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  if (area === "messages") return "Conversation history could not be loaded.";
  if (area === "run") return "Run status could not be refreshed.";
  if (area === "chat") return "Your request could not be sent. Please try again.";
  return "Your dataset sessions could not be loaded. Please try again.";
}

function metadataColumns(session: SessionDetails | null): string[] {
  const value = session?.dataset_meta.columns;
  if (!Array.isArray(value)) return [];
  return value.flatMap((column) => {
    if (typeof column === "string") return [column];
    if (isRecord(column) && typeof column.name === "string") return [column.name];
    return [];
  });
}

function metadataNumber(session: SessionDetails, key: string): string {
  const value = session.dataset_meta[key];
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "—";
}

function dateLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

function conflictRunId(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 409 || !isRecord(error.body)) return null;
  const detail = error.body.detail;
  return isRecord(detail) && typeof detail.run_id === "string" ? detail.run_id : null;
}

function optimisticUserMessage(content: string): AgentMessageRead {
  return {
    id: `pending-${globalThis.crypto.randomUUID()}`,
    role: "user",
    content,
    created_at: new Date().toISOString(),
  };
}

function AgentWorkspacePageContent() {
  const { accessToken, user, logout } = useAuth();
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [sessionsState, setSessionsState] = useState<LoadState>("loading");
  const [sessionsReload, setSessionsReload] = useState(0);
  const [sessionsError, setSessionsError] = useState("");
  const [sessionsMoreError, setSessionsMoreError] = useState("");
  const [sessionsOffset, setSessionsOffset] = useState(0);
  const [sessionsHasMore, setSessionsHasMore] = useState(false);
  const [sessionsLoadingMore, setSessionsLoadingMore] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState("");
  const [messages, setMessages] = useState<AgentMessageRead[]>([]);
  const [messagesState, setMessagesState] = useState<LoadState>("ready");
  const [messagesError, setMessagesError] = useState("");
  const [mode, setMode] = useState<"simple" | "advanced">("simple");
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState<"" | "classification" | "regression">("");
  const [featureSelectionEnabled, setFeatureSelectionEnabled] = useState(false);
  const [featureSelectionMethod, setFeatureSelectionMethod] = useState<FeatureSelectionConfig["method"]>("mutual_information");
  const [featureSelectionTopK, setFeatureSelectionTopK] = useState("20");
  const [featureSelectionMinimum, setFeatureSelectionMinimum] = useState("1");
  const [featureSelectionThreshold, setFeatureSelectionThreshold] = useState("0.95");
  const [featureSelectionTrees, setFeatureSelectionTrees] = useState("50");
  const [draft, setDraft] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [requestNotice, setRequestNotice] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const selectedSessionRef = useRef("");
  const selectionRevision = useRef(0);
  const sendLock = useRef(false);
  const refreshedRunState = useRef("");
  const sessionsPageLock = useRef(false);
  const sessionsPageControllerRef = useRef<AbortController | null>(null);
  const runMonitor = useRunMonitor({
    token: accessToken,
    ownerId: user?.id,
    sessionId: session?.id,
    onUnauthorized: logout,
  });
  const startRun = runMonitor.startRun;

  const refreshMessages = useCallback(async (sessionId: string) => {
    if (!accessToken) return;
    try {
      const result = await listAgentMessages(accessToken, sessionId);
      if (selectedSessionRef.current === sessionId) {
        setMessages(result);
        setMessagesState("ready");
        setMessagesError("");
      }
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
      if (selectedSessionRef.current === sessionId) {
        setMessagesState("error");
        setMessagesError(safeErrorMessage(error, "messages"));
      }
    }
  }, [accessToken, logout]);

  const selectSession = useCallback(async (sessionId: string, knownSession?: SessionDetails, runId?: string) => {
    const id = sessionId.trim();
    const revision = ++selectionRevision.current;
    selectedSessionRef.current = id;
    setSelectedSessionId(id);
    setSession(null);
    setMessages([]);
    setSessionError("");
    setMessagesError("");
    setRequestError("");
    setRequestNotice("");
    setTargetColumn("");
    setTaskType("");
    setSessionLoading(Boolean(id));
    setMessagesState(id ? "loading" : "ready");
    if (id) {
      const runQuery = runId ? `&run_id=${encodeURIComponent(runId)}` : "";
      router.replace(`/workspace/agent?session_id=${encodeURIComponent(id)}${runQuery}`, { scroll: false });
    }
    else router.replace("/workspace/agent", { scroll: false });
    if (!id) return;
    if (!accessToken) {
      setSessionLoading(false);
      setSessionError("Your sign-in session is unavailable. Sign in again to open this dataset.");
      return;
    }

    try {
      const details = knownSession ?? await getAgentSession(accessToken, id);
      if (revision !== selectionRevision.current) return;
      setSession(details);
      setSessions((current) => [details, ...current.filter((item) => item.id !== details.id)]);
      setSessionLoading(false);
      try {
        const history = await listAgentMessages(accessToken, id);
        if (revision !== selectionRevision.current) return;
        setMessages(history);
        setMessagesState("ready");
      } catch (error) {
        if (revision !== selectionRevision.current) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
        setMessagesState("error");
        setMessagesError(safeErrorMessage(error, "messages"));
      }
    } catch (error) {
      if (revision !== selectionRevision.current) return;
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
      setSessionError(safeErrorMessage(error, "sessions"));
      setMessagesState("error");
      setMessagesError("");
      setSessionLoading(false);
    }
  }, [accessToken, logout, router]);

  useEffect(() => {
    if (!accessToken) return;
    let active = true;
    const controller = new AbortController();
    listAgentSessions(accessToken, SESSION_PAGE_SIZE, 0, controller.signal)
      .then(async (result) => {
        if (!active || controller.signal.aborted) return;
        setSessions(result);
        setSessionsOffset(result.length);
        setSessionsHasMore(result.length === SESSION_PAGE_SIZE);
        setSessionsState("ready");
        setSessionsError("");
        setSessionsMoreError("");
        const params = new URLSearchParams(window.location.search);
        const requestedId = params.get("session_id");
        const requestedRunId = params.get("run_id");
        let initialId = requestedId || selectedSessionRef.current || (!requestedRunId ? result[0]?.id : "") || "";
        let verifiedRunId: string | undefined;

        if (requestedRunId) {
          if (!UUID_PATTERN.test(requestedRunId)) {
            setRequestError("The run link contains an invalid identifier. Select a session or return to Results.");
          } else {
            try {
              const run = await getAgentRun(accessToken, requestedRunId, controller.signal);
              if (!active || controller.signal.aborted) return;
              if (requestedId && run.session_id !== requestedId) {
                setRequestError("The selected run does not belong to the selected dataset session.");
                if (requestedId) void selectSession(requestedId);
                return;
              }
              initialId = run.session_id;
              verifiedRunId = run.id;
              startRun(run.id, run.session_id);
            } catch (error) {
              if (!active || controller.signal.aborted) return;
              if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
                logout();
                return;
              }
              setRequestError(safeErrorMessage(error, "run"));
            }
          }
        }
        if (initialId) void selectSession(initialId, undefined, verifiedRunId);
      })
      .catch((error: unknown) => {
        if (!active || controller.signal.aborted) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          logout();
          return;
        }
        setSessionsState("error");
        setSessionsError(safeErrorMessage(error, "sessions"));
      });
    return () => {
      active = false;
      controller.abort();
      sessionsPageControllerRef.current?.abort();
      sessionsPageControllerRef.current = null;
      sessionsPageLock.current = false;
    };
  }, [accessToken, logout, selectSession, sessionsReload, startRun]);

  const loadMoreSessions = useCallback(async () => {
    if (!accessToken || !sessionsHasMore || sessionsPageLock.current) return;
    const offset = sessionsOffset;
    const controller = new AbortController();
    sessionsPageControllerRef.current = controller;
    sessionsPageLock.current = true;
    setSessionsLoadingMore(true);
    setSessionsMoreError("");
    try {
      const page = await listAgentSessions(accessToken, SESSION_PAGE_SIZE, offset, controller.signal);
      if (controller.signal.aborted) return;
      setSessions((current) => {
        const knownIds = new Set(current.map((item) => item.id));
        return [...current, ...page.filter((item) => !knownIds.has(item.id))];
      });
      setSessionsOffset(offset + page.length);
      setSessionsHasMore(page.length === SESSION_PAGE_SIZE);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        logout();
        return;
      }
      setSessionsMoreError(safeErrorMessage(error, "sessions"));
    } finally {
      if (!controller.signal.aborted) {
        sessionsPageLock.current = false;
        sessionsPageControllerRef.current = null;
        setSessionsLoadingMore(false);
      }
    }
  }, [accessToken, logout, sessionsHasMore, sessionsOffset]);

  const retrySessions = useCallback(() => {
    sessionsPageControllerRef.current?.abort();
    sessionsPageControllerRef.current = null;
    sessionsPageLock.current = false;
    setSessionsLoadingMore(false);
    setSessionsMoreError("");
    setSessionsState("loading");
    setSessionsError("");
    setSessionsReload((current) => current + 1);
  }, []);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, runMonitor.run?.status, messagesState]);

  const completedRunId = runMonitor.run?.id;
  const completedRunSessionId = runMonitor.run?.session_id;
  const completedRunStatus = runMonitor.run?.status;

  useEffect(() => {
    const activeSessionId = session?.id;
    if (!accessToken || !activeSessionId || !completedRunId || completedRunSessionId !== activeSessionId || !completedRunStatus) return;
    const responseAvailable = ["succeeded", "partially_completed", "failed", "cancelled", "awaiting_input", "awaiting_approval"].includes(completedRunStatus);
    const refreshKey = `${completedRunId}:${completedRunStatus}`;
    if (refreshedRunState.current === refreshKey) return;
    refreshedRunState.current = refreshKey;
    if (!responseAvailable) return;
    let active = true;
    listAgentMessages(accessToken, activeSessionId)
      .then((result) => {
        if (!active || selectedSessionRef.current !== activeSessionId) return;
        setMessages(result);
        setMessagesState("ready");
        setMessagesError("");
      })
      .catch((error: unknown) => {
        if (!active || selectedSessionRef.current !== activeSessionId) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
        setMessagesState("error");
        setMessagesError(safeErrorMessage(error, "messages"));
      });
    return () => { active = false; };
  }, [accessToken, logout, completedRunId, completedRunSessionId, completedRunStatus, session?.id]);

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || !session || !accessToken || messagesState !== "ready" || isSubmitting || sendLock.current || runMonitor.blocksSubmission) return;

    const selectedOptions: AnalysisOptions = {};
    if (mode === "advanced") {
      if (targetColumn) selectedOptions.target_column = targetColumn;
      if (taskType) selectedOptions.task_type = taskType;
      if (featureSelectionEnabled) {
        const minimum = Number(featureSelectionMinimum);
        const topK = Number(featureSelectionTopK);
        const threshold = Number(featureSelectionThreshold);
        const trees = Number(featureSelectionTrees);
        const hasTopK = featureSelectionMethod !== "variance_filter";
        const thresholdValid = featureSelectionMethod === "variance_filter"
          ? Number.isFinite(threshold) && threshold >= 0 && threshold <= 1_000_000_000_000
          : featureSelectionMethod === "correlation_filter"
            ? Number.isFinite(threshold) && threshold >= 0 && threshold <= 1
            : true;
        if (!Number.isInteger(minimum) || minimum < 1 || minimum > 300
          || (hasTopK && (!Number.isInteger(topK) || topK < 1 || topK > 300 || minimum > topK))
          || !thresholdValid
          || (featureSelectionMethod === "extra_trees" && (!Number.isInteger(trees) || trees < 10 || trees > 200))) {
          setRequestError("Review the feature-selection values: use whole feature counts from 1 to 300, keep the minimum at or below the maximum, and use a valid method threshold.");
          return;
        }
        selectedOptions.feature_selection = {
          method: featureSelectionMethod,
          min_features: minimum,
          ...(featureSelectionMethod === "variance_filter" || featureSelectionMethod === "correlation_filter"
            ? { threshold }
            : {}),
          ...(featureSelectionMethod !== "variance_filter"
            ? { top_k: topK }
            : {}),
          ...(featureSelectionMethod === "extra_trees"
            ? { n_estimators: trees }
            : {}),
        };
      }
    }

    sendLock.current = true;
    setIsSubmitting(true);
    setRequestError("");
    setRequestNotice("");
    try {
      const response = await sendAgentMessage(accessToken, session.id, {
        message: content,
        mode,
        ...(mode === "advanced" && Object.keys(selectedOptions).length > 0
          ? { analysis_options: selectedOptions }
          : {}),
      });
      if (selectedSessionRef.current === session.id) {
        setMessages((current) => [...current, optimisticUserMessage(content)]);
        setDraft("");
      }
      runMonitor.startRun(response.run_id, session.id);
    } catch (error) {
      const existingRunId = conflictRunId(error);
      if (existingRunId) {
        runMonitor.startRun(existingRunId, session.id);
        setRequestNotice("A run is already active for this dataset. Its persisted status and events are shown below.");
        void refreshMessages(session.id);
      } else {
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
        setRequestError(safeErrorMessage(error, "chat"));
      }
    } finally {
      sendLock.current = false;
      setIsSubmitting(false);
    }
  }

  async function handleCreated(created: SessionDetails) {
    setUploadOpen(false);
    setSessionsState("ready");
    setSessionsError("");
    await selectSession(created.id, created);
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  const columns = metadataColumns(session);
  const selectedSessionOptions = [
    { label: "Choose a dataset session", value: "" },
    ...(selectedSessionId && !sessions.some((item) => item.id === selectedSessionId)
      ? [{ label: "Selected session", value: selectedSessionId }]
      : []),
    ...sessions.map((item) => ({ label: item.name, value: item.id })),
  ];
  const isRunActive = runMonitor.blocksSubmission;

  return (
    <>
      <AppShell
        title="AI Workspace"
        eyebrow="PIPELINE AGENT"
        description="Explore your dataset and guide an analysis through a conversation with the Agent."
        breadcrumbs={[{ label: "Workspace", href: "/workspace" }, { label: "AI Workspace" }]}
        activeItem="AI Workspace"
        homeHref="/workspace"
        statusLabel="Agent workspace"
      >
        <div className="agent-workspace">
          {sessionsError && <Alert tone="error" title="Sessions unavailable">{sessionsError}</Alert>}
          {sessionsMoreError && <Alert tone="warning" title="Older sessions unavailable">{sessionsMoreError} You can retry loading them below.</Alert>}
          <div className="agent-session-toolbar">
            <div className="agent-session-picker">
              <SelectField
                label="Dataset session"
                hint="The conversation and saved history belong to the selected session."
                value={selectedSessionId}
                options={selectedSessionOptions}
                disabled={sessionsState === "loading" || isSubmitting}
                onChange={(event) => { void selectSession(event.currentTarget.value); }}
              />
            </div>
            <div className="agent-session-actions">
              {sessionsState === "error" && <Button type="button" variant="outline" leadingIcon={<RotateCw size={14} />} onClick={retrySessions}>Retry sessions</Button>}
              {sessionsHasMore && <Button type="button" variant="ghost" loading={sessionsLoadingMore} disabled={sessionsLoadingMore} onClick={() => { void loadMoreSessions(); }}>{sessionsMoreError ? "Retry older sessions" : "Load older sessions"}</Button>}
              {selectedSessionId && <Link className="button button-outline" href={`/workspace/reports?session_id=${encodeURIComponent(selectedSessionId)}`}>Reports</Link>}
              <Button type="button" variant="outline" disabled={isSubmitting} leadingIcon={<Upload size={14} />} onClick={() => setUploadOpen(true)}>Upload dataset</Button>
            </div>
          </div>

          <div className="agent-mode-switch" role="group" aria-label="Agent mode">
            <button type="button" className={mode === "simple" ? "is-active" : ""} aria-pressed={mode === "simple"} onClick={() => setMode("simple")}>
              <Sparkles size={15} aria-hidden="true" /><span>Simple</span><small>Agent guided</small>
            </button>
            <button type="button" className={mode === "advanced" ? "is-active" : ""} aria-pressed={mode === "advanced"} onClick={() => setMode("advanced")}>
              <FileText size={15} aria-hidden="true" /><span>Advanced</span><small>Set target and task</small>
            </button>
          </div>

          {requestNotice && <Alert tone="info" title="Existing run">{requestNotice}</Alert>}

          <div className="agent-workspace-grid">
            <aside className="agent-context-column" aria-label="Dataset context">
              <Card className="agent-context-card">
                <div className="agent-context-heading">
                  <span className="agent-context-icon"><Database size={16} aria-hidden="true" /></span>
                  <div><span className="section-overline">CURRENT DATASET</span><h2>{session?.name ?? "Dataset context"}</h2></div>
                </div>
                {sessionLoading ? (
                  <div className="agent-context-skeleton" aria-label="Loading dataset metadata"><span /><span /><span /></div>
                ) : session ? (
                  <>
                    <div className="agent-context-stats">
                      <div><strong>{metadataNumber(session, "n_rows")}</strong><span>Rows</span></div>
                      <div><strong>{metadataNumber(session, "n_columns")}</strong><span>Columns</span></div>
                    </div>
                    <div className="agent-context-details">
                      <div><span>Session status</span><Badge tone="blue">{session.status.replaceAll("_", " ")}</Badge></div>
                      <div><span>Added</span><strong>{dateLabel(session.created_at)}</strong></div>
                      {session.target_column && <div><span>Saved target</span><strong className="agent-context-value">{session.target_column}</strong></div>}
                    </div>
                    {columns.length > 0 && (
                      <div className="agent-column-preview">
                        <span className="agent-subheading">Available columns</span>
                        <div>{columns.slice(0, 8).map((column) => <span key={column}>{column}</span>)}</div>
                        {columns.length > 8 && <small>And {columns.length - 8} more</small>}
                      </div>
                    )}
                  </>
                ) : sessionError ? (
                  <div className="agent-context-empty"><p>{sessionError}</p><Button type="button" variant="outline" onClick={() => selectedSessionId && void selectSession(selectedSessionId)}>Retry session</Button></div>
                ) : (
                  <div className="agent-context-empty">
                    <p>Select a dataset session to see its metadata and start a conversation.</p>
                    {sessions.length === 0 && sessionsState === "ready" && <Button type="button" leadingIcon={<Plus size={14} />} onClick={() => setUploadOpen(true)}>Upload a dataset</Button>}
                  </div>
                )}
              </Card>

              {mode === "advanced" && (
                <Card className="agent-advanced-card">
                  <div className="agent-advanced-heading"><span className="section-overline">ANALYSIS SETTINGS</span><h2>Advanced options</h2><p>Leave a setting on automatic for the Agent to decide.</p></div>
                  <SelectField
                    label="Target column"
                    hint={session ? "Column names come from this session's dataset metadata." : "Select a dataset first."}
                    value={targetColumn}
                    disabled={!session || columns.length === 0}
                    options={[
                      { label: "Automatic selection", value: "" },
                      ...columns.map((column) => ({ label: column, value: column })),
                    ]}
                    onChange={(event) => setTargetColumn(event.currentTarget.value)}
                  />
                  <SelectField
                    label="Task type"
                    hint="Choose classification or regression, or let the Agent infer it."
                    value={taskType}
                    disabled={!session}
                    options={[
                      { label: "Automatic inference", value: "" },
                      { label: "Classification", value: "classification" },
                      { label: "Regression", value: "regression" },
                    ]}
                    onChange={(event) => setTaskType(event.currentTarget.value as "" | "classification" | "regression")}
                  />
                  <div className="agent-feature-selection-settings">
                    <Toggle
                      label="Use feature selection"
                      description="Off by default; the selected method is learned again inside every training fold."
                      checked={featureSelectionEnabled}
                      onCheckedChange={setFeatureSelectionEnabled}
                      disabled={!session || isSubmitting}
                    />
                    {featureSelectionEnabled && (
                      <>
                        <SelectField
                          label="Training selector"
                          hint="Applied to preprocessed numeric and one-hot encoded features."
                          value={featureSelectionMethod}
                          disabled={!session || isSubmitting}
                          options={[
                            { label: "Mutual information", value: "mutual_information" },
                            { label: "ANOVA / F-regression", value: "univariate_test" },
                            { label: "Correlation filter", value: "correlation_filter" },
                            { label: "Variance filter", value: "variance_filter" },
                            { label: "Extra Trees importance", value: "extra_trees" },
                          ]}
                          onChange={(event) => {
                            const method = event.currentTarget.value as FeatureSelectionConfig["method"];
                            setFeatureSelectionMethod(method);
                            if (method === "variance_filter") setFeatureSelectionThreshold("0");
                            if (method === "correlation_filter") setFeatureSelectionThreshold("0.95");
                          }}
                        />
                        {(featureSelectionMethod === "mutual_information" || featureSelectionMethod === "univariate_test" || featureSelectionMethod === "extra_trees" || featureSelectionMethod === "correlation_filter") && (
                          <TextInput
                            label="Maximum retained features"
                            type="number"
                            min={1}
                            max={300}
                            step={1}
                            value={featureSelectionTopK}
                            disabled={!session || isSubmitting}
                            onChange={(event) => setFeatureSelectionTopK(event.currentTarget.value)}
                          />
                        )}
                        <TextInput
                          label="Minimum retained features"
                          type="number"
                          min={1}
                          max={300}
                          step={1}
                          value={featureSelectionMinimum}
                          disabled={!session || isSubmitting}
                          onChange={(event) => setFeatureSelectionMinimum(event.currentTarget.value)}
                        />
                        {(featureSelectionMethod === "variance_filter" || featureSelectionMethod === "correlation_filter") && (
                          <TextInput
                            label={featureSelectionMethod === "variance_filter" ? "Variance threshold" : "Absolute correlation threshold"}
                            type="number"
                            min={0}
                            max={featureSelectionMethod === "variance_filter" ? 1000000000000 : 1}
                            step="any"
                            value={featureSelectionThreshold}
                            disabled={!session || isSubmitting}
                            onChange={(event) => setFeatureSelectionThreshold(event.currentTarget.value)}
                          />
                        )}
                        {featureSelectionMethod === "extra_trees" && (
                          <TextInput
                            label="Extra Trees count"
                            type="number"
                            min={10}
                            max={200}
                            step={1}
                            value={featureSelectionTrees}
                            disabled={!session || isSubmitting}
                            onChange={(event) => setFeatureSelectionTrees(event.currentTarget.value)}
                          />
                        )}
                        <p className="agent-feature-selection-note">
                          Preview comparisons are exploratory. Training learns a separate selector in each CV fold; folds can retain different features. The final fitted selector uses training rows only and excludes the holdout. Review the current dataset&apos;s preview in <Link href="/workspace/datasets">Datasets</Link>.
                        </p>
                        <p className="agent-feature-selection-summary" aria-live="polite">
                          Run configuration: {featureSelectionMethod.replaceAll("_", " ")}
                          {featureSelectionMethod !== "variance_filter" ? ` · up to ${featureSelectionTopK || "20"} features` : ` · threshold ${featureSelectionThreshold || "0"}`}
                          {` · minimum ${featureSelectionMinimum || "1"}`}
                          {featureSelectionMethod === "extra_trees" ? ` · ${featureSelectionTrees || "50"} trees` : ""}
                        </p>
                      </>
                    )}
                  </div>
                </Card>
              )}

              <div className="agent-context-note"><CheckCircle2 size={14} aria-hidden="true" /><span>Requests are sent securely through your authenticated workspace.</span></div>
            </aside>

            <section className="agent-conversation card" aria-label="Agent conversation">
              <header className="agent-conversation-header">
                <span className="agent-avatar"><Bot size={18} aria-hidden="true" /></span>
                <div><strong>Pipeline Agent</strong><span>Dataset assistant</span></div>
                <Badge tone={mode === "simple" ? "blue" : "purple"}>{mode === "simple" ? "Simple mode" : "Advanced mode"}</Badge>
              </header>

              {requestError && <div className="agent-inline-alert"><Alert tone="error" title="Message not sent">{requestError}</Alert></div>}
              {sessionError && <div className="agent-inline-alert"><Alert tone="warning" title="Session unavailable">Choose another dataset session to continue.</Alert></div>}
              <div className="agent-conversation-scroll" aria-busy={sessionLoading || messagesState === "loading" || undefined}>
                {sessionsState === "loading" && !selectedSessionId ? (
                  <div className="agent-welcome agent-welcome-loading"><span /><span /><span /></div>
                ) : sessionLoading || messagesState === "loading" ? (
                  <div className="agent-message-skeleton" aria-label="Loading conversation"><span /><span /><span /></div>
                ) : !session ? (
                  <div className="agent-welcome">
                    <span className="agent-welcome-icon"><Sparkles size={21} aria-hidden="true" /></span>
                    <span className="section-overline">YOUR DATA, WITH CONTEXT</span>
                    <h2>Choose a dataset to begin</h2>
                    <p>Select one of your sessions above, or upload a dataset to start a new conversation.</p>
                    {sessions.length === 0 && sessionsState === "ready" && <Button type="button" leadingIcon={<Upload size={14} />} onClick={() => setUploadOpen(true)}>Upload dataset</Button>}
                  </div>
                ) : messagesState === "error" ? (
                  <div className="agent-welcome"><span className="section-overline">CONVERSATION</span><h2>History could not be loaded</h2><p>{messagesError || "Retry loading the saved messages before continuing this session."}</p><Button type="button" variant="outline" onClick={() => void refreshMessages(session.id)}>Retry history</Button></div>
                ) : messages.length > 0 ? (
                  <AgentMessageList messages={messages} />
                ) : (
                  <div className="agent-welcome">
                    <Image
                      className="agent-welcome-illustration"
                      src="/images/automl-agent-illustrations/03-ai-workspace.png"
                      width={815}
                      height={296}
                      alt=""
                      aria-hidden="true"
                    />
                    <span className="agent-welcome-icon"><Sparkles size={21} aria-hidden="true" /></span>
                    <span className="section-overline">READY TO EXPLORE</span>
                    <h2>What would you like to learn?</h2>
                    <p>Ask about the dataset, its quality, or how to prepare an analysis. The Agent will use the selected session.</p>
                    <div className="agent-suggestions" aria-label="Suggested prompts">
                      {SUGGESTED_PROMPTS.map((prompt) => (
                        <button type="button" key={prompt} onClick={() => setDraft(prompt)}>{prompt}</button>
                      ))}
                    </div>
                  </div>
                )}
                <div ref={messageEndRef} />
              </div>

              <form className="agent-composer" onSubmit={(event) => { void handleSend(event); }}>
                <label className="sr-only" htmlFor="agent-message">Message the Pipeline Agent</label>
                <textarea
                  id="agent-message"
                  value={draft}
                  maxLength={20_000}
                  rows={3}
                  placeholder={session ? "Ask about your dataset…" : "Select a dataset session first"}
                  disabled={!session || sessionLoading || messagesState !== "ready" || isSubmitting || isRunActive}
                  onChange={(event) => setDraft(event.currentTarget.value)}
                  onKeyDown={handleComposerKeyDown}
                />
                <div className="agent-composer-footer">
                  <span>Enter to send · Shift+Enter for a new line{draft.length > 18_000 ? ` · ${draft.length.toLocaleString()}/20,000` : ""}</span>
                  <Button type="submit" loading={isSubmitting} disabled={!session || messagesState !== "ready" || !draft.trim() || isRunActive} leadingIcon={<Send size={14} />}>
                    {isRunActive ? "Agent working" : "Send"}
                  </Button>
                </div>
              </form>
              <p className="agent-safety-note">The Agent can make mistakes. Review important analysis decisions.</p>
            </section>
          </div>

          {runMonitor.runId && session && (
            <RunMonitorPanel
              run={runMonitor.run}
              runId={runMonitor.runId}
              sessionName={session.name}
              events={runMonitor.events}
              streamState={runMonitor.streamState}
              isLoading={runMonitor.isLoading}
              error={runMonitor.error}
              historyWarning={runMonitor.historyWarning}
              cancelNotice={runMonitor.cancelNotice}
              isCancelling={runMonitor.isCancelling}
              canCancel={runMonitor.canCancel}
              onReconnect={runMonitor.retry}
              onCancel={() => { void runMonitor.cancelRun(); }}
            />
          )}
        </div>
      </AppShell>

      {uploadOpen && accessToken && (
        <UploadDatasetDialog
          open
          token={accessToken}
          onClose={() => setUploadOpen(false)}
          onCreated={(created) => { void handleCreated(created); }}
        />
      )}
    </>
  );
}

export function AgentWorkspacePage() {
  return <ProtectedRoute><AgentWorkspacePageContent /></ProtectedRoute>;
}
