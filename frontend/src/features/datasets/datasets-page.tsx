"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, Database, FileSearch, FolderKanban, Plus, Search } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert, EmptyState, Skeleton } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { TextInput } from "@/components/ui/form-controls";
import { Badge, Card } from "@/components/ui/surfaces";
import { useAuth } from "@/features/auth/auth-provider";
import { UploadDatasetDialog } from "@/features/dashboard/upload-dataset-dialog";
import { getAgentSession, listAgentSessions } from "@/features/agent-workspace/agent-api";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import type { SessionDetails, SessionListItem } from "@/types/api";
import { Modal } from "@/components/ui/modal";

const SESSION_PAGE_SIZE = 50;

interface SessionPageRequest {
  offset: number;
  generation: number;
}

interface SessionDetailState {
  sessionId: string;
  session?: SessionDetails;
  error?: string;
}

function datasetErrorMessage(error: unknown, detail = false): string {
  if (error instanceof ApiTimeoutError) return "The request took too long. Check your connection and try again.";
  if (error instanceof ApiError) {
    if (error.status === 404) return detail
      ? "This session is no longer available to your account."
      : "Your dataset sessions are unavailable.";
    if (error.status >= 500) return "The service is temporarily unavailable. Please try again shortly.";
  }
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return detail ? "Dataset details could not be loaded." : "Your dataset sessions could not be loaded.";
}

function metadataCount(metadata: Record<string, unknown>, key: string): number | null {
  const value = metadata[key];
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function fileNameFromMetadata(metadata: Record<string, unknown>): string | null {
  const value = metadata.original_filename;
  if (typeof value !== "string" || !value.trim()) return null;
  const name = value.trim().split(/[\\/]/).pop();
  return name && name !== "." && name !== ".." ? name : null;
}

function metadataColumns(metadata: Record<string, unknown>): string[] {
  const value = metadata.columns;
  if (!Array.isArray(value)) return [];
  return value.flatMap((column) => {
    if (typeof column === "string") return [column];
    if (typeof column === "object" && column !== null && !Array.isArray(column)) {
      const name = (column as Record<string, unknown>).name;
      return typeof name === "string" ? [name] : [];
    }
    return [];
  });
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function statusTone(status: string): "neutral" | "blue" | "green" | "amber" | "red" {
  const value = status.toLowerCase();
  if (["created", "ready", "completed"].includes(value)) return "green";
  if (["pending", "processing", "running"].includes(value)) return "blue";
  if (["failed", "error"].includes(value)) return "red";
  if (["cancelling", "awaiting_input", "awaiting_approval"].includes(value)) return "amber";
  return "neutral";
}

function SessionListSkeleton() {
  return (
    <div className="datasets-skeleton-list" role="status" aria-label="Loading dataset sessions">
      <div className="datasets-skeleton-header"><Skeleton /><Skeleton /><Skeleton /></div>
      {Array.from({ length: 5 }, (_, index) => <Skeleton key={index} />)}
    </div>
  );
}

function DatasetDetails({ session }: { session: SessionDetails }) {
  const metadata = session.dataset_meta ?? {};
  const columns = metadataColumns(metadata);
  const dtypes = typeof metadata.dtypes === "object" && metadata.dtypes !== null && !Array.isArray(metadata.dtypes)
    ? metadata.dtypes as Record<string, unknown>
    : {};
  const originalFileName = fileNameFromMetadata(metadata);
  const metrics = [
    { label: "Rows", value: metadataCount(metadata, "n_rows") },
    { label: "Columns", value: metadataCount(metadata, "n_columns") },
    { label: "Missing values", value: metadataCount(metadata, "missing_total") },
    { label: "Duplicate rows", value: metadataCount(metadata, "duplicate_rows") },
  ].filter((metric): metric is { label: string; value: number } => metric.value !== null);

  return (
    <div className="dataset-detail-content">
      <div className="dataset-detail-summary">
        <span className="dataset-detail-icon"><Database size={17} aria-hidden="true" /></span>
        <div>
          <span className="section-overline">DATASET SESSION</span>
          <h3>{session.name}</h3>
          <p>Created {formatDate(session.created_at)}</p>
        </div>
        <Badge tone={statusTone(session.status)}>{session.status.replaceAll("_", " ")}</Badge>
      </div>

      <section className="dataset-detail-section" aria-labelledby="dataset-file-heading">
        <h4 id="dataset-file-heading">File and session</h4>
        <dl className="dataset-detail-list">
          {originalFileName && <div><dt>Source file</dt><dd title={originalFileName}>{originalFileName}</dd></div>}
          <div><dt>Source type</dt><dd>{session.source_type}</dd></div>
          <div><dt>Session ID</dt><dd><code>{session.id}</code></dd></div>
          {session.updated_at && <div><dt>Last updated</dt><dd>{formatDate(session.updated_at)}</dd></div>}
          {session.target_column && <div><dt>Target column</dt><dd>{session.target_column}</dd></div>}
          {session.task_type && <div><dt>Task type</dt><dd>{session.task_type}</dd></div>}
        </dl>
        <Link className="button button-outline dataset-open-workspace" href={`/workspace/agent?session_id=${encodeURIComponent(session.id)}`}>
          Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
        {["trained", "reported"].includes(session.status.toLowerCase()) && (
          <Link className="button button-outline dataset-open-workspace" href={`/workspace/results?session_id=${encodeURIComponent(session.id)}`}>
            View results <ArrowUpRight size={14} aria-hidden="true" />
          </Link>
        )}
        <Link className="button button-outline dataset-open-workspace" href={`/workspace/reports?session_id=${encodeURIComponent(session.id)}`}>
          View reports <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
      </section>

      {metrics.length > 0 && (
        <section className="dataset-detail-section" aria-labelledby="dataset-metadata-heading">
          <h4 id="dataset-metadata-heading">Dataset metadata</h4>
          <dl className="dataset-metrics">
            {metrics.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value.toLocaleString()}</dd></div>)}
          </dl>
        </section>
      )}

      {columns.length > 0 && (
        <section className="dataset-detail-section" aria-labelledby="dataset-columns-heading">
          <div className="dataset-columns-heading">
            <h4 id="dataset-columns-heading">Available columns</h4>
            <span>{columns.length.toLocaleString()} returned</span>
          </div>
          <ul className="dataset-column-list">
            {columns.slice(0, 24).map((column) => {
              const dtype = dtypes[column];
              return <li key={column}><span title={column}>{column}</span>{typeof dtype === "string" && <code>{dtype}</code>}</li>;
            })}
          </ul>
          {columns.length > 24 && <p className="dataset-column-note">Showing the first 24 of {columns.length.toLocaleString()} returned columns.</p>}
        </section>
      )}
    </div>
  );
}

function DatasetsPageContent() {
  const { accessToken, logout } = useAuth();
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [nextOffset, setNextOffset] = useState(0);
  const [listError, setListError] = useState("");
  const [errorOffset, setErrorOffset] = useState<number | null>(null);
  const [pageRequest, setPageRequest] = useState<SessionPageRequest>({ offset: 0, generation: 0 });
  const requestGeneration = useRef(0);
  const createdSessions = useRef<SessionListItem[]>([]);
  const [search, setSearch] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadNotice, setUploadNotice] = useState("");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [detailState, setDetailState] = useState<SessionDetailState | null>(null);
  const [detailRetry, setDetailRetry] = useState(0);

  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    const { offset } = pageRequest;
    listAgentSessions(accessToken, SESSION_PAGE_SIZE, offset, controller.signal)
      .then((page) => {
        if (controller.signal.aborted) return;
        setSessions((current) => {
          const loaded = offset === 0
            ? page
            : [...current, ...page.filter((session) => !current.some((item) => item.id === session.id))];
          const loadedIds = new Set(loaded.map((session) => session.id));
          const newlyCreated = createdSessions.current.filter((session) => !loadedIds.has(session.id));
          return [...newlyCreated, ...loaded];
        });
        setNextOffset(offset + page.length);
        setHasMore(page.length === SESSION_PAGE_SIZE);
        setListError("");
        setErrorOffset(null);
        setInitialLoading(false);
        setLoadingMore(false);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          logout();
          return;
        }
        setListError(datasetErrorMessage(error));
        setErrorOffset(offset);
        setInitialLoading(false);
        setLoadingMore(false);
      });
    return () => controller.abort();
  }, [accessToken, logout, pageRequest]);

  useEffect(() => {
    if (!accessToken || !selectedSessionId) return;
    const controller = new AbortController();
    const sessionId = selectedSessionId;
    getAgentSession(accessToken, sessionId, controller.signal)
      .then((session) => {
        if (!controller.signal.aborted) setDetailState({ sessionId, session });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          logout();
          return;
        }
        setDetailState({ sessionId, error: datasetErrorMessage(error, true) });
      });
    return () => controller.abort();
  }, [accessToken, detailRetry, logout, selectedSessionId]);

  const filteredSessions = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    if (!query) return sessions;
    return sessions.filter((session) => session.name.toLocaleLowerCase().includes(query)
      || session.id.toLocaleLowerCase().includes(query));
  }, [search, sessions]);

  const requestPage = useCallback((offset: number) => {
    setListError("");
    setErrorOffset(null);
    if (offset === 0) {
      setInitialLoading(true);
      setLoadingMore(false);
      setSessions([]);
      setNextOffset(0);
      setHasMore(true);
    } else {
      setLoadingMore(true);
    }
    setPageRequest({ offset, generation: ++requestGeneration.current });
  }, []);

  function handleLoadMore() {
    if (!hasMore || initialLoading || loadingMore) return;
    requestPage(nextOffset);
  }

  function handleRetry() {
    const offset = errorOffset ?? 0;
    requestPage(offset);
  }

  function handleSessionCreated(created: SessionDetails) {
    createdSessions.current = [created, ...createdSessions.current.filter((session) => session.id !== created.id)];
    setSessions((current) => [created, ...current.filter((session) => session.id !== created.id)]);
    setHasMore(true);
    setSearch("");
    setUploadNotice(`“${created.name}” was added to your dataset sessions.`);
  }

  function openSessionDetails(sessionId: string) {
    setDetailState(null);
    setSelectedSessionId(sessionId);
  }

  function closeSessionDetails() {
    setSelectedSessionId(null);
  }

  function retrySessionDetails() {
    setDetailState(null);
    setDetailRetry((current) => current + 1);
  }

  const selectedSession = detailState?.sessionId === selectedSessionId ? detailState.session : undefined;
  const detailError = detailState?.sessionId === selectedSessionId ? detailState.error : undefined;
  const detailLoading = Boolean(selectedSessionId && !selectedSession && !detailError);

  return (
    <>
      <AppShell
        title="Datasets"
        eyebrow="DATA LIBRARY"
        description="Browse your uploaded datasets and the sessions connected to your workspace."
        breadcrumbs={[{ label: "Workspace", href: "/workspace" }, { label: "Datasets" }]}
        activeItem="Datasets"
        homeHref="/workspace"
        statusLabel="Private workspace"
        actions={(
          <Button onClick={() => { setUploadNotice(""); setUploadOpen(true); }} leadingIcon={<Plus size={15} aria-hidden="true" />}>
            Upload dataset
          </Button>
        )}
      >
        <div className="datasets-page">
          {uploadNotice && <Alert tone="success" title="Dataset session created">{uploadNotice}</Alert>}
          {listError && (
            <div className="datasets-list-error">
              <Alert tone="error" title="Dataset sessions unavailable">{listError}</Alert>
              {errorOffset !== null && <Button className="datasets-inline-retry" variant="outline" onClick={handleRetry}>Retry</Button>}
            </div>
          )}

          <Card className="datasets-toolbar-card">
            <div className="datasets-toolbar-copy">
              <span className="datasets-toolbar-icon"><FolderKanban size={17} aria-hidden="true" /></span>
              <div><strong>Your dataset sessions</strong><span>Newest sessions are listed first.</span></div>
            </div>
            <div className="datasets-search-wrap">
              <Search size={15} aria-hidden="true" />
              <TextInput
                label="Search loaded sessions"
                type="search"
                value={search}
                onChange={(event) => setSearch(event.currentTarget.value)}
                placeholder="Search by dataset or session name"
                hint="Search covers loaded sessions. Load more to include older sessions."
              />
            </div>
          </Card>

          <section className="datasets-list-section" aria-labelledby="datasets-list-heading">
            <div className="datasets-list-heading">
              <div><span className="section-overline">YOUR DATA</span><h2 id="datasets-list-heading">Dataset library</h2></div>
              {!initialLoading && <span>{sessions.length.toLocaleString()} loaded</span>}
            </div>

            {initialLoading && sessions.length === 0 ? (
              <Card className="datasets-table-card"><SessionListSkeleton /></Card>
            ) : filteredSessions.length > 0 ? (
              <Card className="datasets-table-card">
                <div className="datasets-table-scroll">
                  <table className="dashboard-table datasets-table">
                    <thead>
                      <tr>
                        <th scope="col">Dataset / session</th>
                        <th scope="col">Status</th>
                        <th scope="col">Target</th>
                        <th scope="col">Task type</th>
                        <th scope="col">Created</th>
                        <th scope="col"><span className="sr-only">Actions</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredSessions.map((session) => (
                        <tr key={session.id}>
                          <th scope="row">
                            <span className="datasets-name" title={session.name}>{session.name}</span>
                            <span className="datasets-session-id" title={session.id}>Session · {session.id}</span>
                          </th>
                          <td><Badge tone={statusTone(session.status)}>{session.status.replaceAll("_", " ")}</Badge></td>
                          <td>{session.target_column || <span className="datasets-muted">—</span>}</td>
                          <td>{session.task_type || <span className="datasets-muted">—</span>}</td>
                          <td><time dateTime={session.created_at}>{formatDate(session.created_at)}</time></td>
                          <td>
                            <div className="datasets-row-actions">
                              <Button type="button" variant="outline" leadingIcon={<FileSearch size={13} aria-hidden="true" />} onClick={() => openSessionDetails(session.id)}>
                                Details
                              </Button>
                              <Link className="button button-ghost" href={`/workspace/agent?session_id=${encodeURIComponent(session.id)}`}>
                                Open session
                              </Link>
                              {["trained", "reported"].includes(session.status.toLowerCase()) && (
                                <Link className="button button-ghost" href={`/workspace/results?session_id=${encodeURIComponent(session.id)}`}>
                                  Results
                                </Link>
                              )}
                              <Link className="button button-ghost" href={`/workspace/reports?session_id=${encodeURIComponent(session.id)}`}>
                                Reports
                              </Link>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            ) : sessions.length > 0 ? (
              <Card className="datasets-empty-card">
                <EmptyState
                  title="No matching sessions"
                  description="Try a different name or load another server page to search older sessions."
                  action={<Button type="button" variant="outline" onClick={() => setSearch("")}>Clear search</Button>}
                />
              </Card>
            ) : !initialLoading && !listError ? (
              <Card className="datasets-empty-card datasets-no-sessions">
                <EmptyState
                  title="No dataset sessions yet"
                  description="Upload a dataset to create a session. It will appear here without starting an analysis."
                  action={<Button onClick={() => setUploadOpen(true)} leadingIcon={<Plus size={14} aria-hidden="true" />}>Upload dataset</Button>}
                  illustration="/images/automl-agent-illustrations/02-datasets.webp"
                />
              </Card>
            ) : null}

            <div className="datasets-pagination" aria-live="polite">
              <p>
                {hasMore
                  ? `Search is limited to the ${sessions.length.toLocaleString()} sessions loaded so far.`
                  : `The server returned ${sessions.length.toLocaleString()} sessions across the loaded pages.`}
              </p>
              {hasMore && sessions.length > 0 && (
                <Button type="button" variant="outline" loading={loadingMore} disabled={initialLoading} onClick={handleLoadMore}>
                  Load more sessions
                </Button>
              )}
              {!hasMore && sessions.length >= SESSION_PAGE_SIZE && <span className="datasets-end-note">No more sessions were returned.</span>}
            </div>
          </section>
        </div>
      </AppShell>

      {uploadOpen && accessToken && (
        <UploadDatasetDialog
          open
          token={accessToken}
          onClose={() => setUploadOpen(false)}
          onCreated={handleSessionCreated}
        />
      )}

      <Modal open={Boolean(selectedSessionId)} onClose={closeSessionDetails} title={selectedSession?.name ?? "Dataset details"}>
        {detailLoading ? (
          <div className="dataset-detail-loading" role="status" aria-label="Loading dataset details"><Skeleton /><Skeleton /><Skeleton /></div>
        ) : detailError ? (
          <div className="dataset-detail-error">
            <Alert tone="error" title="Could not load details">{detailError}</Alert>
            <Button type="button" variant="outline" onClick={retrySessionDetails}>Try again</Button>
          </div>
        ) : selectedSession ? (
          <DatasetDetails session={selectedSession} />
        ) : null}
      </Modal>
    </>
  );
}

export function DatasetsPage() {
  return <DatasetsPageContent />;
}
