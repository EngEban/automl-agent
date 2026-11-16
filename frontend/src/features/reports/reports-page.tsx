"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Download,
  Eye,
  FileBarChart,
  FileText,
  RefreshCw,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert, EmptyState, Skeleton } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { SelectField, TextInput } from "@/components/ui/form-controls";
import { Badge, Card } from "@/components/ui/surfaces";
import { Modal } from "@/components/ui/modal";
import { useAuth } from "@/features/auth/auth-provider";
import { getAgentSession, listAgentSessions } from "@/features/agent-workspace/agent-api";
import {
  downloadSessionReport,
  getLatestSessionReport,
  isPdfFile,
} from "@/features/reports/reports-api";
import type { ReportRecord } from "@/features/reports/reports-api";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";

const SESSION_PAGE_SIZE = 20;
const ARTIFACT_CONCURRENCY = 4;
const REPORT_IMAGE = "/images/automl-agent-illustrations/05-reports.webp";

interface PageRequest {
  offset: number;
  generation: number;
  sessionId: string | null;
}

interface SessionInspection {
  reports: ReportRecord[];
  failures: number;
  authorizationError: unknown | null;
}

function isAuthorizationError(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

function safeLoadError(error: unknown, directSession: boolean): string {
  if (error instanceof ApiTimeoutError) return "The request took too long. Check your connection and try again.";
  if (error instanceof ApiError) {
    if (error.status === 404) return directSession
      ? "This session is unavailable to your account."
      : "Some report records could not be loaded. Refresh to try again.";
    if (error.status >= 500) return "The report service is temporarily unavailable. Please try again shortly.";
  }
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return "Reports could not be loaded. Please try again.";
}

function safeDownloadError(error: unknown): string {
  if (error instanceof ApiTimeoutError) return "The PDF request took too long. Please try again.";
  if (error instanceof ApiError) {
    if (error.status === 401) return "Your sign-in has expired. Sign in again to access this report.";
    if (error.status === 403) return "You do not have permission to access this report.";
    if (error.status === 404) return "The report file is no longer available. Ask the Pipeline Agent to generate it again.";
    if (error.status >= 500) return "The report service is temporarily unavailable. Please try again shortly.";
  }
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return "The report could not be retrieved as a PDF.";
}

function dateLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

async function inspectSessionPage(
  token: string,
  sessions: Awaited<ReturnType<typeof listAgentSessions>>,
  signal: AbortSignal,
): Promise<SessionInspection> {
  const reports: ReportRecord[] = [];
  let failures = 0;

  for (let start = 0; start < sessions.length; start += ARTIFACT_CONCURRENCY) {
    const batch = sessions.slice(start, start + ARTIFACT_CONCURRENCY);
    const results = await Promise.all(batch.map(async (session) => {
      try {
        return { report: await getLatestSessionReport(token, session, signal), error: null as unknown };
      } catch (error) {
        return { report: null, error };
      }
    }));

    if (signal.aborted) return { reports, failures, authorizationError: null };
    for (const result of results) {
      if (result.error) {
        if (isAuthorizationError(result.error)) {
          return { reports, failures, authorizationError: result.error };
        }
        failures += 1;
      } else if (result.report) {
        reports.push(result.report);
      }
    }
  }

  return { reports, failures, authorizationError: null };
}

function mergeReportPages(current: ReportRecord[], incoming: ReportRecord[]): ReportRecord[] {
  const bySession = new Map(current.map((record) => [record.sessionId, record]));
  for (const record of incoming) bySession.set(record.sessionId, record);
  return [...bySession.values()];
}

function ReportDetails({ report }: { report: ReportRecord }) {
  return (
    <div className="reports-detail-content">
      <div className="reports-detail-heading">
        <span className="reports-file-icon"><FileText size={18} aria-hidden="true" /></span>
        <div><span className="section-overline">REGISTERED ARTIFACT</span><h3>PDF analysis report</h3></div>
        <Badge tone="blue">Registered</Badge>
      </div>
      <dl className="reports-detail-list">
        <div><dt>Dataset / session</dt><dd>{report.sessionName}</dd></div>
        <div><dt>Session ID</dt><dd><code>{report.sessionId}</code></dd></div>
        <div><dt>Artifact ID</dt><dd><code>{report.artifactId}</code></dd></div>
        <div><dt>Format</dt><dd>PDF</dd></div>
        <div><dt>Report type</dt><dd>Not returned by the artifact API</dd></div>
        <div><dt>Report creation time</dt><dd>Not returned by the artifact API</dd></div>
        <div><dt>Associated run</dt><dd>Not returned by the artifact API</dd></div>
        <div><dt>Dataset session created</dt><dd>{dateLabel(report.sessionCreatedAt)}</dd></div>
        <div><dt>Dataset session status</dt><dd>{report.sessionStatus.replaceAll("_", " ")}</dd></div>
      </dl>
      <p className="reports-detail-note">The backend download route serves the latest PDF registered for this session. The artifact API does not expose report-specific metadata, file size, or a run identifier.</p>
    </div>
  );
}

function ReportsPageContent({ initialSessionId, contextRunId, selectionError }: { initialSessionId: string | null; contextRunId: string | null; selectionError: string }) {
  const { accessToken, logout } = useAuth();
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [pageRequest, setPageRequest] = useState<PageRequest>({ offset: 0, generation: 0, sessionId: initialSessionId });
  const [pageStatus, setPageStatus] = useState<"loading" | "ready" | "error">("loading");
  const [pageError, setPageError] = useState("");
  const [artifactFailuresByOffset, setArtifactFailuresByOffset] = useState<Record<number, number>>({});
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(!initialSessionId);
  const [nextOffset, setNextOffset] = useState(0);
  const [sessionsScanned, setSessionsScanned] = useState(0);
  const [search, setSearch] = useState("");
  const [sessionFilter, setSessionFilter] = useState(initialSessionId ?? "");
  const [selectedReport, setSelectedReport] = useState<ReportRecord | null>(null);
  const [previewFile, setPreviewFile] = useState<{ artifactId: string; url: string; blob: Blob; filename: string | null } | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [downloadingId, setDownloadingId] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [downloadNotice, setDownloadNotice] = useState("");
  const previewUrlRef = useRef<string | null>(null);
  const previewControllerRef = useRef<AbortController | null>(null);
  const downloadControllerRef = useRef<AbortController | null>(null);
  const downloadResourcesRef = useRef<Array<{ url: string; timer: number }>>([]);
  const artifactFailureCount = Object.values(artifactFailuresByOffset).reduce((total, count) => total + count, 0);

  const revokePreview = useCallback(() => {
    previewControllerRef.current?.abort();
    previewControllerRef.current = null;
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewFile(null);
    setPreviewLoading(false);
  }, []);

  useEffect(() => () => {
    previewControllerRef.current?.abort();
    downloadControllerRef.current?.abort();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    for (const resource of downloadResourcesRef.current) {
      window.clearTimeout(resource.timer);
      URL.revokeObjectURL(resource.url);
    }
  }, []);

  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    const { offset, sessionId } = pageRequest;

    async function loadReports() {
      try {
        if (sessionId) {
          const session = await getAgentSession(accessToken!, sessionId, controller.signal);
          const report = await getLatestSessionReport(accessToken!, session, controller.signal);
          if (controller.signal.aborted) return;
          setReports(report ? [report] : []);
          setSessionsScanned(1);
          setNextOffset(1);
          setHasMore(false);
          setArtifactFailuresByOffset({});
        } else {
          const sessions = await listAgentSessions(accessToken!, SESSION_PAGE_SIZE, offset, controller.signal);
          if (controller.signal.aborted) return;
          const result = await inspectSessionPage(accessToken!, sessions, controller.signal);
          if (controller.signal.aborted) return;
          if (result.authorizationError) {
            setPageError(safeLoadError(result.authorizationError, false));
            setPageStatus("error");
            logout();
            return;
          }
          setReports((current) => offset === 0 ? result.reports : mergeReportPages(current, result.reports));
          setSessionsScanned(offset + sessions.length);
          setNextOffset(offset + sessions.length);
          setHasMore(sessions.length === SESSION_PAGE_SIZE);
          setArtifactFailuresByOffset((current) => {
            const next = { ...current };
            if (result.failures > 0) next[offset] = result.failures;
            else delete next[offset];
            return next;
          });
        }
        setPageError("");
        setPageStatus("ready");
      } catch (error) {
        if (controller.signal.aborted) return;
        if (isAuthorizationError(error)) {
          logout();
          setPageError(error instanceof ApiError && error.status === 401
            ? "Your sign-in has expired. Sign in again to access your reports."
            : "You do not have permission to access these reports.");
        } else {
          setPageError(safeLoadError(error, Boolean(sessionId)));
        }
        setPageStatus("error");
      } finally {
        if (!controller.signal.aborted) setLoadingMore(false);
      }
    }

    void loadReports();
    return () => controller.abort();
  }, [accessToken, logout, pageRequest]);

  const visibleReports = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return reports.filter((report) => {
      if (sessionFilter && report.sessionId !== sessionFilter) return false;
      if (!query) return true;
      return report.sessionName.toLocaleLowerCase().includes(query)
        || report.sessionId.toLocaleLowerCase().includes(query)
        || report.artifactId.toLocaleLowerCase().includes(query);
    });
  }, [reports, search, sessionFilter]);

  const sessionOptions = useMemo(() => {
    const options = [{ label: "All loaded reports", value: "" }];
    if (sessionFilter && !reports.some((report) => report.sessionId === sessionFilter)) {
      options.push({ label: "Selected dataset session", value: sessionFilter });
    }
    for (const report of reports) {
      if (!options.some((option) => option.value === report.sessionId)) {
        options.push({ label: report.sessionName, value: report.sessionId });
      }
    }
    return options;
  }, [reports, sessionFilter]);

  function retryPage() {
    setPageError("");
    setLoadingMore(pageRequest.offset > 0);
    if (pageRequest.offset === 0) setPageStatus("loading");
    setPageRequest((current) => ({ ...current, generation: current.generation + 1 }));
  }

  function refreshReports() {
    setReports([]);
    setPageError("");
    setArtifactFailuresByOffset({});
    setSessionsScanned(0);
    setNextOffset(0);
    setHasMore(!pageRequest.sessionId);
    setPageStatus("loading");
    setPageRequest((current) => ({ ...current, offset: 0, generation: current.generation + 1 }));
  }

  function loadMoreSessions() {
    if (loadingMore || !hasMore || pageStatus === "error") return;
    setLoadingMore(true);
    setPageRequest((current) => ({ ...current, offset: nextOffset, generation: current.generation + 1 }));
  }

  function handleSessionFilter(value: string) {
    if (pageRequest.sessionId && !value) {
      setSessionFilter("");
      setReports([]);
      setSessionsScanned(0);
      setNextOffset(0);
      setHasMore(true);
      setPageError("");
      setPageStatus("loading");
      setPageRequest((current) => ({ offset: 0, generation: current.generation + 1, sessionId: null }));
      return;
    }
    setSessionFilter(value);
  }

  function clearFilters() {
    setSearch("");
    handleSessionFilter("");
  }

  function openDetails(report: ReportRecord) {
    revokePreview();
    setPreviewError("");
    setSelectedReport(report);
  }

  function closeDetails() {
    revokePreview();
    setSelectedReport(null);
    setPreviewError("");
  }

  async function previewReport(report: ReportRecord) {
    if (!accessToken || previewLoading || previewControllerRef.current) return;
    openDetails(report);
    setPreviewLoading(true);
    setPreviewError("");
    const controller = new AbortController();
    previewControllerRef.current = controller;
    try {
      const file = await downloadSessionReport(accessToken, report.sessionId, controller.signal);
      if (controller.signal.aborted) return;
      if (!(await isPdfFile(file.blob))) throw new Error("invalid_pdf");
      if (controller.signal.aborted) return;
      const pdfBlob = new Blob([file.blob], { type: "application/pdf" });
      const url = URL.createObjectURL(pdfBlob);
      previewUrlRef.current = url;
      setPreviewFile({ artifactId: report.artifactId, url, blob: pdfBlob, filename: file.filename });
    } catch (error) {
      if (controller.signal.aborted) return;
      if (isAuthorizationError(error)) {
        logout();
        setPreviewError(error instanceof ApiError && error.status === 401
          ? "Your sign-in has expired. Sign in again to preview this report."
          : "You do not have permission to preview this report.");
      } else if (error instanceof Error && error.message === "invalid_pdf") {
        setPreviewError("The report endpoint did not return a valid PDF file.");
      } else {
        setPreviewError(safeDownloadError(error));
      }
    } finally {
      if (!controller.signal.aborted) setPreviewLoading(false);
      if (previewControllerRef.current === controller) previewControllerRef.current = null;
    }
  }

  async function downloadReport(report: ReportRecord) {
    if (!accessToken || downloadingId || downloadControllerRef.current) return;
    setDownloadingId(report.artifactId);
    setDownloadError("");
    setDownloadNotice("");
    const controller = new AbortController();
    downloadControllerRef.current = controller;
    try {
      const file = previewFile?.artifactId === report.artifactId
        ? { blob: previewFile.blob, filename: previewFile.filename }
        : await downloadSessionReport(accessToken, report.sessionId, controller.signal);
      if (controller.signal.aborted) return;
      if (!(await isPdfFile(file.blob))) throw new Error("invalid_pdf");
      if (controller.signal.aborted) return;
      const pdfBlob = new Blob([file.blob], { type: "application/pdf" });
      const url = URL.createObjectURL(pdfBlob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = file.filename || `report_${report.sessionId}.pdf`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      const timer = window.setTimeout(() => {
        URL.revokeObjectURL(url);
        downloadResourcesRef.current = downloadResourcesRef.current.filter((item) => item.url !== url);
      }, 1000);
      downloadResourcesRef.current.push({ url, timer });
      setDownloadNotice(`The PDF download for “${report.sessionName}” has started.`);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (isAuthorizationError(error)) {
        logout();
        setDownloadError(error instanceof ApiError && error.status === 401
          ? "Your sign-in has expired. Sign in again to download this report."
          : "You do not have permission to download this report.");
      } else if (error instanceof Error && error.message === "invalid_pdf") {
        setDownloadError("The report endpoint did not return a valid PDF file.");
      } else {
        setDownloadError(safeDownloadError(error));
      }
    } finally {
      if (!controller.signal.aborted) setDownloadingId("");
      if (downloadControllerRef.current === controller) downloadControllerRef.current = null;
    }
  }

  const workspaceHref = sessionFilter
    ? `/workspace/agent?session_id=${encodeURIComponent(sessionFilter)}`
    : "/workspace/agent";
  const canShowEmptyState = pageStatus === "ready" && reports.length === 0;

  return (
    <>
      <AppShell
        title="Reports Center"
        eyebrow="ANALYSIS LIBRARY"
        description="Browse and securely retrieve the PDF reports registered for your dataset sessions."
        breadcrumbs={[{ label: "Workspace", href: "/workspace" }, { label: "Reports" }]}
        activeItem="Reports"
        homeHref="/workspace"
        statusLabel="Private workspace"
        actions={<Button type="button" variant="outline" leadingIcon={<RefreshCw size={14} aria-hidden="true" />} onClick={refreshReports}>Refresh reports</Button>}
      >
        <div className="reports-page">
          {selectionError && <Alert tone="warning" title="Invalid report link">{selectionError}</Alert>}
          <Card className="reports-intro-card">
            <span className="reports-intro-icon"><FileBarChart size={18} aria-hidden="true" /></span>
            <div>
              <span className="section-overline">REPORT WORKFLOW</span>
              <h2>Reports follow a completed analysis</h2>
              <p>The backend has no standalone report-generation endpoint. Ask the Pipeline Agent to generate a report after training; registered PDFs appear here when available.</p>
              {contextRunId && <small className="reports-run-context">Opened from run <code>{contextRunId}</code>. The artifact API does not associate reports with run IDs.</small>}
            </div>
            <Link className="button button-ghost" href={workspaceHref}>Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" /></Link>
          </Card>

          {pageError && (
            <div className="reports-error-row">
              <Alert tone="error" title="Reports unavailable">{pageError}</Alert>
              <Button type="button" variant="outline" onClick={retryPage}>Retry</Button>
            </div>
          )}
          {artifactFailureCount > 0 && (
            <Alert tone="warning" title="Some sessions could not be checked">
              Report availability may be incomplete for {artifactFailureCount.toLocaleString()} session artifact request{artifactFailureCount === 1 ? "" : "s"}. Refresh to restart the paginated scan.
            </Alert>
          )}
          {downloadError && <Alert tone="error" title="PDF unavailable">{downloadError}</Alert>}
          {downloadNotice && <Alert tone="success" title="Download started">{downloadNotice}</Alert>}

          <Card className="reports-toolbar-card">
            <div className="reports-toolbar-heading">
              <div><span className="section-overline">YOUR REPORTS</span><h2>Registered PDF reports</h2></div>
              {pageStatus === "ready" && <span className="reports-count">{visibleReports.length.toLocaleString()} shown · {reports.length.toLocaleString()} loaded</span>}
            </div>
            <div className="reports-filters">
              <TextInput
                label="Search loaded reports"
                type="search"
                value={search}
                onChange={(event) => setSearch(event.currentTarget.value)}
                placeholder="Dataset, session, or artifact ID"
                hint="The API does not return report titles; search uses the associated session and artifact IDs."
              />
              <SelectField
                label="Filter by dataset session"
                value={sessionFilter}
                options={sessionOptions}
                onChange={(event) => handleSessionFilter(event.currentTarget.value)}
              />
            </div>
            <p className="reports-data-note">
              {pageRequest.sessionId
                ? "Showing the requested session. Report creation dates, report types, file sizes, and run IDs are not returned by the artifact API."
                : `${sessionsScanned.toLocaleString()} dataset sessions scanned. The API is paginated; older sessions are checked only when you load more. Report rows follow associated session order, not report creation time.`}
            </p>
          </Card>

          {pageStatus === "loading" && reports.length === 0 ? (
            <div className="reports-loading-list" role="status" aria-label="Loading reports">
              <Card><Skeleton /><Skeleton /><Skeleton /></Card>
              <Card><Skeleton /><Skeleton /><Skeleton /></Card>
            </div>
          ) : visibleReports.length > 0 ? (
            <section className="reports-list" aria-label="Report files">
              {visibleReports.map((report) => (
                <Card className="reports-report-card" key={report.artifactId}>
                  <span className="reports-file-icon"><FileText size={19} aria-hidden="true" /></span>
                  <div className="reports-record-copy">
                    <div className="reports-record-title"><h3>PDF analysis report</h3><Badge tone="blue">Registered</Badge></div>
                    <p>{report.sessionName}</p>
                    <small>Dataset session · {dateLabel(report.sessionCreatedAt)}</small>
                    <code title={report.artifactId}>Artifact · {report.artifactId}</code>
                  </div>
                  <div className="reports-record-actions">
                    <Button type="button" variant="ghost" leadingIcon={<Eye size={14} aria-hidden="true" />} onClick={() => { void previewReport(report); }} disabled={previewLoading || Boolean(downloadingId)}>Preview</Button>
                    <Button type="button" variant="outline" leadingIcon={<FileText size={14} aria-hidden="true" />} onClick={() => openDetails(report)} disabled={previewLoading}>Details</Button>
                    <Button type="button" variant="primary" loading={downloadingId === report.artifactId} disabled={Boolean(downloadingId) || previewLoading} leadingIcon={<Download size={14} aria-hidden="true" />} onClick={() => { void downloadReport(report); }}>Download PDF</Button>
                  </div>
                </Card>
              ))}
            </section>
          ) : canShowEmptyState ? (
            <Card className="reports-empty-card">
              <EmptyState
                illustration={REPORT_IMAGE}
                title={reports.length === 0 ? "No report artifacts found" : "No reports match these filters"}
                description={reports.length === 0
                  ? pageRequest.sessionId
                    ? "This session has no registered PDF report yet. You can continue the analysis in the AI Workspace."
                    : hasMore
                      ? "No registered report was found in the sessions scanned so far. Load older sessions to continue searching."
                      : "No PDF report artifacts are registered for the sessions returned by the backend."
                  : "Try another search or choose a different loaded dataset session."}
                action={reports.length === 0
                  ? <Link className="button button-outline" href={workspaceHref}>Open AI Workspace <ArrowUpRight size={14} aria-hidden="true" /></Link>
                  : <Button type="button" variant="outline" onClick={clearFilters}>Clear filters</Button>}
              />
            </Card>
          ) : null}

          {!pageRequest.sessionId && pageStatus === "ready" && hasMore && (
            <div className="reports-pagination">
              <p>{sessionsScanned.toLocaleString()} sessions checked. The backend does not return a report total.</p>
              <Button type="button" variant="outline" loading={loadingMore} disabled={loadingMore} onClick={loadMoreSessions}>Load older sessions</Button>
            </div>
          )}
          {!pageRequest.sessionId && pageStatus === "ready" && !hasMore && sessionsScanned > 0 && (
            <p className="reports-pagination-note">All {sessionsScanned.toLocaleString()} sessions returned by the API have been checked.</p>
          )}
        </div>
      </AppShell>

      <Modal open={Boolean(selectedReport)} onClose={closeDetails} title="Report details">
        {selectedReport && (
          <div className="reports-modal-content">
            <ReportDetails report={selectedReport} />
            <div className="reports-modal-actions">
              {!previewFile && <Button type="button" variant="primary" loading={previewLoading} leadingIcon={<Eye size={14} aria-hidden="true" />} onClick={() => { void previewReport(selectedReport); }}>Preview PDF</Button>}
              <Button type="button" variant="outline" loading={downloadingId === selectedReport.artifactId} disabled={Boolean(downloadingId) || previewLoading} leadingIcon={<Download size={14} aria-hidden="true" />} onClick={() => { void downloadReport(selectedReport); }}>Download PDF</Button>
              <Link className="button button-ghost" href={`/workspace/agent?session_id=${encodeURIComponent(selectedReport.sessionId)}`}>Open session <ArrowUpRight size={14} aria-hidden="true" /></Link>
            </div>
            {previewError && <Alert tone="error" title="Preview unavailable">{previewError}</Alert>}
            {previewLoading && <div className="reports-preview-loading" role="status"><Skeleton /><Skeleton /></div>}
            {previewFile && <iframe className="reports-pdf-frame" src={previewFile.url} title={`PDF report for ${selectedReport.sessionName}`} />}
          </div>
        )}
      </Modal>
    </>
  );
}

export function ReportsPage({ initialSessionId, contextRunId, selectionError }: { initialSessionId: string | null; contextRunId: string | null; selectionError: string }) {
  return <ReportsPageContent initialSessionId={initialSessionId} contextRunId={contextRunId} selectionError={selectionError} />;
}
