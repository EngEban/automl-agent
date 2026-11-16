import Link from "next/link";
import { Activity, ArrowRight, Boxes, CheckCircle2, Clock3, FolderKanban, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EmptyState, Skeleton } from "@/components/ui/feedback";
import { Badge, Card, MetricCard } from "@/components/ui/surfaces";
import type { DashboardSession } from "@/types/api";

export type DashboardLoadState = "loading" | "ready" | "error";

export function DashboardMetrics({ sessionCount, state }: { sessionCount: number; state: DashboardLoadState }) {
  const sessionValue = state === "loading" ? "—" : state === "error" ? "Unavailable" : String(sessionCount);
  const sessionDetail = state === "loading"
    ? "Loading your recent sessions"
    : state === "error"
      ? "Session data could not be loaded"
      : "Most recent sessions shown";

  return (
    <section className="dashboard-metrics" aria-label="Workspace overview">
      <MetricCard label="Recent dataset sessions" value={sessionValue} detail={sessionDetail} icon={<FolderKanban size={16} aria-hidden="true" />} />
      <MetricCard label="Analysis runs" value="—" detail="Run history is not available yet" icon={<Activity size={16} aria-hidden="true" />} />
      <MetricCard label="Completed runs" value="—" detail="Run totals are not available yet" icon={<CheckCircle2 size={16} aria-hidden="true" />} />
      <MetricCard label="Registered models" value="—" detail="Model totals are not available yet" icon={<Boxes size={16} aria-hidden="true" />} />
    </section>
  );
}

export function DashboardQuickActions({ onUpload }: { onUpload: () => void }) {
  return (
    <section className="dashboard-quick-actions" aria-labelledby="quick-actions-title">
      <div className="dashboard-section-heading">
        <div><span className="section-overline">GET STARTED</span><h2 id="quick-actions-title">Quick actions</h2></div>
      </div>
      <div className="dashboard-action-grid">
        <Card className="dashboard-action-card">
          <span className="dashboard-action-icon"><Upload size={17} aria-hidden="true" /></span>
          <div><h3>Upload a dataset</h3><p>Create a dataset session from a local file.</p></div>
          <Button variant="outline" onClick={onUpload}>Upload dataset</Button>
        </Card>
        <Card className="dashboard-action-card">
          <span className="dashboard-action-icon"><Activity size={17} aria-hidden="true" /></span>
          <div><h3>Analysis workspace</h3><p>Open the Agent and choose a dataset session to continue.</p></div>
          <Link className="button button-ghost" href="/workspace/agent">Open AI Workspace <ArrowRight size={14} aria-hidden="true" /></Link>
        </Card>
        <Card className="dashboard-action-card">
          <span className="dashboard-action-icon"><FolderKanban size={17} aria-hidden="true" /></span>
          <div><h3>Recent datasets</h3><p>Review your latest dataset sessions.</p></div>
          <Link className="button button-ghost" href="#recent-datasets">View datasets <ArrowRight size={14} aria-hidden="true" /></Link>
        </Card>
      </div>
    </section>
  );
}

export function RecentActivity({ sessions, state, onUpload }: {
  sessions: DashboardSession[];
  state: DashboardLoadState;
  onUpload: () => void;
}) {
  return (
    <Card className="dashboard-panel">
      <div className="dashboard-panel-heading"><div><span className="section-overline">LATEST EVENTS</span><h2>Recent activity</h2></div></div>
      {state === "loading" ? <ActivitySkeleton /> : state === "error" ? (
        <EmptyState title="Activity is unavailable" description="Use the dashboard retry action to check your recent sessions again." />
      ) : sessions.length === 0 ? (
        <EmptyState title="No activity yet" description="Create a dataset session to see activity here." action={<Button onClick={onUpload}>Upload a dataset</Button>} />
      ) : (
        <ol className="dashboard-activity-list">
          {sessions.slice(0, 5).map((session) => (
            <li key={session.id}>
              <span className="dashboard-activity-marker"><Clock3 size={14} aria-hidden="true" /></span>
              <div className="dashboard-activity-copy"><strong>Dataset session created</strong><span>{session.name} · {sessionStatus(session.status).label}</span></div>
              <time dateTime={session.created_at}>{formatTimestamp(session.created_at)}</time>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

export function RecentAnalyses() {
  return (
    <Card className="dashboard-panel">
      <div className="dashboard-panel-heading"><div><span className="section-overline">RUN HISTORY</span><h2>Recent analyses</h2></div></div>
      <EmptyState
        title="Run history is not available yet"
        description="Your analysis history will appear here when run history becomes available."
        illustration="/assets/illustrations/workflow.svg"
      />
    </Card>
  );
}

export function RecentDatasets({ sessions, state, onUpload }: {
  sessions: DashboardSession[];
  state: DashboardLoadState;
  onUpload: () => void;
}) {
  return (
    <section id="recent-datasets" className="dashboard-datasets" aria-labelledby="recent-datasets-title">
      <div className="dashboard-section-heading">
        <div><span className="section-overline">YOUR DATA</span><h2 id="recent-datasets-title">Recent dataset sessions</h2></div>
        {state === "ready" && sessions.length > 0 && <span className="dashboard-count-note">Showing {Math.min(sessions.length, 5)} latest sessions</span>}
      </div>
      <Card className="dashboard-table-card">
        {state === "loading" ? <DatasetSkeleton /> : state === "error" ? (
          <EmptyState title="Datasets are unavailable" description="Use the dashboard retry action to check your sessions again." />
        ) : sessions.length === 0 ? (
          <EmptyState title="No datasets yet" description="Upload a CSV, TSV, JSON, or Excel file to create your first dataset session." action={<Button onClick={onUpload}>Upload a dataset</Button>} />
        ) : (
          <div className="dashboard-table-scroll">
            <table className="dashboard-table">
              <thead><tr><th scope="col">Dataset</th><th scope="col">Status</th><th scope="col">Target</th><th scope="col">Task type</th><th scope="col">Last updated / created</th></tr></thead>
              <tbody>
                {sessions.slice(0, 5).map((session) => <SessionRow key={session.id} session={session} />)}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </section>
  );
}

function SessionRow({ session }: { session: DashboardSession }) {
  const status = sessionStatus(session.status);
  const rowCount = metadataCount(session.dataset_meta, "n_rows");
  const columnCount = metadataCount(session.dataset_meta, "n_columns");
  const sourceRowCount = metadataCount(nestedMetadata(session.dataset_meta, "loading_metadata"), "original_row_count");
  const timestamp = session.updated_at ?? session.created_at;
  const rowSummary = rowCount === null
    ? "Row count unavailable"
    : sourceRowCount !== null && sourceRowCount > rowCount
      ? `${new Intl.NumberFormat().format(rowCount)} loaded of ${new Intl.NumberFormat().format(sourceRowCount)} source rows`
      : `${new Intl.NumberFormat().format(rowCount)} loaded rows`;
  return (
    <tr>
      <th scope="row">
        <span className="dashboard-dataset-name">{session.name}</span>
        <span className="dashboard-dataset-meta">{rowSummary}{columnCount !== null && ` · ${columnCount} columns`}</span>
      </th>
      <td><Badge tone={status.tone}>{status.label}</Badge></td>
      <td>{session.target_column || "—"}</td>
      <td>{session.task_type || "—"}</td>
      <td><time dateTime={timestamp}>{formatTimestamp(timestamp)}</time><span className="dashboard-date-kind">{session.updated_at ? "Updated" : "Created"}</span></td>
    </tr>
  );
}

function metadataCount(metadata: Record<string, unknown> | undefined, key: string): number | null {
  const value = metadata?.[key];
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function nestedMetadata(metadata: Record<string, unknown> | undefined, key: string): Record<string, unknown> | undefined {
  const value = metadata?.[key];
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function sessionStatus(raw: string): { label: string; tone: "neutral" | "blue" | "green" | "amber" | "red" } {
  const value = raw.toLowerCase();
  if (value === "created" || value === "ready" || value === "completed") {
    return { label: value === "created" ? "Ready" : value[0].toUpperCase() + value.slice(1), tone: "green" };
  }
  if (value === "failed" || value === "error") return { label: "Failed", tone: "red" };
  if (value === "pending" || value === "processing" || value === "running") {
    return { label: value[0].toUpperCase() + value.slice(1), tone: "blue" };
  }
  return { label: "Status unknown", tone: "neutral" };
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function ActivitySkeleton() {
  return <div className="dashboard-skeleton-list" role="status" aria-label="Loading recent activity"><Skeleton /><Skeleton /><Skeleton /></div>;
}

function DatasetSkeleton() {
  return <div className="dashboard-dataset-skeleton" role="status" aria-label="Loading dataset sessions"><Skeleton /><Skeleton /><Skeleton /><Skeleton /></div>;
}
