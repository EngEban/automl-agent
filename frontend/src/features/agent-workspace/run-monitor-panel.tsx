"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Activity, Check, CircleAlert, Clock3, Radio, RotateCw, X } from "lucide-react";
import { Alert } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { Card, StatusIndicator } from "@/components/ui/surfaces";
import { Modal } from "@/components/ui/modal";
import type { AnalysisRunStatus, RunEventRead } from "@/types/api";
import type { RunStreamState } from "@/features/agent-workspace/use-run-monitor";

type Tone = "success" | "warning" | "error" | "info" | "neutral";

function runLabel(status: string): { label: string; tone: Tone; description: string } {
  switch (status) {
    case "queued": return { label: "Queued", tone: "info", description: "The backend accepted this run and queued it for execution." };
    case "running": return { label: "Running", tone: "info", description: "The Agent is processing this run." };
    case "cancelling": return { label: "Cancellation requested", tone: "warning", description: "The backend will stop at a safe execution checkpoint." };
    case "succeeded": return { label: "Completed", tone: "success", description: "The run completed successfully." };
    case "partially_completed": return { label: "Partial success", tone: "warning", description: "Some requested work completed. Review the Agent response and activity below." };
    case "failed": return { label: "Failed", tone: "error", description: "The backend reported that this run failed." };
    case "cancelled": return { label: "Cancelled", tone: "neutral", description: "The backend confirmed that this run was cancelled." };
    case "awaiting_input": return { label: "Needs input", tone: "warning", description: "The Agent is waiting for additional input." };
    case "awaiting_approval": return { label: "Awaiting approval", tone: "warning", description: "This run is waiting for plan approval." };
    default: return { label: status.replaceAll("_", " "), tone: "neutral", description: "The backend returned this run status." };
  }
}

function connectionLabel(state: RunStreamState, terminal: boolean): { label: string; tone: Tone } {
  if (terminal || state === "closed") return { label: "Activity complete", tone: "neutral" };
  switch (state) {
    case "loading": return { label: "Loading activity", tone: "neutral" };
    case "connecting": return { label: "Connecting", tone: "info" };
    case "connected": return { label: "Live events", tone: "success" };
    case "reconnecting": return { label: "Reconnecting", tone: "warning" };
    case "disconnected": return { label: "Disconnected", tone: "warning" };
    case "error": return { label: "Monitoring unavailable", tone: "error" };
    default: return { label: "Waiting for events", tone: "neutral" };
  }
}

function eventTitle(eventType: string): string {
  return eventType
    .split(/[._]/)
    .filter(Boolean)
    .map((word) => `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function eventTone(event: RunEventRead): Tone {
  if (event.event_type.includes("failed") || event.payload.success === false) return "error";
  if (event.event_type.includes("cancelled") || event.event_type === "run.cancellation_requested" || event.event_type === "run.partially_completed" || event.event_type === "run.awaiting_input" || event.event_type === "run.awaiting_approval") return "warning";
  if (event.event_type.includes("succeeded") || event.event_type.endsWith(".completed")) return "success";
  if (event.event_type.endsWith(".started") || event.event_type === "run.queued" || event.event_type === "run.created") return "info";
  return "neutral";
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Time unavailable"
    : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function safeMetadata(event: RunEventRead): string[] {
  const items: string[] = [];
  if (event.payload.success === true) items.push("Tool succeeded");
  else if (event.payload.success === false) items.push("Tool did not succeed");
  if (typeof event.payload.agent_status === "string") items.push(`Agent status: ${event.payload.agent_status}`);
  if (typeof event.payload.analysis_status === "string") items.push(`Analysis: ${event.payload.analysis_status}`);
  if (event.payload.mode === "simple" || event.payload.mode === "advanced") items.push(`${event.payload.mode} mode`);
  if (event.payload.operation_still_running === true) items.push("Operation continues in the backend");
  if (typeof event.payload.error_code === "string") items.push(`Error code: ${event.payload.error_code}`);
  return items;
}

function ActivityTimeline({ events }: { events: RunEventRead[] }) {
  const scrollRef = useRef<HTMLOListElement>(null);
  const nearBottomRef = useRef(true);

  useEffect(() => {
    const list = scrollRef.current;
    if (list && nearBottomRef.current) list.scrollTop = list.scrollHeight;
  }, [events]);

  return (
    <ol
      className="run-event-timeline"
      ref={scrollRef}
      aria-label="Run activity events"
      onScroll={(event) => {
        const list = event.currentTarget;
        nearBottomRef.current = list.scrollHeight - list.scrollTop - list.clientHeight < 72;
      }}
    >
      {events.map((event) => {
        const tone = eventTone(event);
        const metadata = safeMetadata(event);
        const hasProgress = Number.isFinite(event.progress_percent) && event.progress_percent > 0 && event.progress_percent <= 100;
        return (
          <li className={`run-event run-event-${tone}`} key={event.id}>
            <span className="run-event-marker" aria-hidden="true">
              {tone === "success" ? <Check size={13} /> : tone === "error" ? <CircleAlert size={13} /> : tone === "warning" ? <Clock3 size={13} /> : <Activity size={13} />}
            </span>
            <article className="run-event-card">
              <div className="run-event-heading">
                <strong>{eventTitle(event.event_type)}</strong>
                <span>#{event.sequence}</span>
              </div>
              {event.step && <span className="run-event-step">Stage · {event.step}</span>}
              {event.message && <p>{event.message}</p>}
              {metadata.length > 0 && <div className="run-event-metadata">{metadata.map((item) => <span key={item}>{item}</span>)}</div>}
              <footer>
                <time dateTime={event.created_at}>{formatTimestamp(event.created_at)}</time>
                {hasProgress && <span>{event.progress_percent}% at this event</span>}
              </footer>
            </article>
          </li>
        );
      })}
    </ol>
  );
}

export function RunMonitorPanel({
  run,
  runId,
  sessionName,
  events,
  streamState,
  isLoading,
  error,
  historyWarning,
  cancelNotice,
  isCancelling,
  canCancel,
  onReconnect,
  onCancel,
}: {
  run: AnalysisRunStatus | null;
  runId: string;
  sessionName: string;
  events: RunEventRead[];
  streamState: RunStreamState;
  isLoading: boolean;
  error: string;
  historyWarning: string;
  cancelNotice: string;
  isCancelling: boolean;
  canCancel: boolean;
  onReconnect: () => void;
  onCancel: () => void;
}) {
  const [confirmCancel, setConfirmCancel] = useState(false);
  const status = run ? runLabel(run.status) : null;
  const terminal = run ? ["succeeded", "partially_completed", "failed", "cancelled"].includes(run.status) : false;
  const connection = connectionLabel(streamState, terminal);
  const hasCompletedTraining = Boolean(
    run
    && ["succeeded", "partially_completed"].includes(run.status)
    && events.some((event) => event.event_type === "training.completed"),
  );
  const progress = run && Number.isFinite(run.progress_percent) && run.progress_percent >= 0 && run.progress_percent <= 100
    ? run.progress_percent
    : null;

  return (
    <Card className="run-monitor-panel" aria-label="Run monitoring">
      <header className="run-monitor-header">
        <div className="run-monitor-title-row">
          <span className="run-monitor-icon"><Activity size={16} aria-hidden="true" /></span>
          <div className="run-monitor-title-copy"><span className="section-overline">RUN ACTIVITY</span><h2>Live run monitor</h2></div>
          <StatusIndicator label={connection.label} tone={connection.tone} />
        </div>
        <div className="run-monitor-runline">
          {status && <StatusIndicator label={status.label} tone={status.tone} />}
          <span>Dataset · {sessionName}</span>
          <span>Run ID · <code>{run?.id ?? runId}</code></span>
        </div>
        {run && status && <p className="run-monitor-description">{status.description}</p>}
        {progress !== null && (
          <div className="run-progress-block">
            <div><span>Backend-reported progress</span><strong>{progress}%</strong></div>
            <progress max={100} value={progress} aria-label={`Backend-reported run progress ${progress} percent`} />
          </div>
        )}
        <div className="run-monitor-actions">
          {hasCompletedTraining && run && (
            <Link
              className="button button-outline"
              href={`/workspace/results?session_id=${encodeURIComponent(run.session_id)}&run_id=${encodeURIComponent(run.id)}`}
            >
              View results
            </Link>
          )}
          {(streamState === "disconnected" || streamState === "error") && !terminal && (
            <Button type="button" variant="outline" leadingIcon={<RotateCw size={14} />} onClick={onReconnect}>Reconnect</Button>
          )}
          {canCancel && (
            <Button type="button" variant="destructive" disabled={isCancelling} leadingIcon={<X size={14} />} onClick={() => setConfirmCancel(true)}>
              Cancel run
            </Button>
          )}
        </div>
      </header>

      {error && <div className="run-monitor-alert"><Alert tone="error" title="Run monitoring issue">{error}</Alert></div>}
      {historyWarning && <div className="run-monitor-alert"><Alert tone="warning" title="Activity history notice">{historyWarning}</Alert></div>}
      {cancelNotice && <div className="run-monitor-alert"><Alert tone="info" title="Cancellation response">{cancelNotice}</Alert></div>}

      <div className="run-monitor-body">
        <div className="run-timeline-heading">
          <div><span className="section-overline">PERSISTED EVENTS</span><h3>Activity timeline</h3></div>
          <span>{events.length} {events.length === 1 ? "event" : "events"}</span>
        </div>
        {isLoading ? (
          <div className="run-timeline-loading" aria-label="Loading run activity"><span /><span /><span /></div>
        ) : events.length > 0 ? (
          <ActivityTimeline events={events} />
        ) : (
          <div className="run-timeline-empty">
            <Radio size={18} aria-hidden="true" />
            <strong>{terminal ? "No activity events were recorded" : "Waiting for the first run event"}</strong>
            <span>{terminal ? "The backend returned no persisted events for this run." : "New events will appear here as the backend records them."}</span>
          </div>
        )}
      </div>

      <Modal open={confirmCancel} onClose={() => setConfirmCancel(false)} title="Cancel this run?">
        <div className="run-cancel-dialog">
          <p>Cancellation is sent to the backend and may wait for the current operation to reach a safe checkpoint.</p>
          <p>The run will remain active until the backend confirms cancellation.</p>
          <div className="run-cancel-actions">
            <Button type="button" variant="ghost" disabled={isCancelling} onClick={() => setConfirmCancel(false)}>Keep running</Button>
            <Button type="button" variant="destructive" loading={isCancelling} onClick={() => { setConfirmCancel(false); onCancel(); }}>Confirm cancellation</Button>
          </div>
        </div>
      </Modal>
    </Card>
  );
}
