import { API_BASE_URL } from "@/lib/api/client";
import type { RunEventRead } from "@/types/api";

export interface SseFrame {
  id: string;
  event: string;
  data: string;
}

export class RunEventStreamError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "RunEventStreamError";
  }
}

const MAX_SSE_BUFFER_CHARS = 128 * 1024;
const TERMINAL_STATUSES = new Set(["succeeded", "partially_completed", "failed", "cancelled"]);
const RUN_STORAGE_EVENT = "automl-agent:run-storage-change";

export function runStorageKey(ownerId: string, sessionId: string): string {
  return `automl-agent:last-run:${ownerId}:${sessionId}`;
}

export function readStoredRunId(ownerId: string, sessionId: string): string | null {
  try {
    const runId = window.sessionStorage.getItem(runStorageKey(ownerId, sessionId));
    return runId && /^[0-9a-f-]{36}$/i.test(runId) ? runId : null;
  } catch {
    return null;
  }
}

export function subscribeStoredRunId(ownerId: string, sessionId: string, onChange: () => void): () => void {
  const key = runStorageKey(ownerId, sessionId);
  const onStorage = (event: StorageEvent) => {
    if (event.key === key || event.key === null) onChange();
  };
  const onRunStorage = (event: Event) => {
    const detail = (event as CustomEvent<{ key?: unknown }>).detail;
    if (detail?.key === key) onChange();
  };

  window.addEventListener("storage", onStorage);
  window.addEventListener(RUN_STORAGE_EVENT, onRunStorage);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(RUN_STORAGE_EVENT, onRunStorage);
  };
}

function notifyRunStorageChange(key: string): void {
  window.dispatchEvent(new CustomEvent(RUN_STORAGE_EVENT, { detail: { key } }));
}

export function saveStoredRunId(ownerId: string, sessionId: string, runId: string): boolean {
  try {
    const key = runStorageKey(ownerId, sessionId);
    window.sessionStorage.setItem(key, runId);
    notifyRunStorageChange(key);
    return true;
  } catch {
    return false;
  }
}

export function removeStoredRunId(ownerId: string, sessionId: string): void {
  try {
    const key = runStorageKey(ownerId, sessionId);
    window.sessionStorage.removeItem(key);
    notifyRunStorageChange(key);
  } catch {
    // Storage can be disabled by browser policy; monitoring remains available for this page view.
  }
}

export function isRunEvent(value: unknown, runId: string): value is RunEventRead {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const item = value as Record<string, unknown>;
  return typeof item.id === "string"
    && item.run_id === runId
    && typeof item.sequence === "number"
    && Number.isInteger(item.sequence)
    && item.sequence > 0
    && typeof item.event_type === "string"
    && (typeof item.step === "string" || item.step === null)
    && typeof item.progress_percent === "number"
    && typeof item.payload === "object"
    && item.payload !== null
    && !Array.isArray(item.payload)
    && typeof item.created_at === "string"
    && (typeof item.message === "string" || item.message === null);
}

export function statusFromRunEvent(eventType: string): string | null {
  switch (eventType) {
    case "run.created":
    case "run.queued": return "queued";
    case "run.started": return "running";
    case "run.awaiting_input": return "awaiting_input";
    case "run.awaiting_approval": return "awaiting_approval";
    case "run.cancellation_requested": return "cancelling";
    case "run.succeeded": return "succeeded";
    case "run.partially_completed": return "partially_completed";
    case "run.failed": return "failed";
    case "run.cancelled": return "cancelled";
    default: return null;
  }
}

export function isTerminalRunStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export async function openRunEventStream(
  token: string,
  runId: string,
  lastEventId: string | null,
  signal: AbortSignal,
): Promise<Response> {
  const headers = new Headers({
    Accept: "text/event-stream",
    Authorization: `Bearer ${token}`,
  });
  if (lastEventId) headers.set("Last-Event-ID", lastEventId);

  const response = await fetch(`${API_BASE_URL}/runs/${encodeURIComponent(runId)}/events`, {
    method: "GET",
    headers,
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    throw new RunEventStreamError(
      response.status === 401 || response.status === 403
        ? "Authentication is required to read run events."
        : response.status === 404
          ? "Run events are unavailable."
          : "The run event stream could not be opened.",
      response.status,
    );
  }
  if (!response.body || !response.headers.get("content-type")?.toLowerCase().includes("text/event-stream")) {
    throw new RunEventStreamError("The server did not return a run event stream.");
  }
  return response;
}

export async function consumeRunEventStream(
  response: Response,
  signal: AbortSignal,
  onFrame: (frame: SseFrame) => void | Promise<void>,
): Promise<void> {
  if (!response.body) throw new RunEventStreamError("The run event stream has no readable body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];
  let frameDataChars = 0;
  let eventId = "";
  let eventName = "";

  const resetFrame = () => {
    dataLines = [];
    frameDataChars = 0;
    eventId = "";
    eventName = "";
  };

  const dispatchFrame = async () => {
    if (dataLines.length > 0) {
      await onFrame({
        id: eventId,
        event: eventName || "message",
        data: dataLines.join("\n"),
      });
    }
    resetFrame();
  };

  const processLine = async (line: string) => {
    if (line.length > MAX_SSE_BUFFER_CHARS) throw new RunEventStreamError("A run event line exceeded the supported size.");
    if (line === "") {
      await dispatchFrame();
      return;
    }
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "data") {
      frameDataChars += value.length;
      if (frameDataChars > MAX_SSE_BUFFER_CHARS) throw new RunEventStreamError("A run event frame exceeded the supported size.");
      dataLines.push(value);
    } else if (field === "event") eventName = value;
    else if (field === "id" && !value.includes("\0")) eventId = value;
  };

  const consumeCompleteLines = async (atEnd: boolean) => {
    while (buffer.length > 0) {
      let lineEnd = -1;
      let delimiterLength = 0;
      for (let index = 0; index < buffer.length; index += 1) {
        const character = buffer[index];
        if (character === "\n") {
          lineEnd = index;
          delimiterLength = 1;
          break;
        }
        if (character === "\r") {
          if (index === buffer.length - 1 && !atEnd) break;
          lineEnd = index;
          delimiterLength = buffer[index + 1] === "\n" ? 2 : 1;
          break;
        }
      }
      if (lineEnd < 0) {
        if (buffer.length > MAX_SSE_BUFFER_CHARS) throw new RunEventStreamError("A run event frame exceeded the supported size.");
        if (atEnd && buffer.length > 0) {
          const finalLine = buffer;
          buffer = "";
          await processLine(finalLine);
          if (signal.aborted) return;
          await dispatchFrame();
        }
        return;
      }
      const line = buffer.slice(0, lineEnd);
      buffer = buffer.slice(lineEnd + delimiterLength);
      await processLine(line);
      if (signal.aborted) return;
    }
  };

  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        await consumeCompleteLines(true);
        if (dataLines.length > 0) await dispatchFrame();
        return;
      }
      buffer += decoder.decode(value, { stream: true });
      await consumeCompleteLines(false);
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
