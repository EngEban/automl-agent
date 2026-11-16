"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import {
  cancelAgentRun,
  getAgentRun,
  getAgentRunEventPage,
} from "@/features/agent-workspace/agent-api";
import {
  consumeRunEventStream,
  isRunEvent,
  isTerminalRunStatus,
  openRunEventStream,
  readStoredRunId,
  removeStoredRunId,
  saveStoredRunId,
  subscribeStoredRunId,
  statusFromRunEvent,
} from "@/features/agent-workspace/run-monitoring";
import { RunEventStreamError } from "@/features/agent-workspace/run-monitoring";
import type { RunEventRead } from "@/types/api";
import type { AnalysisRunStatus } from "@/types/api";

export type RunStreamState = "idle" | "loading" | "connecting" | "connected" | "reconnecting" | "disconnected" | "closed" | "error";

interface TrackedRun {
  ownerId: string;
  runId: string;
  sessionId: string;
  generation: number;
  storageWarning: boolean;
}

interface RunMonitorViewState {
  identity: string | null;
  run: AnalysisRunStatus | null;
  events: RunEventRead[];
  streamState: RunStreamState;
  isLoading: boolean;
  monitorError: string;
  historyWarning: string;
  cancelNotice: string;
  isCancelling: boolean;
}

function createRunMonitorViewState(
  identity: string | null,
  loading = false,
  historyWarning = "",
): RunMonitorViewState {
  return {
    identity,
    run: null,
    events: [],
    streamState: loading ? "loading" : "idle",
    isLoading: loading,
    monitorError: "",
    historyWarning,
    cancelNotice: "",
    isCancelling: false,
  };
}

interface RunHistoryResult {
  events: RunEventRead[];
  truncated: boolean;
}

const MAX_HISTORY_PAGES = 50;
const MAX_RECONNECTS = 5;
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000, 15000];
const CANCELLABLE_STATUSES = new Set(["queued", "running", "awaiting_input", "awaiting_approval"]);
const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);

function safeMonitorError(error: unknown, area: "status" | "history" | "cancel"): string {
  if (error instanceof ApiTimeoutError) return "The request timed out. Check the connection and try again.";
  if (error instanceof ApiError) {
    if (error.status === 401 || error.status === 403) return "Your sign-in session expired. Sign in again to continue.";
    if (error.status === 404) return "This run is no longer available to this account.";
    if (error.status === 409 && area === "cancel") return "The run changed before cancellation was processed. Its current status was refreshed.";
    if (error.status >= 500) return "The run service is temporarily unavailable.";
  }
  if (error instanceof RunEventStreamError) {
    if (error.status === 401 || error.status === 403) return "Your sign-in session expired. Sign in again to continue.";
    if (error.status === 404) return "Run events are no longer available.";
  }
  if (error instanceof TypeError) return "The service could not be reached. Check your connection.";
  if (area === "history") return "Saved run activity could not be loaded. Live events will try to recover it.";
  if (area === "cancel") return "The cancellation request could not be completed.";
  return "Run status could not be loaded.";
}

function isUnauthorized(error: unknown): boolean {
  return error instanceof ApiError
    ? error.status === 401 || error.status === 403
    : error instanceof RunEventStreamError && (error.status === 401 || error.status === 403);
}

async function loadEventPages(
  token: string,
  runId: string,
  signal: AbortSignal,
  initialCursor?: string | null,
): Promise<RunHistoryResult> {
  const events: RunEventRead[] = [];
  let cursor = initialCursor || undefined;
  let truncated = false;

  for (let pageNumber = 0; pageNumber < MAX_HISTORY_PAGES; pageNumber += 1) {
    const page = await getAgentRunEventPage(token, runId, cursor, signal);
    if (!page || !Array.isArray(page.items) || typeof page.has_more !== "boolean") {
      throw new RunEventStreamError("The run history response was invalid.");
    }
    for (const event of page.items) {
      if (!isRunEvent(event, runId)) throw new RunEventStreamError("The run history contained an invalid event.");
      events.push(event);
    }
    if (!page.has_more) return { events, truncated: false };
    if (!page.next_cursor || page.next_cursor === cursor) {
      truncated = true;
      break;
    }
    cursor = page.next_cursor;
    if (pageNumber === MAX_HISTORY_PAGES - 1) truncated = true;
  }
  return { events, truncated };
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function waitForRetry(delayMs: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(false);
      return;
    }
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve(true);
    }, delayMs);
    const abort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      resolve(false);
    };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

export function useRunMonitor({
  token,
  ownerId,
  sessionId,
  onUnauthorized,
}: {
  token: string | null;
  ownerId: string | undefined;
  sessionId: string | undefined;
  onUnauthorized: () => void;
}) {
  const [manualRun, setManualRun] = useState<TrackedRun | null>(null);
  const [retryVersion, setRetryVersion] = useState(0);
  const runGeneration = useRef(0);
  const subscribe = useCallback((onStoreChange: () => void) => {
    if (!ownerId || !sessionId) return () => undefined;
    return subscribeStoredRunId(ownerId, sessionId, onStoreChange);
  }, [ownerId, sessionId]);
  const getSnapshot = useCallback(() => {
    if (!ownerId || !sessionId) return null;
    return readStoredRunId(ownerId, sessionId);
  }, [ownerId, sessionId]);
  const storedRunId = useSyncExternalStore(subscribe, getSnapshot, () => null);
  const trackedRun = useMemo<TrackedRun | null>(() => {
    if (manualRun && manualRun.ownerId === (ownerId ?? "") && manualRun.sessionId === sessionId) return manualRun;
    if (!ownerId || !sessionId || !storedRunId) return null;
    return { ownerId, runId: storedRunId, sessionId, generation: 0, storageWarning: false };
  }, [manualRun, ownerId, sessionId, storedRunId]);
  const monitorIdentity = trackedRun
    ? JSON.stringify([trackedRun.ownerId, trackedRun.sessionId, trackedRun.runId, trackedRun.generation, retryVersion])
    : null;
  const [monitorState, setMonitorState] = useState<RunMonitorViewState>(() => createRunMonitorViewState(null));
  const updateMonitor = useCallback((patch: Partial<Omit<RunMonitorViewState, "identity">>) => {
    if (!monitorIdentity) return;
    setMonitorState((current) => ({
      ...(current.identity === monitorIdentity
        ? current
        : createRunMonitorViewState(monitorIdentity, true, trackedRun?.storageWarning
          ? "This browser could not save the run ID for recovery after a page reload."
          : "")),
      ...patch,
      identity: monitorIdentity,
    }));
  }, [monitorIdentity, trackedRun?.storageWarning]);
  const visibleMonitorState = monitorState.identity === monitorIdentity
    ? monitorState
    : createRunMonitorViewState(
      monitorIdentity,
      Boolean(trackedRun),
      trackedRun?.storageWarning
        ? "This browser could not save the run ID for recovery after a page reload."
        : "",
    );
  const eventsRef = useRef<RunEventRead[]>([]);
  const eventIdsRef = useRef<Set<string>>(new Set());
  const lastEventIdRef = useRef<string | null>(null);
  const lastSequenceRef = useRef(0);
  const runRef = useRef<AnalysisRunStatus | null>(null);
  const activeStreamRef = useRef<AbortController | null>(null);
  const cancelLock = useRef(false);

  const startRun = useCallback((runId: string, runSessionId: string) => {
    const saved = ownerId ? saveStoredRunId(ownerId, runSessionId, runId) : false;
    activeStreamRef.current?.abort();
    runRef.current = null;
    eventsRef.current = [];
    eventIdsRef.current = new Set();
    lastEventIdRef.current = null;
    lastSequenceRef.current = 0;
    setManualRun({
      ownerId: ownerId ?? "",
      runId,
      sessionId: runSessionId,
      generation: ++runGeneration.current,
      storageWarning: Boolean(ownerId && !saved),
    });
  }, [ownerId]);

  const retry = useCallback(() => {
    setRetryVersion((current) => current + 1);
  }, []);

  useEffect(() => {
    if (!token || !trackedRun || trackedRun.sessionId !== sessionId) return;

    const lifecycle = new AbortController();
    let stopped = false;
    let streamController: AbortController | null = null;
    let eventHistoryComplete = false;
    const { runId, sessionId: trackedSessionId } = trackedRun;

    eventsRef.current = [];
    eventIdsRef.current = new Set();
    lastEventIdRef.current = null;
    lastSequenceRef.current = 0;
    runRef.current = null;

    const updateRun = (next: AnalysisRunStatus) => {
      if (stopped) return;
      runRef.current = next;
      updateMonitor({ run: next });
      if (isTerminalRunStatus(next.status)) {
        activeStreamRef.current?.abort();
        updateMonitor({ streamState: "closed" });
      }
    };

    const mergeEvents = (incoming: RunEventRead[]) => {
      if (stopped || incoming.length === 0) return;
      const merged = new Map(eventsRef.current.map((event) => [event.id, event]));
      for (const event of incoming) {
        if (!isRunEvent(event, runId)) continue;
        if (eventIdsRef.current.has(event.id)) continue;
        eventIdsRef.current.add(event.id);
        merged.set(event.id, event);
      }
      const ordered = [...merged.values()].sort((left, right) => left.sequence - right.sequence);
      eventsRef.current = ordered;
      updateMonitor({ events: ordered });
      const latest = ordered.at(-1);
      if (latest && latest.sequence > lastSequenceRef.current) {
        lastSequenceRef.current = latest.sequence;
        lastEventIdRef.current = latest.id;
      }
      const statusEvent = [...ordered].reverse().find((event) => statusFromRunEvent(event.event_type) !== null);
      const eventStatus = statusEvent ? statusFromRunEvent(statusEvent.event_type) : null;
      if (eventStatus && runRef.current && statusEvent && statusEvent.sequence >= lastSequenceRef.current) {
        updateRun({ ...runRef.current, status: eventStatus });
      } else if (eventStatus && runRef.current && isTerminalRunStatus(eventStatus)) {
        updateRun({ ...runRef.current, status: eventStatus });
      }
    };

    const recoverEventsAfter = async (cursor: string | null) => {
      const result = await loadEventPages(token, runId, lifecycle.signal, cursor);
      mergeEvents(result.events);
      if (result.truncated) {
        updateMonitor({ historyWarning: "Run activity is extensive. Some older events could not be loaded in this pass; the live stream will continue recovery." });
      }
    };

    const refreshStatusAfterDisconnect = async () => {
      const snapshot = await getAgentRun(token, runId, lifecycle.signal);
      updateRun(snapshot);
      if (isTerminalRunStatus(snapshot.status)) {
        await recoverEventsAfter(lastEventIdRef.current);
      }
      return snapshot;
    };

    const handleEventFrame = async (frame: { id: string; event: string; data: string }) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(frame.data) as unknown;
      } catch {
        throw new RunEventStreamError("A run event could not be parsed.");
      }
      if (!isRunEvent(parsed, runId)) throw new RunEventStreamError("The server sent an invalid run event.");
      if (frame.id && frame.id !== parsed.id) throw new RunEventStreamError("The run event ID did not match its event data.");
      if (frame.event !== "message" && frame.event !== parsed.event_type) {
        throw new RunEventStreamError("The run event type did not match its event data.");
      }

      if (lastSequenceRef.current > 0 && parsed.sequence > lastSequenceRef.current + 1) {
        updateMonitor({ historyWarning: "A gap in live events was detected. Recovering the missing events from saved history." });
        await recoverEventsAfter(lastEventIdRef.current);
        updateRun(await getAgentRun(token, runId, lifecycle.signal));
        if (lifecycle.signal.aborted || streamController?.signal.aborted) return;
      }
      mergeEvents([parsed]);
      const status = statusFromRunEvent(parsed.event_type);
      if (status && runRef.current && parsed.sequence >= lastSequenceRef.current) {
        updateRun({ ...runRef.current, status });
      }
    };

    const runStream = async (allowTerminalCatchup: boolean) => {
      let reconnects = 0;
      let catchupPending = allowTerminalCatchup;
      while (!stopped && !lifecycle.signal.aborted) {
        const currentStatus = runRef.current?.status ?? "queued";
        if (!catchupPending && isTerminalRunStatus(currentStatus)) {
          updateMonitor({ streamState: "closed" });
          return;
        }
        catchupPending = false;
        updateMonitor({ streamState: reconnects === 0 ? "connecting" : "reconnecting" });
        streamController = new AbortController();
        activeStreamRef.current = streamController;
        const abortStream = () => streamController?.abort();
        lifecycle.signal.addEventListener("abort", abortStream, { once: true });

        try {
          const response = await openRunEventStream(token, runId, lastEventIdRef.current, streamController.signal);
          if (stopped) return;
          updateMonitor({ streamState: "connected", monitorError: "" });
          await consumeRunEventStream(response, streamController.signal, handleEventFrame);
          if (stopped || lifecycle.signal.aborted) return;
          if (runRef.current && isTerminalRunStatus(runRef.current.status)) {
            updateMonitor({ streamState: "closed" });
            return;
          }
          throw new RunEventStreamError("The event stream ended before the run completed.");
        } catch (error) {
          if (stopped || lifecycle.signal.aborted || isAbort(error)) return;
          if (runRef.current && isTerminalRunStatus(runRef.current.status)) {
            updateMonitor({ streamState: "closed" });
            return;
          }
          if (isUnauthorized(error)) {
            onUnauthorized();
            updateMonitor({ streamState: "error", monitorError: safeMonitorError(error, "status") });
            return;
          }

          if ((error instanceof ApiError || error instanceof RunEventStreamError) && error.status === 400 && lastEventIdRef.current) {
            lastEventIdRef.current = null;
            updateMonitor({ historyWarning: "The saved event cursor was rejected. Replaying persisted events and removing duplicates." });
          }

          try {
            await refreshStatusAfterDisconnect();
          } catch (statusError) {
            if (isUnauthorized(statusError)) {
              onUnauthorized();
              updateMonitor({ streamState: "error", monitorError: safeMonitorError(statusError, "status") });
              return;
            }
            if (statusError instanceof ApiError && statusError.status === 404) {
              removeStoredRunId(ownerId ?? "", trackedSessionId);
              setManualRun((current) => current?.generation === trackedRun.generation
                && current.ownerId === trackedRun.ownerId
                && current.runId === runId ? null : current);
              updateMonitor({ streamState: "error", monitorError: safeMonitorError(statusError, "status") });
              return;
            }
          }
          if (runRef.current && isTerminalRunStatus(runRef.current.status)) {
            updateMonitor({ streamState: "closed" });
            return;
          }
          if (reconnects >= MAX_RECONNECTS) {
            updateMonitor({
              streamState: "disconnected",
              monitorError: "Live updates could not reconnect after several attempts. Check the run status or reconnect manually.",
            });
            return;
          }
          updateMonitor({
            streamState: "reconnecting",
            monitorError: "Live updates disconnected. Reconnecting with the last processed event.",
          });
          const delay = RECONNECT_DELAYS_MS[reconnects] ?? RECONNECT_DELAYS_MS.at(-1) ?? 15000;
          reconnects += 1;
          const shouldRetry = await waitForRetry(delay, lifecycle.signal);
          if (!shouldRetry) return;
        } finally {
          lifecycle.signal.removeEventListener("abort", abortStream);
          if (activeStreamRef.current === streamController) activeStreamRef.current = null;
        }
      }
    };

    const initialize = async () => {
      try {
        const snapshot = await getAgentRun(token, runId, lifecycle.signal);
        if (stopped) return;
        if (snapshot.id !== runId || snapshot.session_id !== trackedSessionId) {
          removeStoredRunId(ownerId ?? "", trackedSessionId);
          setManualRun((current) => current?.generation === trackedRun.generation
            && current.ownerId === trackedRun.ownerId
            && current.runId === runId ? null : current);
          updateMonitor({
            isLoading: false,
            streamState: "error",
            monitorError: "This run does not belong to the selected dataset session.",
          });
          return;
        }
        updateRun(snapshot);

        let historyFailed = false;
        try {
          const history = await loadEventPages(token, runId, lifecycle.signal);
          if (stopped) return;
          mergeEvents(history.events);
          eventHistoryComplete = !history.truncated;
          if (history.truncated) {
            updateMonitor({ historyWarning: "Run activity spans many pages. Remaining events will continue to load from the live stream." });
          }
        } catch (error) {
          if (stopped) return;
          if (isUnauthorized(error)) {
            onUnauthorized();
            updateMonitor({
              monitorError: safeMonitorError(error, "history"),
              streamState: "error",
              isLoading: false,
            });
            return;
          }
          historyFailed = true;
          updateMonitor({ historyWarning: safeMonitorError(error, "history") });
        }
        updateMonitor({ isLoading: false });

        const latestStatusEvent = [...eventsRef.current].reverse()
          .map((event) => statusFromRunEvent(event.event_type))
          .find((status): status is string => status !== null);
        const historyStatus = latestStatusEvent;
        const statusToUse = historyStatus && eventHistoryComplete ? historyStatus : snapshot.status;
        if (statusToUse !== snapshot.status) updateRun({ ...snapshot, status: statusToUse });
        const hasCatchup = historyFailed || !eventHistoryComplete;
        if (!hasCatchup && isTerminalRunStatus(statusToUse)) {
          updateMonitor({ streamState: "closed" });
          return;
        }
        await runStream(hasCatchup && isTerminalRunStatus(statusToUse));
      } catch (error) {
        if (stopped || lifecycle.signal.aborted) return;
        if (isUnauthorized(error)) onUnauthorized();
        if (error instanceof ApiError && error.status === 404) {
          removeStoredRunId(ownerId ?? "", trackedSessionId);
          setManualRun((current) => current?.generation === trackedRun.generation
            && current.ownerId === trackedRun.ownerId
            && current.runId === runId ? null : current);
        }
        updateMonitor({
          isLoading: false,
          streamState: "error",
          monitorError: safeMonitorError(error, "status"),
        });
      }
    };

    void initialize();
    return () => {
      stopped = true;
      lifecycle.abort();
      streamController?.abort();
      if (activeStreamRef.current === streamController) activeStreamRef.current = null;
    };
  }, [onUnauthorized, ownerId, retryVersion, sessionId, token, trackedRun, updateMonitor]);

  const cancelRun = useCallback(async () => {
    if (!token || !runRef.current || cancelLock.current || !CANCELLABLE_STATUSES.has(runRef.current.status)) return;
    const runId = runRef.current.id;
    cancelLock.current = true;
    updateMonitor({ isCancelling: true, cancelNotice: "" });
    try {
      const response = await cancelAgentRun(token, runId);
      const latest = runRef.current;
      if (latest?.id === response.run_id) {
        const updated = { ...latest, status: response.status };
        runRef.current = updated;
        updateMonitor({ run: updated });
        if (isTerminalRunStatus(updated.status)) {
          activeStreamRef.current?.abort();
          updateMonitor({ streamState: "closed" });
        }
      }
      updateMonitor({ cancelNotice: response.message });
    } catch (error) {
      if (isUnauthorized(error)) onUnauthorized();
      if (error instanceof ApiError && error.status === 409 && token) {
        try {
          const latest = await getAgentRun(token, runId);
          runRef.current = latest;
          updateMonitor({ run: latest });
          if (isTerminalRunStatus(latest.status)) {
            activeStreamRef.current?.abort();
            updateMonitor({ streamState: "closed" });
          }
          updateMonitor({ cancelNotice: "The run changed before cancellation was processed. Its latest status is shown above." });
        } catch {
          updateMonitor({ cancelNotice: "The run changed before cancellation was processed. Refresh its status and try again if it is still active." });
        }
      } else {
        updateMonitor({ cancelNotice: safeMonitorError(error, "cancel") });
      }
    } finally {
      cancelLock.current = false;
      updateMonitor({ isCancelling: false });
    }
  }, [onUnauthorized, token, updateMonitor]);

  const isForCurrentSession = Boolean(sessionId && trackedRun?.sessionId === sessionId);
  const currentRun = isForCurrentSession ? visibleMonitorState.run : null;
  const currentEvents = isForCurrentSession ? visibleMonitorState.events : [];
  const hasTrackedRun = isForCurrentSession && Boolean(trackedRun);
  const blocksSubmission = Boolean(hasTrackedRun && (!currentRun || ACTIVE_STATUSES.has(currentRun.status)));
  const canCancel = Boolean(currentRun && CANCELLABLE_STATUSES.has(currentRun.status));

  return {
    run: currentRun,
    runId: isForCurrentSession ? trackedRun?.runId ?? null : null,
    events: currentEvents,
    streamState: isForCurrentSession ? visibleMonitorState.streamState : "idle" as RunStreamState,
    isLoading: isForCurrentSession && visibleMonitorState.isLoading,
    error: isForCurrentSession ? visibleMonitorState.monitorError : "",
    historyWarning: isForCurrentSession ? visibleMonitorState.historyWarning : "",
    cancelNotice: isForCurrentSession ? visibleMonitorState.cancelNotice : "",
    isCancelling: isForCurrentSession && visibleMonitorState.isCancelling,
    blocksSubmission,
    canCancel,
    startRun,
    retry,
    cancelRun,
  };
}
