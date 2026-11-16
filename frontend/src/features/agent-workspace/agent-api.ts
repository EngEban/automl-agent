import { apiRequest } from "@/lib/api/client";
import type {
  AgentMessageRead,
  AnalysisRunStatus,
  ChatAcceptedResponse,
  ChatRequest,
  RunCancellationResponse,
  RunEventHistoryPage,
  SessionDetails,
  SessionListItem,
} from "@/types/api";

export function listAgentSessions(
  token: string,
  limit = 50,
  offset = 0,
  signal?: AbortSignal,
): Promise<SessionListItem[]> {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return apiRequest<SessionListItem[]>(`/sessions?${query.toString()}`, { token, signal });
}

export function getAgentSession(token: string, sessionId: string, signal?: AbortSignal): Promise<SessionDetails> {
  return apiRequest<SessionDetails>(`/sessions/${encodeURIComponent(sessionId)}`, { token, signal });
}

export function listAgentMessages(token: string, sessionId: string): Promise<AgentMessageRead[]> {
  return apiRequest<AgentMessageRead[]>(
    `/chat/${encodeURIComponent(sessionId)}/messages?limit=200`,
    { token },
  );
}

export function sendAgentMessage(
  token: string,
  sessionId: string,
  request: ChatRequest,
): Promise<ChatAcceptedResponse> {
  return apiRequest<ChatAcceptedResponse, ChatRequest>(`/chat/${encodeURIComponent(sessionId)}`, {
    method: "POST",
    token,
    body: request,
    headers: { "Idempotency-Key": globalThis.crypto.randomUUID() },
  });
}

export function getAgentRun(token: string, runId: string, signal?: AbortSignal): Promise<AnalysisRunStatus> {
  return apiRequest<AnalysisRunStatus>(`/runs/${encodeURIComponent(runId)}`, { token, signal });
}

export function getAgentRunEventPage(
  token: string,
  runId: string,
  cursor?: string,
  signal?: AbortSignal,
): Promise<RunEventHistoryPage> {
  const query = new URLSearchParams({ limit: "200" });
  if (cursor) query.set("cursor", cursor);
  return apiRequest<RunEventHistoryPage>(
    `/runs/${encodeURIComponent(runId)}/events/history?${query.toString()}`,
    { token, signal },
  );
}

export function cancelAgentRun(token: string, runId: string): Promise<RunCancellationResponse> {
  return apiRequest<RunCancellationResponse>(`/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
    token,
  });
}
