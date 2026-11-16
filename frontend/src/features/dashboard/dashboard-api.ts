import { apiRequest } from "@/lib/api/client";
import type { DashboardSession, SessionDetails, SessionListItem } from "@/types/api";

const SESSION_PAGE_SIZE = 50;

export function listRecentSessions(token: string): Promise<SessionListItem[]> {
  return apiRequest<SessionListItem[]>(`/sessions?limit=${SESSION_PAGE_SIZE}&offset=0`, { token });
}

export async function listDashboardSessions(token: string): Promise<DashboardSession[]> {
  const sessions = await listRecentSessions(token);
  const metadataById = new Map<string, SessionDetails>();
  const recent = sessions.slice(0, 5);
  const details = await Promise.all(recent.map(async (session) => {
    try {
      return await apiRequest<SessionDetails>(`/sessions/${encodeURIComponent(session.id)}`, { token });
    } catch (error) {
      if (error instanceof Error && "status" in error) {
        const status = (error as Error & { status: number }).status;
        if (status === 401 || status === 403) throw error;
      }
      return null;
    }
  }));
  for (const detail of details) {
    if (detail) metadataById.set(detail.id, detail);
  }
  return sessions.map((session) => {
    const detail = metadataById.get(session.id);
    return detail ? { ...session, updated_at: detail.updated_at, dataset_meta: detail.dataset_meta } : session;
  });
}

export function uploadDatasetSession(
  token: string,
  values: { name: string; file: File; targetColumn: string },
): Promise<SessionDetails> {
  const body = new FormData();
  body.set("name", values.name);
  body.set("file", values.file);
  if (values.targetColumn.trim()) body.set("target_column", values.targetColumn.trim());

  return apiRequest<SessionDetails, FormData>("/sessions/upload", {
    method: "POST",
    token,
    body,
  });
}
