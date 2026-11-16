import { apiDownloadFile, apiRequest } from "@/lib/api/client";
import type { SessionArtifact, SessionDetails, SessionListItem } from "@/types/api";

export interface ReportRecord {
  artifactId: string;
  sessionId: string;
  sessionName: string;
  sessionStatus: string;
  sessionCreatedAt: string;
}

/**
 * The artifacts endpoint returns rows in creation order, while its download
 * route resolves only the latest report for a session. Keep that same latest
 * report row as the one actionable item for each session.
 */
export async function getLatestSessionReport(
  token: string,
  session: SessionListItem | SessionDetails,
  signal?: AbortSignal,
): Promise<ReportRecord | null> {
  const artifacts = await apiRequest<SessionArtifact[]>(
    `/artifacts/${encodeURIComponent(session.id)}`,
    { token, signal },
  );
  const report = artifacts.filter((artifact) => artifact.artifact_type === "report").at(-1);
  if (!report) return null;

  return {
    artifactId: report.id,
    sessionId: session.id,
    sessionName: session.name,
    sessionStatus: session.status,
    sessionCreatedAt: session.created_at,
  };
}

export function downloadSessionReport(
  token: string,
  sessionId: string,
  signal?: AbortSignal,
) {
  return apiDownloadFile(
    `/artifacts/${encodeURIComponent(sessionId)}/report`,
    { token, signal },
  );
}

export async function isPdfFile(blob: Blob): Promise<boolean> {
  return (await blob.slice(0, 5).text()) === "%PDF-";
}
