import { apiDownload, apiRequest } from "@/lib/api/client";
import type { RegisteredModelVersion, SessionArtifact } from "@/types/api";

export type DownloadableArtifactType = "report" | "notebook" | "cleaned_data";

const artifactDownloadPaths: Record<DownloadableArtifactType, string> = {
  report: "report",
  notebook: "notebook",
  cleaned_data: "cleaned-dataset",
};

export function listSessionModelVersions(
  token: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<RegisteredModelVersion[]> {
  return apiRequest<RegisteredModelVersion[]>(
    `/predict/models/${encodeURIComponent(sessionId)}`,
    { token, signal },
  );
}

export function listSessionArtifacts(
  token: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<SessionArtifact[]> {
  return apiRequest<SessionArtifact[]>(`/artifacts/${encodeURIComponent(sessionId)}`, { token, signal });
}

export function downloadSessionArtifact(
  token: string,
  sessionId: string,
  artifactType: DownloadableArtifactType,
  signal?: AbortSignal,
): Promise<Blob> {
  return apiDownload(
    `/artifacts/${encodeURIComponent(sessionId)}/${artifactDownloadPaths[artifactType]}`,
    { token, signal },
  );
}
