import { apiRequest } from "@/lib/api/client";
import type {
  AgentRole,
  ProviderAssignment,
  ProviderCatalogResponse,
  ProviderConnectionTest,
  ProviderProfile,
  ProviderProfileCreate,
  ProviderProfileUpdate,
} from "./types";

const providerPath = "settings/ai-providers";

export function getProviderCatalog(token: string, signal?: AbortSignal): Promise<ProviderCatalogResponse> {
  return apiRequest(`${providerPath}/catalog`, { token, signal });
}

export function listProviderProfiles(token: string, signal?: AbortSignal): Promise<ProviderProfile[]> {
  return apiRequest(`${providerPath}/profiles`, { token, signal });
}

export function createProviderProfile(
  token: string,
  payload: ProviderProfileCreate,
): Promise<ProviderProfile> {
  return apiRequest(`${providerPath}/profiles`, { token, method: "POST", body: payload });
}

export function updateProviderProfile(
  token: string,
  profileId: string,
  payload: ProviderProfileUpdate,
): Promise<ProviderProfile> {
  return apiRequest(`${providerPath}/profiles/${encodeURIComponent(profileId)}`, {
    token,
    method: "PATCH",
    body: payload,
  });
}

export function deleteProviderProfile(token: string, profileId: string): Promise<void> {
  return apiRequest(`${providerPath}/profiles/${encodeURIComponent(profileId)}`, {
    token,
    method: "DELETE",
  });
}

export function replaceProviderCredential(
  token: string,
  profileId: string,
  credential: string,
): Promise<ProviderProfile> {
  return apiRequest(`${providerPath}/profiles/${encodeURIComponent(profileId)}/credential`, {
    token,
    method: "PUT",
    body: { credential },
  });
}

export function removeProviderCredential(token: string, profileId: string): Promise<ProviderProfile> {
  return apiRequest(`${providerPath}/profiles/${encodeURIComponent(profileId)}/credential`, {
    token,
    method: "DELETE",
  });
}

export function testProviderConnection(
  token: string,
  profileId: string,
): Promise<ProviderConnectionTest> {
  return apiRequest(`${providerPath}/profiles/${encodeURIComponent(profileId)}/test-connection`, {
    token,
    method: "POST",
  });
}

export function listProviderAssignments(
  token: string,
  signal?: AbortSignal,
): Promise<ProviderAssignment[]> {
  return apiRequest(`${providerPath}/assignments`, { token, signal });
}

export function assignProvider(
  token: string,
  role: AgentRole,
  profileId: string,
): Promise<ProviderAssignment> {
  return apiRequest(`${providerPath}/assignments/${role}`, {
    token,
    method: "PUT",
    body: { profile_id: profileId },
  });
}

export function clearProviderAssignment(token: string, role: AgentRole): Promise<void> {
  return apiRequest(`${providerPath}/assignments/${role}`, { token, method: "DELETE" });
}
