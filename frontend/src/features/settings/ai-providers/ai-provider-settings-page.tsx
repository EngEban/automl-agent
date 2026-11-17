"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";
import {
  Check, Cloud, Cpu, KeyRound, Pencil, Plus, RefreshCw, ShieldCheck,
  Trash2, Unplug, Zap,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert, Skeleton } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { TextInput } from "@/components/ui/form-controls";
import { Badge, Card } from "@/components/ui/surfaces";
import { Modal } from "@/components/ui/modal";
import { useAuth } from "@/features/auth/auth-provider";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import {
  assignProvider,
  clearProviderAssignment,
  createProviderProfile,
  deleteProviderProfile,
  getProviderCatalog,
  listProviderAssignments,
  listProviderProfiles,
  removeProviderCredential,
  replaceProviderCredential,
  testProviderConnection,
  updateProviderProfile,
} from "./api";
import type {
  AgentRole,
  CapabilityState,
  ProviderAssignment,
  ProviderCatalogEntry,
  ProviderConnectionTest,
  ProviderProfile,
  ProviderProfileCreate,
  ProviderProfileUpdate,
  SettingsNotice,
} from "./types";

interface SettingsData {
  catalog: ProviderCatalogEntry[];
  profiles: ProviderProfile[];
  assignments: ProviderAssignment[];
}

type ProfileDraft = ProviderProfileCreate;

type Confirmation =
  | { kind: "delete-profile"; profile: ProviderProfile }
  | { kind: "remove-credential"; profile: ProviderProfile }
  | { kind: "clear-assignment"; role: AgentRole };

const ROLE_LABEL: Record<AgentRole, string> = {
  pipeline_agent: "Pipeline Agent",
  report_agent: "Report Agent",
};

const CAPABILITY_LABEL: Record<string, string> = {
  chat_completion: "Chat",
  tool_calling: "Tool calling",
  structured_output: "Structured output",
  json_schema: "JSON schema",
  streaming: "Streaming",
};

const CONFIGURATION_LABEL: Record<string, string> = {
  temperature: "Temperature",
  max_tokens: "Maximum tokens",
  top_p: "Top P",
  frequency_penalty: "Frequency penalty",
  presence_penalty: "Presence penalty",
};

function capabilityText(value: CapabilityState | undefined): string {
  if (value === "supported") return "Supported";
  if (value === "unsupported") return "Unsupported";
  return "Unknown · not tested";
}

function capabilityTone(value: CapabilityState | undefined): "green" | "red" | "amber" {
  if (value === "supported") return "green";
  if (value === "unsupported") return "red";
  return "amber";
}

function errorMessage(error: unknown, secret = ""): string {
  let message = "The request could not be completed. Please try again.";
  if (error instanceof ApiTimeoutError) {
    message = "The request timed out. Check the service connection and try again.";
  } else if (error instanceof ApiError) {
    if (error.status === 401) message = "Your sign-in has expired. Sign in again to manage AI providers.";
    else if (error.status === 403) message = "You do not have permission to manage this provider.";
    else if (error.status === 409 || error.status === 422) message = error.message;
    else if (error.status >= 500) message = "The provider service is temporarily unavailable. Please try again.";
    else message = error.message;
  } else if (error instanceof TypeError) {
    message = "We could not reach the API. Check your connection and try again.";
  }
  return secret ? message.split(secret).join("[redacted]") : message;
}

function noticeForError(error: unknown): SettingsNotice {
  const expired = error instanceof ApiError && error.status === 401;
  return {
    tone: "error",
    title: expired ? "Sign-in required" : "Could not complete the request",
    message: errorMessage(error),
    sessionExpired: expired,
  };
}

function displayDate(value: string | null): string {
  if (!value) return "Not tested yet";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function roleRequirements(
  data: SettingsData,
  profile: ProviderProfile | undefined,
  role: AgentRole,
): string[] {
  const provider = data.catalog.find((entry) => entry.provider_id === profile?.provider_id)
    ?? data.catalog[0];
  return provider?.agent_requirements[role] ?? [];
}

function isAssignable(
  profile: ProviderProfile,
  role: AgentRole,
  data: SettingsData,
): { eligible: boolean; reason: string; unverified: string[] } {
  const provider = data.catalog.find((entry) => entry.provider_id === profile.provider_id);
  if (!provider) return { eligible: false, reason: "Provider metadata is unavailable.", unverified: [] };
  if (provider.api_key_requirement === "required" && !profile.credential_configured) {
    return { eligible: false, reason: "Configure its credential first.", unverified: [] };
  }
  if (profile.connection_status !== "connected") {
    return { eligible: false, reason: "Test the connection successfully first.", unverified: [] };
  }
  const required = provider.agent_requirements[role] ?? [];
  if (required.some((name) => profile.capability_status[name] === "unsupported")) {
    return { eligible: false, reason: "A required capability is unsupported.", unverified: [] };
  }
  return {
    eligible: true,
    reason: "",
    unverified: required.filter((name) => (profile.capability_status[name] ?? "unknown") === "unknown"),
  };
}

function SessionNotice({ notice }: { notice: SettingsNotice }) {
  return (
    <div className="provider-notice">
      <Alert tone={notice.tone} title={notice.title}>{notice.message}</Alert>
      {notice.sessionExpired && <Link className="provider-signin-link" href="/login">Go to sign in</Link>}
    </div>
  );
}

async function loadSettingsData(token: string, signal?: AbortSignal): Promise<SettingsData> {
  const [catalog, profiles, assignments] = await Promise.all([
    getProviderCatalog(token, signal),
    listProviderProfiles(token, signal),
    listProviderAssignments(token, signal),
  ]);
  return { catalog: catalog.providers, profiles, assignments };
}

export function AIProviderSettingsPage() {
  const { accessToken } = useAuth();
  const [data, setData] = useState<SettingsData | null>(null);
  const [loadNotice, setLoadNotice] = useState<SettingsNotice | null>(null);
  const [notice, setNotice] = useState<SettingsNotice | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  const [editor, setEditor] = useState<ProviderProfile | null | undefined>(undefined);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [assignmentStateStale, setAssignmentStateStale] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, ProviderConnectionTest>>({});
  const [roleSelections, setRoleSelections] = useState<Partial<Record<AgentRole, string>>>({});

  useEffect(() => {
    const controller = new AbortController();
    if (!accessToken) return () => controller.abort();
    loadSettingsData(accessToken, controller.signal)
      .then((nextData) => {
        setData(nextData);
        setAssignmentStateStale(false);
        setRoleSelections({});
        setLoadNotice(null);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setLoadNotice(noticeForError(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [accessToken, reloadKey]);

  const assignmentsByRole = useMemo(() => {
    const map = new Map<AgentRole, ProviderAssignment>();
    data?.assignments.forEach((assignment) => map.set(assignment.agent_role, assignment));
    return map;
  }, [data?.assignments]);

  function mergeProfile(profile: ProviderProfile) {
    setData((current) => {
      if (!current) return current;
      const exists = current.profiles.some((item) => item.id === profile.id);
      const profiles = exists
        ? current.profiles.map((item) => item.id === profile.id ? profile : item)
        : [...current.profiles, profile];
      return { ...current, profiles };
    });
  }

  function markAssignmentStateStale() {
    setAssignmentStateStale(true);
    setRoleSelections({});
  }

  async function refreshCanonicalState(token: string): Promise<boolean> {
    markAssignmentStateStale();
    try {
      const [profiles, assignments] = await Promise.all([
        listProviderProfiles(token),
        listProviderAssignments(token),
      ]);
      setData((current) => current ? { ...current, profiles, assignments } : current);
      setAssignmentStateStale(false);
      setRoleSelections({});
      return true;
    } catch {
      return false;
    }
  }

  function assignmentSyncNotice(): SettingsNotice {
    return {
      tone: "warning",
      title: "Assignment state could not be verified",
      message: "The latest provider request could not be fully confirmed, and Agent assignments could not be reloaded. Retry the refresh before changing an assignment.",
    };
  }

  async function retryAssignmentRefresh() {
    if (!accessToken) return;
    setPendingAction("refresh-assignments");
    setNotice(null);
    try {
      const refreshed = await refreshCanonicalState(accessToken);
      setNotice(refreshed
        ? { tone: "success", title: "Assignments refreshed", message: "Provider profiles and Agent assignments now match the backend." }
        : assignmentSyncNotice());
    } finally {
      setPendingAction(null);
    }
  }

  async function saveProfile(draft: ProfileDraft, existingProfile: ProviderProfile | null) {
    if (!accessToken) throw new Error("Sign in again to manage AI providers.");
    const { credential, ...profileFields } = draft;
    try {
      if (existingProfile) {
        const updates: ProviderProfileUpdate = {
          display_name: profileFields.display_name,
          model_identifier: profileFields.model_identifier,
          endpoint_url: profileFields.endpoint_url,
          configuration: profileFields.configuration,
        };
        const updated = await updateProviderProfile(accessToken, existingProfile.id, updates);
        markAssignmentStateStale();
        mergeProfile(updated);
        if (credential) {
          const credentialUpdated = await replaceProviderCredential(accessToken, existingProfile.id, credential);
          markAssignmentStateStale();
          mergeProfile(credentialUpdated);
        }
      } else {
        const created = await createProviderProfile(accessToken, draft);
        markAssignmentStateStale();
        mergeProfile(created);
      }
      const refreshed = await refreshCanonicalState(accessToken);
      setNotice({
        ...(refreshed
          ? {
            tone: "success" as const,
            title: existingProfile ? "Provider profile saved" : "Provider added",
            message: existingProfile
              ? "The saved profile and Agent assignments now match the backend."
              : "The provider was added. Test its connection before assigning it to an Agent.",
          }
          : assignmentSyncNotice()),
      });
      setEditor(undefined);
    } catch (error) {
      const refreshed = await refreshCanonicalState(accessToken);
      if (!refreshed) setNotice(assignmentSyncNotice());
      throw error;
    }
  }

  async function runConnectionTest(profile: ProviderProfile) {
    if (!accessToken) return;
    setPendingAction("test:" + profile.id);
    setNotice(null);
    try {
      const result = await testProviderConnection(accessToken, profile.id);
      setTestResults((current) => ({ ...current, [profile.id]: result }));
      markAssignmentStateStale();
      const testedProfile = {
        ...profile,
        connection_status: result.status,
        connection_error_code: result.status === "failed" ? result.category : null,
        capability_status: result.capabilities,
        last_connection_checked_at: result.checked_at,
      } satisfies ProviderProfile;
      mergeProfile(testedProfile);
      const refreshed = await refreshCanonicalState(accessToken);
      setNotice(!refreshed
        ? assignmentSyncNotice()
        : {
          tone: result.status === "connected" ? "success" : "error",
          title: result.status === "connected" ? "Connection test completed" : "Connection test failed",
          message: result.message,
        });
    } catch (error) {
      await refreshCanonicalState(accessToken);
      setNotice(noticeForError(error));
    } finally {
      setPendingAction(null);
    }
  }

  async function saveAssignment(role: AgentRole, profileId: string) {
    if (!accessToken) return;
    setPendingAction("assignment:" + role);
    setNotice(null);
    try {
      const assignment = await assignProvider(accessToken, role, profileId);
      const refreshed = await refreshCanonicalState(accessToken);
      setNotice(!refreshed
        ? assignmentSyncNotice()
        : {
          tone: assignment.unverified_capabilities.length ? "warning" : "success",
          title: ROLE_LABEL[role] + " assignment saved",
          message: assignment.unverified_capabilities.length
            ? "Assignment saved. Some required capabilities remain unverified: " + assignment.unverified_capabilities.map((name) => CAPABILITY_LABEL[name] ?? name).join(", ") + "."
            : "The selected provider is assigned to this Agent.",
        });
    } catch (error) {
      await refreshCanonicalState(accessToken);
      setNotice(noticeForError(error));
    } finally {
      setPendingAction(null);
    }
  }

  async function confirmAction() {
    if (!accessToken || !confirmation) return;
    const action = confirmation;
    setPendingAction("confirm");
    setNotice(null);
    try {
      if (action.kind === "delete-profile") {
        await deleteProviderProfile(accessToken, action.profile.id);
        markAssignmentStateStale();
        setData((current) => current ? {
          ...current,
          profiles: current.profiles.filter((profile) => profile.id !== action.profile.id),
        } : current);
        const refreshed = await refreshCanonicalState(accessToken);
        setNotice(refreshed
          ? { tone: "success", title: "Provider removed", message: "The provider profile and related assignments were refreshed from the backend." }
          : assignmentSyncNotice());
      } else if (action.kind === "remove-credential") {
        const updated = await removeProviderCredential(accessToken, action.profile.id);
        markAssignmentStateStale();
        mergeProfile(updated);
        const refreshed = await refreshCanonicalState(accessToken);
        setNotice(refreshed
          ? { tone: "success", title: "Credential removed", message: "The credential and Agent assignments were refreshed from the backend." }
          : assignmentSyncNotice());
      } else {
        await clearProviderAssignment(accessToken, action.role);
        markAssignmentStateStale();
        const refreshed = await refreshCanonicalState(accessToken);
        setNotice(refreshed
          ? { tone: "success", title: "Assignment cleared", message: ROLE_LABEL[action.role] + " assignment state now matches the backend." }
          : assignmentSyncNotice());
      }
      setConfirmation(null);
    } catch (error) {
      await refreshCanonicalState(accessToken);
      setNotice(noticeForError(error));
    } finally {
      setPendingAction(null);
    }
  }

  const roleNames: AgentRole[] = ["pipeline_agent", "report_agent"];
  const confirmationTitle = confirmation?.kind === "delete-profile"
    ? "Remove provider profile?"
    : confirmation?.kind === "remove-credential"
      ? "Remove saved credential?"
      : "Clear Agent assignment?";

  return (
    <AppShell
      title="AI provider settings"
      eyebrow="SETTINGS"
      description="Connect your own models and choose which provider powers each Agent."
      breadcrumbs={[{ label: "Workspace", href: "/workspace" }, { label: "Settings" }, { label: "AI providers" }]}
      activeItem="Settings"
      homeHref="/workspace"
      statusLabel="Provider configuration"
      actions={<Button onClick={() => { setNotice(null); setEditor(null); }} leadingIcon={<Plus size={16} />} disabled={!data}>Add provider</Button>}
    >
      <div className="provider-settings-page">
        {notice && <SessionNotice notice={notice} />}
        {loadNotice && <SessionNotice notice={loadNotice} />}
        {loadNotice && (
          <Button variant="outline" onClick={() => { setLoading(true); setReloadKey((key) => key + 1); }} leadingIcon={<RefreshCw size={15} />}>
            Retry loading
          </Button>
        )}
        {loading && !data ? (
          <div className="provider-loading-list" aria-label="Loading AI provider settings">
            <Card><Skeleton /><Skeleton /><Skeleton /></Card>
            <Card><Skeleton /><Skeleton /></Card>
          </div>
        ) : data && (
          <>
            <section className="provider-section" aria-labelledby="provider-profiles-title">
              <div className="provider-section-heading">
                <div>
                  <p className="section-overline">YOUR CONFIGURATION</p>
                  <h2 id="provider-profiles-title">Provider profiles</h2>
                  <p>Credentials stay private. Capability checks run only when you request them.</p>
                </div>
                <span className="provider-count">{data.profiles.length} {data.profiles.length === 1 ? "profile" : "profiles"}</span>
              </div>
              {!data.profiles.length ? (
                <Card className="provider-empty-card">
                  <div className="provider-empty-icon"><KeyRound size={21} /></div>
                  <div>
                    <h3>Connect an AI provider to get started</h3>
                    <p>Add a cloud, local, or compatible endpoint profile. You can assign it to an Agent after its connection test succeeds.</p>
                    <Button onClick={() => { setNotice(null); setEditor(null); }} leadingIcon={<Plus size={15} />}>Add your first provider</Button>
                  </div>
                </Card>
              ) : (
                <div className="provider-profile-list">
                  {data.profiles.map((profile) => {
                    const provider = data.catalog.find((item) => item.provider_id === profile.provider_id);
                    const capabilities = Object.keys(provider?.capabilities ?? profile.capability_status);
                    const testResult = testResults[profile.id];
                    const testing = pendingAction === "test:" + profile.id;
                    return (
                      <Card className="provider-profile-card" key={profile.id}>
                        <div className="provider-profile-head">
                          <span className="provider-kind-icon" aria-hidden="true">
                            {provider?.category === "cloud" ? <Cloud size={18} /> : <Cpu size={18} />}
                          </span>
                          <div className="provider-profile-title">
                            <div className="provider-name-line">
                              <h3>{profile.display_name}</h3>
                              <Badge tone={provider?.category === "cloud" ? "blue" : "purple"}>
                                {provider?.display_name ?? profile.provider_id}
                              </Badge>
                              {provider && <Badge tone="neutral">{provider.category.replace("_", " ")}</Badge>}
                            </div>
                            <p><span>Model</span><code>{profile.model_identifier}</code></p>
                            {provider?.endpoint_requirement === "required" && profile.endpoint_url && (
                              <p><span>Endpoint</span><code className="provider-endpoint">{profile.endpoint_url}</code></p>
                            )}
                          </div>
                          <div className="provider-profile-status">
                            <Badge tone={profile.credential_configured ? "green" : provider?.api_key_requirement === "not_used" ? "neutral" : "amber"}>
                              {provider?.api_key_requirement === "not_used"
                                ? "Credential not required"
                                : profile.credential_configured ? "Credential configured" : "Credential not configured"}
                            </Badge>
                            <Badge tone={profile.connection_status === "connected" ? "green" : profile.connection_status === "failed" ? "red" : "amber"}>
                              {profile.connection_status === "connected" ? "Connected" : profile.connection_status === "failed" ? "Connection failed" : "Not tested"}
                            </Badge>
                          </div>
                        </div>
                        {capabilities.length > 0 && (
                          <div className="provider-capability-block">
                            <div className="provider-subheading"><strong>Model capabilities</strong><span>Last checked: {displayDate(profile.last_connection_checked_at)}</span></div>
                            <ul className="provider-capability-list">
                              {capabilities.map((name) => {
                                const value = profile.capability_status[name] ?? "unknown";
                                return <li key={name}><span>{CAPABILITY_LABEL[name] ?? name.replaceAll("_", " ")}</span><Badge tone={capabilityTone(value)}>{capabilityText(value)}</Badge></li>;
                              })}
                            </ul>
                          </div>
                        )}
                        {testResult && (
                          <div className="provider-test-result" role="status">
                            <strong>{testResult.status === "connected" ? "Connection test passed" : "Connection test did not pass"}</strong>
                            <p>{testResult.message}</p>
                            <small>Category: {testResult.category} · Checked {displayDate(testResult.checked_at)}</small>
                          </div>
                        )}
                        <div className="provider-profile-actions">
                          <Button variant="outline" onClick={() => { setNotice(null); setEditor(profile); }} leadingIcon={<Pencil size={14} />} disabled={pendingAction !== null}>Edit</Button>
                          <Button variant="outline" onClick={() => void runConnectionTest(profile)} leadingIcon={<RefreshCw size={14} />} loading={testing} disabled={pendingAction !== null && !testing}>Test connection</Button>
                          {profile.credential_configured && <Button variant="ghost" onClick={() => setConfirmation({ kind: "remove-credential", profile })} leadingIcon={<Unplug size={14} />} disabled={pendingAction !== null}>Remove credential</Button>}
                          <Button variant="destructive" onClick={() => setConfirmation({ kind: "delete-profile", profile })} leadingIcon={<Trash2 size={14} />} disabled={pendingAction !== null}>Remove profile</Button>
                        </div>
                      </Card>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="provider-section" aria-labelledby="provider-assignment-title">
              <div className="provider-section-heading">
                <div>
                  <p className="section-overline">AGENT CONFIGURATION</p>
                  <h2 id="provider-assignment-title">Agent assignments</h2>
                  <p>Each Agent can use a different profile. The backend validates tested capabilities before saving.</p>
                </div>
                <span className="provider-assignment-icon"><ShieldCheck size={19} /></span>
              </div>
              {assignmentStateStale && (
                <div className="provider-assignment-sync-error">
                  <Alert tone="warning" title="Assignment state not verified">
                    Current assignments could not be confirmed after the latest request. Refresh them before making another assignment.
                  </Alert>
                  <Button
                    variant="outline"
                    onClick={() => void retryAssignmentRefresh()}
                    loading={pendingAction === "refresh-assignments"}
                    disabled={pendingAction !== null && pendingAction !== "refresh-assignments"}
                    leadingIcon={<RefreshCw size={14} />}
                  >
                    Refresh assignments
                  </Button>
                </div>
              )}
              {!data.profiles.length ? (
                <Card className="provider-assignment-empty">
                  <p>Add a provider profile and test its connection before assigning a model to either Agent.</p>
                </Card>
              ) : (
                <div className="provider-assignment-grid">
                  {roleNames.map((role) => {
                    const assignment = assignmentStateStale ? undefined : assignmentsByRole.get(role);
                    const selectedId = assignmentStateStale ? "" : roleSelections[role] ?? assignment?.profile_id ?? "";
                    const selectedProfile = data.profiles.find((profile) => profile.id === selectedId);
                    const requirements = roleRequirements(data, assignment?.profile ?? selectedProfile, role);
                    const eligibility = selectedProfile ? isAssignable(selectedProfile, role, data) : null;
                    const currentIsSelected = selectedId === assignment?.profile_id;
                    return (
                      <Card className="provider-assignment-card" key={role}>
                        <div className="provider-assignment-heading">
                          <span className="provider-role-icon">{role === "pipeline_agent" ? <Zap size={17} /> : <ShieldCheck size={17} />}</span>
                          <div><h3>{ROLE_LABEL[role]}</h3><p>{role === "pipeline_agent" ? "Runs data analysis and tool-based workflows." : "Creates structured, evidence-based reports."}</p></div>
                        </div>
                        <label className="field">
                          <span className="field-label">Assigned provider</span>
                          <div className="select-wrap">
                            <select
                              className="input select-input"
                              value={selectedId}
                              onChange={(event) => setRoleSelections((current) => ({ ...current, [role]: event.target.value }))}
                              aria-label={ROLE_LABEL[role] + " provider"}
                              disabled={pendingAction !== null || assignmentStateStale}
                            >
                              <option value="">Choose a provider profile</option>
                              {data.profiles.map((profile) => {
                                const valid = isAssignable(profile, role, data);
                                return (
                                  <option key={profile.id} value={profile.id} disabled={!valid.eligible && profile.id !== assignment?.profile_id}>
                                    {profile.display_name + " · " + profile.model_identifier + (valid.eligible && valid.unverified.length ? " · capabilities unverified" : !valid.eligible ? " · " + valid.reason : "")}
                                  </option>
                                );
                              })}
                            </select>
                          </div>
                        </label>
                        <div className="provider-role-requirements">
                          <span className="field-label">Required capabilities</span>
                          <ul>
                            {requirements.map((name) => {
                              const value = assignment?.profile_id === selectedId
                                ? assignment.profile.capability_status[name] ?? "unknown"
                                : selectedProfile?.capability_status[name] ?? "unknown";
                              return <li key={name}><span>{CAPABILITY_LABEL[name] ?? name.replaceAll("_", " ")}</span><Badge tone={capabilityTone(value)}>{capabilityText(value)}</Badge></li>;
                            })}
                          </ul>
                        </div>
                        {assignment && (
                          <div className="provider-current-assignment">
                            <Check size={14} aria-hidden="true" />
                            <span>Currently using <strong>{assignment.profile.display_name}</strong> · {assignment.profile.model_identifier}</span>
                          </div>
                        )}
                        {eligibility?.eligible && eligibility.unverified.length > 0 && (
                          <p className="provider-unverified-note">The backend allows this assignment, but these required capabilities are still unknown: {eligibility.unverified.map((name) => CAPABILITY_LABEL[name] ?? name).join(", ")}.</p>
                        )}
                        {selectedProfile && eligibility && !eligibility.eligible && (
                          <p className="provider-eligibility-note">{eligibility.reason}</p>
                        )}
                        <div className="provider-assignment-actions">
                          <Button
                            onClick={() => void saveAssignment(role, selectedId)}
                            loading={pendingAction === "assignment:" + role}
                            disabled={!selectedProfile || !eligibility?.eligible || currentIsSelected || pendingAction !== null || assignmentStateStale}
                            leadingIcon={<Check size={14} />}
                          >
                            {currentIsSelected ? "Assigned" : "Assign provider"}
                          </Button>
                          {assignment && (
                            <Button
                              variant="ghost"
                              onClick={() => setConfirmation({ kind: "clear-assignment", role })}
                              disabled={pendingAction !== null}
                            >
                              Clear assignment
                            </Button>
                          )}
                        </div>
                      </Card>
                    );
                  })}
                </div>
              )}
            </section>
          </>
        )}
      </div>

      {editor !== undefined && data && (
        <ProviderProfileForm
          key={editor?.id ?? "new-profile"}
          catalog={data.catalog}
          profile={editor}
          busy={pendingAction === "save-profile"}
          onClose={() => setEditor(undefined)}
          onSave={async (draft) => {
            setPendingAction("save-profile");
            try {
              await saveProfile(draft, editor);
            } finally {
              setPendingAction(null);
            }
          }}
        />
      )}

      <Modal open={confirmation !== null} onClose={() => { if (pendingAction !== "confirm") setConfirmation(null); }} title={confirmationTitle ?? "Confirm action"}>
        <div className="provider-confirmation">
          <p>
            {confirmation?.kind === "delete-profile" && "This removes the provider profile from your account. This action cannot be undone."}
            {confirmation?.kind === "remove-credential" && "The saved credential will be deleted. The provider will need a new successful connection test before it can be assigned."}
            {confirmation?.kind === "clear-assignment" && "The selected Agent will no longer have a provider assigned."}
          </p>
          <div className="provider-modal-actions">
            <Button variant="outline" onClick={() => setConfirmation(null)} disabled={pendingAction === "confirm"}>Cancel</Button>
            <Button variant="destructive" onClick={() => void confirmAction()} loading={pendingAction === "confirm"}>Confirm</Button>
          </div>
        </div>
      </Modal>
    </AppShell>
  );
}

function ProviderProfileForm({
  catalog,
  profile,
  busy,
  onClose,
  onSave,
}: {
  catalog: ProviderCatalogEntry[];
  profile: ProviderProfile | null;
  busy: boolean;
  onClose: () => void;
  onSave: (draft: ProfileDraft) => Promise<void>;
}) {
  const [providerId, setProviderId] = useState(profile?.provider_id ?? catalog[0]?.provider_id ?? "");
  const [displayName, setDisplayName] = useState(profile?.display_name ?? "");
  const [modelIdentifier, setModelIdentifier] = useState(profile?.model_identifier ?? "");
  const [endpointUrl, setEndpointUrl] = useState(profile?.endpoint_url ?? "");
  const [configuration, setConfiguration] = useState<Record<string, string>>(() => Object.fromEntries(
    Object.entries(profile?.configuration ?? {}).map(([key, value]) => [key, String(value)]),
  ));
  const [formNotice, setFormNotice] = useState<SettingsNotice | null>(null);
  const credentialRef = useRef<HTMLInputElement>(null);
  const provider = catalog.find((item) => item.provider_id === providerId);
  const editId = profile?.id ?? "create";

  useEffect(() => () => {
    if (credentialRef.current) credentialRef.current.value = "";
  }, []);

  function parseConfiguration(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const name of provider?.supported_configuration ?? []) {
      const value = configuration[name]?.trim();
      if (!value) continue;
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) throw new Error((CONFIGURATION_LABEL[name] ?? name) + " must be a number.");
      if (name === "temperature" && (parsed < 0 || parsed > 2)) throw new Error("Temperature must be between 0 and 2.");
      if (name === "top_p" && (parsed <= 0 || parsed > 1)) throw new Error("Top P must be greater than 0 and at most 1.");
      if (name === "max_tokens" && (!Number.isInteger(parsed) || parsed < 1 || parsed > 131072)) throw new Error("Maximum tokens must be an integer from 1 to 131072.");
      if ((name === "frequency_penalty" || name === "presence_penalty") && (parsed < -2 || parsed > 2)) {
        throw new Error((CONFIGURATION_LABEL[name] ?? name) + " must be between -2 and 2.");
      }
      result[name] = parsed;
    }
    return result;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormNotice(null);
    const rawCredential = credentialRef.current?.value ?? "";
    const credential = rawCredential.trim();
    try {
      if (!provider) throw new Error("Select a supported provider.");
      if (provider.endpoint_requirement === "required" && !endpointUrl.trim()) {
        throw new Error("Enter the endpoint URL required by this provider.");
      }
      if (!profile && provider.api_key_requirement === "required" && !credential) {
        throw new Error("Enter a provider credential to create this profile.");
      }
      const draft: ProfileDraft = {
        display_name: displayName.trim(),
        provider_id: provider.provider_id,
        model_identifier: modelIdentifier.trim(),
        endpoint_url: provider.endpoint_requirement === "required" ? endpointUrl.trim() : null,
        configuration: parseConfiguration(),
        ...(credential ? { credential } : {}),
      };
      await onSave(draft);
    } catch (error) {
      const expired = error instanceof ApiError && error.status === 401;
      setFormNotice({
        tone: "error",
        title: expired ? "Sign-in required" : "Provider could not be saved",
        message: errorMessage(error, credential),
        sessionExpired: expired,
      });
    } finally {
      if (credentialRef.current) credentialRef.current.value = "";
    }
  }

  return (
    <Modal open onClose={onClose} title={profile ? "Edit provider profile" : "Add provider profile"}>
      <form className="provider-form" onSubmit={submit}>
        {formNotice && <SessionNotice notice={formNotice} />}
        <div className="provider-form-intro">
          <p>{profile ? "Update the model settings or enter a new credential to replace the saved one." : "Add a provider profile for one of your Agents. Model IDs are entered manually because available models vary by account."}</p>
        </div>
        {!profile && (
          <label className="field">
            <span className="field-label">Provider</span>
            <div className="select-wrap">
              <select className="input select-input" value={providerId} onChange={(event) => {
                setProviderId(event.target.value);
                setEndpointUrl("");
                setConfiguration({});
              }} required>
                {catalog.map((item) => <option key={item.provider_id} value={item.provider_id}>{item.display_name}</option>)}
              </select>
            </div>
          </label>
        )}
        {provider && (
          <div className="provider-metadata-note">
            <div><Badge tone={provider.category === "cloud" ? "blue" : "purple"}>{provider.category.replace("_", " ")}</Badge><span>{provider.description}</span></div>
            <small>Model format: {provider.model_identifier_format}</small>
          </div>
        )}
        <TextInput label="Profile name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} maxLength={120} autoComplete="off" required />
        <TextInput
          label="Model ID"
          value={modelIdentifier}
          onChange={(event) => setModelIdentifier(event.target.value)}
          hint="Enter the exact model identifier accepted by your provider."
          maxLength={255}
          autoComplete="off"
          required
        />
        {provider?.endpoint_requirement === "required" && (
          <TextInput
            label="Base URL"
            type="url"
            value={endpointUrl}
            onChange={(event) => setEndpointUrl(event.target.value)}
            hint={provider.description}
            maxLength={2048}
            autoComplete="url"
            required
          />
        )}
        {provider && provider.api_key_requirement !== "not_used" && (
          <div className="field">
            <label className="field-label" htmlFor={"provider-credential-" + editId}>
              {provider.api_key_requirement === "required" ? "API key or credential" : "API key or credential (optional)"}
            </label>
            <input
              ref={credentialRef}
              className="input"
              id={"provider-credential-" + editId}
              type="password"
              autoComplete="new-password"
              placeholder={profile?.credential_configured ? "Enter a new value to replace the saved credential" : "Enter provider credential"}
              required={!profile && provider.api_key_requirement === "required"}
              maxLength={8192}
            />
            <span className="field-hint">
              {profile?.credential_configured
                ? "The saved value is never displayed. Leave this field empty to keep it unchanged; use Remove credential to delete it."
                : provider.api_key_requirement === "required"
                  ? "This provider requires a credential. It is sent securely to the backend and is never returned."
                  : "Optional for this provider. A value is never returned after it is saved."}
            </span>
          </div>
        )}
        {!!provider?.supported_configuration.length && (
          <fieldset className="provider-config-fields">
            <legend>Model parameters <span>Optional</span></legend>
            {provider.supported_configuration.map((name) => {
              const bounds: Record<string, { min?: number; max?: number; step?: number }> = {
                temperature: { min: 0, max: 2, step: 0.1 },
                max_tokens: { min: 1, max: 131072, step: 1 },
                top_p: { min: 0.00001, max: 1, step: 0.01 },
                frequency_penalty: { min: -2, max: 2, step: 0.1 },
                presence_penalty: { min: -2, max: 2, step: 0.1 },
              };
              return (
                <TextInput
                  key={name}
                  label={CONFIGURATION_LABEL[name] ?? name.replaceAll("_", " ")}
                  type="number"
                  value={configuration[name] ?? ""}
                  onChange={(event) => setConfiguration((current) => ({ ...current, [name]: event.target.value }))}
                  min={bounds[name]?.min}
                  max={bounds[name]?.max}
                  step={bounds[name]?.step}
                  placeholder={name === "temperature" ? "0.2" : name === "max_tokens" ? "2048" : ""}
                  autoComplete="off"
                />
              );
            })}
          </fieldset>
        )}
        <div className="provider-security-note"><ShieldCheck size={15} /><span>Credentials are managed by the backend. They are not saved in browser storage or shown after saving.</span></div>
        <div className="provider-modal-actions">
          <Button type="button" variant="outline" onClick={onClose} disabled={busy}>Cancel</Button>
          <Button type="submit" loading={busy} leadingIcon={<Check size={15} />}>{profile ? "Save changes" : "Add provider"}</Button>
        </div>
      </form>
    </Modal>
  );
}
