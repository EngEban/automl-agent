export type AgentRole = "pipeline_agent" | "report_agent";

export type CapabilityState = "supported" | "unsupported" | "unknown";

export interface ProviderCatalogEntry {
  provider_id: string;
  display_name: string;
  category: string;
  endpoint_mode: string;
  api_key_requirement: "required" | "optional" | "not_used";
  endpoint_requirement: "managed" | "required";
  model_identifier_format: string;
  supported_configuration: string[];
  description: string;
  capabilities: Record<string, CapabilityState>;
  capability_note: string;
  agent_requirements: Record<AgentRole, string[]>;
}

export interface ProviderCatalogResponse {
  providers: ProviderCatalogEntry[];
}

export interface ProviderProfile {
  id: string;
  display_name: string;
  provider_id: string;
  model_identifier: string;
  endpoint_url: string | null;
  configuration: Record<string, number>;
  credential_configured: boolean;
  capability_status: Record<string, CapabilityState>;
  connection_status: "connected" | "failed" | null;
  connection_error_code: string | null;
  last_connection_checked_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProviderProfileCreate {
  display_name: string;
  provider_id: string;
  model_identifier: string;
  endpoint_url: string | null;
  configuration: Record<string, number>;
  credential?: string;
}

export interface ProviderProfileUpdate {
  display_name?: string;
  model_identifier?: string;
  endpoint_url?: string | null;
  configuration?: Record<string, number>;
}

export interface ProviderConnectionTest {
  status: "connected" | "failed";
  category: string;
  message: string;
  capabilities: Record<string, CapabilityState>;
  checked_at: string;
}

export interface ProviderAssignment {
  agent_role: AgentRole;
  profile_id: string;
  created_at: string;
  updated_at: string;
  profile: ProviderProfile;
  unverified_capabilities: string[];
}

export type SettingsNotice = {
  tone: "success" | "warning" | "error" | "info";
  title: string;
  message: string;
  sessionExpired?: boolean;
};
