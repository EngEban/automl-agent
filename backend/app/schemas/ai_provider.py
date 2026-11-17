"""Request and response schemas for user-managed AI provider profiles."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator


AgentRole = Literal["pipeline_agent", "report_agent"]
CapabilityState = Literal["supported", "unsupported", "unknown"]


def _clean_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} must be non-empty printable text.")
    return value


class ProviderProfileCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    provider_id: str = Field(min_length=1, max_length=50)
    model_identifier: str = Field(min_length=1, max_length=255)
    endpoint_url: str | None = Field(default=None, max_length=2048)
    configuration: dict = Field(default_factory=dict)
    credential: SecretStr | None = None

    @field_validator("display_name", "provider_id", "model_identifier")
    @classmethod
    def validate_text(cls, value: str, info):
        return _clean_text(value, info.field_name)

    @field_validator("credential")
    @classmethod
    def validate_optional_credential(cls, value: SecretStr | None):
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("credential must not be blank.")
        return value


class ProviderProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    model_identifier: str | None = Field(default=None, min_length=1, max_length=255)
    endpoint_url: str | None = Field(default=None, max_length=2048)
    configuration: dict | None = None

    @field_validator("display_name", "model_identifier")
    @classmethod
    def validate_optional_text(cls, value: str | None, info):
        return None if value is None else _clean_text(value, info.field_name)


class ProviderCredentialReplace(BaseModel):
    credential: SecretStr

    @field_validator("credential")
    @classmethod
    def validate_credential(cls, value: SecretStr):
        secret = value.get_secret_value()
        if not secret.strip() or len(secret) > 8192:
            raise ValueError("credential must contain between 1 and 8192 characters.")
        return value


class ProviderProfileRead(BaseModel):
    id: str
    display_name: str
    provider_id: str
    model_identifier: str
    endpoint_url: str | None
    configuration: dict
    credential_configured: bool
    capability_status: dict[str, CapabilityState]
    connection_status: Literal["connected", "failed"] | None
    connection_error_code: str | None
    last_connection_checked_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ProviderCatalogEntry(BaseModel):
    provider_id: str
    display_name: str
    category: str
    endpoint_mode: str
    api_key_requirement: Literal["required", "optional", "not_used"]
    endpoint_requirement: Literal["managed", "required"]
    model_identifier_format: str
    supported_configuration: list[str]
    description: str
    capabilities: dict[str, CapabilityState]
    capability_note: str
    agent_requirements: dict[str, list[str]]


class ProviderCatalogRead(BaseModel):
    providers: list[ProviderCatalogEntry]


class ProviderAssignmentSet(BaseModel):
    profile_id: str = Field(min_length=1, max_length=36)


class ProviderAssignmentRead(BaseModel):
    agent_role: AgentRole
    profile_id: str
    created_at: datetime
    updated_at: datetime
    profile: ProviderProfileRead
    unverified_capabilities: list[str]


class ProviderConnectionTestRead(BaseModel):
    status: Literal["connected", "failed"]
    category: str
    message: str
    capabilities: dict[str, CapabilityState]
    checked_at: datetime
