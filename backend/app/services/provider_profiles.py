"""Ownership-scoped CRUD for encrypted user provider profiles."""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.db.models import AIProviderProfile, UserAgentProviderAssignment
from app.services.provider_credentials import encrypt_provider_credential
from app.services.provider_endpoint_security import validate_endpoint_syntax
from app.services.provider_errors import ProviderManagementError
from app.services.provider_registry import get_provider, supported_configuration_for


_SENSITIVE_CONFIG_KEY = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|credential|"
    r"private[_-]?key|passphrase)",
    re.IGNORECASE,
)
_ALLOWED_CONFIGURATION_KEYS = frozenset({
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
})


def validate_configuration(configuration: Any, *, provider_id: str | None = None) -> dict:
    if not isinstance(configuration, dict):
        raise ProviderManagementError(
            "invalid_configuration", "Provider configuration must be a JSON object.", 422,
        )
    if len(configuration) > 32:
        raise ProviderManagementError(
            "invalid_configuration", "Provider configuration contains too many options.", 422,
        )

    allowed_configuration = (
        frozenset(supported_configuration_for(provider_id))
        if provider_id is not None
        else _ALLOWED_CONFIGURATION_KEYS
    )
    clean: dict[str, Any] = {}
    for key, value in configuration.items():
        if not isinstance(key, str) or not key or len(key) > 80:
            raise ProviderManagementError(
                "invalid_configuration", "Provider configuration contains an invalid option.", 422,
            )
        if key not in allowed_configuration:
            if _SENSITIVE_CONFIG_KEY.search(key):
                raise ProviderManagementError(
                    "sensitive_configuration_rejected",
                    "Credentials must be supplied through the encrypted credential field.",
                    422,
                )
            raise ProviderManagementError(
                "unsupported_configuration_option",
                "Provider configuration contains an unsupported option.",
                422,
            )
        clean[key] = value

    try:
        encoded = json.dumps(clean, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise ProviderManagementError(
            "invalid_configuration", "Provider configuration must contain JSON values.", 422,
        ) from None
    if len(encoded.encode("utf-8")) > 16_384:
        raise ProviderManagementError(
            "invalid_configuration", "Provider configuration is too large.", 422,
        )

    temperature = clean.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise ProviderManagementError(
            "invalid_configuration", "temperature must be a number between 0 and 2.", 422,
        )
    top_p = clean.get("top_p")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0 < top_p <= 1
    ):
        raise ProviderManagementError(
            "invalid_configuration", "top_p must be a number greater than 0 and at most 1.", 422,
        )
    max_tokens = clean.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 131_072
    ):
        raise ProviderManagementError(
            "invalid_configuration", "max_tokens must be an integer between 1 and 131072.", 422,
        )
    for option in ("frequency_penalty", "presence_penalty"):
        value = clean.get(option)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not -2 <= value <= 2
        ):
            raise ProviderManagementError(
                "invalid_configuration",
                f"{option} must be a number between -2 and 2.",
                422,
            )
    return clean


def validate_profile_fields(
    *,
    display_name: str,
    provider_id: str,
    model_identifier: str,
    endpoint_url: str | None,
    configuration: dict,
) -> tuple[str, str | None, dict]:
    provider_key = provider_id.strip().casefold()
    definition = get_provider(provider_key)
    if definition is None:
        raise ProviderManagementError(
            "unsupported_provider", "Select a provider from the supported catalog.", 422,
        )

    display = display_name.strip()
    model = model_identifier.strip()
    if (
        not display or len(display) > 120
        or any(ord(character) < 32 for character in display)
    ):
        raise ProviderManagementError(
            "invalid_display_name", "Display name must be non-empty printable text.", 422,
        )
    if (
        not model or len(model) > 255
        or any(ord(character) < 32 for character in model)
    ):
        raise ProviderManagementError(
            "invalid_model_identifier", "Model identifier must be non-empty printable text.", 422,
        )

    normalized_endpoint = validate_endpoint_syntax(provider_key, endpoint_url)
    clean_configuration = validate_configuration(configuration, provider_id=provider_key)
    return provider_key, normalized_endpoint, clean_configuration


def get_owned_profile(
    db: DBSession, user_id: str, profile_id: str,
) -> AIProviderProfile:
    profile = db.scalar(
        select(AIProviderProfile).where(
            AIProviderProfile.id == profile_id,
            AIProviderProfile.user_id == user_id,
        )
    )
    if profile is None:
        raise ProviderManagementError(
            "provider_profile_not_found", "Provider profile not found.", 404,
        )
    return profile


def list_profiles(db: DBSession, user_id: str) -> list[AIProviderProfile]:
    return list(db.scalars(
        select(AIProviderProfile)
        .where(AIProviderProfile.user_id == user_id)
        .order_by(AIProviderProfile.created_at.desc(), AIProviderProfile.id)
    ))


def create_profile(
    db: DBSession,
    *,
    user_id: str,
    display_name: str,
    provider_id: str,
    model_identifier: str,
    endpoint_url: str | None,
    configuration: dict,
    credential: str | None,
) -> AIProviderProfile:
    provider_key, normalized_endpoint, clean_configuration = validate_profile_fields(
        display_name=display_name,
        provider_id=provider_id,
        model_identifier=model_identifier,
        endpoint_url=endpoint_url,
        configuration=configuration,
    )
    encrypted = None
    key_version = None
    if credential is not None:
        encrypted, key_version = encrypt_provider_credential(credential)

    profile = AIProviderProfile(
        user_id=user_id,
        display_name=display_name.strip(),
        provider_id=provider_key,
        model_identifier=model_identifier.strip(),
        endpoint_url=normalized_endpoint,
        configuration_json=clean_configuration,
        credential_ciphertext=encrypted,
        credential_key_version=key_version,
    )
    db.add(profile)
    try:
        db.commit()
        db.refresh(profile)
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "provider_profile_conflict", "The provider profile could not be saved.", 409,
        ) from None
    return profile


def update_profile(
    db: DBSession,
    *,
    user_id: str,
    profile_id: str,
    updates: dict,
) -> AIProviderProfile:
    profile = get_owned_profile(db, user_id, profile_id)
    values = {
        "display_name": profile.display_name,
        "model_identifier": profile.model_identifier,
        "endpoint_url": profile.endpoint_url,
        "configuration": profile.configuration_json,
    }
    unknown_fields = set(updates) - set(values)
    if unknown_fields:
        raise ProviderManagementError(
            "invalid_profile_update", "The provider profile update contains unsupported fields.", 422,
        )
    prior_values = values.copy()
    values.update(updates)
    _, normalized_endpoint, clean_configuration = validate_profile_fields(
        display_name=values["display_name"],
        provider_id=profile.provider_id,
        model_identifier=values["model_identifier"],
        endpoint_url=values["endpoint_url"],
        configuration=values["configuration"],
    )
    profile.display_name = values["display_name"].strip()
    profile.model_identifier = values["model_identifier"].strip()
    profile.endpoint_url = normalized_endpoint
    profile.configuration_json = clean_configuration
    technical_configuration_changed = any(
        values[name] != prior_values[name]
        for name in ("model_identifier", "endpoint_url", "configuration")
    )
    if technical_configuration_changed:
        db.execute(delete(UserAgentProviderAssignment).where(
            UserAgentProviderAssignment.user_id == user_id,
            UserAgentProviderAssignment.profile_id == profile.id,
        ))
        profile.connection_status = None
        profile.connection_error_code = None
        profile.last_connection_checked_at = None
        profile.capability_status_json = {
            "chat_completion": "unknown",
            "tool_calling": "unknown",
            "structured_output": "unknown",
            "json_schema": "unknown",
            "streaming": "unknown",
        }
    try:
        db.commit()
        db.refresh(profile)
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "provider_profile_conflict", "The provider profile could not be updated.", 409,
        ) from None
    return profile


def replace_credential(
    db: DBSession, *, user_id: str, profile_id: str, credential: str,
) -> AIProviderProfile:
    profile = get_owned_profile(db, user_id, profile_id)
    ciphertext, key_version = encrypt_provider_credential(credential)
    profile.credential_ciphertext = ciphertext
    profile.credential_key_version = key_version
    db.execute(delete(UserAgentProviderAssignment).where(
        UserAgentProviderAssignment.user_id == user_id,
        UserAgentProviderAssignment.profile_id == profile.id,
    ))
    profile.connection_status = None
    profile.connection_error_code = None
    profile.last_connection_checked_at = None
    try:
        db.commit()
        db.refresh(profile)
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "provider_credential_update_failed",
            "The provider credential could not be updated.",
            409,
        ) from None
    return profile


def remove_credential(
    db: DBSession, *, user_id: str, profile_id: str,
) -> AIProviderProfile:
    profile = get_owned_profile(db, user_id, profile_id)
    profile.credential_ciphertext = None
    profile.credential_key_version = None
    db.execute(delete(UserAgentProviderAssignment).where(
        UserAgentProviderAssignment.user_id == user_id,
        UserAgentProviderAssignment.profile_id == profile.id,
    ))
    profile.connection_status = None
    profile.connection_error_code = None
    profile.last_connection_checked_at = None
    try:
        db.commit()
        db.refresh(profile)
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "provider_credential_remove_failed",
            "The provider credential could not be removed.",
            409,
        ) from None
    return profile


def delete_profile(db: DBSession, *, user_id: str, profile_id: str) -> None:
    profile = get_owned_profile(db, user_id, profile_id)
    db.execute(delete(UserAgentProviderAssignment).where(
        UserAgentProviderAssignment.user_id == user_id,
        UserAgentProviderAssignment.profile_id == profile.id,
    ))
    db.delete(profile)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ProviderManagementError(
            "provider_profile_delete_failed", "The provider profile could not be deleted.", 409,
        ) from None


def record_connection_result(
    db: DBSession,
    profile: AIProviderProfile,
    *,
    connected: bool,
    error_code: str | None,
    capability_status: dict[str, str],
    checked_at,
) -> None:
    profile.connection_status = "connected" if connected else "failed"
    profile.connection_error_code = error_code
    profile.last_connection_checked_at = checked_at
    profile.capability_status_json = capability_status
    db.commit()
    db.refresh(profile)
