"""Explicit, bounded provider connection and capability probes."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config.settings import settings
from app.db.models import AIProviderProfile
from app.services.provider_credentials import (
    ProviderCredentialConfigurationError,
    decrypt_provider_credential,
)
from app.services.provider_endpoint_security import (
    PinnedEndpoint,
    resolve_and_pin_endpoint_async,
)
from app.services.provider_errors import ProviderManagementError
from app.services.provider_registry import (
    CAPABILITIES,
    get_provider,
    resolve_litellm_model,
)

logger = logging.getLogger(__name__)
_MAX_RESPONSE_BYTES = 64 * 1024
_PROBE_TOOL_NAME = "provider_capability_probe"
_JSON_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProviderConnectionOutcome:
    status: str
    category: str
    message: str
    capabilities: dict[str, str]
    checked_at: datetime


class _ProbeFailure(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


def _unknown_capabilities() -> dict[str, str]:
    return {name: "unknown" for name in CAPABILITIES}


def _get(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _first_message(response: Any) -> Any | None:
    choices = _get(response, "choices", [])
    if not choices:
        return None
    return _get(choices[0], "message")


def _classify_provider_exception(exc: Exception) -> tuple[str, str]:
    status_code = getattr(exc, "status_code", None)
    error_type = type(exc).__name__.casefold()
    if status_code in (401, 403) or "authentication" in error_type or "permission" in error_type:
        return "authentication_failed", "The provider rejected the configured credential."
    if status_code == 404 or "modelnotfound" in error_type or "model_not_found" in error_type:
        return "unsupported_model", "The provider does not recognize this model identifier."
    if status_code == 429 or "ratelimit" in error_type:
        return "rate_limited", "The provider is rate limiting this connection test."
    if status_code in (408, 500, 502, 503, 504) or any(
        marker in error_type for marker in ("timeout", "connect", "network")
    ):
        return "network_failure", "The provider could not be reached before the timeout."
    if status_code in (400, 422) or "badrequest" in error_type or "invalidrequest" in error_type:
        return "capability_failure", "The provider rejected this capability probe."
    return "provider_error", "The provider could not complete the connection test."


def _http_status_failure(status_code: int, *, capability_probe: bool) -> _ProbeFailure:
    if status_code in (401, 403):
        return _ProbeFailure(
            "authentication_failed", "The provider rejected the configured credential.",
        )
    if status_code == 404:
        return _ProbeFailure(
            "unsupported_model", "The provider does not recognize this model identifier.",
        )
    if status_code == 429:
        return _ProbeFailure("rate_limited", "The provider is rate limiting this connection test.")
    if status_code in (301, 302, 303, 307, 308):
        return _ProbeFailure(
            "unsafe_redirect", "The endpoint attempted a redirect, which was not followed.",
        )
    if status_code in (400, 422) and capability_probe:
        return _ProbeFailure(
            "capability_failure", "The model does not accept this capability probe.",
        )
    if status_code >= 500:
        return _ProbeFailure(
            "network_failure", "The provider returned a temporary service error.",
        )
    return _ProbeFailure(
        "provider_error", "The provider rejected the connection test request.",
    )


async def _litellm_acompletion(**kwargs):
    import litellm

    return await litellm.acompletion(**kwargs)


def _litellm_payload(profile: AIProviderProfile, credential: str | None, probe: str) -> dict:
    if probe == "chat":
        payload = {
            "messages": [
                {"role": "system", "content": "Reply only with the word OK."},
                {"role": "user", "content": "Connection check."},
            ],
        }
    elif probe == "tools":
        payload = {
            "messages": [
                {"role": "system", "content": "Call the requested harmless probe tool."},
                {"role": "user", "content": "Call the tool now."},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": _PROBE_TOOL_NAME,
                    "description": "Harmless capability check. Do not perform external actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": _PROBE_TOOL_NAME},
            },
        }
    else:
        payload = {
            "messages": [
                {"role": "system", "content": "Return a JSON object with ok set to true."},
                {"role": "user", "content": "Return the requested JSON object."},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "provider_capability_probe",
                    "strict": True,
                    "schema": _JSON_PROBE_SCHEMA,
                },
            },
        }

    payload.update({
        "model": resolve_litellm_model(profile.provider_id, profile.model_identifier),
        "temperature": 0,
        "max_tokens": 64,
        "timeout": settings.PROVIDER_TEST_TIMEOUT_SECONDS,
        "stream": False,
        "num_retries": 0,
    })
    if credential:
        payload["api_key"] = credential
    return payload


def _compatible_endpoint_payload(profile: AIProviderProfile, probe: str) -> dict:
    payload: dict[str, Any] = {
        "model": profile.model_identifier,
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }
    if probe == "chat":
        payload["messages"] = [
            {"role": "system", "content": "Reply only with the word OK."},
            {"role": "user", "content": "Connection check."},
        ]
    elif probe == "tools":
        payload.update({
            "messages": [
                {"role": "system", "content": "Call the requested harmless probe tool."},
                {"role": "user", "content": "Call the tool now."},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": _PROBE_TOOL_NAME,
                    "description": "Harmless capability check. Do not perform external actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": _PROBE_TOOL_NAME},
            },
        })
    else:
        payload.update({
            "messages": [
                {"role": "system", "content": "Return a JSON object with ok set to true."},
                {"role": "user", "content": "Return the requested JSON object."},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "provider_capability_probe",
                    "strict": True,
                    "schema": _JSON_PROBE_SCHEMA,
                },
            },
        })
    return payload


def _completion_url(endpoint: PinnedEndpoint) -> str:
    parsed = urlsplit(endpoint.base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        completion_path = path
    elif path.endswith("/v1"):
        completion_path = path + "/chat/completions"
    else:
        completion_path = path + "/v1/chat/completions"
    return f"{endpoint.base_url[:len(endpoint.base_url) - len(parsed.path)]}{completion_path}"


def _create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(
            float(settings.PROVIDER_TEST_TIMEOUT_SECONDS),
            connect=min(5.0, float(settings.PROVIDER_TEST_TIMEOUT_SECONDS)),
        ),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
    )


async def _post_compatible(
    client: httpx.AsyncClient,
    endpoint: PinnedEndpoint,
    credential: str | None,
    probe: str,
    profile: AIProviderProfile,
) -> dict:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Host": endpoint.host_header,
    }
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    extensions = {}
    if endpoint.sni_hostname:
        extensions["sni_hostname"] = endpoint.sni_hostname
    request = client.build_request(
        "POST",
        _completion_url(endpoint),
        headers=headers,
        json=_compatible_endpoint_payload(profile, probe),
        extensions=extensions,
    )
    try:
        response = await client.send(request, stream=True, follow_redirects=False)
    except httpx.TimeoutException:
        raise _ProbeFailure(
            "network_failure", "The endpoint did not respond before the configured timeout.",
        ) from None
    except httpx.RequestError:
        raise _ProbeFailure(
            "network_failure", "The configured endpoint could not be reached.",
        ) from None

    try:
        if response.is_redirect:
            raise _ProbeFailure(
                "unsafe_redirect", "The endpoint attempted a redirect, which was not followed.",
            )
        if response.status_code >= 400:
            raise _http_status_failure(
                response.status_code, capability_probe=probe != "chat",
            )
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise _ProbeFailure(
                    "invalid_response", "The endpoint returned an oversized response.",
                )
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise _ProbeFailure(
                "invalid_response", "The endpoint returned an invalid JSON response.",
            ) from None
        if not isinstance(payload, dict):
            raise _ProbeFailure(
                "invalid_response", "The endpoint returned an invalid response shape.",
            )
        return payload
    finally:
        await response.aclose()


async def _call_probe(
    profile: AIProviderProfile,
    credential: str | None,
    probe: str,
    *,
    endpoint: PinnedEndpoint | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    if endpoint is not None:
        assert client is not None
        return await _post_compatible(client, endpoint, credential, probe, profile)

    try:
        return await _litellm_acompletion(**_litellm_payload(profile, credential, probe))
    except Exception as exc:
        category, message = _classify_provider_exception(exc)
        raise _ProbeFailure(category, message) from None


def _tool_probe_succeeded(response: Any) -> bool:
    message = _first_message(response)
    tool_calls = _get(message, "tool_calls", []) if message is not None else []
    for tool_call in tool_calls or []:
        function = _get(tool_call, "function", {})
        if _get(function, "name") == _PROBE_TOOL_NAME:
            return True
    return False


def _json_probe_succeeded(response: Any) -> bool:
    message = _first_message(response)
    if message is None:
        return False
    content = _get(message, "content")
    if content is None:
        parsed = _get(message, "parsed")
        return isinstance(parsed, dict) and parsed.get("ok") is True
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
    except ValueError:
        return False
    return isinstance(parsed, dict) and parsed.get("ok") is True


def _failure_outcome(
    category: str,
    message: str,
    *,
    capabilities: dict[str, str] | None = None,
) -> ProviderConnectionOutcome:
    return ProviderConnectionOutcome(
        status="failed",
        category=category,
        message=message,
        capabilities=capabilities or _unknown_capabilities(),
        checked_at=datetime.now(timezone.utc),
    )


async def test_provider_profile(
    profile: AIProviderProfile,
) -> ProviderConnectionOutcome:
    """Explicitly test only a saved profile with short synthetic prompts."""
    definition = get_provider(profile.provider_id)
    if definition is None:
        return _failure_outcome(
            "unsupported_provider", "This provider is not in the supported catalog.",
        )
    if definition.api_key_requirement == "required" and not profile.credential_configured:
        return _failure_outcome(
            "missing_api_key", "Configure this provider's API credential before testing.",
        )

    credential = None
    if profile.credential_configured:
        try:
            credential = decrypt_provider_credential(
                profile.credential_ciphertext,
                profile.credential_key_version,
            )
        except ProviderCredentialConfigurationError as exc:
            return _failure_outcome("credential_configuration_error", str(exc))

    endpoint = None
    client = None
    if definition.endpoint_mode in {
        "local_openai_compatible", "custom_openai_compatible",
    }:
        try:
            endpoint = await resolve_and_pin_endpoint_async(
                profile.provider_id,
                profile.endpoint_url or "",
            )
        except ProviderManagementError as exc:
            return _failure_outcome(exc.code, exc.message)
        client = _create_http_client()

    capabilities = _unknown_capabilities()
    try:
        if endpoint is not None:
            assert client is not None
            async with client:
                chat_response = await _call_probe(
                    profile, credential, "chat", endpoint=endpoint, client=client,
                )
                message = _first_message(chat_response)
                if message is None:
                    return _failure_outcome(
                        "invalid_response", "The provider returned no chat response.",
                    )
                capabilities["chat_completion"] = "supported"
                return await _run_capability_probes(
                    profile, credential, endpoint, client, capabilities,
                )
        else:
            chat_response = await _call_probe(profile, credential, "chat")
            if _first_message(chat_response) is None:
                return _failure_outcome(
                    "invalid_response", "The provider returned no chat response.",
                )
            capabilities["chat_completion"] = "supported"
            return await _run_capability_probes(
                profile, credential, None, None, capabilities,
            )
    except _ProbeFailure as exc:
        logger.info(
            "AI provider connection test failed (provider=%s category=%s)",
            profile.provider_id,
            exc.category,
        )
        return _failure_outcome(exc.category, exc.message, capabilities=capabilities)
    except ProviderManagementError as exc:
        return _failure_outcome(exc.code, exc.message, capabilities=capabilities)
    except Exception as exc:
        # Keep provider exception text, headers, and request payload out of logs.
        logger.warning(
            "AI provider connection test failed (provider=%s error_type=%s)",
            profile.provider_id,
            type(exc).__name__,
        )
        return _failure_outcome(
            "provider_error",
            "The provider could not complete the connection test.",
            capabilities=capabilities,
        )
    finally:
        if client is not None and not client.is_closed:
            await client.aclose()


async def _run_capability_probes(
    profile: AIProviderProfile,
    credential: str | None,
    endpoint: PinnedEndpoint | None,
    client: httpx.AsyncClient | None,
    capabilities: dict[str, str],
) -> ProviderConnectionOutcome:
    probe_failures: list[str] = []
    try:
        tool_response = await _call_probe(
            profile, credential, "tools", endpoint=endpoint, client=client,
        )
        capabilities["tool_calling"] = (
            "supported" if _tool_probe_succeeded(tool_response) else "unsupported"
        )
    except _ProbeFailure as exc:
        if exc.category == "capability_failure":
            capabilities["tool_calling"] = "unsupported"
        else:
            probe_failures.append(exc.category)

    try:
        schema_response = await _call_probe(
            profile, credential, "json_schema", endpoint=endpoint, client=client,
        )
        schema_supported = _json_probe_succeeded(schema_response)
        capabilities["structured_output"] = (
            "supported" if schema_supported else "unsupported"
        )
        capabilities["json_schema"] = (
            "supported" if schema_supported else "unsupported"
        )
    except _ProbeFailure as exc:
        if exc.category == "capability_failure":
            capabilities["structured_output"] = "unsupported"
            capabilities["json_schema"] = "unsupported"
        else:
            probe_failures.append(exc.category)

    if probe_failures:
        return ProviderConnectionOutcome(
            status="connected",
            category="capability_validation_incomplete",
            message=(
                "Chat completion succeeded, but one or more capability probes "
                "could not be verified."
            ),
            capabilities=capabilities,
            checked_at=datetime.now(timezone.utc),
        )
    return ProviderConnectionOutcome(
        status="connected",
        category="connected",
        message="Provider connection and supported capability probes completed.",
        capabilities=capabilities,
        checked_at=datetime.now(timezone.utc),
    )
