"""Invocation-scoped provider resolution and secure model calls."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx
from sqlalchemy.orm import Session as DBSession

from app.config.settings import settings
from app.db import crud
from app.services.provider_assignments import get_assignment
from app.services.provider_credentials import (
    ProviderCredentialConfigurationError,
    decrypt_provider_credential,
)
from app.services.provider_endpoint_security import (
    PinnedEndpoint,
    resolve_and_pin_endpoint,
    resolve_and_pin_endpoint_async,
    validate_endpoint_syntax,
)
from app.services.provider_errors import ProviderManagementError
from app.services.provider_profiles import validate_configuration
from app.services.provider_registry import (
    ROLE_REQUIRED_CAPABILITIES,
    get_provider,
    litellm_api_base_for,
    resolve_litellm_model,
    supported_configuration_for,
)

logger = logging.getLogger(__name__)
_MAX_COMPLETION_BYTES = 2 * 1024 * 1024


class ProviderRuntimeError(RuntimeError):
    """Safe, actionable provider execution error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_request_failed",
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class RuntimeProvider:
    """A single execution's provider data; never store or serialize this object."""

    user_id: str
    agent_role: str
    provider_id: str
    model_identifier: str
    model: str
    endpoint_url: str | None
    configuration: Mapping[str, Any]
    capabilities: Mapping[str, str]
    unknown_required_capabilities: tuple[str, ...]
    credential: str | None = field(default=None, repr=False, compare=False)

    @property
    def display_name(self) -> str:
        definition = get_provider(self.provider_id)
        return definition.display_name if definition else "selected provider"


def _required_configuration_message(agent_role: str) -> str:
    role_label = "Pipeline Agent" if agent_role == "pipeline_agent" else "Report Agent"
    return (
        f"Configure and successfully test a provider for the {role_label} in AI Provider Settings."
    )


def resolve_assigned_provider(
    db: DBSession,
    *,
    session_id: str,
    expected_user_id: str,
    agent_role: str,
    run_id: str | None = None,
) -> RuntimeProvider:
    """Resolve an assignment only after rechecking trusted session/run ownership."""
    if agent_role not in ROLE_REQUIRED_CAPABILITIES:
        raise ProviderRuntimeError(
            "The requested Agent provider role is not supported.", code="invalid_agent_role",
        )
    if not expected_user_id:
        raise ProviderRuntimeError(
            "This Agent execution has no authenticated session owner.",
            code="provider_owner_required",
        )

    session = crud.get_session(db, session_id)
    if session is None or str(session.user_id) != str(expected_user_id):
        raise ProviderRuntimeError(
            "The session is not available to the authenticated owner.",
            code="provider_owner_mismatch",
        )
    if run_id is not None:
        run = crud.get_run(db, run_id)
        if (
            run is None
            or str(run.session_id) != str(session_id)
            or str(run.requested_by) != str(expected_user_id)
        ):
            raise ProviderRuntimeError(
                "The analysis run is not available to the authenticated session owner.",
                code="provider_run_owner_mismatch",
            )

    assignment_row = get_assignment(
        db, user_id=str(expected_user_id), agent_role=agent_role,
    )
    if assignment_row is None:
        raise ProviderRuntimeError(
            _required_configuration_message(agent_role),
            code="provider_assignment_required",
        )
    assignment, profile, _unverified = assignment_row
    if assignment.agent_role != agent_role or str(profile.user_id) != str(expected_user_id):
        raise ProviderRuntimeError(
            "The selected provider is not available to this Agent owner.",
            code="provider_owner_mismatch",
        )
    if profile.connection_status != "connected":
        raise ProviderRuntimeError(
            _required_configuration_message(agent_role),
            code="provider_connection_test_required",
        )

    definition = get_provider(profile.provider_id)
    if definition is None:
        raise ProviderRuntimeError(
            "The selected provider is no longer supported. Choose another provider profile.",
            code="unsupported_provider",
        )

    capabilities = dict(profile.capability_status_json or {})
    required = ROLE_REQUIRED_CAPABILITIES[agent_role]
    unsupported = [
        capability for capability in required
        if capabilities.get(capability, "unknown") == "unsupported"
    ]
    if unsupported:
        raise ProviderRuntimeError(
            "The selected model does not support capabilities required by this Agent. "
            "Choose a compatible model and test its capabilities.",
            code="incompatible_provider_capabilities",
        )
    unknown = tuple(
        capability for capability in required
        if capabilities.get(capability, "unknown") == "unknown"
    )

    try:
        configuration = validate_configuration(
            dict(profile.configuration_json or {}), provider_id=profile.provider_id,
        )
        endpoint_url = validate_endpoint_syntax(
            profile.provider_id, profile.endpoint_url,
        )
    except ProviderManagementError:
        raise ProviderRuntimeError(
            "The selected provider profile has invalid configuration. Review and save it again.",
            code="invalid_provider_configuration",
        ) from None

    if definition.api_key_requirement == "required" and not profile.credential_configured:
        raise ProviderRuntimeError(
            "Add a provider credential, then test the provider connection.",
            code="provider_credential_required",
        )
    credential = None
    if profile.credential_configured:
        try:
            credential = decrypt_provider_credential(
                profile.credential_ciphertext,
                profile.credential_key_version,
            )
        except ProviderCredentialConfigurationError:
            raise ProviderRuntimeError(
                "The saved provider credential cannot be decrypted. Re-enter it and test the connection.",
                code="provider_credential_unavailable",
            ) from None

    try:
        model = (
            resolve_litellm_model(profile.provider_id, profile.model_identifier)
            if definition.litellm_provider
            else profile.model_identifier
        )
    except ValueError:
        raise ProviderRuntimeError(
            "The selected provider profile has an invalid model configuration.",
            code="invalid_provider_model",
        ) from None

    return RuntimeProvider(
        user_id=str(expected_user_id),
        agent_role=agent_role,
        provider_id=profile.provider_id,
        model_identifier=profile.model_identifier,
        model=model,
        endpoint_url=endpoint_url,
        configuration=MappingProxyType(configuration),
        capabilities=MappingProxyType(capabilities),
        unknown_required_capabilities=unknown,
        credential=credential,
    )


def _generation_parameters(
    provider: RuntimeProvider,
    *,
    role: str,
    timeout: float,
) -> dict[str, Any]:
    allowed = supported_configuration_for(provider.provider_id)
    invalid = set(provider.configuration) - set(allowed)
    if invalid:
        raise ProviderRuntimeError(
            "This provider does not support one or more saved generation settings. "
            "Edit the profile and remove unsupported settings.",
            code="unsupported_provider_parameter",
        )
    defaults = {
        "temperature": 0.2 if role == "report_agent" else settings.LLM_TEMPERATURE,
        "max_tokens": settings.REPORT_LLM_MAX_TOKENS
        if role == "report_agent" else settings.LLM_MAX_TOKENS,
    }
    parameters = {
        key: provider.configuration.get(key, defaults.get(key))
        for key in allowed
        if provider.configuration.get(key, defaults.get(key)) is not None
    }
    parameters["timeout"] = float(timeout)
    return parameters


def _check_operation_capabilities(
    provider: RuntimeProvider,
    *,
    tools: list[dict] | None,
    response_format: dict | None,
) -> None:
    requested = []
    if tools:
        requested.append("tool_calling")
    if response_format:
        requested.extend(("structured_output", "json_schema"))
    unsupported = [
        capability for capability in requested
        if provider.capabilities.get(capability, "unknown") == "unsupported"
    ]
    if unsupported:
        raise ProviderRuntimeError(
            "The selected model does not support a capability required for this request. "
            "Choose a compatible model and test its capabilities.",
            code="incompatible_provider_capabilities",
        )


def _openai_sdk_client(provider: RuntimeProvider, *, timeout: float, asynchronous: bool):
    """Build an invocation-only OpenAI client so LiteLLM cannot retain its API key cache-side."""
    if provider.provider_id != "openai":
        return None
    if not provider.credential:
        raise ProviderRuntimeError(
            "Add an OpenAI credential to this provider profile before running the Agent.",
            code="provider_credential_required",
        )
    try:
        if asynchronous:
            from openai import AsyncOpenAI

            return AsyncOpenAI(
                api_key=provider.credential,
                base_url=litellm_api_base_for(provider.provider_id),
                timeout=timeout,
                max_retries=0,
            )
        from openai import OpenAI

        return OpenAI(
            api_key=provider.credential,
            base_url=litellm_api_base_for(provider.provider_id),
            timeout=timeout,
            max_retries=0,
        )
    except Exception as exc:
        raise _safe_provider_failure(
            provider=provider, error_type=type(exc).__name__,
        ) from None


async def _close_openai_async_client(client, provider: RuntimeProvider) -> None:
    if client is None:
        return
    try:
        await client.close()
    except Exception as exc:
        logger.warning(
            "Invocation provider client cleanup failed (provider=%s error_type=%s)",
            provider.provider_id,
            type(exc).__name__,
        )


def _close_openai_sync_client(client, provider: RuntimeProvider) -> None:
    if client is None:
        return
    try:
        client.close()
    except Exception as exc:
        logger.warning(
            "Invocation provider client cleanup failed (provider=%s error_type=%s)",
            provider.provider_id,
            type(exc).__name__,
        )


def _safe_provider_failure(
    *,
    provider: RuntimeProvider,
    status_code: int | None = None,
    error_type: str = "",
) -> ProviderRuntimeError:
    error_name = error_type.casefold()
    retryable = status_code in {408, 429, 500, 502, 503, 504} or any(
        item in error_name for item in ("timeout", "connect", "network", "serviceunavailable")
    )
    if status_code in {401, 403} or any(
        item in error_name for item in ("authentication", "permission", "unauthorized")
    ):
        return ProviderRuntimeError(
            f"{provider.display_name} rejected the saved credential. Update it and test the connection.",
            code="provider_authentication_failed", status_code=status_code,
        )
    if status_code == 404 or "modelnotfound" in error_name:
        return ProviderRuntimeError(
            f"{provider.display_name} did not recognize this model identifier. Check the profile model name.",
            code="provider_model_not_found", status_code=status_code,
        )
    if status_code == 429 or "ratelimit" in error_name:
        return ProviderRuntimeError(
            f"{provider.display_name} is rate limiting this request. Retry later.",
            code="provider_rate_limited", status_code=status_code, retryable=True,
        )
    if status_code is not None and status_code >= 500:
        return ProviderRuntimeError(
            f"{provider.display_name} returned a temporary server error. Retry later.",
            code="provider_server_error", status_code=status_code, retryable=True,
        )
    if status_code in {400, 422}:
        message = (
            f"{provider.display_name} rejected the required Agent capability request. "
            "Run the profile capability test or choose a compatible model."
        )
        return ProviderRuntimeError(
            message, code="provider_capability_rejected", status_code=status_code,
        )
    if "timeout" in error_name:
        return ProviderRuntimeError(
            f"{provider.display_name} did not respond before the configured timeout.",
            code="provider_timeout", retryable=True,
        )
    if any(item in error_name for item in ("connect", "network")):
        return ProviderRuntimeError(
            f"Could not connect to {provider.display_name}. Test the saved endpoint and retry.",
            code="provider_network_error", retryable=True,
        )
    return ProviderRuntimeError(
        f"The {provider.display_name} request failed. Test the profile and check model compatibility.",
        code="provider_request_failed", retryable=retryable,
    )


def _completion_url(endpoint: PinnedEndpoint) -> str:
    parsed = urlsplit(endpoint.base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        target_path = path
    elif path.endswith("/v1"):
        target_path = path + "/chat/completions"
    else:
        target_path = path + "/v1/chat/completions"
    return f"{endpoint.base_url[:len(endpoint.base_url) - len(parsed.path)]}{target_path}"


def _compatible_payload(
    provider: RuntimeProvider,
    *,
    messages: list[dict[str, Any]],
    parameters: dict[str, Any],
    tools: list[dict] | None,
    tool_choice: Any,
    response_format: dict | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model_identifier,
        "messages": messages,
        "stream": False,
        **parameters,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if response_format:
        payload["response_format"] = response_format
    return payload


def _raise_status_error(provider: RuntimeProvider, status_code: int) -> None:
    if status_code in {301, 302, 303, 307, 308}:
        raise ProviderRuntimeError(
            "The configured provider endpoint attempted a redirect, which was not followed.",
            code="unsafe_provider_redirect", status_code=status_code,
        )
    if status_code >= 400:
        raise _safe_provider_failure(provider=provider, status_code=status_code)


def _decode_bounded_response(response, provider: RuntimeProvider) -> dict[str, Any]:
    _raise_status_error(provider, response.status_code)
    if response.is_redirect:
        raise ProviderRuntimeError(
            "The configured provider endpoint attempted a redirect, which was not followed.",
            code="unsafe_provider_redirect", status_code=response.status_code,
        )
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_COMPLETION_BYTES:
            raise ProviderRuntimeError(
                "The provider response exceeded the allowed size.",
                code="provider_response_too_large",
            )
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise ProviderRuntimeError(
            "The provider returned an invalid response. Test the profile and model compatibility.",
            code="provider_invalid_response",
        ) from None
    if not isinstance(payload, dict):
        raise ProviderRuntimeError(
            "The provider returned an invalid response shape.", code="provider_invalid_response",
        )
    return payload


async def _direct_completion_async(
    provider: RuntimeProvider,
    *,
    messages: list[dict[str, Any]],
    parameters: dict[str, Any],
    tools: list[dict] | None,
    tool_choice: Any,
    response_format: dict | None,
) -> dict[str, Any]:
    assert provider.endpoint_url is not None
    try:
        endpoint = await resolve_and_pin_endpoint_async(provider.provider_id, provider.endpoint_url)
    except ProviderManagementError:
        raise ProviderRuntimeError(
            "The configured provider endpoint failed security validation.",
            code="unsafe_provider_endpoint",
        ) from None
    timeout = httpx.Timeout(parameters["timeout"], connect=min(5.0, parameters["timeout"]))
    try:
        async with httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=timeout,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        ) as client:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Host": endpoint.host_header,
            }
            if provider.credential:
                headers["Authorization"] = f"Bearer {provider.credential}"
            extensions = {"sni_hostname": endpoint.sni_hostname} if endpoint.sni_hostname else {}
            request = client.build_request(
                "POST",
                _completion_url(endpoint),
                headers=headers,
                json=_compatible_payload(
                    provider,
                    messages=messages,
                    parameters={key: value for key, value in parameters.items() if key != "timeout"},
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                ),
                extensions=extensions,
            )
            response = await client.send(request, stream=True, follow_redirects=False)
            try:
                _raise_status_error(provider, response.status_code)
                if response.is_redirect:
                    _raise_status_error(provider, response.status_code)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_COMPLETION_BYTES:
                        raise ProviderRuntimeError(
                            "The provider response exceeded the allowed size.",
                            code="provider_response_too_large",
                        )
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    raise ProviderRuntimeError(
                        "The provider returned an invalid response. Test the profile and model compatibility.",
                        code="provider_invalid_response",
                    ) from None
                if not isinstance(payload, dict):
                    raise ProviderRuntimeError(
                        "The provider returned an invalid response shape.",
                        code="provider_invalid_response",
                    )
                return payload
            finally:
                await response.aclose()
    except ProviderRuntimeError:
        raise
    except httpx.TimeoutException as exc:
        raise _safe_provider_failure(provider=provider, error_type=type(exc).__name__) from None
    except httpx.RequestError as exc:
        raise _safe_provider_failure(provider=provider, error_type=type(exc).__name__) from None


def _direct_completion_sync(
    provider: RuntimeProvider,
    *,
    messages: list[dict[str, Any]],
    parameters: dict[str, Any],
    tools: list[dict] | None,
    tool_choice: Any,
    response_format: dict | None,
) -> dict[str, Any]:
    assert provider.endpoint_url is not None
    try:
        endpoint = resolve_and_pin_endpoint(provider.provider_id, provider.endpoint_url)
    except ProviderManagementError:
        raise ProviderRuntimeError(
            "The configured provider endpoint failed security validation.",
            code="unsafe_provider_endpoint",
        ) from None
    timeout = httpx.Timeout(parameters["timeout"], connect=min(5.0, parameters["timeout"]))
    try:
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=timeout,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        ) as client:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Host": endpoint.host_header,
            }
            if provider.credential:
                headers["Authorization"] = f"Bearer {provider.credential}"
            extensions = {"sni_hostname": endpoint.sni_hostname} if endpoint.sni_hostname else {}
            request = client.build_request(
                "POST",
                _completion_url(endpoint),
                headers=headers,
                json=_compatible_payload(
                    provider,
                    messages=messages,
                    parameters={key: value for key, value in parameters.items() if key != "timeout"},
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                ),
                extensions=extensions,
            )
            response = client.send(request, stream=True, follow_redirects=False)
            try:
                return _decode_bounded_response(response, provider)
            finally:
                response.close()
    except ProviderRuntimeError:
        raise
    except httpx.TimeoutException as exc:
        raise _safe_provider_failure(provider=provider, error_type=type(exc).__name__) from None
    except httpx.RequestError as exc:
        raise _safe_provider_failure(provider=provider, error_type=type(exc).__name__) from None


async def complete_async(
    provider: RuntimeProvider | None,
    *,
    role: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    response_format: dict | None = None,
    timeout: float,
) -> Any:
    if provider is None:
        raise ProviderRuntimeError(
            "Configure a provider for this Agent before starting a run.",
            code="provider_assignment_required",
        )
    if provider.agent_role != role:
        raise ProviderRuntimeError(
            "The selected provider assignment is not configured for this Agent.",
            code="provider_role_mismatch",
        )
    _check_operation_capabilities(provider, tools=tools, response_format=response_format)
    parameters = _generation_parameters(provider, role=role, timeout=timeout)
    definition = get_provider(provider.provider_id)
    if definition is None:
        raise ProviderRuntimeError("The selected provider is not supported.", code="unsupported_provider")
    if definition.litellm_provider is None:
        return await _direct_completion_async(
            provider,
            messages=messages,
            parameters=parameters,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "api_base": litellm_api_base_for(provider.provider_id),
        "messages": messages,
        "timeout": parameters.pop("timeout"),
        "stream": False,
        "num_retries": 0,
        **parameters,
    }
    if provider.credential:
        kwargs["api_key"] = provider.credential
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if response_format:
        kwargs["response_format"] = response_format
    try:
        import litellm

        client = _openai_sdk_client(
            provider, timeout=kwargs["timeout"], asynchronous=True,
        )
        if client is not None:
            kwargs["client"] = client
        return await litellm.acompletion(**kwargs)
    except ProviderRuntimeError:
        raise
    except Exception as exc:
        safe = _safe_provider_failure(
            provider=provider,
            status_code=getattr(exc, "status_code", None),
            error_type=type(exc).__name__,
        )
        logger.warning(
            "User provider request failed (provider=%s role=%s error_type=%s code=%s)",
            provider.provider_id,
            role,
            type(exc).__name__,
            safe.code,
        )
        raise safe from None
    finally:
        if "client" in locals():
            await _close_openai_async_client(client, provider)


def complete_sync(
    provider: RuntimeProvider,
    *,
    role: str,
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    response_format: dict | None = None,
    timeout: float,
) -> Any:
    if provider.agent_role != role:
        raise ProviderRuntimeError(
            "The selected provider assignment is not configured for this Agent.",
            code="provider_role_mismatch",
        )
    _check_operation_capabilities(provider, tools=tools, response_format=response_format)
    parameters = _generation_parameters(provider, role=role, timeout=timeout)
    definition = get_provider(provider.provider_id)
    if definition is None:
        raise ProviderRuntimeError("The selected provider is not supported.", code="unsupported_provider")
    if definition.litellm_provider is None:
        return _direct_completion_sync(
            provider,
            messages=messages,
            parameters=parameters,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "api_base": litellm_api_base_for(provider.provider_id),
        "messages": messages,
        "timeout": parameters.pop("timeout"),
        "stream": False,
        "num_retries": 0,
        **parameters,
    }
    if provider.credential:
        kwargs["api_key"] = provider.credential
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if response_format:
        kwargs["response_format"] = response_format
    try:
        import litellm

        client = _openai_sdk_client(
            provider, timeout=kwargs["timeout"], asynchronous=False,
        )
        if client is not None:
            kwargs["client"] = client
        return litellm.completion(**kwargs)
    except Exception as exc:
        if isinstance(exc, ProviderRuntimeError):
            raise
        safe = _safe_provider_failure(
            provider=provider,
            status_code=getattr(exc, "status_code", None),
            error_type=type(exc).__name__,
        )
        logger.warning(
            "User provider request failed (provider=%s role=%s error_type=%s code=%s)",
            provider.provider_id,
            role,
            type(exc).__name__,
            safe.code,
        )
        raise safe from None
    finally:
        if "client" in locals():
            _close_openai_sync_client(client, provider)
