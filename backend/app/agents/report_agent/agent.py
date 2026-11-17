"""Lazy LiteLLM client adapter used only by report graph nodes."""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config.settings import settings
from app.services.provider_runtime import RuntimeProvider, ProviderRuntimeError, complete_sync

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ReportAgentConfigurationError(RuntimeError):
    """Raised when report-provider configuration is incomplete."""


class ReportProviderError(RuntimeError):
    """Raised for provider, timeout, rate-limit, or malformed response errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "report_provider_error",
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after_seconds = retry_after_seconds


_STRICT_SCHEMA_KEYS = frozenset({
    "$defs", "$ref", "type", "properties", "items", "anyOf", "enum", "description",
})


def _strict_json_schema(value: dict) -> dict:
    """Normalize Pydantic's schema for strict structured-output providers."""
    def clean(node):
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, item in node.items():
            if key not in _STRICT_SCHEMA_KEYS:
                continue
            if key in {"properties", "$defs"} and isinstance(item, dict):
                result[key] = {name: clean(definition) for name, definition in item.items()}
            else:
                result[key] = clean(item)
        if result.get("type") == "object":
            properties = result.get("properties", {})
            result["required"] = list(properties)
            result["additionalProperties"] = False
        return result

    return clean(value)


def resolve_report_model(provider: RuntimeProvider | None) -> str:
    """Return the selected invocation's already-resolved model identifier."""
    if provider is None or provider.agent_role != "report_agent":
        raise ReportAgentConfigurationError(
            "A Report Agent provider profile is required for report generation."
        )
    return provider.model


def _get(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _invoke_provider(provider: RuntimeProvider, **kwargs):
    return complete_sync(
        provider,
        role="report_agent",
        timeout=settings.REPORT_LLM_TIMEOUT_SECONDS,
        **kwargs,
    )


def _parse_json(content: str) -> dict:
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ReportProviderError(
            "The selected model returned malformed JSON for the requested report schema.",
            error_code="report_malformed_json",
        ) from None
    if not isinstance(value, dict):
        raise ReportProviderError(
            "The selected model returned an invalid report response shape.",
            error_code="report_invalid_response_shape",
        )
    return value


def invoke_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[SchemaT],
    provider: RuntimeProvider | None = None,
) -> SchemaT:
    """Call one invocation-scoped provider and validate JSON with Pydantic."""
    if provider is None or provider.agent_role != "report_agent":
        raise ReportAgentConfigurationError(
            "Configure and test a provider for the Report Agent in AI Provider Settings before generating reports."
        )
    try:
        response = _invoke_provider(
            provider,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-z0-9_]", "_", response_schema.__name__.lower()),
                    "strict": True,
                    "schema": _strict_json_schema(response_schema.model_json_schema()),
                },
            },
        )
    except ProviderRuntimeError as exc:
        raise ReportProviderError(
            str(exc), error_code=f"report_{exc.code}",
        ) from None
    except Exception as exc:
        logger.warning(
            "Report provider request failed (provider=%s error_type=%s)",
            provider.provider_id,
            type(exc).__name__,
        )
        raise ReportProviderError(
            "The selected report provider failed. Test its profile and model compatibility.",
            error_code="report_provider_request_failed",
        ) from None

    choices = _get(response, "choices", [])
    if not choices:
        raise ReportProviderError(
            "The selected model returned no report response choices.", error_code="report_empty_response"
        )
    message = _get(choices[0], "message", {})
    content = _get(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise ReportProviderError(
            "The selected model returned empty report content.", error_code="report_empty_response"
        )
    try:
        return response_schema.model_validate(_parse_json(content))
    except ValidationError as exc:
        raise ReportProviderError(
            "The report response did not pass Pydantic schema validation.",
            error_code="report_schema_validation_failed",
        ) from None
