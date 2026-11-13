"""Lazy LiteLLM client adapter used only by report graph nodes."""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings

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


SUPPORTED_REPORT_MODELS = {
    "groq": frozenset({"openai/gpt-oss-20b"}),
}

_GROQ_SCHEMA_KEYS = frozenset({
    "$defs", "$ref", "type", "properties", "items", "anyOf", "enum", "description",
})


def _groq_strict_schema(value: dict) -> dict:
    """Reduce Pydantic's schema to Groq's strict Structured Outputs subset."""
    def clean(node):
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, item in node.items():
            if key not in _GROQ_SCHEMA_KEYS:
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


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0, math.ceil(float(raw)))
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(str(raw))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return max(0, math.ceil((deadline - datetime.now(timezone.utc)).total_seconds()))
        except (TypeError, ValueError, OverflowError):
            return None


def _safe_provider_error(exc: Exception) -> ReportProviderError:
    """Map LiteLLM/Groq errors to safe, actionable messages without echoing bodies."""
    status = _status_code(exc)
    name = type(exc).__name__.lower()
    detail = str(exc).lower()  # Used only for classification; never returned or logged.
    quota_markers = (
        "quota", "insufficient_quota", "credit", "billing", "tokens per day",
        "requests per day", "daily limit", "usage limit",
    )
    if status == 402 or any(marker in detail for marker in quota_markers):
        retry_after = _retry_after_seconds(exc)
        wait = f" Groq advises retrying after {retry_after} seconds." if retry_after is not None else ""
        return ReportProviderError(
            "Groq reports that the account quota or credits are exhausted." + wait + " The report request stopped; no retry was made.",
            error_code="report_quota_exceeded",
            retry_after_seconds=retry_after,
        )
    if status == 401 or "authentication" in name or "invalid_api_key" in detail:
        return ReportProviderError(
            "Groq rejected the configured API key (HTTP 401). Verify REPORT_LLM_API_KEY in the local environment.",
            error_code="report_authentication_failed",
        )
    if status == 403 or "permission" in name:
        return ReportProviderError(
            "Groq denied access to this model. Verify that the account can use openai/gpt-oss-20b.",
            error_code="report_model_access_denied",
        )
    if status == 404 or ("model" in detail and ("not found" in detail or "does not exist" in detail)):
        return ReportProviderError(
            "Groq could not find openai/gpt-oss-20b for this account. Verify model access and configuration.",
            error_code="report_model_not_found",
        )
    if status == 429 or "ratelimit" in name or "rate_limit" in detail:
        retry_after = _retry_after_seconds(exc)
        wait = f" Retry after {retry_after} seconds, as directed by Groq." if retry_after is not None else ""
        return ReportProviderError(
            "Groq rate limit reached." + wait + " The report request stopped without an automatic retry.",
            error_code="report_rate_limited",
            retry_after_seconds=retry_after,
        )
    if status == 408 or "timeout" in name or "timed out" in detail:
        return ReportProviderError(
            "The Groq request timed out. No automatic retry was made.",
            error_code="report_provider_timeout",
        )
    if "connection" in name or "network" in name or "connecterror" in name:
        return ReportProviderError(
            "Could not connect to Groq. Check network access and try again later.",
            error_code="report_network_error",
        )
    if status == 400:
        return ReportProviderError(
            "Groq rejected the request (HTTP 400). Check the strict JSON Schema and model parameters; no unstructured fallback was attempted.",
            error_code="report_structured_request_rejected",
        )
    if status is not None and status >= 500:
        return ReportProviderError(
            f"Groq returned a server error (HTTP {status}). The report request stopped without an automatic retry.",
            error_code="report_provider_server_error",
        )
    return ReportProviderError(
        "The Groq report request failed. Check provider configuration and connectivity; no automatic retry was made.",
        error_code="report_provider_error",
    )


def resolve_report_model() -> str:
    provider = str(settings.REPORT_LLM_PROVIDER or "").strip().lower()
    model = str(settings.REPORT_LLM_MODEL or "").strip()
    prefix = f"{provider}/"
    normalized_model = model.removeprefix(prefix) if provider else model
    if provider not in SUPPORTED_REPORT_MODELS:
        raise ReportAgentConfigurationError(
            "REPORT_LLM_PROVIDER must be 'groq' for the configured Report Agent."
        )
    if normalized_model not in SUPPORTED_REPORT_MODELS[provider]:
        raise ReportAgentConfigurationError(
            "REPORT_LLM_MODEL must be 'openai/gpt-oss-20b' for the configured Groq Report Agent."
        )
    return f"{provider}/{normalized_model}"


def _litellm_completion(**kwargs):
    # Import lazily: app startup and unrelated tests do not need provider setup.
    import litellm

    return litellm.completion(**kwargs)


def _get(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _parse_json(content: str) -> dict:
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ReportProviderError(
            "Groq returned malformed JSON for the requested report schema.",
            error_code="report_malformed_json",
        ) from None
    if not isinstance(value, dict):
        raise ReportProviderError(
            "Groq returned an invalid report response shape.",
            error_code="report_invalid_response_shape",
        )
    return value


def invoke_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: type[SchemaT],
) -> SchemaT:
    """Call the configured provider once and validate its JSON with Pydantic."""
    model = resolve_report_model()
    api_key = str(settings.REPORT_LLM_API_KEY or "").strip()
    if not api_key:
        raise ReportAgentConfigurationError(
            "REPORT_LLM_API_KEY is missing. Configure it with the existing Groq API token to enable the Report Agent."
        )
    try:
        response = _litellm_completion(
            model=model,
            api_key=api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-z0-9_]", "_", response_schema.__name__.lower()),
                    "strict": True,
                    "schema": _groq_strict_schema(response_schema.model_json_schema()),
                },
            },
            temperature=0.2,
            max_tokens=int(settings.REPORT_LLM_MAX_TOKENS),
            timeout=float(settings.REPORT_LLM_TIMEOUT_SECONDS),
            stream=False,
            num_retries=0,
        )
    except ReportAgentConfigurationError:
        raise
    except Exception as exc:
        safe_error = _safe_provider_error(exc)
        logger.warning(
            "Report provider request failed (error_type=%s, error_code=%s, status=%s)",
            type(exc).__name__, safe_error.error_code, _status_code(exc),
        )
        raise safe_error from None

    choices = _get(response, "choices", [])
    if not choices:
        raise ReportProviderError(
            "Groq returned no report response choices.", error_code="report_empty_response"
        )
    message = _get(choices[0], "message", {})
    content = _get(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise ReportProviderError(
            "Groq returned empty report content.", error_code="report_empty_response"
        )
    try:
        return response_schema.model_validate(_parse_json(content))
    except ValidationError as exc:
        raise ReportProviderError(
            "Groq's report response did not pass Pydantic schema validation.",
            error_code="report_schema_validation_failed",
        ) from None
