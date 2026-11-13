"""
LiteLLM client and response normalisation for the agent.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import litellm
from litellm import acompletion
from litellm.utils import token_counter as original_token_counter

# ==============================================================================
# MONKEY PATCH: Fix LiteLLM internal crash when using Ollama tool calls.
# LiteLLM converts arguments to dict internally and crashes in token_counter.
# This patch forces it to serialize back to string before counting.
# ==============================================================================
def _safe_token_counter(*args, **kwargs):
    """Keep LiteLLM's non-critical usage counter from blocking Ollama calls."""
    call_args = list(args)
    call_kwargs = dict(kwargs)

    # token_counter's positional signature is:
    # model, custom_tokenizer, text, messages, ...
    messages = call_kwargs.get("messages")
    messages_position = 3
    if messages is None and len(call_args) > messages_position:
        messages = call_args[messages_position]

    has_tool_calls = False
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if isinstance(message, Mapping) and message.get("tool_calls"):
                has_tool_calls = True

    # LiteLLM 1.55.x concatenates tool arguments without coercing them and can
    # also raise a duplicate-model TypeError in this internal accounting path.
    # Ollama supplies prompt_eval_count/eval_count itself, so token accounting
    # must never be allowed to fail the actual completion request.
    if has_tool_calls:
        return 0

    # LiteLLM 1.55.x's token_counter concatenates function arguments directly
    # and has provider-specific signature issues. A zero fallback is safe for
    # this non-critical accounting path and must never block the model response.
    try:
        return original_token_counter(*call_args, **call_kwargs)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "LiteLLM token counting failed; returning zero usage tokens: %s",
            exc,
        )
        return 0

litellm.token_counter = _safe_token_counter
# ==============================================================================

from app.config import settings

logger = logging.getLogger(__name__)

_PROVIDER_PREFIX = {
    "groq": "groq/",
    "openai": "openai/",
    "anthropic": "anthropic/",
    "google": "gemini/",
    "ollama": "ollama_chat/",
}

_OLLAMA_TOOL_MODEL_FAMILIES = {
    "gpt-oss",
    "qwen3",
    "qwen2.5",
    "qwen2.5-coder",
    "llama3.1",
    "llama4",
    "devstral",
}


class AgentConfigurationError(RuntimeError):
    """The configured Pipeline Agent model cannot satisfy its tool contract."""


class MalformedAgentResponse(RuntimeError):
    """The provider returned a malformed assistant or tool-call message."""


def supports_native_tool_calling(provider: str, model: str) -> bool:
    """Check known Ollama model families without downloading or contacting a provider."""
    if provider.strip().lower() != "ollama":
        return True
    normalized = model.strip().lower()
    if normalized.startswith("ollama_chat/"):
        normalized = normalized.removeprefix("ollama_chat/")
    family = normalized.split(":", 1)[0]
    return family in _OLLAMA_TOOL_MODEL_FAMILIES


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _json_arguments(arguments: Any) -> str:
    if arguments is None:
        value: Any = {}
    elif isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            value = {}
        else:
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                value = arguments
    else:
        value = arguments

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def normalize_tool_call(tool_call: Any, index: int = 0) -> dict[str, Any]:
    function = _value(tool_call, "function", {}) or {}
    name = _value(function, "name") or _value(tool_call, "name") or ""
    arguments = _value(
        function,
        "arguments",
        _value(tool_call, "arguments", {}),
    )
    call_id = _value(tool_call, "id")

    return {
        "id": str(call_id) if call_id is not None else "",
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _json_arguments(arguments),
        },
    }


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if tool_calls is None:
        return None

    if isinstance(tool_calls, Mapping):
        tool_calls = [tool_calls]
    elif not isinstance(tool_calls, (list, tuple)):
        return None

    normalized = [
        normalize_tool_call(tool_call, index=index)
        for index, tool_call in enumerate(tool_calls)
        if tool_call is not None
    ]
    return normalized or None


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []

    for original in messages:
        message = dict(original)

        if "tool_calls" in message:
            tool_calls = normalize_tool_calls(message.get("tool_calls"))
            if tool_calls:
                message["tool_calls"] = tool_calls
            else:
                message.pop("tool_calls", None)

        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif not isinstance(content, str):
            message["content"] = json.dumps(
                content,
                ensure_ascii=False,
                default=str,
            )

        normalized_messages.append(message)

    return normalized_messages


def _resolve_model() -> str:
    provider = settings.LLM_PROVIDER.lower().strip()
    model = settings.LLM_MODEL.strip()

    prefix = _PROVIDER_PREFIX.get(provider)
    if prefix is None:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Choose one of: {', '.join(_PROVIDER_PREFIX)}"
        )

    if provider == "groq" and model.startswith("groq/"):
        return model
    if "/" in model:
        return model

    return f"{prefix}{model}"


def _build_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }

    if settings.LLM_API_KEY:
        kwargs["api_key"] = settings.LLM_API_KEY

    if settings.LLM_BASE_URL:
        kwargs["api_base"] = settings.LLM_BASE_URL
        if "ngrok" in settings.LLM_BASE_URL:
            kwargs["extra_headers"] = {
                "ngrok-skip-browser-warning": "true"
            }

    return kwargs


async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_retries: int = 2,
) -> Any:
    model = _resolve_model()
    kwargs = _build_kwargs()

    if tools:
        if not supports_native_tool_calling(settings.LLM_PROVIDER, settings.LLM_MODEL):
            raise AgentConfigurationError(
                f"The configured Ollama model '{settings.LLM_MODEL}' is not in the "
                "supported tool-calling model families. Configure an installed "
                "Ollama model with native tool support; the Agent will not download one."
            )
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    request_messages = _normalise_messages(messages)
    attempts = max(1, min(int(max_retries), 2))
    kwargs["timeout"] = settings.AGENT_LLM_TIMEOUT_SECONDS
    kwargs["num_retries"] = 0

    for attempt in range(1, attempts + 1):
        try:
            return await acompletion(
                model=model,
                messages=request_messages,
                **kwargs,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = (
                isinstance(status_code, int)
                and (status_code == 408 or status_code == 429 or status_code >= 500)
            ) or any(
                marker in type(exc).__name__.casefold()
                for marker in ("timeout", "connect", "connection", "serviceunavailable")
            )
            if retryable and attempt < attempts:
                await asyncio.sleep(0.5)
                continue

            logger.warning(
                "Pipeline Agent model request failed (provider=%s model=%s error_type=%s attempt=%s)",
                settings.LLM_PROVIDER,
                settings.LLM_MODEL,
                type(exc).__name__,
                attempt,
            )
            if settings.LLM_PROVIDER.strip().lower() == "ollama":
                detail = str(exc).casefold()
                if "model" in detail and ("not found" in detail or "does not exist" in detail):
                    raise AgentConfigurationError(
                        f"Ollama does not have model '{settings.LLM_MODEL}' available. "
                        "Install that model manually, then retry; the Agent will not download it."
                    ) from None
                if any(marker in type(exc).__name__.casefold() for marker in ("connect", "connection")):
                    raise RuntimeError(
                        "The local Ollama service is unavailable. Start Ollama and retry."
                    ) from None
                if "timeout" in type(exc).__name__.casefold():
                    raise RuntimeError(
                        "The local Ollama request timed out. Check the service and model load."
                    ) from None
            raise RuntimeError(
                "The Pipeline Agent model request failed. Check the configured provider and model."
            ) from None

    raise RuntimeError("The Pipeline Agent model request stopped after its retry limit.")


def extract_message(response: Any) -> dict[str, Any]:
    choices = _value(response, "choices")
    if not choices:
        raise MalformedAgentResponse("The model returned no response choices.")

    message = _value(choices[0], "message")
    if message is None:
        raise MalformedAgentResponse("The model response did not contain a message.")

    raw_content = _value(message, "content", "")

    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content
    else:
        content = json.dumps(
            raw_content,
            ensure_ascii=False,
            default=str,
        )

    raw_calls = _value(message, "tool_calls")
    normalized_calls = normalize_tool_calls(raw_calls)
    if raw_calls is not None and not normalized_calls:
        raise MalformedAgentResponse("The model returned malformed tool calls.")
    for call in normalized_calls or []:
        function = call.get("function") or {}
        if not call.get("id") or not function.get("name"):
            raise MalformedAgentResponse(
                "A model tool call is missing its call ID or registered tool name."
            )

    return {
        "content": content,
        "tool_calls": normalized_calls,
    }


def verify_connection() -> None:
    model = _resolve_model()

    if (
        not settings.LLM_API_KEY
        and settings.LLM_PROVIDER.lower() != "ollama"
    ):
        raise RuntimeError(
            f"LLM_API_KEY is empty but provider "
            f"'{settings.LLM_PROVIDER}' requires one."
        )

    logger.info(
        "LLM configured: provider=%s, model=%s",
        settings.LLM_PROVIDER,
        model,
    )
