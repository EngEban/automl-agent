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

from app.config.settings import settings
from app.services.provider_runtime import (
    ProviderRuntimeError,
    RuntimeProvider,
    complete_async,
)

logger = logging.getLogger(__name__)

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


async def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    max_retries: int = 2,
    provider: RuntimeProvider | None = None,
) -> Any:
    if provider is None:
        raise AgentConfigurationError(
            "Configure and test a Pipeline Agent provider in AI Provider Settings before starting a run."
        )

    request_messages = _normalise_messages(messages)
    attempts = max(1, min(int(max_retries), 2))

    for attempt in range(1, attempts + 1):
        try:
            return await complete_async(
                provider,
                role="pipeline_agent",
                messages=request_messages,
                tools=tools,
                tool_choice=tool_choice,
                timeout=settings.AGENT_LLM_TIMEOUT_SECONDS,
            )
        except ProviderRuntimeError as exc:
            if exc.retryable and attempt < attempts:
                await asyncio.sleep(0.5)
                continue
            logger.warning(
                "Pipeline Agent model request failed (provider=%s model=%s error_type=%s attempt=%s)",
                provider.provider_id,
                provider.model_identifier,
                exc.code,
                attempt,
            )
            if exc.code == "provider_assignment_required":
                raise AgentConfigurationError(str(exc)) from None
            raise RuntimeError(str(exc)) from None

    raise RuntimeError(
        "The selected Pipeline Agent provider did not respond after the bounded retry limit."
    )


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
    logger.info(
        "User-managed Pipeline Agent providers are resolved for each owned session."
    )
