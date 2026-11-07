"""
Conversation memory loaded from PostgreSQL and normalised for LiteLLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session as DBSession

from app.agent.llm import normalize_tool_calls
from app.db.crud import list_messages
from app.db.models import Message

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 4000
_MAX_HISTORY_MESSAGES = 8
_LLM_ROLES = {"system", "user", "assistant", "tool"}


def _truncate(value: Any, limit: int = _MAX_TOOL_RESULT_CHARS) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars]"


def _stored_tool_calls(value: Any) -> list[dict[str, Any]] | None:
    """Read the supported database envelopes and normalise their calls."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed persisted tool_calls JSON")
            return None
    if isinstance(value, dict):
        value = value.get("calls", value.get("tool_calls", value))
    return normalize_tool_calls(value)


def _message_to_llm(msg: Message) -> list[dict[str, Any]]:
    """Convert one database message to strict LiteLLM message dictionaries."""
    role = str(msg.role or "")
    if role not in _LLM_ROLES:
        logger.warning("Skipping unsupported persisted message role %r", role)
        return []

    content = msg.content or ""
    if not isinstance(content, str):
        content = str(content)

    if role == "assistant":
        entry: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = _stored_tool_calls(msg.tool_calls)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return [entry]

    if role == "tool":
        tool_results = msg.tool_results
        if isinstance(tool_results, str):
            try:
                tool_results = json.loads(tool_results)
            except json.JSONDecodeError:
                tool_results = None
        tool_call_id = (
            tool_results.get("tool_call_id")
            if isinstance(tool_results, dict)
            else None
        )
        if not tool_call_id:
            logger.warning("Skipping tool message without tool_call_id")
            return []
        return [{
            "role": "tool",
            "tool_call_id": str(tool_call_id),
            "content": _truncate(content),
        }]

    return [{"role": role, "content": content}]


def load_history(db: DBSession, session_id: str) -> list[dict[str, Any]]:
    """Return recent messages in the exact structure accepted by LiteLLM."""
    rows = list_messages(db, session_id)[-_MAX_HISTORY_MESSAGES:]
    history: list[dict[str, Any]] = []
    for row in rows:
        history.extend(_message_to_llm(row))

    # A row-window can begin in the middle of an assistant/tool exchange.
    # Leading tool messages are invalid without their assistant tool call, so
    # drop only that orphaned prefix while retaining the rest of the history.
    while history and history[0].get("role") == "tool":
        history.pop(0)
    return history


def format_tool_result(tool_name: str, result: Any) -> str:
    """Serialise a tool result for persistence and the next LLM turn."""
    try:
        payload = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        payload = str(result)
    return f"[{tool_name}] {_truncate(payload)}"
