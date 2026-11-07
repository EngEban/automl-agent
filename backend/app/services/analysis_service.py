"""
Glue layer used by the chat route: runs the agent and shapes the response.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session as DBSession

from app.agent.agent import run_agent
from app.schemas.chat import ChatResponse, ToolCallRecord

logger = logging.getLogger(__name__)


async def handle_user_message(
    db: DBSession, session_id: str, user_message: str,
) -> ChatResponse:
    """Run the agent for one user turn and shape the response."""
    raw = await run_agent(session_id=session_id, user_message=user_message, db=db)

    tool_records = [
        ToolCallRecord(
            name=tc["function"]["name"],
            arguments=_maybe_json(tc["function"].get("arguments")),
            result=None,
        )
        for tc in raw.get("tool_calls", [])
    ]

    return ChatResponse(
        session_id=session_id,
        content=raw["content"],
        iterations=raw["iterations"],
        tool_calls=tool_records,
    )


def _maybe_json(raw: Any) -> dict:
    """Best-effort JSON decode for tool arguments."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        import json
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}