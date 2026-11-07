"""
Pydantic schemas for chat.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """User sends a message to the agent."""
    message: str = Field(..., min_length=1)


class ToolCallRecord(BaseModel):
    """One tool invocation performed during an agent turn."""
    name: str
    arguments: dict[str, Any] = {}
    result: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Agent's reply for one turn."""
    session_id: str
    content: str
    iterations: int
    tool_calls: list[ToolCallRecord] = []


class MessageRead(BaseModel):
    """One persisted chat message."""
    id: str
    role: str
    content: str
    tool_calls: Optional[dict[str, Any]] = None
    tool_results: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}