"""
Pydantic schemas for chat.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisOptions(BaseModel):
    """Training options currently supported by the shared AutoML backend."""

    model_config = ConfigDict(extra="forbid")

    target_column: str | None = Field(default=None, min_length=1)
    task_type: Literal["classification", "regression"] | None = None


class ChatRequest(BaseModel):
    """User sends a message to the agent."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=20_000)
    mode: Literal["simple", "advanced"] = "simple"
    analysis_options: AnalysisOptions | None = None

    @field_validator("message")
    @classmethod
    def message_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank.")
        return value.strip()

    @field_validator("analysis_options")
    @classmethod
    def options_require_advanced_mode(cls, value, info):
        if value is not None and info.data.get("mode", "simple") != "advanced":
            raise ValueError("analysis_options are available only in advanced mode.")
        return value


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
