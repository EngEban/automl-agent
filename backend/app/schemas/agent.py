"""
Pydantic schemas for agent introspection.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Current agent configuration (safe to expose — no key)."""
    provider: str
    model: str
    temperature: float
    max_iterations: int
    has_api_key: bool


class ToolInfo(BaseModel):
    """Description of one available tool."""
    name: str
    description: str
    parameters: dict[str, Any]