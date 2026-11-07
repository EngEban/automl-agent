"""Pydantic schemas for API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., description="Direct URL to CSV/JSON/Excel")
    target_column: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str = "queued"


class TaskStatus(BaseModel):
    task_id: str
    status: str
    step: Optional[str] = None
    progress: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None