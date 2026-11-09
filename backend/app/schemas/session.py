"""
Pydantic schemas for sessions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreateURL(BaseModel):
    """Create a session from a URL."""
    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., description="Direct URL to CSV/JSON/Excel")
    target_column: Optional[str] = None
    notes: Optional[str] = None


class SessionRead(BaseModel):
    """Full session view."""
    id: str
    name: str
    source_type: str
    source_path: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    status: str
    dataset_meta: dict[str, Any] = {}
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    """Lightweight session view for list endpoints."""
    id: str
    name: str
    status: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

class SessionCreateKaggle(BaseModel):
    name: str
    kaggle_ref: str = Field(..., description="Kaggle dataset ID (e.g., 'uciml/iris')")
    target_column: Optional[str] = None
    notes: Optional[str] = None
    