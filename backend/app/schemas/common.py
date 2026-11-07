"""
Common response wrappers used across endpoints.
"""
from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Ok(BaseModel, Generic[T]):
    """Standard success response."""
    ok: bool = True
    data: T


class Err(BaseModel):
    """Standard error response."""
    ok: bool = False
    error: str
    detail: Optional[str] = None