from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.profile import DatasetProfile


class FeatureAnalysisRequest(BaseModel):
    target_column: str | None = Field(default=None, min_length=1, max_length=255)
    task_type: Literal["classification", "regression"] | None = None
    max_features: int = Field(default=10, ge=1, le=50)


class FeatureAnalysisRead(BaseModel):
    n_rows: int
    n_columns: int
    profiled_rows: int
    sampled: bool
    profile: DatasetProfile
    relationships: dict[str, Any]
    selection_comparison: dict[str, Any]
