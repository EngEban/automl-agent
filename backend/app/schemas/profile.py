from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_constant: bool
    is_high_cardinality: bool
    likely_id: bool
    detected_type: str = "unknown"
    cardinality_percentage: float = 0.0
    is_near_constant: bool = False
    is_date_like: bool = False
    duplicate_of: str | None = None
    potential_issues: list[str] = Field(default_factory=list)
    numeric_statistics: dict[str, Any] | None = None
    outlier_count: int | None = None
    top_categories: list[dict[str, Any]] = Field(default_factory=list)
    category_frequency: list[dict[str, Any]] = Field(default_factory=list)
    rare_category_count: int | None = None


class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    quality_score: float
    quality_details: dict[str, Any]
    warnings: list[str]
    numeric_count: int = 0
    categorical_count: int = 0
    boolean_count: int = 0
    date_count: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0
    duplicate_row_count: int = 0
    memory_usage_bytes: int | None = None
    target_info: dict[str, Any] | None = None
