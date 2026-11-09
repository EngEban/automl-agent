from __future__ import annotations
from typing import Any
from pydantic import BaseModel

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    is_constant: bool
    is_high_cardinality: bool
    likely_id: bool

class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    quality_score: float
    quality_details: dict[str, Any]
    warnings: list[str]