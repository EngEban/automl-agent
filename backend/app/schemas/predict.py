from __future__ import annotations
from typing import Any, Union
from pydantic import BaseModel

class PredictRequest(BaseModel):
    records: list[dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: list[Union[str, float, int]]
    probabilities: list[list[float]] | None = None
    model_version_id: str
    model_name: str