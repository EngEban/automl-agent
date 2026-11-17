"""Strict configuration for training-pipeline feature selection."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FeatureSelectionMethod = Literal[
    "variance_filter",
    "correlation_filter",
    "mutual_information",
    "univariate_test",
    "extra_trees",
]


class FeatureSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    method: FeatureSelectionMethod
    threshold: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    top_k: int | None = Field(default=None, ge=1, le=300)
    min_features: int = Field(default=1, ge=1, le=300)
    n_estimators: int | None = Field(default=None, ge=10, le=200)

    @model_validator(mode="after")
    def validate_method_parameters(self):
        if self.method == "variance_filter":
            if self.threshold is None:
                object.__setattr__(self, "threshold", 0.0)
            if self.top_k is not None or self.n_estimators is not None:
                raise ValueError("variance_filter accepts threshold and min_features only.")
        elif self.method == "correlation_filter":
            if self.threshold is None:
                object.__setattr__(self, "threshold", 0.95)
            elif self.threshold > 1:
                raise ValueError("correlation_filter threshold must be between 0 and 1.")
            if self.n_estimators is not None:
                raise ValueError("correlation_filter does not accept n_estimators.")
        elif self.method in {"mutual_information", "univariate_test"}:
            if self.top_k is None:
                raise ValueError(f"{self.method} requires top_k.")
            if self.threshold is not None or self.n_estimators is not None:
                raise ValueError(f"{self.method} accepts top_k and min_features only.")
        elif self.method == "extra_trees":
            if self.top_k is None:
                raise ValueError("extra_trees requires top_k.")
            if self.threshold is not None:
                raise ValueError("extra_trees does not accept threshold.")
            if self.n_estimators is None:
                object.__setattr__(self, "n_estimators", 50)

        if self.top_k is not None and self.min_features > self.top_k:
            raise ValueError("min_features cannot exceed top_k.")
        return self
