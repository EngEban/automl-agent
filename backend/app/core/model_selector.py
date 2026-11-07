"""
Detect task type (classification/regression) and return candidate models.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "regression"]


def detect_task_type(y: pd.Series, original: pd.Series) -> TaskType:
    """Infer task type from target column."""
    if original.dtype == "object" or original.dtype.name == "category":
        return "classification"
    if original.dtype == "bool":
        return "classification"
    if y.nunique() < 20:
        return "classification"
    return "regression"


def get_candidate_models(task_type: TaskType) -> list[tuple[str, object]]:
    """Return 4-5 candidate models per task type."""
    if task_type == "classification":
        return [
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
            ("DecisionTree", DecisionTreeClassifier(random_state=42)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ]
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(random_state=42)),
        ("DecisionTree", DecisionTreeRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]


def auto_detect_target(df: pd.DataFrame) -> str | None:
    """Guess the target column."""
    common = ["target", "label", "class", "y", "output", "result", "price", "value"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in common:
        if name in lower_map:
            return lower_map[name]
    if len(df.columns) > 0:
        return df.columns[-1]
    return None