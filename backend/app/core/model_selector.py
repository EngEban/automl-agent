"""
Model Catalog & Intelligent Selection.
Provides a registry of supported estimators and selects candidates based on
task type, dataset size, and feature count to prevent resource exhaustion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.config import settings

logger = logging.getLogger(__name__)

TaskType = str  # "classification" or "regression"


@dataclass
class ModelSpec:
    """Metadata for a candidate model."""
    name: str
    factory: Callable[[], Any]
    task_type: TaskType
    requires_scaling: bool
    supports_multiclass: bool
    is_fast: bool


# ==========================================
# MODEL REGISTRY
# ==========================================
MODEL_REGISTRY: dict[str, list[ModelSpec]] = {
    "classification": [
        ModelSpec(
            name="LogisticRegression",
            factory=lambda: LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE),
            task_type="classification",
            requires_scaling=True,
            supports_multiclass=True,
            is_fast=True,
        ),
        ModelSpec(
            name="DecisionTree",
            factory=lambda: DecisionTreeClassifier(random_state=settings.RANDOM_STATE),
            task_type="classification",
            requires_scaling=False,
            supports_multiclass=True,
            is_fast=True,
        ),
        ModelSpec(
            name="RandomForest",
            factory=lambda: RandomForestClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1),
            task_type="classification",
            requires_scaling=False,
            supports_multiclass=True,
            is_fast=False,
        ),
        ModelSpec(
            name="HistGradientBoosting",
            factory=lambda: HistGradientBoostingClassifier(random_state=settings.RANDOM_STATE),
            task_type="classification",
            requires_scaling=False,
            supports_multiclass=True,
            is_fast=True,
        ),
        ModelSpec(
            name="GradientBoosting",
            factory=lambda: GradientBoostingClassifier(random_state=settings.RANDOM_STATE),
            task_type="classification",
            requires_scaling=False,
            supports_multiclass=True,
            is_fast=False,
        ),
    ],
    "regression": [
        ModelSpec(
            name="LinearRegression",
            factory=lambda: LinearRegression(),
            task_type="regression",
            requires_scaling=True,
            supports_multiclass=False,
            is_fast=True,
        ),
        ModelSpec(
            name="Ridge",
            factory=lambda: Ridge(random_state=settings.RANDOM_STATE),
            task_type="regression",
            requires_scaling=True,
            supports_multiclass=False,
            is_fast=True,
        ),
        ModelSpec(
            name="DecisionTree",
            factory=lambda: DecisionTreeRegressor(random_state=settings.RANDOM_STATE),
            task_type="regression",
            requires_scaling=False,
            supports_multiclass=False,
            is_fast=True,
        ),
        ModelSpec(
            name="RandomForest",
            factory=lambda: RandomForestRegressor(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1),
            task_type="regression",
            requires_scaling=False,
            supports_multiclass=False,
            is_fast=False,
        ),
        ModelSpec(
            name="HistGradientBoosting",
            factory=lambda: HistGradientBoostingRegressor(random_state=settings.RANDOM_STATE),
            task_type="regression",
            requires_scaling=False,
            supports_multiclass=False,
            is_fast=True,
        ),
        ModelSpec(
            name="GradientBoosting",
            factory=lambda: GradientBoostingRegressor(random_state=settings.RANDOM_STATE),
            task_type="regression",
            requires_scaling=False,
            supports_multiclass=False,
            is_fast=False,
        ),
    ],
}


def detect_task_type(y: pd.Series, original: pd.Series) -> TaskType:
    """Infer task type from target column."""
    if original.dtype == "object" or original.dtype.name == "category":
        return "classification"
    if original.dtype == "bool":
        return "classification"
    if y.nunique() < 20:
        return "classification"
    return "regression"


def get_candidate_models(
    task_type: TaskType,
    n_samples: int = 1000,
    n_classes: int = 2,
) -> list[tuple[str, Any]]:
    """
    Select candidate models based on task type and data characteristics.
    Returns a list of (name, model_instance) tuples.
    """
    if task_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}")

    candidates = MODEL_REGISTRY[task_type].copy()

    # Filter based on multiclass support if needed
    if task_type == "classification" and n_classes > 2:
        candidates = [c for c in candidates if c.supports_multiclass]

    # Resource-aware filtering:
    # If dataset is huge (> 100k rows), prioritize fast models (like HistGradientBoosting)
    # and drop slow ones to prevent user timeouts.
    if n_samples > 100_000:
        logger.info("Large dataset detected. Prioritizing fast models.")
        candidates = [c for c in candidates if c.is_fast]

    # If dataset is tiny (< 50 rows), drop heavy ensemble models that might overfit
    elif n_samples < 50:
        logger.info("Small dataset detected. Dropping heavy ensembles.")
        candidates = [c for c in candidates if c.name in ("LogisticRegression", "LinearRegression", "Ridge", "DecisionTree")]

    # Limit to settings.MAX_CANDIDATE_MODELS (default is 5)
    max_models = settings.MAX_CANDIDATE_MODELS
    if len(candidates) > max_models:
        candidates = candidates[:max_models]

    return [(spec.name, spec.factory()) for spec in candidates]


def auto_detect_target(df: pd.DataFrame) -> dict:
    """
    Suggest target columns with confidence scores based on heuristics.
    Returns a dictionary with the best candidate and alternatives.
    """
    candidates = []
    
    # Names that commonly indicate a target
    target_keywords = ["target", "label", "class", "y", "output", "result", "price", "value", "species", "diagnosis"]
    
    for col in df.columns:
        score = 0
        reasons = []
        series = df[col]
        unique_count = series.nunique()
        
        # 1. Name check
        if col.lower() in target_keywords:
            score += 50
            reasons.append("Column name is a common target name.")
            
        # 2. Position (last column)
        if col == df.columns[-1]:
            score += 20
            reasons.append("It is the last column in the dataset.")
            
        # 3. Cardinality
        if series.dtype == 'object' or unique_count < 20:
            score += 30
            reasons.append("Low cardinality suggests classification.")
        elif unique_count > 20 and series.dtype in ['int64', 'float64']:
            score += 20
            reasons.append("Continuous numeric data suggests regression.")
            
        # 4. ID Check (if it's unique for every row, it's a bad target)
        if unique_count == len(df):
            score -= 40
            reasons.append("All values are unique (likely an ID).")
            
        if score > 0:
            candidates.append({
                "column": col,
                "score": min(100, score),
                "reasons": reasons,
            })
            
    # Sort candidates by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    if not candidates:
        return {"best": df.columns[-1], "confidence": 0, "candidates": []}
        
    best = candidates[0]
    return {
        "best": best["column"],
        "confidence": best["score"],
        "candidates": candidates[:3], # Return top 3
    }