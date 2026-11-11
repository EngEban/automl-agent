"""
Model Catalog & Intelligent Selection.
Provides a registry of supported estimators and selects candidates based on
task type, dataset size, and feature count to prevent resource exhaustion.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
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

_SCALING_REQUIREMENTS: dict[type, bool] = {}
for _model_specs in MODEL_REGISTRY.values():
    for _model_spec in _model_specs:
        _SCALING_REQUIREMENTS[type(_model_spec.factory())] = _model_spec.requires_scaling


def model_requires_scaling(model: Any) -> bool:
    """Return the registered preprocessing requirement for a candidate estimator."""
    return _SCALING_REQUIREMENTS.get(type(model), False)


_TARGET_NAME_SIGNALS = {
    "target", "label", "class", "outcome", "result", "response", "output",
    "prediction", "diagnosis", "species", "survived", "churn", "fraud",
    "default", "approved", "purchase", "price", "saleprice",
}
_TARGET_NAME_PARTS = {
    "target", "label", "class", "outcome", "result", "response", "output", "diagnosis",
    "species", "survived", "churn", "fraud", "default", "approved", "price",
}
_ID_NAME_PARTS = {"id", "uuid", "guid", "key", "index", "identifier"}
_CLASSIFICATION_NAME_PARTS = {
    "class", "label", "category", "status", "type", "segment", "group",
    "outcome", "result", "diagnosis", "species", "grade", "severity",
}
_REGRESSION_NAME_PARTS = {
    "price", "cost", "amount", "count", "quantity", "age", "salary", "income",
    "revenue", "distance", "duration", "weight", "height", "temperature",
    "measurement", "rate", "value", "score",
}


def _name_tokens(name: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(name).casefold())


def is_categorical_target_schema(series: pd.Series | None) -> bool:
    """Return whether the target's schema denotes categorical labels."""
    if series is None:
        return False
    return bool(
        pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or series.dtype.name == "string"
    )


def _is_identifier_column(name: Any, series: pd.Series) -> bool:
    tokens = _name_tokens(name)
    if any(token in _ID_NAME_PARTS for token in tokens) or str(name).casefold().startswith("unnamed"):
        return True

    values = series.dropna()
    if len(values) < 5 or values.nunique() / len(values) < 0.98:
        return False
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return bool(np.all(np.diff(numeric) == 1) or np.all(np.diff(numeric) == -1))
    if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
        sample = values.astype(str).head(20)
        return bool(sample.str.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}").all())
    return False


def validate_target_column(df: pd.DataFrame, target: Any) -> str:
    """Validate an explicitly selected supervised target without replacing it."""
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Choose a non-empty target column name.")
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' does not exist. Choose one of: "
            f"{', '.join(map(str, df.columns))}."
        )
    series = df[target].dropna()
    if series.empty:
        raise ValueError(
            f"Target column '{target}' contains no usable values. Choose a column "
            "with at least two non-missing target values."
        )
    if series.nunique(dropna=True) < 2:
        raise ValueError(
            f"Target column '{target}' has only one usable value. Choose a target "
            "with at least two distinct values."
        )
    if (
        pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError(
            f"Target column '{target}' has unsupported type '{series.dtype}'. "
            "Convert it to numeric measurements or categorical labels first."
        )
    return target


def detect_target_column(df: pd.DataFrame, explicit_target: str | None = None) -> dict[str, Any]:
    """Resolve an explicit target or conservatively rank automatic candidates."""
    if explicit_target is not None:
        selected = validate_target_column(df, explicit_target)
        return {
            "status": "selected",
            "best": selected,
            "selected": selected,
            "confidence": 100,
            "candidates": [{"column": selected, "score": 100, "reasons": ["Explicitly selected by the user."]}],
            "reason": "The selected target exists and contains at least two usable values.",
            "message": f"Using the explicitly selected target column '{selected}'.",
        }

    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return {
            "status": "not_found", "best": None, "selected": None,
            "confidence": 0, "candidates": [],
            "reason": "The dataset has no columns to use as a target.",
            "message": "No target column can be detected because the dataset has no columns.",
        }

    ranked: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        if series.dropna().nunique() < 2 or _is_identifier_column(column, series):
            continue
        tokens = _name_tokens(column)
        token_set = set(tokens)
        normalized = "".join(tokens)
        reasons: list[str] = []
        score = 0
        if normalized in _TARGET_NAME_SIGNALS:
            score = 95
            reasons.append("Column name is a recognized target or outcome name.")
        elif token_set & _TARGET_NAME_PARTS:
            score = 82
            reasons.append("Column name contains a recognized target or outcome term.")
        elif any(token.startswith("is") or token.startswith("has") for token in tokens) and len(tokens) > 1:
            score = 55
            reasons.append("Column name may describe a yes/no outcome; user confirmation is needed.")
        elif pd.api.types.is_bool_dtype(series.dtype):
            score = 25
            reasons.append("Boolean schema is compatible with a classification target, but is not sufficient by itself.")
        elif (
            pd.api.types.is_object_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or series.dtype.name == "string"
        ):
            score = 20
            reasons.append("Categorical schema is compatible with a target, but may also be a feature.")
        elif pd.api.types.is_numeric_dtype(series.dtype) and series.nunique() == 2:
            score = 18
            reasons.append("Binary numeric schema is compatible with a target, but may also be a feature.")
        if score:
            ranked.append({"column": str(column), "score": score, "reasons": reasons})

    ranked.sort(key=lambda item: (-item["score"], item["column"].casefold()))
    strong = [candidate for candidate in ranked if candidate["score"] >= 60]
    if strong:
        top = strong[0]
        tied = [candidate for candidate in strong if top["score"] - candidate["score"] < 12]
        if len(tied) > 1:
            names = [candidate["column"] for candidate in tied]
            message = (
                "Several target columns are similarly plausible: "
                f"{', '.join(names)}. Choose the column you want to predict."
            )
            return {
                "status": "ambiguous", "best": None, "selected": None,
                "confidence": top["score"], "candidates": tied,
                "reason": "Multiple columns have similarly strong target-name evidence.",
                "message": message,
            }
        if top["score"] >= 75:
            return {
                "status": "selected", "best": top["column"], "selected": top["column"],
                "confidence": top["score"], "candidates": strong[:5],
                "reason": top["reasons"][0],
                "message": f"Detected '{top['column']}' as the likely target. {top['reasons'][0]}",
            }

    suggestions = ranked[:5]
    if suggestions:
        message = "No target is sufficiently clear. Choose the column you want to predict."
        reason = "Available columns have only weak target evidence, such as dtype or binary values."
        status = "ambiguous"
    else:
        message = (
            "No reliable target column was found. Set the target column explicitly "
            "to the value you want the model to predict."
        )
        reason = "No non-identifier column has sufficiently strong target-name evidence."
        status = "not_found"
    return {
        "status": status, "best": None, "selected": None, "confidence": 0,
        "candidates": suggestions, "reason": reason, "message": message,
    }


def auto_detect_target(df: pd.DataFrame) -> dict[str, Any]:
    """Backward-compatible name for conservative target detection."""
    return detect_target_column(df)


def infer_task_type(
    y: pd.Series,
    *,
    target_name: str | None = None,
    target_is_categorical: bool | None = None,
    explicit_task_type: str | None = None,
) -> dict[str, Any]:
    """Infer classification/regression from schema and supplied training labels."""
    usable = y.dropna()
    if usable.empty:
        raise ValueError("The target has no usable training values. Provide more non-missing target data.")

    if explicit_task_type is not None and explicit_task_type != "auto":
        if usable.nunique(dropna=True) < 2:
            raise ValueError("The training target has only one usable value; add target variation.")
        if explicit_task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Unsupported task type '{explicit_task_type}'. Choose 'classification' or 'regression'."
            )
        return {
            "status": "resolved", "task_type": explicit_task_type,
            "confidence": 100, "reason": "The task type was explicitly selected.",
        }

    tokens = set(_name_tokens(target_name or y.name or ""))
    if (
        target_is_categorical is True
        or is_categorical_target_schema(y)
    ):
        return {
            "status": "resolved", "task_type": "classification", "confidence": 95,
            "reason": "Boolean, string, or categorical target schema indicates class labels.",
        }

    if tokens & _REGRESSION_NAME_PARTS:
        return {
            "status": "resolved", "task_type": "regression", "confidence": 90,
            "reason": "The target name describes a numeric measurement or amount.",
        }
    if tokens & _CLASSIFICATION_NAME_PARTS:
        return {
            "status": "resolved", "task_type": "classification", "confidence": 90,
            "reason": "The target name describes a class, label, status, or category.",
        }
    if pd.api.types.is_float_dtype(y.dtype):
        return {
            "status": "resolved", "task_type": "regression", "confidence": 90,
            "reason": "Floating-point target values are treated as continuous measurements.",
        }
    if pd.api.types.is_integer_dtype(y.dtype):
        distinct = usable.nunique(dropna=True)
        if distinct < 2:
            return {
                "status": "ambiguous", "task_type": None, "confidence": 0,
                "reason": "The training-only probe contains one numeric value.",
                "message": (
                    f"Could not infer a task type for target '{target_name or y.name}' "
                    "from the training rows. Add more varied examples or explicitly "
                    "choose classification or regression."
                ),
            }
        values = np.sort(pd.to_numeric(usable, errors="coerce").unique())
        if distinct == 2:
            return {
                "status": "resolved", "task_type": "classification", "confidence": 90,
                "reason": "A two-valued integer target is treated as binary class labels.",
            }
        compact_codes = (
            distinct <= 20
            and np.all(np.diff(values) == 1)
            and int(values[0]) in {0, 1}
        )
        if compact_codes:
            return {
                "status": "resolved", "task_type": "classification", "confidence": 80,
                "reason": "Low-cardinality consecutive integers starting at 0 or 1 look like encoded classes.",
            }
        if distinct > 20:
            return {
                "status": "resolved", "task_type": "regression", "confidence": 85,
                "reason": "The high-cardinality integer target is treated as a numeric measurement.",
            }

    candidates = ["classification", "regression"]
    return {
        "status": "ambiguous", "task_type": None, "confidence": 0,
        "candidates": candidates,
        "reason": "The numeric target values could represent either class codes or measurements.",
        "message": (
            f"Could not reliably infer a task type for target '{target_name or y.name}'. "
            "Specify 'classification' or 'regression' to continue."
        ),
    }


def detect_task_type(
    y: pd.Series,
    original: pd.Series | None = None,
    *,
    explicit_task_type: str | None = None,
    target_name: str | None = None,
) -> TaskType:
    """Compatibility wrapper returning one resolved task type or a clear error."""
    inference = infer_task_type(
        y,
        target_name=target_name or getattr(original, "name", None) or y.name,
        target_is_categorical=is_categorical_target_schema(original),
        explicit_task_type=explicit_task_type,
    )
    if inference["status"] == "ambiguous":
        raise ValueError(inference["message"])
    return inference["task_type"]


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
