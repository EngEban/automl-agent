"""
Train candidate models, compare them, and return the best fitted pipeline.

The input frame is always raw feature data.  Every candidate owns a cloned
preprocessor inside its sklearn Pipeline, so preprocessing is fitted only on
the training data and is applied consistently to validation/test data.
"""
from __future__ import annotations

import base64
import copy
import logging
import math
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from app.config import settings
from app.core.model_selector import get_candidate_models

logger = logging.getLogger(__name__)

_SMALL_DATASET_ROWS = 20


def _to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _clone_or_copy(value: Any) -> Any:
    """Clone sklearn estimators without requiring a caller-owned fit state."""
    try:
        return clone(value)
    except Exception:
        return copy.deepcopy(value)


def _pipeline(preprocessor: Any, model: Any) -> Pipeline:
    return Pipeline([
        ("prep", _clone_or_copy(preprocessor)),
        ("model", _clone_or_copy(model)),
    ])


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a safe holdout split, including for very small datasets."""
    test_count = max(1, math.ceil(len(X) * settings.TEST_SIZE))
    test_count = min(test_count, len(X) - 1)
    train_count = len(X) - test_count

    class_count = int(y.nunique(dropna=False)) if task_type == "classification" else 0
    can_stratify = (
        task_type == "classification"
        and class_count > 1
        and int(y.value_counts(dropna=False).min()) >= 2
        and test_count >= class_count
        and train_count >= class_count
    )

    return train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=settings.RANDOM_STATE,
        stratify=y if can_stratify else None,
    )


def _holdout_metrics(
    y_test: pd.Series,
    predictions: Any,
    task_type: str,
) -> dict[str, float]:
    if task_type == "classification":
        return {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "F1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2_value = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0
    return {
        "R2": r2_value if np.isfinite(r2_value) else 0.0,
        "RMSE": rmse,
    }


def _cv_configuration(
    y: pd.Series,
    task_type: str,
) -> tuple[Any, int] | None:
    """Return a valid CV splitter, or None when the data cannot support CV."""
    if len(y) < 2:
        return None

    if task_type == "classification":
        smallest_class = int(y.value_counts(dropna=False).min())
        folds = min(5, smallest_class)
        if folds < 2:
            return None
        return (
            StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=settings.RANDOM_STATE,
            ),
            folds,
        )

    folds = min(5, len(y))
    if folds < 2:
        return None
    return (
        KFold(n_splits=folds, shuffle=True, random_state=settings.RANDOM_STATE),
        folds,
    )


def _cross_validation_metrics(
    preprocessor: Any,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    splitter: Any,
) -> dict[str, float]:
    """Evaluate a candidate with preprocessing inside every CV fold."""
    pipe = _pipeline(preprocessor, model)
    if task_type == "classification":
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=splitter,
            scoring={"Accuracy": "accuracy", "F1": "f1_weighted"},
            error_score="raise",
        )
        return {
            "Accuracy": float(np.nanmean(scores["test_Accuracy"])),
            "F1": float(np.nanmean(scores["test_F1"])),
        }

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=splitter,
        scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error"},
        error_score="raise",
    )
    r2_values = scores["test_R2"]
    r2_value = float(np.nanmean(r2_values)) if np.any(np.isfinite(r2_values)) else 0.0
    return {
        "R2": r2_value if np.isfinite(r2_value) else 0.0,
        "RMSE": float(-np.nanmean(scores["test_RMSE"])),
    }


def _fit_holdout_candidate(
    preprocessor: Any,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit one raw-data pipeline on X_train and predict raw X_test."""
    pipe = _pipeline(preprocessor, model)
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    return pipe, _holdout_metrics(y_test, predictions, task_type)


def _fit_full_pipeline(preprocessor: Any, model: Any, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = _pipeline(preprocessor, model)
    pipe.fit(X, y)
    return pipe


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    task_type: str,
) -> dict[str, Any]:
    """Evaluate candidate models and return the best fitted raw-data pipeline."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame containing raw features.")
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")
    if len(X) < 2:
        raise ValueError("At least two rows are required for model training.")
    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"Unsupported task type '{task_type}'. "
            "Expected 'classification' or 'regression'."
        )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    candidates = get_candidate_models(task_type)
    if not candidates:
        raise RuntimeError("No candidate models are configured.")

    # Small datasets deliberately use one simple holdout split.  This avoids
    # cross-validation folds that are too small to fit classifiers reliably.
    holdout: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None = None
    splitter: Any | None = None
    use_cross_validation = len(X) >= _SMALL_DATASET_ROWS
    if use_cross_validation:
        cv_config = _cv_configuration(y, task_type)
        if cv_config is None:
            use_cross_validation = False
        else:
            splitter = cv_config[0]

    if not use_cross_validation:
        holdout = _split_data(X, y, task_type)
    else:
        # Keep a fallback holdout ready if one particular estimator cannot
        # support CV (for example because its fold-level data is degenerate).
        try:
            holdout = _split_data(X, y, task_type)
        except ValueError:
            holdout = None

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    best_score = -np.inf
    best_model: Pipeline | None = None
    best_name = ""

    for name, model in candidates:
        try:
            if use_cross_validation and splitter is not None:
                try:
                    metrics = _cross_validation_metrics(
                        preprocessor,
                        model,
                        X,
                        y,
                        task_type,
                        splitter,
                    )
                except Exception as cv_exc:
                    if holdout is None:
                        raise
                    logger.warning(
                        "CV failed for %s; falling back to holdout evaluation: %s",
                        name,
                        cv_exc,
                    )
                    pipeline, metrics = _fit_holdout_candidate(
                        preprocessor,
                        model,
                        *holdout,
                        task_type,
                    )
                else:
                    pipeline = _fit_full_pipeline(preprocessor, model, X, y)
            else:
                if holdout is None:
                    raise RuntimeError("Unable to create a training/validation split.")
                pipeline, metrics = _fit_holdout_candidate(
                    preprocessor,
                    model,
                    *holdout,
                    task_type,
                )

            results.append({"model": name, **metrics})
            primary = metrics["Accuracy"] if task_type == "classification" else metrics["R2"]
            if np.isfinite(primary) and primary > best_score:
                best_score = float(primary)
                best_model = pipeline
                best_name = name
        except Exception as exc:
            logger.exception("Model %s failed: %s", name, exc)
            failures.append({
                "model": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    if best_model is None:
        details = "; ".join(
            f"{failure['model']}: {failure['error']}" for failure in failures
        )
        raise RuntimeError(f"All candidate models failed. {details}")

    metric = "Accuracy" if task_type == "classification" else "R2"
    metric_names = [key for key in results[0] if key != "model"]
    chart = _comparison_chart(results, metric)

    return {
        "results": results,
        "best_model": best_model,
        "best_model_name": best_name,
        "best_score": float(best_score),
        "metric": metric,
        "metric_names": metric_names,
        "comparison_chart": chart,
        "failures": failures,
    }


def _comparison_chart(results: list[dict[str, Any]], metric: str) -> str:
    names = [str(result["model"]) for result in results]
    values = [float(result[metric]) for result in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, values, color="#3498db")
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison by {metric}")
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.3f}", va="center")
    return _to_b64(fig)
