"""
Train candidate models, evaluate them, tune the winner, and generate advanced charts.
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
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.core.model_selector import (
    get_candidate_models,
    infer_task_type,
    is_categorical_target_schema,
    model_requires_scaling,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _clone_or_copy(value: Any) -> Any:
    try:
        return clone(value)
    except Exception:
        return copy.deepcopy(value)


def _pipeline(preprocessor: Any, model: Any) -> Pipeline:
    fitted_preprocessor = _clone_or_copy(preprocessor)
    if hasattr(fitted_preprocessor, "scale_numeric"):
        fitted_preprocessor.set_params(scale_numeric=model_requires_scaling(model))
    return Pipeline([("prep", fitted_preprocessor), ("model", _clone_or_copy(model))])


def _split_data(X, y, task_type):
    if not 0 < settings.TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1 (exclusive).")
    if len(X) != len(y):
        raise ValueError("Features and target must contain the same number of rows.")

    test_count = max(2, math.ceil(len(X) * settings.TEST_SIZE))
    train_count = len(X) - test_count
    if test_count >= len(X):
        raise ValueError("Too few rows to create an independent holdout set.")

    stratify = None
    if task_type == "classification":
        class_counts = y.value_counts(dropna=False)
        class_count = len(class_counts)
        if class_count < 2:
            raise ValueError("Classification holdout requires at least two target classes.")
        if int(class_counts.min()) < 3:
            raise ValueError(
                "A reliable classification holdout and training-only cross-validation "
                "require at least 3 examples in every class. Collect more examples "
                "for the rare class or use a different evaluation design."
            )
        if test_count < class_count or train_count < class_count:
            raise ValueError(
                "The configured test size cannot place every class in both the "
                "training and holdout partitions. Increase the dataset size or "
                "adjust TEST_SIZE."
            )
        stratify = y
    elif task_type == "regression":
        if train_count < 4:
            raise ValueError(
                "Regression evaluation requires at least 4 training rows after "
                "the independent holdout split for defensible cross-validation."
            )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=settings.RANDOM_STATE,
        stratify=stratify,
    )
    if task_type == "classification" and int(y_train.value_counts().min()) < 2:
        raise ValueError(
            "The holdout split leaves too few examples of a rare class for "
            "training-only cross-validation. Collect more examples or adjust TEST_SIZE."
        )
    return X_train, X_test, y_train, y_test


def _infer_task_type_from_training_data(
    y_train,
    target_is_categorical=None,
    target_name=None,
):
    """Infer automatic task type from training labels only."""
    result = infer_task_type(
        y_train,
        target_name=target_name,
        target_is_categorical=target_is_categorical,
    )
    if result["status"] == "ambiguous":
        raise ValueError(result["message"])
    return result["task_type"]


def _task_inference_training_partition(X, y):
    """Reserve a training-only inference probe before the final holdout split."""
    if not 0 < settings.TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1 (exclusive).")
    if len(X) != len(y):
        raise ValueError("Features and target must contain the same number of rows.")
    final_test_count = max(2, math.ceil(len(X) * settings.TEST_SIZE))
    probe_count = len(X) - (2 * final_test_count)
    if probe_count < 4:
        raise ValueError("Too few rows to infer task type from a training partition.")
    X_probe, X_remainder, y_probe, y_remainder = train_test_split(
        X,
        y,
        train_size=probe_count,
        random_state=settings.RANDOM_STATE,
    )
    return X_probe, X_remainder, y_probe, y_remainder, final_test_count


def _split_data_after_training_inference(
    X,
    y,
    target_is_categorical=None,
    target_name=None,
):
    """Infer from probe rows retained in training, then draw the final holdout."""
    X_probe, X_remainder, y_probe, y_remainder, test_count = (
        _task_inference_training_partition(X, y)
    )
    task_type = _infer_task_type_from_training_data(
        y_probe,
        target_is_categorical=target_is_categorical,
        target_name=target_name,
    )

    stratify = None
    if task_type == "classification":
        counts = y_remainder.value_counts(dropna=False)
        if len(counts) < 2 or int(counts.min()) < 2:
            raise ValueError(
                "The held-out rows cannot be stratified across every inferred class. "
                "Add examples for rare classes or choose an explicit evaluation design."
            )
        if test_count < len(counts) or len(y_remainder) - test_count < len(counts):
            raise ValueError(
                "The configured holdout is too small to represent every class. "
                "Increase the dataset size or adjust TEST_SIZE."
            )
        stratify = y_remainder

    X_train_rest, X_test, y_train_rest, y_test = train_test_split(
        X_remainder,
        y_remainder,
        test_size=test_count,
        random_state=settings.RANDOM_STATE,
        stratify=stratify,
    )
    X_train = pd.concat([X_probe, X_train_rest], axis=0)
    y_train = pd.concat([y_probe, y_train_rest], axis=0)
    if task_type == "classification" and int(y_train.value_counts().min()) < 2:
        raise ValueError(
            "The training rows contain too few examples of an inferred class for "
            "cross-validation. Add examples for the rare class."
        )
    return X_train, X_test, y_train, y_test, task_type


def _calculate_metrics(y_true, predictions, task_type):
    if task_type == "classification":
        return {
            "Accuracy": float(accuracy_score(y_true, predictions)),
            "F1_Weighted": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
            "F1_Macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    r2_value = float(r2_score(y_true, predictions)) if len(y_true) > 1 else 0.0
    return {"R2": r2_value if np.isfinite(r2_value) else 0.0, "RMSE": rmse, "MAE": mae}


def _cv_configuration(y, task_type):
    if task_type == "classification":
        smallest_class = int(y.value_counts(dropna=False).min())
        # Keep at least two examples of every class in each fold's training
        # partition; otherwise even two-fold CV can reduce a class to a single
        # training example and produce an unstable comparison.
        folds = next(
            (
                candidate
                for candidate in range(min(5, smallest_class), 1, -1)
                if smallest_class - math.ceil(smallest_class / candidate) >= 2
            ),
            0,
        )
        if folds < 2:
            return None
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=settings.RANDOM_STATE,
        )
    folds = min(5, len(y) // 2)
    if folds < 2:
        return None
    return KFold(n_splits=folds, shuffle=True, random_state=settings.RANDOM_STATE)


def _apply_training_class_weights(candidates, y_train, task_type):
    """Enable built-in balancing only when the training labels are imbalanced."""
    if task_type != "classification":
        return candidates, None
    counts = pd.Series(y_train).value_counts(dropna=False)
    if len(counts) < 2 or not counts.iloc[-1]:
        return candidates, None
    imbalance_ratio = float(counts.iloc[0] / counts.iloc[-1])
    minority_percentage = float(counts.iloc[-1] / counts.sum() * 100)
    if imbalance_ratio < 3 and minority_percentage >= 20:
        return candidates, None

    balanced = []
    adjusted_models = []
    for name, model in candidates:
        try:
            params = model.get_params(deep=False)
        except (AttributeError, TypeError):
            params = {}
        if "class_weight" in params:
            model.set_params(class_weight="balanced")
            adjusted_models.append(str(name))
        balanced.append((name, model))
    decision = {
        "type": "class_imbalance_handling",
        "imbalance_ratio": round(imbalance_ratio, 3),
        "minority_class_percentage": round(minority_percentage, 2),
        "strategy": "class_weight='balanced' for estimators that support it; no resampling",
        "models_adjusted": adjusted_models,
    }
    return balanced, decision


def _cross_validation_metrics(preprocessor, model, X, y, task_type, splitter):
    if splitter is None:
        raise ValueError("No defensible training-only cross-validation split is available.")

    fold_metrics = []
    for train_indices, validation_indices in splitter.split(X, y):
        fold_pipeline = _pipeline(preprocessor, model)
        X_fold_train = X.iloc[train_indices]
        y_fold_train = y.iloc[train_indices]
        X_fold_validation = X.iloc[validation_indices]
        y_fold_validation = y.iloc[validation_indices]
        fold_pipeline.fit(X_fold_train, y_fold_train)
        validation_predictions = fold_pipeline.predict(X_fold_validation)
        fold_metrics.append(
            _calculate_metrics(y_fold_validation, validation_predictions, task_type)
        )

    metric_names = fold_metrics[0].keys()
    return {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in metric_names
    }


def _fit_full_pipeline(preprocessor, model, X, y):
    pipe = _pipeline(preprocessor, model)
    pipe.fit(X, y)
    return pipe


# ==========================================
# PHASE 2.1: OPTUNA TUNING
# ==========================================
def _tune_with_optuna(preprocessor, model, X_train, y_train, task_type, primary_metric):
    """Tune RandomForest or HistGradientBoosting using Optuna."""
    model_name = type(model).__name__
    
    # We only tune tree-based models that have meaningful hyperparameters
    if "RandomForest" not in model_name and "HistGradientBoosting" not in model_name:
        return _clone_or_copy(model), {}

    splitter = _cv_configuration(y_train, task_type)
    if splitter is None:
        raise ValueError(
            "Optuna tuning requires a defensible cross-validation split inside "
            "the training partition."
        )

    def objective(trial):
        if "RandomForest" in model_name:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
        else: # HistGradientBoosting
            params = {
                "max_iter": trial.suggest_int("max_iter", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
            }
            
        trial_model = _clone_or_copy(model)
        trial_model.set_params(**params)
        metrics = _cross_validation_metrics(
            preprocessor,
            trial_model,
            X_train,
            y_train,
            task_type,
            splitter,
        )
        return metrics[primary_metric]

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, timeout=60) # Max 10 trials or 60 seconds
        
        best_params = dict(study.best_params)
        tuned_model = _clone_or_copy(model)
        tuned_model.set_params(**best_params)
        return tuned_model, best_params
    except Exception as e:
        logger.warning("Optuna tuning failed; keeping CV-selected defaults: %s", e)
        return _clone_or_copy(model), {}


# ==========================================
# PHASE 2.3: ADVANCED CHARTS
# ==========================================
def _plot_confusion_matrix(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Independent Holdout)")
    return _to_b64(fig)


def _plot_regression_eval(y_true, y_pred) -> tuple[str, str]:
    # 1. Predicted vs Actual
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(y_true, y_pred, alpha=0.5, color="#3498db")
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual (Independent Holdout)")
    chart1 = _to_b64(fig1)

    # 2. Residuals
    residuals = y_true - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_pred, residuals, alpha=0.5, color="#e74c3c")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot (Independent Holdout)")
    chart2 = _to_b64(fig2)
    
    return chart1, chart2


# ==========================================
# MAIN FUNCTION
# ==========================================
def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    task_type: str,
    target_is_categorical: bool | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if task_type is None:
        task_type = "auto"
    if task_type == "auto":
        X_train, X_test, y_train, y_test, task_type = (
            _split_data_after_training_inference(
                X,
                y,
                target_is_categorical=target_is_categorical,
                target_name=target_name,
            )
        )
    else:
        # Reuse the central validator so invalid explicit values produce the
        # same actionable error as all other inference entry points.
        infer_task_type(y, explicit_task_type=task_type)
        # Freeze the independent holdout before candidate selection or tuning.
        X_train, X_test, y_train, y_test = _split_data(X, y, task_type)

    if task_type == "regression" and not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError(
            "Regression requires a numeric target. Convert the selected target "
            "to numeric measurements or choose classification."
        )

    target_encoder = None
    if task_type == "classification" and is_categorical_target_schema(y):
        # Learn the output-label mapping from training rows only. The final
        # stratified split ensures every class is represented in training.
        target_encoder = LabelEncoder()
        try:
            target_encoder.fit(y_train)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Categorical target labels must use one consistent scalar type. "
                "Clean the target values or select another target column."
            ) from exc
        y_train = pd.Series(
            target_encoder.transform(y_train), index=y_train.index, name=y.name
        )
        y_test = pd.Series(
            target_encoder.transform(y_test), index=y_test.index, name=y.name
        )
        y = pd.Series(target_encoder.transform(y), index=y.index, name=y.name)
    splitter = _cv_configuration(y_train, task_type)
    if splitter is None:
        raise ValueError(
            "The independent holdout was created, but the remaining training "
            "partition is too small for reliable cross-validation. Add more "
            "training data or adjust TEST_SIZE."
        )

    n_classes = int(y_train.nunique()) if task_type == "classification" else 2
    candidates = get_candidate_models(
        task_type,
        n_samples=len(X_train),
        n_classes=n_classes,
    )
    candidates, imbalance_operation = _apply_training_class_weights(
        candidates, y_train, task_type
    )
    results, failures = [], []
    best_score, selected_model, best_name = -np.inf, None, ""
    primary_metric = "F1_Weighted" if task_type == "classification" else "R2"

    # Candidate comparison uses only CV folds within the training partition.
    for name, model in candidates:
        try:
            metrics = _cross_validation_metrics(
                preprocessor,
                model,
                X_train,
                y_train,
                task_type,
                splitter,
            )
            results.append({"model": name, **metrics})
            score = metrics.get(primary_metric, -np.inf)
            if np.isfinite(score) and score > best_score:
                best_score, selected_model, best_name = float(score), model, name
        except Exception as exc:
            failures.append({"model": name, "error": str(exc)})

    if selected_model is None:
        raise RuntimeError(
            "All candidate models failed training-only cross-validation; "
            "no model was selected."
        )

    # Tune only on the training partition; the holdout remains inaccessible.
    tuning_params = {}
    if len(X_train) > 50:
        logger.info("Tuning %s with training-only cross-validation", best_name)
        selected_model_before_tuning = selected_model
        selected_model, tuning_params = _tune_with_optuna(
            preprocessor,
            selected_model,
            X_train,
            y_train,
            task_type,
            primary_metric,
        )
        if tuning_params:
            try:
                tuned_cv_metrics = _cross_validation_metrics(
                    preprocessor,
                    selected_model,
                    X_train,
                    y_train,
                    task_type,
                    splitter,
                )
            except Exception:
                logger.exception(
                    "Tuned model failed its final training-only CV comparison; "
                    "keeping the previously selected model"
                )
                selected_model = selected_model_before_tuning
                tuning_params = {}
            else:
                best_score = float(tuned_cv_metrics[primary_metric])
                for result in results:
                    if result["model"] == best_name:
                        result.update(tuned_cv_metrics)
                        break

    # Evaluate exactly one training-fitted pipeline on the untouched holdout.
    evaluation_pipeline = _pipeline(preprocessor, selected_model)
    evaluation_pipeline.fit(X_train, y_train)
    holdout_predictions = evaluation_pipeline.predict(X_test)
    final_test_metrics = _calculate_metrics(y_test, holdout_predictions, task_type)

    # Charts and permutation importance describe the same independent holdout.
    extra_charts = {}
    if task_type == "classification":
        extra_charts["confusion_matrix"] = _plot_confusion_matrix(
            y_test,
            holdout_predictions,
        )
    else:
        pva_chart, residual_chart = _plot_regression_eval(
            y_test,
            holdout_predictions,
        )
        extra_charts["predicted_vs_actual"] = pva_chart
        extra_charts["residuals"] = residual_chart

    # The evaluation pipeline is still fitted on training rows here.
    feature_importances, importance_chart = {}, ""
    try:
        importance_result = permutation_importance(
            evaluation_pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
        )
        feature_importances = {
            str(name): float(value)
            for name, value in zip(X_test.columns, importance_result.importances_mean)
        }
        feature_importances = dict(
            sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:15]
        )
        importance_chart = _importance_chart(feature_importances)
    except Exception:
        logger.exception("Holdout permutation importance could not be calculated")

    metric_names = [k for k in results[0] if k != "model"]
    comparison_chart = _comparison_chart(results, primary_metric)

    # Only after all independent evaluation work is complete, refit a fresh
    # production pipeline on every available row.
    production_pipeline = _fit_full_pipeline(preprocessor, selected_model, X, y)
    fitted_preprocessing = list(
        getattr(production_pipeline.named_steps["prep"], "operations_", [])
    )
    if imbalance_operation is not None:
        fitted_preprocessing.append(imbalance_operation)
    evaluation = {
        "available": True,
        "method": (
            "stratified_holdout" if task_type == "classification" else "random_holdout"
        ),
        "test_size": float(settings.TEST_SIZE),
        "test_rows": int(len(X_test)),
        "test_fraction": float(len(X_test) / len(X)),
        "random_state": int(settings.RANDOM_STATE),
        "metrics": final_test_metrics,
    }

    return {
        "task_type": task_type,
        "target_encoder": target_encoder,
        "preprocessing_operations": fitted_preprocessing,
        "results": results,
        "best_model": production_pipeline,
        "best_model_name": best_name,
        "best_score": float(best_score),
        "metric": primary_metric,
        "selection_metric": primary_metric,
        "cv_selection_score": float(best_score),
        "metric_names": metric_names,
        "evaluation": evaluation,
        "final_test_metrics": final_test_metrics,
        "comparison_chart": comparison_chart,
        "importance_chart": importance_chart,
        "feature_importances": feature_importances,
        "extra_charts": extra_charts,
        "tuning_params": tuning_params,
        "failures": failures,
    }


def _importance_chart(importances: dict[str, float]) -> str:
    top_items = list(importances.items())[:10]
    names = [str(i[0]) for i in top_items][::-1]
    values = [float(i[1]) for i in top_items][::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, values, color="#2ecc71")
    ax.set_xlabel("Mean decrease in model score")
    ax.set_title("Permutation Importance (Independent Holdout)")
    for i, v in enumerate(values): ax.text(v, i, f" {v:.3f}", va="center")
    return _to_b64(fig)


def _comparison_chart(results: list[dict[str, Any]], metric: str) -> str:
    names = [str(r["model"]) for r in results]
    values = [float(r.get(metric, 0.0)) for r in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, values, color="#3498db")
    ax.set_xlabel(metric)
    ax.set_title(f"Cross-validation model comparison by {metric}")
    for index, value in enumerate(values): ax.text(value, index, f" {value:.3f}", va="center")
    return _to_b64(fig)
