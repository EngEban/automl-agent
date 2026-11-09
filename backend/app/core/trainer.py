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
    KFold, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

from app.config import settings
from app.core.model_selector import get_candidate_models

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_SMALL_DATASET_ROWS = 20


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
    return Pipeline([("prep", _clone_or_copy(preprocessor)), ("model", _clone_or_copy(model))])


def _split_data(X, y, task_type):
    test_count = max(1, math.ceil(len(X) * settings.TEST_SIZE))
    test_count = min(test_count, len(X) - 1)
    class_count = int(y.nunique(dropna=False)) if task_type == "classification" else 0
    can_stratify = (
        task_type == "classification" and class_count > 1
        and int(y.value_counts(dropna=False).min()) >= 2
        and test_count >= class_count and len(X) - test_count >= class_count
    )
    return train_test_split(
        X, y, test_size=test_count, random_state=settings.RANDOM_STATE,
        stratify=y if can_stratify else None,
    )


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
    if len(y) < 2: return None
    if task_type == "classification":
        smallest_class = int(y.value_counts(dropna=False).min())
        folds = min(5, smallest_class)
        if folds < 2: return None
        return (StratifiedKFold(n_splits=folds, shuffle=True, random_state=settings.RANDOM_STATE), folds)
    folds = min(5, len(y))
    if folds < 2: return None
    return (KFold(n_splits=folds, shuffle=True, random_state=settings.RANDOM_STATE), folds)


def _cross_validation_metrics(preprocessor, model, X, y, task_type, splitter):
    pipe = _pipeline(preprocessor, model)
    if task_type == "classification":
        scores = cross_validate(pipe, X, y, cv=splitter, scoring={"Accuracy": "accuracy", "F1_Weighted": "f1_weighted", "F1_Macro": "f1_macro"}, error_score="raise")
        return {k.replace("test_", ""): float(np.nanmean(v)) for k, v in scores.items() if k.startswith("test_")}
    scores = cross_validate(pipe, X, y, cv=splitter, scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}, error_score="raise")
    res = {k.replace("test_", ""): float(np.nanmean(v)) for k, v in scores.items() if k.startswith("test_")}
    res["RMSE"] = -res["RMSE"]; res["MAE"] = -res["MAE"]
    return res


def _fit_holdout_candidate(preprocessor, model, X_train, X_test, y_train, y_test, task_type):
    pipe = _pipeline(preprocessor, model)
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    return pipe, _calculate_metrics(y_test, predictions, task_type)


def _fit_full_pipeline(preprocessor, model, X, y):
    pipe = _pipeline(preprocessor, model)
    pipe.fit(X, y)
    return pipe


# ==========================================
# PHASE 2.1: OPTUNA TUNING
# ==========================================
def _tune_with_optuna(best_pipeline, X_train, y_train, task_type, primary_metric):
    """Tune RandomForest or HistGradientBoosting using Optuna."""
    model = best_pipeline.named_steps["model"]
    model_name = type(model).__name__
    
    # We only tune tree-based models that have meaningful hyperparameters
    if "RandomForest" not in model_name and "HistGradientBoosting" not in model_name:
        return best_pipeline, {}

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
            
        test_pipe = _pipeline(best_pipeline.named_steps["prep"], model.set_params(**params))
        
        # Use CV for tuning
        cv_config = _cv_configuration(y_train, task_type)
        if cv_config is None:
            return -1.0 # Fallback if data is too small
            
        splitter = cv_config[0]
        metrics = _cross_validation_metrics(best_pipeline.named_steps["prep"], model.set_params(**params), X_train, y_train, task_type, splitter)
        return metrics.get(primary_metric, 0.0)

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, timeout=60) # Max 10 trials or 60 seconds
        
        best_params = study.best_params
        tuned_model = model.set_params(**best_params)
        tuned_pipeline = _pipeline(best_pipeline.named_steps["prep"], tuned_model)
        tuned_pipeline.fit(X_train, y_train)
        return tuned_pipeline, best_params
    except Exception as e:
        logger.warning(f"Optuna tuning failed: {e}")
        return best_pipeline, {}


# ==========================================
# PHASE 2.3: ADVANCED CHARTS
# ==========================================
def _plot_confusion_matrix(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Test Set)")
    return _to_b64(fig)


def _plot_regression_eval(y_true, y_pred) -> tuple[str, str]:
    # 1. Predicted vs Actual
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(y_true, y_pred, alpha=0.5, color="#3498db")
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual (Test Set)")
    chart1 = _to_b64(fig1)

    # 2. Residuals
    residuals = y_true - y_pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(y_pred, residuals, alpha=0.5, color="#e74c3c")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot (Test Set)")
    chart2 = _to_b64(fig2)
    
    return chart1, chart2


# ==========================================
# MAIN FUNCTION
# ==========================================
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, preprocessor: Any, task_type: str) -> dict[str, Any]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    n_samples = len(X)
    n_classes = int(y.nunique()) if task_type == "classification" else 2
    candidates = get_candidate_models(task_type, n_samples=n_samples, n_classes=n_classes)
    
    holdout = None
    splitter = None
    use_cross_validation = len(X) >= _SMALL_DATASET_ROWS
    
    if use_cross_validation:
        cv_config = _cv_configuration(y, task_type)
        if cv_config is not None: splitter = cv_config[0]
        else: use_cross_validation = False

    if not use_cross_validation:
        holdout = _split_data(X, y, task_type)
    else:
        try: holdout = _split_data(X, y, task_type)
        except ValueError: holdout = None

    results, failures = [], []
    best_score, best_model, best_name = -np.inf, None, ""
    primary_metric = "F1_Weighted" if task_type == "classification" else "R2"

    # 1. Evaluate candidates
    for name, model in candidates:
        try:
            if use_cross_validation and splitter is not None:
                try:
                    metrics = _cross_validation_metrics(preprocessor, model, X, y, task_type, splitter)
                    pipeline = _fit_full_pipeline(preprocessor, model, X, y)
                except Exception:
                    pipeline, metrics = _fit_holdout_candidate(preprocessor, model, *holdout, task_type)
            else:
                pipeline, metrics = _fit_holdout_candidate(preprocessor, model, *holdout, task_type)

            results.append({"model": name, **metrics})
            score = metrics.get(primary_metric, -np.inf)
            if np.isfinite(score) and score > best_score:
                best_score, best_model, best_name = float(score), pipeline, name
        except Exception as exc:
            failures.append({"model": name, "error": str(exc)})

    if best_model is None:
        raise RuntimeError("All candidate models failed.")

    # 2. Tune the winner using Optuna (Phase 2.1)
    tuning_params = {}
    X_train, X_test, y_train, y_test = holdout if holdout else (X, X, y, y)
    if use_cross_validation and len(X_train) > 50: # Only tune if enough data
        logger.info(f"Tuning {best_name} with Optuna...")
        tuned_model, tuning_params = _tune_with_optuna(best_model, X_train, y_train, task_type, primary_metric)
        if tuning_params:
            best_model = tuned_model
            # Re-evaluate tuned model on test set
            test_pred = best_model.predict(X_test)
            test_metrics = _calculate_metrics(y_test, test_pred, task_type)
            best_score = test_metrics[primary_metric]
            
            # Update results table to reflect tuned score
            for r in results:
                if r["model"] == best_name:
                    r.update(test_metrics)
                    break

    # 3. Generate Advanced Charts (Phase 2.3)
    extra_charts = {}
    if holdout is not None:
        y_pred = best_model.predict(X_test)
        if task_type == "classification":
            extra_charts["confusion_matrix"] = _plot_confusion_matrix(y_test, y_pred)
        else:
            pva_chart, residual_chart = _plot_regression_eval(y_test, y_pred)
            extra_charts["predicted_vs_actual"] = pva_chart
            extra_charts["residuals"] = residual_chart

    # 4. Feature Importance (Phase 2.2 - already done)
    feature_importances, importance_chart = {}, ""
    if holdout is not None:
        try:
            result = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=settings.RANDOM_STATE, n_jobs=-1)
            feature_importances = {str(n): float(i) for n, i in zip(X_test.columns, result.importances_mean)}
            feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:15])
            importance_chart = _importance_chart(feature_importances)
        except Exception: pass

    metric_names = [k for k in results[0] if k != "model"]
    comparison_chart = _comparison_chart(results, primary_metric)

    return {
        "results": results,
        "best_model": best_model,
        "best_model_name": best_name,
        "best_score": float(best_score),
        "metric": primary_metric,
        "metric_names": metric_names,
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
    ax.set_title("Feature Importance (Test Set)")
    for i, v in enumerate(values): ax.text(v, i, f" {v:.3f}", va="center")
    return _to_b64(fig)


def _comparison_chart(results: list[dict[str, Any]], metric: str) -> str:
    names = [str(r["model"]) for r in results]
    values = [float(r.get(metric, 0.0)) for r in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, values, color="#3498db")
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison by {metric}")
    for index, value in enumerate(values): ax.text(value, index, f" {value:.3f}", va="center")
    return _to_b64(fig)