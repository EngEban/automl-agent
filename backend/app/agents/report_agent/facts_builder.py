"""Build a small, privacy-conscious report context from trusted backend output."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from app.agents.report_agent.schemas import ReportFacts

ALLOWED_FIGURES = {
    "missing", "numeric_dist", "categorical_dist", "correlation", "outliers",
    "target", "model_comparison", "feature_importance", "confusion_matrix",
    "predicted_vs_actual", "residuals", "shap_summary",
}


class ReportFactsError(ValueError):
    """Raised when the completed analysis does not contain usable report facts."""


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump(mode="json")
        return result if isinstance(result, dict) else {}
    return {}


def _text(value: Any, *, limit: int = 500) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    value = str(value).strip()
    return value[:limit] if value else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _safe_metrics(value: Any) -> dict[str, int | float | None]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float | None] = {}
    for key, raw in list(value.items())[:30]:
        label = _text(key, limit=100)
        if label:
            result[label] = _number(raw)
    return result


def _operation_labels(training: dict[str, Any], operations: Any) -> list[str]:
    entries = training.get("preprocessing_operations")
    if not isinstance(entries, (list, tuple)) or not entries:
        entries = operations if isinstance(operations, (list, tuple)) else []
    labels = []
    for entry in entries[:50]:
        if isinstance(entry, str):
            label = _text(entry, limit=180)
        elif isinstance(entry, Mapping):
            kind = _text(entry.get("type") or entry.get("op_type"), limit=80)
            details = _mapping(entry.get("details"))
            # Keep operation labels and recorded strategies, while omitting
            # column values and any row-level payloads.
            strategy = next(
                (
                    _text(details[key], limit=80)
                    for key in ("strategy", "method", "action")
                    if _text(details.get(key), limit=80)
                ),
                None,
            )
            label = ": ".join(item for item in (kind, strategy) if item)
        else:
            label = None
        if label and label not in labels:
            labels.append(label)
    return labels


def _safe_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, raw in list(value.items())[:20]:
        name = _text(key, limit=80)
        if not name:
            continue
        if isinstance(raw, bool | int | float | str) or raw is None:
            result[name] = raw if not isinstance(raw, float) or math.isfinite(raw) else None
    return result


def build_report_facts(
    *,
    report_id: str,
    analysis_id: str,
    report_type: str,
    language: str,
    data: Any,
    eda: Any,
    operations: Any,
    training: Any,
    task_type: str,
    target_column: str,
    figures: Any = None,
    experiment: Any = None,
    model_registry: Any = None,
) -> ReportFacts:
    """Extract only validated aggregates; never accept or serialize a dataframe."""
    data = _mapping(data)
    training = _mapping(training)
    eda = _mapping(eda)
    summary = _mapping(eda.get("summary"))
    profile = _mapping(data.get("profile") or summary.get("profile"))
    loading = _mapping(data.get("loading_metadata") or training.get("loading_metadata"))

    loaded_rows = _integer(data.get("n_rows"))
    if loaded_rows is None:
        raise ReportFactsError("The loaded dataset row count is unavailable.")
    source_rows = _integer(loading.get("original_row_count"))
    metadata_loaded_rows = _integer(loading.get("loaded_row_count"))
    if metadata_loaded_rows is not None and metadata_loaded_rows != loaded_rows:
        raise ReportFactsError(
            "Dataset loading metadata does not match the loaded row count."
        )

    resolved_task = _text(task_type, limit=40)
    target = _text(target_column, limit=256)
    best_model = _text(training.get("best_model_name"), limit=160)
    selection_metric = _text(
        training.get("selection_metric") or training.get("metric"), limit=100
    )
    if resolved_task not in {"classification", "regression"}:
        raise ReportFactsError("The resolved task type is unavailable or invalid.")
    if not target:
        raise ReportFactsError("The selected target column is unavailable.")
    if not best_model or not selection_metric:
        raise ReportFactsError(
            "The best model or its cross-validation selection metric is unavailable."
        )

    metric_names = training.get("metric_names")
    if not isinstance(metric_names, (list, tuple)):
        metric_names = []
    normalized_rows = []
    raw_results = training.get("results")
    if not isinstance(raw_results, (list, tuple)):
        raw_results = []
    for row in raw_results[:50]:
        row = _mapping(row)
        model_name = _text(row.get("model"), limit=160)
        if not model_name:
            continue
        selected_metrics = (
            {str(name): row.get(str(name)) for name in metric_names}
            if metric_names
            else {key: value for key, value in row.items() if key != "model"}
        )
        normalized_rows.append(
            {"model": model_name, "metrics": _safe_metrics(selected_metrics)}
        )
    if not normalized_rows:
        raise ReportFactsError("Cross-validation model comparison results are unavailable.")

    best_score = _number(
        training.get("cv_selection_score", training.get("best_score"))
    )
    if best_score is None:
        raise ReportFactsError("The cross-validation selection score is unavailable.")

    evaluation = _mapping(training.get("evaluation"))
    holdout_available = evaluation.get("available") is True
    holdout_metrics = (
        _safe_metrics(evaluation.get("metrics") or training.get("final_test_metrics"))
        if holdout_available
        else {}
    )
    test_size = _number(evaluation.get("test_size"))
    holdout = {
        "available": holdout_available,
        "method": _text(evaluation.get("method"), limit=100),
        "test_rows": _integer(evaluation.get("test_rows")),
        "test_size": test_size,
        "random_state": _integer(evaluation.get("random_state")),
        "metrics": holdout_metrics,
    }

    profile_warnings = profile.get("warnings")
    if not isinstance(profile_warnings, (list, tuple)):
        profile_warnings = summary.get("important_findings", [])
    findings = [
        text
        for item in list(profile_warnings or [])[:20]
        if (text := _text(item, limit=400))
    ]
    safe_figures = _mapping(figures if figures is not None else eda.get("figures"))
    available_figures = sorted(
        str(name) for name, value in safe_figures.items()
        if str(name) in ALLOWED_FIGURES and (value is True or isinstance(value, str) and value)
    )
    explainability_available = bool(
        {"feature_importance", "shap_summary"}.intersection(available_figures)
    )
    report_experiment = {
        key: _text(_mapping(experiment).get(key), limit=160)
        for key in ("id", "name", "status")
        if _text(_mapping(experiment).get(key), limit=160)
    }
    report_registry = {
        key: _text(_mapping(model_registry).get(key), limit=160)
        for key in ("model_name", "lifecycle_status", "version_id")
        if _text(_mapping(model_registry).get(key), limit=160)
    }
    preprocessing = _operation_labels(training, operations)
    parameters = _safe_parameters(training.get("tuning_params"))
    optimization_performed = bool(parameters)

    limitations = []
    if source_rows is None:
        limitations.append("The original source row count was not recorded.")
    if loading.get("sampling_applied") is True:
        limitations.append(
            "The analysis, profiling, and reported evaluation use the loaded sample; rare groups may be absent."
        )
    if not holdout_available:
        limitations.append("Independent holdout evaluation was unavailable for this run.")
    if not explainability_available:
        limitations.append("No explainability output was recorded for this run.")
    if not preprocessing:
        limitations.append("No preprocessing operation details were recorded.")

    evidence = {
        "dataset.loaded_rows": f"Loaded rows: {loaded_rows}.",
        "dataset.profile_findings": (
            "Profile findings were recorded." if findings else "No profile findings were recorded."
        ),
        "training.best_model": f"Selected model: {best_model}.",
        "training.cv_results": (
            f"CV selection metric: {selection_metric}; selection score: {best_score}."
        ),
        "training.holdout": (
            "An independent holdout evaluation is available."
            if holdout_available
            else "Independent holdout evaluation is unavailable."
        ),
        "preprocessing.operations": (
            "Recorded preprocessing operations are available."
            if preprocessing
            else "Preprocessing details are unavailable."
        ),
        "explainability.outputs": (
            "A saved explainability chart is available."
            if explainability_available
            else "Explainability outputs are unavailable."
        ),
        "analysis.limitations": "Limitations are derived from the recorded analysis facts.",
    }
    if source_rows is not None:
        evidence["dataset.source_rows"] = f"Original source rows: {source_rows}."
    if report_experiment:
        evidence["registry.experiment"] = "Experiment metadata is available."
    if report_registry:
        evidence["registry.model"] = "Model Registry metadata is available."

    return ReportFacts.model_validate(
        {
            "report_id": report_id,
            "analysis_id": analysis_id,
            "report_type": report_type,
            "language": language,
            "task_type": resolved_task,
            "target_column": target,
            "dataset": {
                "loaded_rows": loaded_rows,
                "source_rows": source_rows,
                "sampling_applied": loading.get("sampling_applied"),
                "sampling_method": _text(loading.get("sampling_method"), limit=80),
                "sampling_seed": _integer(loading.get("sampling_seed")),
                "chunk_size": _integer(loading.get("chunk_size")),
                "column_count": _integer(data.get("n_columns")),
                "missing_values": _integer(data.get("missing_total")),
                "duplicate_rows": _integer(data.get("duplicate_rows")),
                "profile_findings": findings,
            },
            "best_model": best_model,
            "selection_metric": selection_metric,
            "cv_selection_score": best_score,
            "cv_results": normalized_rows,
            "holdout": holdout,
            "preprocessing_operations": preprocessing,
            "explainability_available": explainability_available,
            "available_figures": available_figures,
            "optimization_performed": optimization_performed,
            "parameters": parameters,
            "experiment": report_experiment,
            "model_registry": report_registry,
            "limitations": limitations,
            "evidence": evidence,
        }
    )
