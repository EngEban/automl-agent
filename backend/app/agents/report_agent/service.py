"""Synchronous integration point for legacy and LangGraph report generation."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)
_REPORT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _metadata_path(report_id: str) -> Path:
    return settings.REPORT_DIR / f"{report_id}.report.json"


def _save_failure_metadata(
    *, report_id: str, analysis_id: str, report_type: str, language: str,
    code: str, message: str, provider: str, model: str | None,
) -> str | None:
    metadata = {
        "report_id": report_id,
        "analysis_id": analysis_id,
        "report_type": report_type,
        "language": language,
        "provider": provider,
        "model": model,
        "generation_status": "failed",
        "validation_status": "not_run",
        "html_path": None,
        "pdf_path": None,
        "pdf_status": "not_run",
        "error_code": code,
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path = _metadata_path(report_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except OSError as exc:
        logger.warning("Report failure metadata could not be saved (error_type=%s)", type(exc).__name__)
        return None


def _result_from_state(state: dict[str, Any]) -> dict[str, Any]:
    report_id = state.get("report_id", "")
    pdf_path = state.get("pdf_path") or None
    html_path = state.get("html_path") or None
    metadata_path = state.get("metadata_path") or None
    return {
        "status": state.get("status", "failed"),
        "report_id": report_id,
        "analysis_id": state.get("analysis_id"),
        "report_type": state.get("report_type"),
        "language": state.get("language"),
        "provider": settings.REPORT_LLM_PROVIDER,
        "model": _resolved_model_or_none(),
        "validation_status": state.get("validation_status", "not_run"),
        "validation_errors": state.get("validation_errors", []),
        "revision_count": int(state.get("revision_count", 0)),
        "html_path": html_path,
        "pdf_path": pdf_path,
        "metadata_path": metadata_path,
        "pdf_status": state.get("pdf_status", "not_run"),
        "error_code": state.get("error_code") or None,
        "message": state.get("message") or None,
    }


def _resolved_model_or_none() -> str | None:
    try:
        from app.agents.report_agent.agent import resolve_report_model

        return resolve_report_model()
    except Exception:
        return None


def _compact_analysis_inputs(
    *, data: Any, eda: Any, operations: Any, training: Any,
    task_type: str, target_column: str, chart_paths: dict[str, str],
    experiment: Any, model_registry: Any,
) -> dict[str, Any]:
    """Keep graph state to aggregate facts, IDs, and staged file paths."""
    data = data if isinstance(data, dict) else {}
    training = training if isinstance(training, dict) else {}
    eda = eda if isinstance(eda, dict) else {}
    loading = data.get("loading_metadata")
    loading = loading if isinstance(loading, dict) else {}
    profile = data.get("profile")
    if not isinstance(profile, dict):
        profile = {}
    warnings = profile.get("warnings", [])
    if not isinstance(warnings, (list, tuple)):
        warnings = []
    summary = eda.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    eda_findings = summary.get("important_findings", [])
    if not isinstance(eda_findings, (list, tuple)):
        eda_findings = []
    compact_data = {
        key: data.get(key)
        for key in ("n_rows", "n_columns", "missing_total", "duplicate_rows")
    }
    compact_data["loading_metadata"] = {
        key: loading.get(key)
        for key in (
            "original_row_count", "loaded_row_count", "sampling_applied",
            "sampling_method", "sampling_seed", "chunk_size",
        )
    }
    compact_data["profile"] = {"warnings": list(warnings[:20])}
    compact_training = {
        key: training.get(key)
        for key in (
            "best_model_name", "best_score", "selection_metric", "cv_selection_score",
            "metric", "metric_names", "results", "evaluation", "final_test_metrics",
            "preprocessing_operations", "tuning_params",
        )
    }
    compact_operations = []
    if isinstance(operations, (list, tuple)):
        for operation in operations[:50]:
            if isinstance(operation, dict):
                details = operation.get("details")
                details = details if isinstance(details, dict) else {}
                compact_operations.append({
                    "type": operation.get("type"),
                    "details": {
                        key: details.get(key)
                        for key in ("strategy", "method", "action")
                        if details.get(key) is not None
                    },
                })
    return {
        "data": compact_data,
        "eda": {"summary": {"important_findings": list(eda_findings[:20])}},
        "operations": compact_operations,
        "training": compact_training,
        "task_type": task_type,
        "target_column": target_column,
        "figures": {name: True for name in chart_paths},
        "experiment": experiment if isinstance(experiment, dict) else {},
        "model_registry": model_registry if isinstance(model_registry, dict) else {},
    }


def generate_report(
    *,
    report_id: str | None = None,
    task_id: str | None = None,
    analysis_id: str | None = None,
    report_type: str = "machine_learning_analysis",
    language: str = "English",
    data: dict[str, Any],
    eda: dict[str, Any],
    operations: list[dict[str, Any]],
    training: dict[str, Any],
    task_type: str,
    target_column: str,
    figures: dict[str, Any] | None = None,
    experiment: dict[str, Any] | None = None,
    model_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a report without allowing report failures to escape into ML status."""
    identifier = report_id or task_id or ""
    analysis_identifier = analysis_id or identifier
    if not _REPORT_ID.fullmatch(identifier) or not _REPORT_ID.fullmatch(analysis_identifier):
        return {
            "status": "failed", "error_code": "invalid_report_id",
            "message": "Report and analysis identifiers are invalid.",
            "html_path": None, "pdf_path": None,
        }
    language = str(language or "English")[:48]

    if not settings.REPORT_AGENT_ENABLED:
        try:
            from app.core.report_generator import generate_report as generate_legacy_report

            pdf_path = generate_legacy_report(
                task_id=identifier,
                data=data,
                eda=eda,
                operations=operations,
                training=training,
                task_type=task_type,
                target_column=target_column,
            )
            metadata = {
                "report_id": identifier,
                "analysis_id": analysis_identifier,
                "report_type": report_type,
                "language": language,
                "provider": "legacy",
                "model": None,
                "generation_status": "success",
                "validation_status": "not_applicable",
                "html_path": None,
                "pdf_path": str(pdf_path),
                "pdf_status": "success",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_file = _metadata_path(identifier)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "status": "success", "report_id": identifier,
                "analysis_id": analysis_identifier, "report_type": report_type,
                "language": language, "provider": "legacy", "model": None,
                "validation_status": "not_applicable", "pdf_status": "success",
                "pdf_path": str(pdf_path), "html_path": None,
                "metadata_path": str(metadata_file), "error_code": None,
                "message": "Legacy report generated.",
            }
        except Exception as exc:
            logger.warning("Legacy report generation failed (error_type=%s)", type(exc).__name__)
            message = "The legacy report could not be generated. The analysis results remain available."
            metadata_file = _save_failure_metadata(
                report_id=identifier,
                analysis_id=analysis_identifier,
                report_type=report_type,
                language=language,
                code="legacy_report_failed",
                message=message,
                provider="legacy",
                model=None,
            )
            return {
                "status": "failed", "report_id": identifier,
                "analysis_id": analysis_identifier, "report_type": report_type,
                "language": language, "provider": "legacy", "model": None,
                "validation_status": "not_applicable", "pdf_status": "failed",
                "html_path": None, "pdf_path": None,
                "metadata_path": metadata_file, "error_code": "legacy_report_failed",
                "message": message,
            }

    try:
        from app.agents.report_agent.graph import build_report_graph
        from app.agents.report_agent.html_renderer import stage_figures

        eda = eda if isinstance(eda, dict) else {}
        graph_figures = figures if figures is not None else eda.get("figures", {})
        chart_paths = stage_figures(identifier, graph_figures)
        initial_state = {
            "report_id": identifier,
            "analysis_id": analysis_identifier,
            "report_type": report_type,
            "language": language,
            "input_facts": _compact_analysis_inputs(
                data=data,
                eda=eda,
                operations=operations,
                training=training,
                task_type=task_type,
                target_column=target_column,
                chart_paths=chart_paths,
                experiment=experiment,
                model_registry=model_registry,
            ),
            "chart_paths": chart_paths,
            "revision_count": 0,
            "status": "pending",
        }
        state = build_report_graph().invoke(initial_state, {"recursion_limit": 20})
        result = _result_from_state(state)
        if not result.get("metadata_path"):
            result["metadata_path"] = _save_failure_metadata(
                report_id=identifier,
                analysis_id=analysis_identifier,
                report_type=report_type,
                language=language,
                code=result.get("error_code") or "report_metadata_missing",
                message=result.get("message") or "Report metadata could not be saved.",
                provider=settings.REPORT_LLM_PROVIDER,
                model=result.get("model"),
            )
        return result
    except Exception as exc:
        logger.warning("Report Agent failed (error_type=%s)", type(exc).__name__)
        message = "Report Agent generation failed. The completed analysis results remain available."
        model = _resolved_model_or_none()
        metadata_file = _save_failure_metadata(
            report_id=identifier,
            analysis_id=analysis_identifier,
            report_type=report_type,
            language=language,
            code="report_agent_failed",
            message=message,
            provider=settings.REPORT_LLM_PROVIDER,
            model=model,
        )
        return {
            "status": "failed", "report_id": identifier,
            "analysis_id": analysis_identifier, "report_type": report_type,
            "language": language, "provider": settings.REPORT_LLM_PROVIDER,
            "model": model, "validation_status": "not_run", "pdf_status": "not_run",
            "html_path": None, "pdf_path": None, "metadata_path": metadata_file,
            "error_code": "report_agent_failed", "message": message,
        }


def generate_report_safely(**kwargs) -> dict[str, Any]:
    """Keep unexpected reporting failures separate from completed ML analysis."""
    try:
        return generate_report(**kwargs)
    except Exception as exc:
        logger.warning("Report service failed (error_type=%s)", type(exc).__name__)
        return {
            "status": "failed",
            "report_id": kwargs.get("report_id") or kwargs.get("task_id"),
            "analysis_id": kwargs.get("analysis_id"),
            "error_code": "report_service_failed",
            "message": "The analysis completed, but report generation failed.",
            "html_path": None,
            "pdf_path": None,
        }
