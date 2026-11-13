"""LangGraph workflow for facts, outline, report validation, and rendering."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.report_agent import agent as report_model
from app.agents.report_agent.facts_builder import ReportFactsError, build_report_facts
from app.agents.report_agent.html_renderer import render_html, render_pdf
from app.agents.report_agent.prompts import PLAN_USER, REPORT_SYSTEM, REVISE_USER, WRITE_USER
from app.agents.report_agent.state import ReportAgentState
from app.agents.report_agent.schemas import (
    REPORT_SECTIONS,
    ReportDocument,
    ReportFacts,
    ReportOutline,
)
from app.agents.report_agent.validator import validate_document, validate_facts, validate_outline
from app.config import settings

logger = logging.getLogger(__name__)
MAX_REPORT_REVISIONS = 2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def build_report_context(state: ReportAgentState) -> dict[str, Any]:
    raw = state.get("input_facts", {})
    try:
        facts = build_report_facts(
            report_id=state["report_id"],
            analysis_id=state["analysis_id"],
            report_type=state["report_type"],
            language=state["language"],
            data=raw.get("data"),
            eda=raw.get("eda"),
            operations=raw.get("operations"),
            training=raw.get("training"),
            task_type=raw.get("task_type"),
            target_column=raw.get("target_column"),
            figures=raw.get("figures"),
            experiment=raw.get("experiment"),
            model_registry=raw.get("model_registry"),
        )
        return {"facts": facts.model_dump(mode="json"), "status": "validating_facts"}
    except (ReportFactsError, ValueError, TypeError, KeyError) as exc:
        return {
            "status": "failed",
            "validation_status": "failed",
            "error_code": "invalid_analysis_facts",
            "message": str(exc)[:500],
            "validation_errors": [str(exc)[:500]],
        }


def route_after_context(state: ReportAgentState) -> str:
    return "validate_report_facts" if state.get("status") == "validating_facts" else "save_report"


def validate_report_facts(state: ReportAgentState) -> dict[str, Any]:
    result = validate_facts(state.get("facts", {}))
    if not result.valid:
        return {
            "status": "failed",
            "validation_status": "failed",
            "error_code": "invalid_analysis_facts",
            "message": " ".join(result.errors) or "Required report facts are missing or inconsistent.",
            "validation_errors": result.errors,
        }
    return {"status": "planning", "validation_status": "facts_valid"}


def route_after_fact_validation(state: ReportAgentState) -> str:
    return "plan_report" if state.get("status") == "planning" else "save_report"


def plan_report(state: ReportAgentState) -> dict[str, Any]:
    facts = ReportFacts.model_validate(state["facts"])
    catalog = REPORT_SECTIONS[facts.report_type]
    prompt = PLAN_USER.format(
        report_type=facts.report_type.replace("_", " "),
        language=facts.language,
        task_type=facts.task_type,
        section_catalog=_json(catalog),
        facts_json=_json(facts.model_dump(mode="json")),
    )
    try:
        outline = report_model.invoke_structured(
            system_prompt=REPORT_SYSTEM,
            user_prompt=prompt,
            response_schema=ReportOutline,
        )
        checked = validate_outline(outline, facts)
        if not checked.valid:
            return {
                "status": "failed",
                "validation_status": "failed",
                "error_code": "invalid_report_outline",
                "message": "The report outline failed structural validation.",
                "validation_errors": checked.errors,
            }
        return {"outline": outline.model_dump(mode="json"), "status": "writing"}
    except report_model.ReportAgentConfigurationError as exc:
        return {
            "status": "failed",
            "validation_status": "not_run",
            "error_code": "report_configuration_missing",
            "message": str(exc),
        }
    except report_model.ReportProviderError as exc:
        return {
            "status": "failed",
            "validation_status": "not_run",
            "error_code": exc.error_code,
            "message": str(exc),
        }


def route_after_planning(state: ReportAgentState) -> str:
    return "write_report" if state.get("status") == "writing" else "save_report"


def _write_document(state: ReportAgentState, *, revision: bool = False) -> dict[str, Any]:
    facts = ReportFacts.model_validate(state["facts"])
    outline = ReportOutline.model_validate(state["outline"])
    if revision:
        prompt = REVISE_USER.format(
            errors_json=_json(state.get("validation_errors", [])),
            document_json=_json(state.get("document", {})),
            outline_json=_json(outline.model_dump(mode="json")),
            facts_json=_json(facts.model_dump(mode="json")),
        )
    else:
        prompt = WRITE_USER.format(
            language=facts.language,
            outline_json=_json(outline.model_dump(mode="json")),
            facts_json=_json(facts.model_dump(mode="json")),
        )
    document = report_model.invoke_structured(
        system_prompt=REPORT_SYSTEM,
        user_prompt=prompt,
        response_schema=ReportDocument,
    )
    return {"document": document.model_dump(mode="json"), "status": "validating_report"}


def write_report(state: ReportAgentState) -> dict[str, Any]:
    try:
        return _write_document(state)
    except report_model.ReportAgentConfigurationError as exc:
        return {
            "status": "failed",
            "error_code": "report_configuration_missing",
            "message": str(exc),
        }
    except report_model.ReportProviderError as exc:
        return {
            "status": "failed",
            "error_code": exc.error_code,
            "message": str(exc),
        }


def validate_report(state: ReportAgentState) -> dict[str, Any]:
    result = validate_document(state.get("document", {}), state.get("facts", {}), state.get("outline", {}))
    return {
        "validation_errors": result.errors,
        "validation_status": "passed" if result.valid else "failed",
        "status": "rendering" if result.valid else "revision_needed",
    }


def route_after_report_validation(state: ReportAgentState) -> str:
    if state.get("validation_status") == "passed":
        return "render_html"
    if int(state.get("revision_count", 0)) < MAX_REPORT_REVISIONS:
        return "revise_report"
    return "report_validation_exhausted"


def report_validation_exhausted(state: ReportAgentState) -> dict[str, Any]:
    return {
        "status": "failed",
        "error_code": "report_validation_failed",
        "message": "The report remained invalid after the allowed revisions.",
    }


def revise_report(state: ReportAgentState) -> dict[str, Any]:
    next_count = int(state.get("revision_count", 0)) + 1
    try:
        update = _write_document(state, revision=True)
        update["revision_count"] = next_count
        return update
    except report_model.ReportAgentConfigurationError as exc:
        return {
            "status": "failed",
            "error_code": "report_configuration_missing",
            "message": str(exc),
            "revision_count": next_count,
        }
    except report_model.ReportProviderError as exc:
        return {
            "status": "failed",
            "error_code": exc.error_code,
            "message": str(exc),
            "revision_count": next_count,
        }


def route_after_revision(state: ReportAgentState) -> str:
    return "validate_report" if state.get("status") == "validating_report" else "save_report"


def render_report_html(state: ReportAgentState) -> dict[str, Any]:
    try:
        path = render_html(
            state["report_id"],
            ReportDocument.model_validate(state["document"]),
            ReportFacts.model_validate(state["facts"]),
            state.get("chart_paths", {}),
        )
        return {"html_path": str(path), "status": "html_ready"}
    except Exception as exc:
        logger.warning("Report HTML rendering failed (error_type=%s)", type(exc).__name__)
        return {
            "status": "failed",
            "error_code": "html_render_failed",
            "message": "The validated report could not be rendered as HTML.",
        }


def route_after_html(state: ReportAgentState) -> str:
    return "render_pdf" if state.get("status") == "html_ready" else "save_report"


def render_report_pdf(state: ReportAgentState) -> dict[str, Any]:
    try:
        result = render_pdf(state["report_id"], Path(state["html_path"]))
    except Exception as exc:
        logger.warning("Report PDF rendering failed (error_type=%s)", type(exc).__name__)
        result = {"status": "unavailable", "path": None, "reason": "browser_render_failed"}
    status = "success" if result["status"] == "success" else "html_only"
    return {
        "pdf_path": result.get("path") or "",
        "pdf_status": str(result["status"]),
        "status": status,
        "message": (
            "Report HTML and PDF generated."
            if status == "success"
            else "Validated report HTML was saved, but a PDF could not be rendered in this environment."
        ),
        "error_code": "" if status == "success" else str(result.get("reason") or "pdf_unavailable"),
    }


def save_report(state: ReportAgentState) -> dict[str, Any]:
    report_id = state.get("report_id", "")
    metadata_path = settings.REPORT_DIR / f"{report_id}.report.json"
    try:
        model = report_model.resolve_report_model()
    except report_model.ReportAgentConfigurationError:
        model = None
    metadata = {
        "report_id": report_id,
        "analysis_id": state.get("analysis_id"),
        "report_type": state.get("report_type"),
        "language": state.get("language"),
        "provider": settings.REPORT_LLM_PROVIDER,
        "model": model,
        "generation_status": state.get("status", "failed"),
        "validation_status": state.get("validation_status", "not_run"),
        "revision_count": int(state.get("revision_count", 0)),
        "html_path": state.get("html_path") or None,
        "pdf_path": state.get("pdf_path") or None,
        "pdf_status": state.get("pdf_status", "not_run"),
        "error_code": state.get("error_code") or None,
        "message": state.get("message") or None,
        "validation_errors": state.get("validation_errors", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"metadata_path": str(metadata_path)}
    except OSError as exc:
        logger.warning("Report metadata could not be saved (error_type=%s)", type(exc).__name__)
        return {"metadata_path": "", "metadata_save_error": True}


def build_report_graph():
    """Construct and compile the finite report workflow."""
    builder = StateGraph(ReportAgentState)
    builder.add_node("build_report_context", build_report_context)
    builder.add_node("validate_report_facts", validate_report_facts)
    builder.add_node("plan_report", plan_report)
    builder.add_node("write_report", write_report)
    builder.add_node("validate_report", validate_report)
    builder.add_node("revise_report", revise_report)
    builder.add_node("report_validation_exhausted", report_validation_exhausted)
    builder.add_node("render_html", render_report_html)
    builder.add_node("render_pdf", render_report_pdf)
    builder.add_node("save_report", save_report)

    builder.add_edge(START, "build_report_context")
    builder.add_conditional_edges("build_report_context", route_after_context)
    builder.add_conditional_edges("validate_report_facts", route_after_fact_validation)
    builder.add_conditional_edges("plan_report", route_after_planning)
    builder.add_edge("write_report", "validate_report")
    builder.add_conditional_edges("validate_report", route_after_report_validation)
    builder.add_conditional_edges("revise_report", route_after_revision)
    builder.add_edge("report_validation_exhausted", "save_report")
    builder.add_conditional_edges("render_html", route_after_html)
    builder.add_edge("render_pdf", "save_report")
    builder.add_edge("save_report", END)
    return builder.compile()
