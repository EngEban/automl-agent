"""Compact shared state for one report generation run."""
from __future__ import annotations

from typing import Any, TypedDict


class ReportAgentState(TypedDict, total=False):
    report_id: str
    analysis_id: str
    report_type: str
    language: str
    input_facts: dict[str, Any]
    facts: dict[str, Any]
    outline: dict[str, Any]
    document: dict[str, Any]
    chart_paths: dict[str, str]
    html_path: str
    pdf_path: str
    metadata_path: str
    validation_errors: list[str]
    validation_status: str
    revision_count: int
    pdf_status: str
    status: str
    error_code: str
    message: str
