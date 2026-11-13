"""Run one live Groq Report Agent smoke test using synthetic facts only.

Run from ``backend`` with the existing virtual environment:
    .\.venv\Scripts\python.exe scripts\smoke_test_report_agent.py

This script intentionally makes real Groq requests. Do not run it from pytest.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pypdf import PdfReader

from app.agents.report_agent.agent import ReportAgentConfigurationError, resolve_report_model
from app.agents.report_agent.schemas import REPORT_SECTIONS
from app.agents.report_agent.service import generate_report
from app.config import settings


class _HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self._active = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag in {"h1", "h2"}:
            self._active = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2"} and self._active:
            heading = re.sub(r"\s+", " ", unescape("".join(self._parts))).strip()
            if heading:
                self.headings.append(heading)
            self._active = False
            self._parts = []


def _synthetic_context() -> dict:
    return {
        "report_type": "technical_report",
        "language": "English",
        "data": {
            "n_rows": 80,
            "n_columns": 3,
            "missing_total": 0,
            "duplicate_rows": 0,
            "loading_metadata": {
                "original_row_count": 80,
                "loaded_row_count": 80,
                "sampling_applied": False,
            },
            "profile": {"warnings": ["Synthetic smoke-test facts only."]},
        },
        "eda": {"summary": {"important_findings": []}, "figures": {}},
        "operations": [
            {"type": "imputation", "details": {"strategy": "median"}},
            {"type": "encoding", "details": {"strategy": "one-hot"}},
        ],
        "training": {
            "best_model_name": "RandomForestClassifier",
            "best_score": 0.91,
            "cv_selection_score": 0.91,
            "selection_metric": "F1_Weighted",
            "metric": "F1_Weighted",
            "metric_names": ["F1_Weighted", "Accuracy"],
            "results": [
                {"model": "RandomForestClassifier", "F1_Weighted": 0.91, "Accuracy": 0.93},
                {"model": "LogisticRegression", "F1_Weighted": 0.87, "Accuracy": 0.90},
            ],
            "evaluation": {
                "available": True,
                "method": "stratified holdout",
                "test_rows": 16,
                "test_size": 0.2,
                "random_state": 42,
                "metrics": {"F1_Weighted": 0.86, "Accuracy": 0.88},
            },
            "preprocessing_operations": ["median imputation", "one-hot encoding"],
            "tuning_params": {},
        },
        "task_type": "classification",
        "target_column": "class_label",
    }


def _verify_pdf(result: dict, output_dir: Path, api_key: str) -> dict:
    html_path = Path(result["html_path"])
    pdf_path = Path(result["pdf_path"])
    metadata_path = Path(result["metadata_path"])
    for path in (html_path, pdf_path, metadata_path):
        path.resolve().relative_to(output_dir.resolve())
        if not path.is_file():
            raise RuntimeError("A Report Agent output file is missing.")

    html_bytes = html_path.read_bytes()
    pdf_bytes = pdf_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    if api_key.encode("utf-8") in html_bytes + pdf_bytes + metadata_bytes:
        raise RuntimeError("A generated report artifact contained credential material.")
    if len(pdf_bytes) == 0 or not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError("The PDF is empty or lacks a valid PDF signature.")

    reader = PdfReader(str(pdf_path), strict=True)
    if not reader.pages:
        raise RuntimeError("The generated PDF contains no pages.")
    pdf_text = re.sub(
        r"\s+",
        " ",
        " ".join(page.extract_text() or "" for page in reader.pages),
    ).strip()
    html = html_bytes.decode("utf-8")
    headings = _HeadingParser()
    headings.feed(html)
    required_sections = REPORT_SECTIONS["technical_report"]["required"]
    section_keys = set(re.findall(r'<section id="([^"]+)"', html))
    if not set(required_sections) <= section_keys:
        raise RuntimeError("The rendered HTML is missing a required report section.")
    if not headings.headings or any(heading not in pdf_text for heading in headings.headings):
        raise RuntimeError("The PDF is missing the report title or one or more section headings.")

    normalized = pdf_text.lower()
    cv_marker = normalized.find("cross-validation results")
    holdout_marker = normalized.find("independent holdout results")
    cv_value = normalized.find("0.91")
    holdout_value = normalized.find("0.86")
    if not (0 <= cv_marker < cv_value < holdout_marker < holdout_value):
        raise RuntimeError("CV and holdout values were missing or not separated in the PDF text.")
    for recorded_step in ("median imputation", "one-hot encoding"):
        if recorded_step not in normalized:
            raise RuntimeError("A recorded synthetic preprocessing step is missing from the PDF.")
    if any(term in normalized for term in ("smote", "outlier removal", "shap", "xgboost")):
        raise RuntimeError("The PDF includes an unsupported method or model claim.")

    return {
        "pdf_path": str(pdf_path),
        "pdf_bytes": len(pdf_bytes),
        "pdf_pages": len(reader.pages),
        "pdf_signature_valid": True,
        "title_and_sections_present": True,
        "cv_holdout_values_separate": True,
        "preprocessing_facts_present": True,
        "no_credential_material": True,
    }


def main() -> int:
    try:
        if settings.REPORT_LLM_PROVIDER != "groq":
            raise ReportAgentConfigurationError("Set REPORT_LLM_PROVIDER=groq in the local environment.")
        model = resolve_report_model()
        api_key = str(settings.REPORT_LLM_API_KEY or "").strip()
        if not api_key:
            raise ReportAgentConfigurationError("REPORT_LLM_API_KEY is missing from the local environment.")
        if not settings.REPORT_AGENT_ENABLED:
            raise ReportAgentConfigurationError("Set REPORT_AGENT_ENABLED=true in the local environment.")

        output_dir = settings.REPORT_DIR / "smoke-tests" / datetime.now(timezone.utc).strftime("task11-1-%Y%m%dT%H%M%SZ")
        output_dir.mkdir(parents=True, exist_ok=False)
        settings.REPORT_DIR = output_dir
        settings.REPORT_LLM_MAX_TOKENS = min(int(settings.REPORT_LLM_MAX_TOKENS), 1400)
        settings.REPORT_LLM_TIMEOUT_SECONDS = min(int(settings.REPORT_LLM_TIMEOUT_SECONDS), 60)

        from app.agents.report_agent import graph

        # Keep this one live verification to a single bounded correction at most.
        graph.MAX_REPORT_REVISIONS = min(graph.MAX_REPORT_REVISIONS, 1)
        report_id = "task11-1-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = generate_report(
            report_id=report_id,
            analysis_id=report_id,
            **_synthetic_context(),
        )
        summary = {
            "provider": settings.REPORT_LLM_PROVIDER,
            "model": model,
            "report_id": report_id,
            "status": result.get("status"),
            "validation_status": result.get("validation_status"),
            "error_code": result.get("error_code"),
            "message": result.get("message"),
            "html_path": result.get("html_path"),
            "pdf_path": result.get("pdf_path"),
            "pdf_status": result.get("pdf_status"),
            "metadata_path": result.get("metadata_path"),
        }
        if result.get("status") != "success" or result.get("validation_status") != "passed":
            summary_path = output_dir / "smoke_test_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return 1

        summary.update(_verify_pdf(result, output_dir, api_key))
        summary_path = output_dir / "smoke_test_summary.json"
        summary["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except ReportAgentConfigurationError as exc:
        print(f"configuration_error={exc}")
        return 2
    except Exception as exc:
        # Never print exception bodies: HTTP client errors may contain request details.
        print(f"smoke_test_failed error_type={type(exc).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
