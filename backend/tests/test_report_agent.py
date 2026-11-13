from __future__ import annotations

import base64
import importlib
import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from app.agents.report_agent import agent as report_model
from app.agents.report_agent.facts_builder import build_report_facts
from app.agents.report_agent.graph import MAX_REPORT_REVISIONS, build_report_graph
from app.agents.report_agent.html_renderer import render_html, render_pdf, stage_figures
from app.agents.report_agent.schemas import (
    OutlineSection,
    ReportDocument,
    ReportFacts,
    ReportOutline,
    ReportSection,
)
from app.agents.report_agent.service import generate_report, generate_report_safely
from app.agents.report_agent.validator import (
    validate_document,
    validate_facts,
    validate_outline,
)
from app.config import settings


def _analysis_inputs() -> dict:
    return {
        "data": {
            "n_rows": 80,
            "n_columns": 3,
            "missing_total": 2,
            "duplicate_rows": 1,
            "loading_metadata": {
                "original_row_count": 120,
                "loaded_row_count": 80,
                "sampling_applied": True,
                "sampling_method": "reservoir",
                "sampling_seed": 7,
                "chunk_size": 10,
            },
            "profile": {"warnings": ["The target distribution is imbalanced."]},
        },
        "eda": {"summary": {"important_findings": ["The target distribution is imbalanced."]}, "figures": {}},
        "operations": [],
        "training": {
            "best_model_name": "RandomForestClassifier",
            "best_score": 0.88,
            "cv_selection_score": 0.88,
            "selection_metric": "F1_Weighted",
            "metric": "F1_Weighted",
            "metric_names": ["Accuracy", "F1_Weighted"],
            "results": [
                {"model": "RandomForestClassifier", "Accuracy": 0.85, "F1_Weighted": 0.88},
                {"model": "LogisticRegression", "Accuracy": 0.80, "F1_Weighted": 0.82},
            ],
            "evaluation": {
                "available": True,
                "method": "stratified_holdout",
                "test_rows": 16,
                "test_size": 0.2,
                "random_state": 42,
                "metrics": {"Accuracy": 0.81, "F1_Weighted": 0.79},
            },
            "final_test_metrics": {"Accuracy": 0.81, "F1_Weighted": 0.79},
            "preprocessing_operations": [],
            "tuning_params": {},
            "best_model": object(),
        },
        "task_type": "classification",
        "target_column": "outcome",
    }


def _facts(**overrides) -> ReportFacts:
    args = _analysis_inputs()
    args.update(overrides)
    return build_report_facts(
        report_id="report-1",
        analysis_id="run-1",
        report_type="model_comparison",
        language="English",
        **args,
    )


def _outline(facts: ReportFacts | None = None) -> ReportOutline:
    refs = [
        "training.best_model",
        "training.cv_results",
        "training.holdout",
        "analysis.limitations",
    ]
    keys = ["executive_summary", "model_comparison", "evaluation", "limitations", "conclusion"]
    return ReportOutline(
        title="Model Comparison Report",
        sections=[
            OutlineSection(key=key, title=key.replace("_", " ").title(), purpose="Summarize recorded analysis evidence.", fact_refs=[refs[index % len(refs)]])
            for index, key in enumerate(keys)
        ],
    )


def _document(*, prose: str = "The section summarizes the recorded analysis evidence.") -> ReportDocument:
    evidence = {
        "executive_summary": ["training.best_model"],
        "model_comparison": ["training.cv_results"],
        "evaluation": ["training.holdout"],
        "limitations": ["analysis.limitations"],
        "conclusion": ["training.best_model"],
    }
    keys = ["executive_summary", "model_comparison", "evaluation", "limitations", "conclusion"]
    return ReportDocument(
        title="Model Comparison Report",
        subtitle="Evidence-based summary",
        sections=[
            ReportSection(
                key=key,
                title=key.replace("_", " ").title(),
                paragraphs=[prose],
                evidence_refs=evidence[key],
            )
            for key in keys
        ],
    )


def _mock_model(monkeypatch, *, invalid_documents: int = 0):
    facts = _facts()
    outline = _outline(facts)
    calls = {"outline": 0, "document": 0}
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")

    def invoke_structured(**kwargs):
        schema = kwargs["response_schema"]
        if schema is ReportOutline:
            calls["outline"] += 1
            return outline
        calls["document"] += 1
        prose = "This result is 17." if calls["document"] <= invalid_documents else "The section summarizes the recorded analysis evidence."
        return _document(prose=prose)

    monkeypatch.setattr(report_model, "invoke_structured", invoke_structured)
    return calls


def _invoke_service(tmp_path, monkeypatch, *, invalid_documents: int = 0):
    _mock_model(monkeypatch, invalid_documents=invalid_documents)
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "unit-test-token")
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)

    from app.agents.report_agent import graph

    def fake_pdf(report_id, _html_path):
        path = tmp_path / f"{report_id}.pdf"
        path.write_bytes(b"%PDF-test")
        return {"status": "success", "path": str(path), "reason": None}

    monkeypatch.setattr(graph, "render_pdf", fake_pdf)
    args = _analysis_inputs()
    return generate_report(
        report_id="report-1",
        analysis_id="run-1",
        report_type="model_comparison",
        language="English",
        **args,
    )


def test_report_graph_compiles_with_conditional_validation_routes():
    graph = build_report_graph()
    node_names = set(graph.get_graph().nodes)

    assert {
        "build_report_context", "validate_report_facts", "plan_report",
        "write_report", "validate_report", "revise_report", "render_html",
        "render_pdf", "save_report", "report_validation_exhausted",
    } <= node_names


def test_facts_keep_source_loaded_cv_and_holdout_values_separate():
    facts = _facts()

    assert facts.dataset.source_rows == 120
    assert facts.dataset.loaded_rows == 80
    assert facts.dataset.sampling_applied is True
    assert facts.cv_results[0].metrics["F1_Weighted"] == pytest.approx(0.88)
    assert facts.holdout.metrics["F1_Weighted"] == pytest.approx(0.79)
    assert facts.best_model == "RandomForestClassifier"
    assert facts.target_column == "outcome"
    assert facts.task_type == "classification"
    assert validate_facts(facts).valid is True


def test_missing_metrics_remain_unavailable_and_holdout_is_not_inferred():
    args = _analysis_inputs()
    args["training"]["results"][0]["F1_Weighted"] = None
    args["training"]["evaluation"] = {"available": False, "metrics": {}}
    args["training"]["final_test_metrics"] = {"F1_Weighted": 0.99}
    facts = _facts(**args)

    assert facts.cv_results[0].metrics["F1_Weighted"] is None
    assert facts.holdout.available is False
    assert facts.holdout.metrics == {}


def test_inconsistent_sampling_facts_are_rejected():
    facts = _facts()
    broken = facts.model_dump(mode="python")
    broken["dataset"]["source_rows"] = 70

    result = validate_facts(broken)

    assert result.valid is False
    assert any("source rows" in error.lower() for error in result.errors)


def test_invalid_essential_facts_stop_before_model_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "")
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    args = _analysis_inputs()
    args["training"]["results"] = []

    result = generate_report(
        report_id="report-invalid", analysis_id="run-1", **args
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "invalid_analysis_facts"
    assert "cross-validation" in result["message"].lower()


def test_outline_requires_report_type_sections_and_valid_fact_references():
    facts = _facts()
    invalid = ReportOutline(
        title="Report",
        sections=[
            OutlineSection(key="invented", title="Invented", purpose="No source.", fact_refs=["made.up"]),
        ] * 3,
    )

    result = validate_outline(invalid, facts)

    assert result.valid is False
    assert any("required sections" in error for error in result.errors)
    assert any("references facts" in error for error in result.errors)


def test_numeric_claims_and_unavailable_methods_are_rejected():
    facts = _facts()
    outline = _outline(facts)
    numeric_result = validate_document(_document(prose="The score equals 0.91."), facts, outline)
    assert numeric_result.valid is False
    assert any("numeric claims" in error for error in numeric_result.errors)

    facts_without_explainability = facts.model_copy(update={"explainability_available": False})
    explainability_document = _document(prose="SHAP proves the feature effect.")
    explainability_result = validate_document(
        explainability_document, facts_without_explainability, outline
    )
    assert any("explainability" in error.lower() for error in explainability_result.errors)


def test_missing_preprocessing_metadata_is_not_invented():
    facts = _facts()
    result = validate_document(
        _document(prose="Missing values were imputed before model fitting."),
        facts,
        _outline(facts),
    )

    assert any("preprocessing" in error.lower() for error in result.errors)


def test_graph_runs_outline_write_validation_html_and_pdf_with_mocked_model(tmp_path, monkeypatch):
    calls = _mock_model(monkeypatch)
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "unit-test-token")
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    from app.agents.report_agent import graph

    def fake_pdf(report_id, _html_path):
        path = tmp_path / f"{report_id}.pdf"
        path.write_bytes(b"%PDF-test")
        return {"status": "success", "path": str(path), "reason": None}

    monkeypatch.setattr(graph, "render_pdf", fake_pdf)
    args = _analysis_inputs()
    result = generate_report(
        report_id="report-valid", analysis_id="run-valid",
        report_type="model_comparison", language="French", **args,
    )

    assert calls == {"outline": 1, "document": 1}
    assert result["status"] == "success"
    assert result["validation_status"] == "passed"
    assert result["pdf_status"] == "success"
    assert Path(result["html_path"]).is_file()
    assert Path(result["pdf_path"]).read_bytes().startswith(b"%PDF")
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["language"] == "French"
    assert metadata["provider"] == "groq"
    assert "token" not in json.dumps(metadata).lower()


def test_graph_revises_at_most_twice_then_returns_controlled_failure(tmp_path, monkeypatch):
    calls = _mock_model(monkeypatch, invalid_documents=MAX_REPORT_REVISIONS + 1)
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "unit-test-token")
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    from app.agents.report_agent import graph

    monkeypatch.setattr(
        graph,
        "render_pdf",
        lambda *_args: pytest.fail("Invalid report content must not render."),
    )
    args = _analysis_inputs()
    result = generate_report(
        report_id="report-retries", analysis_id="run-1",
        report_type="model_comparison", **args,
    )

    assert calls["document"] == MAX_REPORT_REVISIONS + 1
    assert result["revision_count"] == MAX_REPORT_REVISIONS
    assert result["status"] == "failed"
    assert result["validation_status"] == "failed"
    assert result["error_code"] == "report_validation_failed"
    assert not (tmp_path / "report-retries.pdf").exists()


def test_report_html_escapes_untrusted_text_and_uses_no_external_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    facts = _facts()
    document = _document(prose="User supplied text: <script>alert(1)</script>")

    html_path = render_html("safe-report", document, facts)
    html = html_path.read_text(encoding="utf-8")

    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "https://" not in html
    assert "<iframe" not in html
    assert "counter(page)" in html
    assert "Contents" in html


def test_figures_are_whitelisted_pngs_and_staged_inside_report_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    png = b"\x89PNG\r\n\x1a\nsynthetic"
    encoded = base64.b64encode(png).decode("ascii")

    paths = stage_figures(
        "figures-1",
        {"feature_importance": encoded, "untrusted": encoded, "missing": "not-base64"},
    )

    assert set(paths) == {"feature_importance"}
    assert Path(paths["feature_importance"]).read_bytes() == png
    assert Path(paths["feature_importance"]).is_relative_to(tmp_path)


def test_pdf_renderer_success_and_unavailable_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    html_path = tmp_path / "pdf-check.html"
    html_path.write_text("<html><body>Report</body></html>", encoding="utf-8")
    executable = tmp_path / "chromium.exe"
    executable.write_bytes(b"test")

    class Page:
        def set_default_timeout(self, _value):
            pass

        def set_content(self, *_args, **_kwargs):
            pass

        def pdf(self, *, path, **_kwargs):
            Path(path).write_bytes(b"%PDF-test")

    class Browser:
        def new_page(self):
            return Page()

        def close(self):
            pass

    class Chromium:
        executable_path = str(executable)

        def launch(self, **_kwargs):
            return Browser()

    class Playwright:
        chromium = Chromium()

    class Manager:
        def __enter__(self):
            return Playwright()

        def __exit__(self, *_args):
            return False

    import playwright.sync_api

    monkeypatch.setattr(playwright.sync_api, "sync_playwright", lambda: Manager())
    success = render_pdf("pdf-check", html_path)
    assert success["status"] == "success"
    assert Path(success["path"]).read_bytes().startswith(b"%PDF")

    class MissingChromium:
        executable_path = str(tmp_path / "missing-chromium.exe")

    Playwright.chromium = MissingChromium()
    unavailable = render_pdf("pdf-unavailable", html_path)
    assert unavailable == {"status": "unavailable", "path": None, "reason": "chromium_not_installed"}
    assert not (tmp_path / "pdf-unavailable.pdf").exists()


def test_report_provider_uses_existing_settings_and_suppresses_secret_errors(monkeypatch, caplog):
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "unit-secret-token")
    monkeypatch.setattr(report_model, "_litellm_completion", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unit-secret-token")))

    with pytest.raises(report_model.ReportProviderError) as error:
        report_model.invoke_structured(
            system_prompt="system", user_prompt="user", response_schema=ReportOutline
        )

    assert report_model.resolve_report_model() == "groq/openai/gpt-oss-20b"
    assert "unit-secret-token" not in str(error.value)
    assert "unit-secret-token" not in caplog.text


def test_groq_provider_model_prefix_and_pipeline_provider_are_preserved(monkeypatch):
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    assert report_model.resolve_report_model() == "groq/openai/gpt-oss-20b"

    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "groq/openai/gpt-oss-20b")
    assert report_model.resolve_report_model() == "groq/openai/gpt-oss-20b"

    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "groq/groq/openai/gpt-oss-20b")
    with pytest.raises(report_model.ReportAgentConfigurationError, match="REPORT_LLM_MODEL"):
        report_model.resolve_report_model()

    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "xai")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "grok-4.7")
    with pytest.raises(report_model.ReportAgentConfigurationError, match="REPORT_LLM_PROVIDER"):
        report_model.resolve_report_model()

    assert settings.LLM_PROVIDER == "ollama"


class _FakeProviderError(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


@pytest.mark.parametrize(
    ("status", "detail", "expected_code", "message_fragment"),
    [
        (401, "invalid api key: unit-secret-value", "report_authentication_failed", "HTTP 401"),
        (402, "insufficient credits: unit-secret-value", "report_quota_exceeded", "quota or credits"),
        (403, "permission denied: unit-secret-value", "report_model_access_denied", "denied access"),
        (404, "model not found: unit-secret-value", "report_model_not_found", "could not find"),
        (408, "request timed out: unit-secret-value", "report_provider_timeout", "timed out"),
        (429, "rate limit reached: unit-secret-value", "report_rate_limited", "rate limit"),
        (429, "daily quota exceeded: unit-secret-value", "report_quota_exceeded", "quota or credits"),
        (500, "internal server error: unit-secret-value", "report_provider_server_error", "HTTP 500"),
        (400, "unsupported JSON schema: unit-secret-value", "report_structured_request_rejected", "HTTP 400"),
    ],
)
def test_groq_provider_errors_are_classified_without_echoing_secrets(
    monkeypatch, status, detail, expected_code, message_fragment
):
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "configured-secret")
    monkeypatch.setattr(
        report_model,
        "_litellm_completion",
        lambda **_kwargs: (_ for _ in ()).throw(_FakeProviderError(status, detail)),
    )

    with pytest.raises(report_model.ReportProviderError) as caught:
        report_model.invoke_structured(
            system_prompt="system", user_prompt="user", response_schema=ReportOutline
        )

    assert caught.value.error_code == expected_code
    assert message_fragment.lower() in str(caught.value).lower()
    assert "unit-secret-value" not in str(caught.value)
    assert "configured-secret" not in str(caught.value)


def test_groq_retry_after_is_reported_without_retrying(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "configured-secret")

    def fail_once(**_kwargs):
        calls.append(True)
        raise _FakeProviderError(429, "rate limit", {"Retry-After": "23"})

    monkeypatch.setattr(report_model, "_litellm_completion", fail_once)
    with pytest.raises(report_model.ReportProviderError) as caught:
        report_model.invoke_structured(
            system_prompt="system", user_prompt="user", response_schema=ReportOutline
        )

    assert len(calls) == 1
    assert caught.value.retry_after_seconds == 23
    assert "Retry after 23 seconds" in str(caught.value)


@pytest.mark.parametrize(
    ("provider_exception", "expected_code"),
    [
        (TimeoutError("request timed out"), "report_provider_timeout"),
        (ConnectionError("network unavailable"), "report_network_error"),
    ],
)
def test_groq_timeout_and_network_failures_are_actionable(
    monkeypatch, provider_exception, expected_code
):
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "configured-secret")
    monkeypatch.setattr(
        report_model,
        "_litellm_completion",
        lambda **_kwargs: (_ for _ in ()).throw(provider_exception),
    )

    with pytest.raises(report_model.ReportProviderError) as caught:
        report_model.invoke_structured(
            system_prompt="system", user_prompt="user", response_schema=ReportOutline
        )

    assert caught.value.error_code == expected_code


def test_groq_strict_schema_closes_objects_and_requires_every_property():
    schema = report_model._groq_strict_schema(ReportDocument.model_json_schema())
    allowed = report_model._GROQ_SCHEMA_KEYS | {"required", "additionalProperties"}

    def walk(node):
        if isinstance(node, dict):
            assert set(node) <= allowed
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
            for value in node.get("properties", {}).values():
                walk(value)
            for value in node.get("$defs", {}).values():
                walk(value)
            if "items" in node:
                walk(node["items"])
            for value in node.get("anyOf", []):
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert schema["properties"]["subtitle"]["anyOf"][-1] == {"type": "null"}


@pytest.mark.parametrize(
    ("choices", "expected_code"),
    [
        ([], "report_empty_response"),
        ([SimpleNamespace(message=SimpleNamespace(content=""))], "report_empty_response"),
        ([SimpleNamespace(message=SimpleNamespace(content="not-json"))], "report_malformed_json"),
        ([SimpleNamespace(message=SimpleNamespace(content="{}"))], "report_schema_validation_failed"),
    ],
)
def test_groq_incomplete_or_invalid_structured_responses_fail_safely(
    monkeypatch, choices, expected_code
):
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "configured-secret")
    monkeypatch.setattr(
        report_model,
        "_litellm_completion",
        lambda **_kwargs: SimpleNamespace(choices=choices),
    )

    with pytest.raises(report_model.ReportProviderError) as caught:
        report_model.invoke_structured(
            system_prompt="system", user_prompt="user", response_schema=ReportOutline
        )

    assert caught.value.error_code == expected_code
    assert "configured-secret" not in str(caught.value)


def test_missing_report_key_is_lazy_and_application_import_still_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", True)
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "")
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)
    env = os.environ.copy()
    env.update({
        "REPORT_AGENT_ENABLED": "true",
        "REPORT_LLM_PROVIDER": "groq",
        "REPORT_LLM_MODEL": "openai/gpt-oss-20b",
        "REPORT_LLM_API_KEY": "",
        "KAGGLE_CONFIG_DIR": str(tmp_path / "empty-kaggle-config"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    })
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(name, None)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.config import settings; assert not settings.REPORT_LLM_API_KEY; "
            "from app.main import app; assert app.title == 'AutoML Agent API'",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    args = _analysis_inputs()

    result = generate_report(
        report_id="no-key", analysis_id="run-1", report_type="model_comparison", **args
    )

    assert result["status"] == "failed"
    assert result["error_code"] == "report_configuration_missing"
    assert "REPORT_LLM_API_KEY" in result["message"]

def test_litellm_sends_groq_strict_schema_and_pydantic_validates_response(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "title": "Test outline",
            "sections": [
                {"key": "one", "title": "One", "purpose": "Purpose", "fact_refs": ["known"]},
                {"key": "two", "title": "Two", "purpose": "Purpose", "fact_refs": ["known"]},
                {"key": "three", "title": "Three", "purpose": "Purpose", "fact_refs": ["known"]},
            ],
        })))]
    )
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(settings, "REPORT_LLM_API_KEY", "test-token")
    monkeypatch.setattr(settings, "REPORT_LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "REPORT_LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(report_model, "_litellm_completion", fake_completion)

    outline = report_model.invoke_structured(
        system_prompt="system", user_prompt="user", response_schema=ReportOutline
    )

    assert isinstance(outline, ReportOutline)
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["type"] == "object"
    assert captured["model"] == "groq/openai/gpt-oss-20b"
    assert captured["timeout"] == settings.REPORT_LLM_TIMEOUT_SECONDS
    assert captured["stream"] is False
    assert captured["num_retries"] == 0
    assert captured["api_key"] == "test-token"
    assert "tools" not in captured


def test_disabled_agent_uses_legacy_pdf_and_marks_generator(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORT_AGENT_ENABLED", False)
    monkeypatch.setattr(settings, "REPORT_DIR", tmp_path)

    def fake_legacy(**kwargs):
        path = tmp_path / f"{kwargs['task_id']}.pdf"
        path.write_bytes(b"%PDF-legacy")
        return path

    legacy = importlib.import_module("app.core.report_generator")
    monkeypatch.setattr(legacy, "generate_report", fake_legacy)
    args = _analysis_inputs()

    result = generate_report(report_id="legacy-report", **args)

    assert result["status"] == "success"
    assert result["provider"] == "legacy"
    assert result["pdf_status"] == "success"
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["provider"] == "legacy"


def test_unexpected_report_failure_is_separate_from_analysis_result(monkeypatch):
    from app.agents.report_agent import service

    monkeypatch.setattr(service, "generate_report", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private")))

    result = generate_report_safely(report_id="r-1", analysis_id="a-1")

    assert result["status"] == "failed"
    assert result["error_code"] == "report_service_failed"
    assert "private" not in result["message"]


def test_graph_state_is_compact_and_does_not_include_dataset_or_model_objects():
    from app.agents.report_agent.service import _compact_analysis_inputs

    args = _analysis_inputs()
    args["data"]["sample_rows"] = [{"sensitive": "row"}]
    compact = _compact_analysis_inputs(
        data=args["data"], eda=args["eda"], operations=args["operations"],
        training=args["training"], task_type=args["task_type"],
        target_column=args["target_column"], chart_paths={"target": "C:/reports/chart.png"},
        experiment={}, model_registry={},
    )

    assert "sample_rows" not in compact["data"]
    assert "best_model" not in compact["training"]
    assert compact["figures"] == {"target": True}
    assert compact["training"]["results"] == args["training"]["results"]


def test_sampled_report_facts_include_explicit_limitation():
    facts = _facts()

    assert any("loaded sample" in limitation for limitation in facts.limitations)

