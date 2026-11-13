"""Safe self-contained HTML and browser PDF rendering for validated reports."""
from __future__ import annotations

import base64
import binascii
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment

from app.agents.report_agent.facts_builder import ALLOWED_FIGURES
from app.agents.report_agent.schemas import ReportDocument, ReportFacts
from app.config import settings

logger = logging.getLogger(__name__)
MAX_FIGURE_BYTES = 8 * 1024 * 1024
_REPORT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

REPORT_HTML = """<!doctype html>
<html lang="{{ language_code }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ document.title }}</title>
<style>
  @page { size: A4; margin: 20mm 18mm 20mm; @bottom-right { content: "Page " counter(page) " of " counter(pages); color:#657080; font:9pt Arial,sans-serif; } }
  :root { color-scheme: light; }
  body { color:#202a36; font: 10.5pt/1.58 Georgia, "Times New Roman", serif; margin:0 auto; max-width: 850px; }
  header { border-bottom:2px solid #233a56; margin-bottom:24px; padding-bottom:16px; }
  .eyebrow { color:#526b86; font:700 9pt Arial,sans-serif; letter-spacing:.08em; text-transform:uppercase; }
  h1 { color:#182b42; font:700 25pt/1.2 Arial,sans-serif; margin:10px 0; }
  .subtitle { color:#4d5d70; font:12pt/1.4 Arial,sans-serif; }
  h2 { border-bottom:1px solid #cbd4df; color:#233a56; font:700 16pt Arial,sans-serif; margin:26px 0 10px; padding-bottom:5px; page-break-after:avoid; }
  h3 { color:#2c4159; font:700 12pt Arial,sans-serif; page-break-after:avoid; }
  p { margin:8px 0 12px; }
  .summary { background:#f3f6f9; border-left:4px solid #42678e; margin:14px 0 22px; padding:12px 16px; }
  .toc { background:#fafbfd; border:1px solid #d9e0e8; padding:10px 16px; page-break-inside:avoid; }
  .toc a { color:#233a56; text-decoration:none; }
  table { border-collapse:collapse; font:9.5pt Arial,sans-serif; margin:12px 0 18px; page-break-inside:avoid; width:100%; }
  th { background:#e8edf3; color:#233a56; text-align:left; }
  th,td { border:1px solid #cbd4df; padding:7px 9px; vertical-align:top; }
  tr { page-break-inside:avoid; }
  ul { margin:7px 0 14px; padding-left:22px; }
  li { margin:4px 0; }
  .figure { margin:18px auto; max-width:100%; page-break-inside:avoid; text-align:center; }
  .figure img { height:auto; max-width:100%; }
  .caption,.evidence,.muted { color:#596777; font:9pt/1.45 Arial,sans-serif; }
  .evidence { border-left:2px solid #d6dee7; margin:10px 0 22px; padding-left:9px; }
  .page-break { page-break-before:always; }
  footer { border-top:1px solid #cbd4df; color:#657080; font:9pt Arial,sans-serif; margin-top:30px; padding-top:8px; }
  @media print { body { max-width:none; } a { color:inherit; } }
</style>
</head>
<body>
<header>
  <div class="eyebrow">AutoML Agent · {{ report_type_label }}</div>
  <h1>{{ document.title }}</h1>
  {% if document.subtitle %}<div class="subtitle">{{ document.subtitle }}</div>{% endif %}
</header>
<div class="summary">
  <strong>Analysis context</strong><br>
  Task: {{ facts.task_type }} · Target: {{ facts.target_column }} · Loaded rows: {{ facts.dataset.loaded_rows }}
</div>
{% if show_toc %}
<nav class="toc" aria-label="Table of contents"><h3>Contents</h3><ol>
{% for section in document.sections %}<li><a href="#{{ section.key }}">{{ section.title }}</a></li>{% endfor %}
</ol></nav>
{% endif %}
{% for section in document.sections %}
<section id="{{ section.key }}">
  <h2>{{ section.title }}</h2>
  {% for paragraph in section.paragraphs %}<p>{{ paragraph }}</p>{% endfor %}
  {% if section.bullets %}<ul>{% for item in section.bullets %}<li>{{ item }}</li>{% endfor %}</ul>{% endif %}
  {% if section.key == 'dataset' %}
  <table><tbody>
    <tr><th>Original source rows</th><td>{{ facts.dataset.source_rows if facts.dataset.source_rows is not none else "Unavailable" }}</td></tr>
    <tr><th>Loaded rows</th><td>{{ facts.dataset.loaded_rows }}</td></tr>
    <tr><th>Sampling applied</th><td>{{ facts.dataset.sampling_applied if facts.dataset.sampling_applied is not none else "Unknown" }}</td></tr>
    {% if facts.dataset.sampling_applied %}<tr><th>Sampling method</th><td>{{ facts.dataset.sampling_method or "Unavailable" }}</td></tr>{% endif %}
    <tr><th>Columns</th><td>{{ facts.dataset.column_count if facts.dataset.column_count is not none else "Unavailable" }}</td></tr>
    <tr><th>Missing values in loaded data</th><td>{{ facts.dataset.missing_values if facts.dataset.missing_values is not none else "Unavailable" }}</td></tr>
    <tr><th>Duplicate rows in loaded data</th><td>{{ facts.dataset.duplicate_rows if facts.dataset.duplicate_rows is not none else "Unavailable" }}</td></tr>
  </tbody></table>
  {% if facts.dataset.sampling_applied %}<p class="muted">Profile findings and evaluation describe the loaded sample. Rare groups may not be represented.</p>{% endif %}
  {% endif %}
  {% if section.key == 'data_quality' and facts.dataset.profile_findings %}<ul>{% for finding in facts.dataset.profile_findings %}<li>{{ finding }}</li>{% endfor %}</ul>{% endif %}
  {% if section.key == 'model_comparison' %}
  <p class="muted">The selection metric is {{ facts.selection_metric }}. These values are cross-validation results and are separate from the holdout evaluation.</p>
  <table><thead><tr><th>Model</th>{% for metric in metric_names %}<th>{{ metric }}</th>{% endfor %}</tr></thead><tbody>
  {% for row in facts.cv_results %}<tr><td>{{ row.model }}{% if row.model == facts.best_model %} · selected{% endif %}</td>{% for metric in metric_names %}<td>{{ row.metrics.get(metric) if row.metrics.get(metric) is not none else "Unavailable" }}</td>{% endfor %}</tr>{% endfor %}
  </tbody></table>
  <p class="muted">Selected model: {{ facts.best_model }} · CV selection score: {{ facts.cv_selection_score }} ({{ facts.selection_metric }}).</p>
  {% endif %}
  {% if section.key == 'evaluation' %}
    {% if facts.holdout.available %}
    <p class="muted">Independent holdout results for this run; they are estimates and do not guarantee future performance.</p>
    <table><tbody><tr><th>Evaluation method</th><td>{{ facts.holdout.method or "Unavailable" }}</td></tr><tr><th>Holdout rows</th><td>{{ facts.holdout.test_rows if facts.holdout.test_rows is not none else "Unavailable" }}</td></tr>
    {% for metric,value in facts.holdout.metrics.items() %}<tr><th>{{ metric }}</th><td>{{ value if value is not none else "Unavailable" }}</td></tr>{% endfor %}</tbody></table>
    {% else %}<p class="muted">Independent holdout evaluation was unavailable for this run.</p>{% endif %}
  {% endif %}
  {% if section.key == 'preprocessing' %}
    {% if facts.preprocessing_operations %}<ul>{% for operation in facts.preprocessing_operations %}<li>{{ operation }}</li>{% endfor %}</ul>{% else %}<p>Preprocessing operation details were not recorded.</p>{% endif %}
  {% endif %}
  {% if section.key == 'limitations' and facts.limitations %}<ul>{% for limitation in facts.limitations %}<li>{{ limitation }}</li>{% endfor %}</ul>{% endif %}
  {% if section.key == 'explainability' %}
    {% for name,path in charts.items() if name in ['feature_importance','shap_summary'] %}<figure class="figure"><img src="{{ path }}" alt="{{ name }}"><figcaption class="caption">Recorded {{ name|replace('_',' ') }} output. Importance is descriptive and does not establish causation.</figcaption></figure>{% endfor %}
    {% if not facts.explainability_available %}<p>No explainability output was recorded for this analysis.</p>{% endif %}
  {% endif %}
  {% if section.key in ['data_quality','methodology'] %}
    {% for name,path in charts.items() if name in ['missing','numeric_dist','categorical_dist','correlation','outliers','target'] %}<figure class="figure"><img src="{{ path }}" alt="{{ name|replace('_',' ') }}"><figcaption class="caption">Recorded exploratory analysis: {{ name|replace('_',' ') }}.</figcaption></figure>{% endfor %}
  {% endif %}
  <div class="evidence"><strong>Evidence:</strong> {% for ref in section.evidence_refs %}{{ facts.evidence.get(ref, "") }}{% if not loop.last %} {% endif %}{% endfor %}</div>
</section>
{% endfor %}
<footer>Generated from validated AutoML analysis facts. Numeric results above are supplied by the analysis backend.</footer>
</body></html>"""


def _validate_report_id(report_id: str) -> str:
    if not _REPORT_ID.fullmatch(report_id):
        raise ValueError("Report ID contains unsupported characters.")
    return report_id


def _storage_path(path: Path) -> Path:
    root = settings.REPORT_DIR.resolve()
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def stage_figures(report_id: str, figures: Any) -> dict[str, str]:
    """Store validated PNGs and return compact paths for graph state."""
    report_id = _validate_report_id(report_id)
    if not isinstance(figures, dict):
        return {}
    assets_dir = _storage_path(settings.REPORT_DIR / f"{report_id}.assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for name, encoded in figures.items():
        name = str(name)
        if name not in ALLOWED_FIGURES or not isinstance(encoded, str):
            continue
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            continue
        if not decoded.startswith(b"\x89PNG\r\n\x1a\n") or len(decoded) > MAX_FIGURE_BYTES:
            continue
        path = _storage_path(assets_dir / f"{name}.png")
        path.write_bytes(decoded)
        staged[name] = str(path)
    return staged


def _inline_chart(path_value: str) -> str:
    path = _storage_path(Path(path_value))
    if not path.is_file() or path.stat().st_size > MAX_FIGURE_BYTES:
        raise ValueError("A report chart file is unavailable or too large.")
    contents = path.read_bytes()
    if not contents.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("A report chart file is not a PNG image.")
    return "data:image/png;base64," + base64.b64encode(contents).decode("ascii")


def render_html(
    report_id: str,
    document: ReportDocument,
    facts: ReportFacts,
    chart_paths: dict[str, str] | None = None,
) -> Path:
    report_id = _validate_report_id(report_id)
    chart_paths = chart_paths or {}
    charts = {
        name: _inline_chart(path)
        for name, path in chart_paths.items()
        if name in ALLOWED_FIGURES
    }
    report_type_label = facts.report_type.replace("_", " ").title()
    language_code = facts.language.split("-", 1)[0].lower()
    html = Environment(autoescape=True).from_string(REPORT_HTML).render(
        document=document.model_dump(mode="json"),
        facts=facts.model_dump(mode="json"),
        charts=charts,
        metric_names=list(dict.fromkeys(
            metric for row in facts.cv_results for metric in row.metrics
        )),
        show_toc=len(document.sections) >= 5,
        report_type_label=report_type_label,
        language_code=language_code,
    )
    # The template is static, text is autoescaped, and chart URLs are validated
    # data URIs. Reject accidental active elements before preserving the file.
    if re.search(r"<(script|iframe|object|embed|form)\b|https?://|javascript\s*:", html, re.I):
        raise ValueError("Rendered report HTML contains an unsafe active resource.")
    output = _storage_path(settings.REPORT_DIR / f"{report_id}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def render_pdf(report_id: str, html_path: Path) -> dict[str, str | None]:
    """Render PDF with the installed Chromium runtime; never download a browser."""
    report_id = _validate_report_id(report_id)
    html_path = _storage_path(html_path)
    output = _storage_path(settings.REPORT_DIR / f"{report_id}.pdf")
    browser = None
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                return {"status": "unavailable", "path": None, "reason": "chromium_not_installed"}
            browser = playwright.chromium.launch(timeout=30_000)
            page = browser.new_page()
            page.set_default_timeout(30_000)
            page.set_content(html_path.read_text(encoding="utf-8"), wait_until="load", timeout=30_000)
            try:
                page.pdf(
                    path=str(output), format="A4", print_background=True,
                    display_header_footer=True,
                    header_template="<span></span>",
                    footer_template=(
                        '<div style="font-size:8px;width:100%;text-align:right;'
                        'padding-right:18mm;color:#657080">Page <span class="pageNumber"></span>'
                        ' of <span class="totalPages"></span></div>'
                    ),
                    margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"},
                    timeout=30_000,
                )
            finally:
                browser.close()
                browser = None
        if not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            return {"status": "failed", "path": None, "reason": "empty_pdf_output"}
        return {"status": "success", "path": str(output), "reason": None}
    except Exception as exc:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("Report PDF rendering failed (error_type=%s)", type(exc).__name__)
        return {
            "status": "unavailable",
            "path": None,
            "reason": f"browser_render_failed:{type(exc).__name__}",
        }
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
