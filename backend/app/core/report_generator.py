"""
English PDF report generator using WeasyPrint + Jinja2.
"""
from __future__ import annotations

import base64
import binascii
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape
from weasyprint import HTML

from app.config import settings

logger = logging.getLogger(__name__)

TASK_TYPE_EN = {
    "classification": "Classification",
    "regression": "Regression",
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="ltr" lang="en">
<head><meta charset="UTF-8">
<style>
  @page { size: A4; margin: 1.5cm; }
  body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color:#2c3e50; line-height:1.6; font-size: 11px; }
  h1 { color:#1a252f; border-bottom:3px solid #3498db; padding-bottom:10px; font-size: 26px; text-align: center; }
  h2 { color:#2980b9; margin-top:30px; border-left:5px solid #3498db; padding-left:10px; font-size: 18px; page-break-after: avoid; }
  h3 { color:#34495e; font-size: 15px; }
  table { width:100%; border-collapse:collapse; margin:15px 0; font-size:11px; page-break-inside: avoid; }
  th { background:#3498db; color:#fff; padding:10px; text-align:left; }
  td { padding:10px; border-bottom:1px solid #ecf0f1; vertical-align: top; }
  tr:nth-child(even) { background:#f9fbfd; }
  .figure { text-align:center; margin:20px 0; page-break-inside: avoid; }
  .figure img { max-width:100%; height:auto; border:1px solid #ecf0f1; border-radius:5px; padding:5px; }
  .summary-box { background:#e8f4f8; padding:20px; border:1px solid #bce8f1; border-radius:8px; margin:20px 0; page-break-inside: avoid; }
  .ok { color:#27ae60; font-weight: bold; } .warn { color:#e67e22; font-weight: bold; }
  ul { padding-left:20px; }
  .footer { text-align:center; color:#95a5a6; font-size:10px; margin-top:50px; border-top:1px solid #ecf0f1; padding-top:10px; }
</style></head>
<body>
  <h1>Automated Data Analysis Report</h1>
  <p style="text-align: center; color: #7f8c8d;">Generated: {{ timestamp }}</p>

  <div class="summary-box">
    <h3>Executive Summary</h3>
    <ul>
      <li>Rows: <strong>{{ data.n_rows }}</strong> | Columns: <strong>{{ data.n_columns }}</strong></li>
      <li>Target column: <strong>{{ target_column }}</strong></li>
      <li>Task type: <strong>{{ task_type_en }}</strong></li>
      <li>Best model: <strong>{{ training.best_model_name }}</strong></li>
      <li>{{ training.metric }}: <strong>{{ training.best_score|number }}</strong></li>
    </ul>
  </div>

  <h2>1. Data Overview</h2>
  <table>
    <tr><th>Property</th><th>Value</th></tr>
    <tr><td>Columns</td><td>{{ data.columns|join(', ') }}</td></tr>
    <tr><td>Missing values</td><td>{{ data.missing_total }}</td></tr>
    <tr><td>Duplicate rows</td><td>{{ data.duplicate_rows }}</td></tr>
    <tr><td>Numeric columns</td><td>{{ data.numeric_columns|length }}</td></tr>
    <tr><td>Categorical columns</td><td>{{ data.categorical_columns|length }}</td></tr>
  </table>

  <h2>2. Data Readiness Assessment</h2>
  <table>
    <tr><th>Requirement</th><th>Status</th></tr>
    {% for k, v in readiness.items() %}
    <tr>
      <td>{{ k }}</td>
      <td>{% if v is sameas true %}<span class="warn">Yes</span>
          {% elif v is sameas false %}<span class="ok">No</span>
          {% else %}{{ v }}{% endif %}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>3. Exploratory Data Analysis</h2>
  {% for name in ['missing', 'numeric_dist', 'categorical_dist', 'correlation', 'outliers'] %}
    {% if name in figures %}
    <div class="figure">
      <img src="data:image/png;base64,{{ figures[name] }}" alt="{{ name }}">
    </div>
    {% endif %}
  {% endfor %}

  <h2>4. Target Variable Analysis</h2>
  {% if 'target' in figures %}
  <div class="figure">
    <img src="data:image/png;base64,{{ figures['target'] }}" alt="Target Distribution">
  </div>
  {% endif %}

  <h2>5. Applied Data Operations</h2>
  <table>
    <tr><th style="width: 30%;">Operation Type</th><th>Details</th></tr>
    {% for op in operations %}
    <tr>
      <td>{{ op.type }}</td>
      <td>
        {% for key, value in op.details.items() %}
          <strong>{{ key }}:</strong> {{ value }}{% if not loop.last %}, {% endif %}
        {% endfor %}
      </td>
    </tr>
    {% endfor %}
  </table>

  <h2>6. Model Comparison Results</h2>
  <table>
    <tr>
      <th>Model</th>
      {% for m in training.metric_names %}<th>{{ m }}</th>{% endfor %}
    </tr>
    {% for r in training.results %}
    <tr>
      <td>{{ r.model }}</td>
      {% for m in training.metric_names %}
        <td>{{ r[m]|number }}</td>
      {% endfor %}
    </tr>
    {% endfor %}
  </table>

  {% if 'model_comparison' in figures %}
  <div class="figure">
    <img src="data:image/png;base64,{{ figures['model_comparison'] }}" alt="Model Comparison">
  </div>
  {% endif %}

  <div class="summary-box">
    <h3>Winning Model: {{ training.best_model_name }}</h3>
    <p>This model achieved the highest {{ training.metric }} score of <strong>{{ training.best_score|number }}</strong>.</p>
  </div>

  <h2>7. Recommendations</h2>
  <ul>{% for rec in recommendations %}<li>{{ rec }}</li>{% endfor %}</ul>

  <div class="footer">This report was generated automatically by AutoML Agent</div>
</body></html>
"""


def generate_report(
    task_id: str,
    data: dict[str, Any],
    eda: dict[str, Any],
    operations: list[dict[str, Any]],
    training: dict[str, Any],
    task_type: str,
    target_column: str,
) -> Path:
    """Generate a PDF report, tolerating incomplete optional report data."""
    data = _normalize_data(data)
    training = _normalize_training(training)
    operations = _normalize_operations(operations)

    eda = eda if isinstance(eda, dict) else {}
    summary = eda.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    readiness = summary.get("readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    figures = eda.get("figures")
    figures = figures if isinstance(figures, dict) else {}
    figures = {
        str(name): value
        for name, value in figures.items()
        if _is_png_base64(value)
    }

    recs = _recommendations(data, training, task_type)
    environment = Environment(autoescape=select_autoescape(["html", "xml"]))
    environment.filters["number"] = _format_number
    html = environment.from_string(HTML_TEMPLATE).render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data=data,
        target_column=target_column or "Not specified",
        task_type_en=TASK_TYPE_EN.get(task_type, task_type),
        readiness=readiness,
        figures=figures,
        operations=operations,
        training=training,
        recommendations=recs,
    )

    out = settings.REPORT_DIR / f"{task_id}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(out))

    logger.info("Report saved: %s", out)
    return out


def _normalize_data(data: Any) -> dict[str, Any]:
    """Provide template-safe defaults for missing or malformed data summaries."""
    result = dict(data) if isinstance(data, dict) else {}
    result.setdefault("n_rows", 0)
    result.setdefault("n_columns", 0)
    result.setdefault("columns", [])
    result.setdefault("missing_total", 0)
    result.setdefault("duplicate_rows", 0)
    result.setdefault("numeric_columns", [])
    result.setdefault("categorical_columns", [])

    if not isinstance(result["columns"], (list, tuple)):
        result["columns"] = []
    else:
        result["columns"] = [str(column) for column in result["columns"]]

    for key in ("numeric_columns", "categorical_columns"):
        if not isinstance(result[key], (list, tuple)):
            result[key] = []

    for key in ("n_rows", "n_columns", "missing_total", "duplicate_rows"):
        try:
            result[key] = max(0, int(result[key]))
        except (TypeError, ValueError, OverflowError):
            result[key] = 0

    return result


def _normalize_training(training: Any) -> dict[str, Any]:
    """Normalize metrics so malformed or partial results still render clearly."""
    result = dict(training) if isinstance(training, dict) else {}
    result.setdefault("best_model_name", "Unavailable")
    result.setdefault("best_score", None)
    result.setdefault("metric", "Metric")

    metric_names = result.get("metric_names")
    if not isinstance(metric_names, (list, tuple)):
        metric_names = []
    result["metric_names"] = [str(metric) for metric in metric_names]

    rows = result.get("results")
    if not isinstance(rows, (list, tuple)):
        rows = []
    normalized_rows = []
    for row in rows:
        row = row if isinstance(row, dict) else {}
        normalized = {"model": str(row.get("model", "Unknown model"))}
        for metric in result["metric_names"]:
            normalized[metric] = row.get(metric)
        normalized_rows.append(normalized)
    result["results"] = normalized_rows

    if result["best_model_name"] is None:
        result["best_model_name"] = "Unavailable"
    if result["metric"] is None:
        result["metric"] = "Metric"
    return result


def _normalize_operations(operations: Any) -> list[dict[str, Any]]:
    """Ensure operation rows always have the fields expected by the template."""
    if not isinstance(operations, (list, tuple)):
        return []

    normalized = []
    for operation in operations:
        operation = operation if isinstance(operation, dict) else {}
        details = operation.get("details")
        normalized.append({
            "type": str(operation.get("type", "Unknown operation")),
            "details": details if isinstance(details, dict) else {},
        })
    return normalized


def _format_number(value: Any) -> str:
    """Format finite numeric report values, displaying unavailable values safely."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    return f"{number:.4f}" if math.isfinite(number) else "N/A"


def _is_png_base64(value: Any) -> bool:
    """Ignore malformed EDA images so one corrupt figure cannot break the PDF."""
    if not isinstance(value, str) or not value:
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return decoded.startswith(b"\x89PNG\r\n\x1a\n")


def _recommendations(data: dict, training: dict, task_type: str) -> list[str]:
    recs: list[str] = []
    if data.get("missing_total", 0) > 0:
        recs.append(f"Handled {data['missing_total']} missing values automatically.")
    if data.get("duplicate_rows", 0) > 0:
        recs.append(f"Removed {data['duplicate_rows']} duplicate rows.")
    try:
        best_score = float(training.get("best_score"))
    except (TypeError, ValueError, OverflowError):
        best_score = float("nan")

    if not math.isfinite(best_score):
        recs.append("Model performance could not be assessed from the available metrics.")
    elif best_score < 0.7:
        recs.append("Model performance is below 70% - consider collecting more data or feature engineering.")
    else:
        recs.append(
            f"The winning model ({training.get('best_model_name', 'Unavailable')}) "
            "achieved good performance."
        )
    if task_type == "classification":
        recs.append("Try Boosting models (XGBoost/LightGBM) to potentially improve performance.")
    else:
        recs.append("Consider hyperparameter tuning to further improve performance.")
    return recs
