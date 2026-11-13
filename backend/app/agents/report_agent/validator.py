"""Deterministic validation for report facts, outlines, and authored content."""
from __future__ import annotations

import re
import math
from typing import Any

from pydantic import ValidationError

from app.agents.report_agent.schemas import (
    REPORT_SECTIONS,
    ReportDocument,
    ReportFacts,
    ReportOutline,
    ValidationResult,
)

_ACTIVE_MARKUP = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|form|svg|img|link|meta|style)\b|javascript\s*:",
    re.IGNORECASE,
)
_PREPROCESSING_TERMS: dict[str, tuple[str, ...]] = {
    "imputation": ("imput", "fill missing", "missing-value replacement"),
    "encoding": ("encod", "one-hot", "label transform"),
    "scaling": ("scal", "standardiz", "normaliz"),
    "outlier": ("outlier remov", "outlier eliminat"),
    "sampling": ("oversampl", "undersampl", "resampl"),
    "feature engineering": ("feature engineer",),
    "class imbalance": ("class weight", "class imbalance handling"),
}
_KNOWN_MODEL_NAMES = {
    "randomforest", "randomforestclassifier", "logisticregression",
    "decisiontree", "decisiontreeclassifier", "gradientboosting",
    "histgradientboosting", "extratrees", "adaboost", "kneighbors",
    "kneighborsclassifier", "svc", "svm", "linearregression", "ridge",
    "lasso", "xgboost", "lightgbm", "catboost", "gaussiannb",
}


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else {}


def validate_facts(value: Any) -> ValidationResult:
    try:
        facts = value if isinstance(value, ReportFacts) else ReportFacts.model_validate(value)
    except ValidationError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Report facts are invalid: {error['loc'][-1]}" for error in exc.errors()[:10]],
        )

    errors = []
    data = facts.dataset
    if data.loaded_rows < 0:
        errors.append("Loaded row count cannot be negative.")
    if data.chunk_size is not None and data.chunk_size <= 0:
        errors.append("CSV chunk size must be a positive integer when present.")
    if data.source_rows is not None and data.source_rows < data.loaded_rows:
        errors.append("Original source rows cannot be fewer than loaded rows.")
    if data.sampling_applied is True and data.source_rows is not None:
        if data.source_rows <= data.loaded_rows:
            errors.append("Sampling metadata conflicts with the source and loaded row counts.")
    if data.sampling_applied is False and data.source_rows is not None:
        if data.source_rows != data.loaded_rows:
            errors.append("Unsampled row counts must match the source row count.")

    model_names = {row.model for row in facts.cv_results}
    if facts.best_model not in model_names:
        errors.append("The selected model is absent from cross-validation comparison results.")
    if facts.cv_selection_score is None:
        errors.append("The cross-validation selection score is unavailable.")
    elif not math.isfinite(float(facts.cv_selection_score)):
        errors.append("The cross-validation selection score must be finite.")
    if not facts.selection_metric:
        errors.append("The cross-validation selection metric is unavailable.")
    selected_row = next((row for row in facts.cv_results if row.model == facts.best_model), None)
    if selected_row is not None:
        selected_score = selected_row.metrics.get(facts.selection_metric)
        if selected_score is not None and facts.cv_selection_score is not None:
            if not math.isclose(
                float(selected_score), float(facts.cv_selection_score),
                rel_tol=1e-6, abs_tol=1e-6,
            ):
                errors.append(
                    "The selected model score does not match its recorded CV selection metric."
                )
    if facts.holdout.available and not facts.holdout.metrics:
        errors.append("Holdout evaluation is marked available but has no recorded metrics.")
    if not facts.holdout.available and facts.holdout.metrics:
        errors.append("Holdout metrics must remain separate and unavailable when no holdout ran.")
    if facts.holdout.available and facts.holdout.test_rows is not None and facts.holdout.test_rows <= 0:
        errors.append("An available holdout must contain at least one test row.")
    if facts.holdout.test_size is not None and not 0 < facts.holdout.test_size < 1:
        errors.append("Holdout test fraction must be between zero and one.")
    for row in facts.cv_results:
        if any(value is not None and not math.isfinite(float(value)) for value in row.metrics.values()):
            errors.append("Cross-validation metrics must be finite numbers or unavailable.")
            break
    if any(value is not None and not math.isfinite(float(value)) for value in facts.holdout.metrics.values()):
        errors.append("Holdout metrics must be finite numbers or unavailable.")
    return ValidationResult(valid=not errors, errors=errors)


def validate_outline(value: Any, facts_value: Any) -> ValidationResult:
    try:
        outline = value if isinstance(value, ReportOutline) else ReportOutline.model_validate(value)
        facts = facts_value if isinstance(facts_value, ReportFacts) else ReportFacts.model_validate(facts_value)
    except ValidationError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Report outline or facts are invalid: {error['loc'][-1]}" for error in exc.errors()[:10]],
        )

    spec = REPORT_SECTIONS[facts.report_type]
    keys = [section.key for section in outline.sections]
    errors = []
    if len(keys) != len(set(keys)):
        errors.append("The outline contains duplicate section keys.")
    allowed = set(spec["required"] + spec["optional"])
    if any(key not in allowed for key in keys):
        errors.append("The outline contains a section that is not supported for this report type.")
    missing = set(spec["required"]) - set(keys)
    if missing:
        errors.append("The outline is missing required sections: " + ", ".join(sorted(missing)) + ".")
    if facts.limitations and "limitations" not in keys:
        errors.append("The outline must include recorded limitations and warnings.")
    known_refs = set(facts.evidence)
    if any(ref not in known_refs for section in outline.sections for ref in section.fact_refs):
        errors.append("The outline references facts that were not supplied by the analysis.")
    return ValidationResult(valid=not errors, errors=errors)


def validate_document(
    value: Any,
    facts_value: Any,
    outline_value: Any,
) -> ValidationResult:
    try:
        document = value if isinstance(value, ReportDocument) else ReportDocument.model_validate(value)
        facts = facts_value if isinstance(facts_value, ReportFacts) else ReportFacts.model_validate(facts_value)
        outline = outline_value if isinstance(outline_value, ReportOutline) else ReportOutline.model_validate(outline_value)
    except ValidationError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Report content is invalid: {error['loc'][-1]}" for error in exc.errors()[:10]],
        )

    errors: list[str] = []
    keys = [section.key for section in document.sections]
    outline_keys = [section.key for section in outline.sections]
    if len(keys) != len(set(keys)) or set(keys) != set(outline_keys):
        errors.append("Report sections must match the approved outline exactly, without duplicates.")

    allowed_refs = set(facts.evidence)
    body_parts = [document.title, document.subtitle or ""]
    for section in document.sections:
        body_parts.extend([section.title, *section.paragraphs, *section.bullets])
        if not section.evidence_refs:
            errors.append(f"Section '{section.key}' has no supporting fact references.")
        elif any(ref not in allowed_refs for ref in section.evidence_refs):
            errors.append(f"Section '{section.key}' references an unavailable fact.")
    body = "\n".join(body_parts)

    if re.search(r"\d", body):
        errors.append(
            "Authored prose contains numeric claims; Python-rendered fact tables must supply all numeric values."
        )
    if _ACTIVE_MARKUP.search(body):
        errors.append("Authored content contains unsafe markup or a scriptable URL.")

    normalized_body = re.sub(r"[^a-z0-9]+", "", body.lower())
    if not facts.preprocessing_operations:
        if "preprocess" in normalized_body or any(
            re.sub(r"[^a-z0-9]+", "", term) in normalized_body
            for terms in _PREPROCESSING_TERMS.values()
            for term in terms
        ):
            errors.append("The report claims preprocessing that is absent from recorded operation metadata.")
    else:
        operation_text = " ".join(facts.preprocessing_operations).lower()
        normalized_operations = re.sub(r"[^a-z0-9]+", "", operation_text)
        for concept, terms in _PREPROCESSING_TERMS.items():
            if any(re.sub(r"[^a-z0-9]+", "", term) in normalized_body for term in terms):
                aliases = (concept, *terms)
                if not any(re.sub(r"[^a-z0-9]+", "", alias) in normalized_operations for alias in aliases):
                    errors.append(f"The report claims unrecorded {concept}.")

    if not facts.explainability_available and re.search(
        r"\b(SHAP|permutation importance|feature importance|explainability)\b",
        body,
        re.IGNORECASE,
    ):
        errors.append("The report claims explainability results that were not recorded.")
    if not facts.optimization_performed and re.search(
        r"\b(tuned|tuning|optimized|optimisation|optimization|hyperparameter search)\b",
        body,
        re.IGNORECASE,
    ):
        errors.append("The report claims optimization that was not recorded.")

    candidates = {re.sub(r"[^a-z0-9]+", "", name.lower()) for name in (row.model for row in facts.cv_results)}
    for known_model in _KNOWN_MODEL_NAMES:
        if known_model in normalized_body and not any(
            known_model in candidate or candidate in known_model for candidate in candidates
        ):
            errors.append("The report names a model absent from the recorded comparison results.")
            break

    required = set(REPORT_SECTIONS[facts.report_type]["required"])
    if required - set(keys):
        errors.append("Report content is missing one or more required sections.")
    if facts.limitations and "limitations" not in keys:
        errors.append("Report content is missing recorded limitations and warnings.")
    if facts.dataset.sampling_applied and "analysis.limitations" not in allowed_refs:
        errors.append("Sample-based reports must include a supported sampling limitation.")
    return ValidationResult(valid=not errors, errors=list(dict.fromkeys(errors)))


def validate_report_content(value: Any, facts: Any, outline: Any) -> list[str]:
    """Compatibility helper returning only the deterministic validation errors."""
    return validate_document(value, facts, outline).errors
