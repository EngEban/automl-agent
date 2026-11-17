"""Strict data contracts used between the report graph stages."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

ReportType = Literal[
    "machine_learning_analysis",
    "data_analysis",
    "model_comparison",
    "technical_report",
    "executive_summary",
]

REPORT_SECTIONS: dict[str, dict[str, list[str]]] = {
    "machine_learning_analysis": {
        "required": ["executive_summary", "dataset", "methodology", "model_comparison", "evaluation", "limitations", "conclusion"],
        "optional": ["data_quality", "preprocessing", "explainability", "recommendations"],
    },
    "data_analysis": {
        "required": ["executive_summary", "dataset", "data_quality", "conclusion"],
        "optional": ["methodology", "preprocessing", "explainability", "limitations", "recommendations"],
    },
    "model_comparison": {
        "required": ["executive_summary", "model_comparison", "evaluation", "limitations", "conclusion"],
        "optional": ["dataset", "methodology", "preprocessing", "explainability", "recommendations"],
    },
    "technical_report": {
        "required": ["executive_summary", "dataset", "methodology", "preprocessing", "model_comparison", "evaluation", "limitations", "conclusion"],
        "optional": ["data_quality", "explainability", "recommendations"],
    },
    "executive_summary": {
        "required": ["executive_summary", "outcomes", "limitations", "recommendations", "conclusion"],
        "optional": ["dataset", "evaluation"],
    },
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DatasetFacts(StrictModel):
    loaded_rows: StrictInt
    source_rows: StrictInt | None = None
    sampling_applied: StrictBool | None = None
    sampling_method: StrictStr | None = None
    sampling_seed: StrictInt | None = None
    chunk_size: StrictInt | None = None
    column_count: StrictInt | None = None
    missing_values: StrictInt | None = None
    duplicate_rows: StrictInt | None = None
    profile_findings: list[StrictStr] = Field(default_factory=list, max_length=20)


class ModelMetrics(StrictModel):
    model: StrictStr = Field(min_length=1, max_length=160)
    metrics: dict[StrictStr, float | int | None]


class EvaluationFacts(StrictModel):
    available: StrictBool
    method: StrictStr | None = None
    test_rows: StrictInt | None = None
    test_size: float | None = None
    random_state: StrictInt | None = None
    metrics: dict[StrictStr, float | int | None] = Field(default_factory=dict)


class SelectedFeatureFact(StrictModel):
    name: StrictStr
    source_feature: StrictStr | None = None
    score: float | None = None
    rank: StrictInt | None = None


class FeatureSelectionFacts(StrictModel):
    available: StrictBool = False
    enabled: StrictBool = False
    method: StrictStr | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    fit_rows: StrictInt | None = None
    original_source_features: StrictInt | None = None
    input_features: StrictInt | None = None
    retained_features: StrictInt | None = None
    selected_feature_names_truncated: StrictInt = 0
    cross_validation_fold_count: StrictInt | None = None
    folds_selected_different_features: StrictBool | None = None
    selected_features: list[SelectedFeatureFact] = Field(default_factory=list, max_length=300)


class ReportFacts(StrictModel):
    report_id: StrictStr = Field(min_length=1, max_length=128)
    analysis_id: StrictStr = Field(min_length=1, max_length=128)
    report_type: ReportType
    language: StrictStr = Field(min_length=1, max_length=48)
    task_type: Literal["classification", "regression"]
    target_column: StrictStr = Field(min_length=1, max_length=256)
    dataset: DatasetFacts
    best_model: StrictStr = Field(min_length=1, max_length=160)
    selection_metric: StrictStr = Field(min_length=1, max_length=100)
    cv_selection_score: float | None
    cv_results: list[ModelMetrics] = Field(min_length=1, max_length=50)
    holdout: EvaluationFacts
    feature_selection: FeatureSelectionFacts = Field(default_factory=FeatureSelectionFacts)
    preprocessing_operations: list[StrictStr] = Field(default_factory=list, max_length=50)
    explainability_available: StrictBool
    available_figures: list[StrictStr] = Field(default_factory=list, max_length=20)
    optimization_performed: StrictBool
    parameters: dict[str, Any] = Field(default_factory=dict)
    experiment: dict[str, str | None] = Field(default_factory=dict)
    model_registry: dict[str, str | None] = Field(default_factory=dict)
    limitations: list[StrictStr] = Field(default_factory=list, max_length=30)
    evidence: dict[StrictStr, StrictStr]


class OutlineSection(StrictModel):
    key: StrictStr = Field(min_length=1, max_length=48)
    title: StrictStr = Field(min_length=1, max_length=120)
    purpose: StrictStr = Field(min_length=1, max_length=500)
    fact_refs: list[StrictStr] = Field(min_length=1, max_length=12)


class ReportOutline(StrictModel):
    title: StrictStr = Field(min_length=1, max_length=160)
    sections: list[OutlineSection] = Field(min_length=3, max_length=12)


class ReportSection(StrictModel):
    key: StrictStr = Field(min_length=1, max_length=48)
    title: StrictStr = Field(min_length=1, max_length=120)
    paragraphs: list[StrictStr] = Field(default_factory=list, max_length=12)
    bullets: list[StrictStr] = Field(default_factory=list, max_length=16)
    evidence_refs: list[StrictStr] = Field(min_length=1, max_length=12)


class ReportDocument(StrictModel):
    title: StrictStr = Field(min_length=1, max_length=160)
    subtitle: StrictStr | None = Field(default=None, max_length=240)
    sections: list[ReportSection] = Field(min_length=3, max_length=12)


class ValidationResult(StrictModel):
    valid: StrictBool
    errors: list[StrictStr] = Field(default_factory=list, max_length=50)
