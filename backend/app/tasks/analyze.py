"""
Celery task: full analysis pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from app.config import settings
from app.agents.report_agent.service import generate_report_safely
from app.core.data_loader import describe, load_from_path, load_from_url
from app.core.eda import run_eda
from app.core.model_selector import detect_target_column, is_categorical_target_schema
from app.core.memory import estimate_dataframe_memory
from app.core.preprocessing import build_preprocessor, prepare_xy_for_training
from app.core.trainer import train_and_evaluate
from app.worker import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="analyze_dataset")
def analyze_dataset(
    self,
    source_type: str,
    source: str,
    target_column: str | None = None,
) -> dict:
    """
    source_type: "upload" (local file path) or "url"
    """
    try:
        self.update_state(state="PROGRESS", meta={"step": "loading", "progress": 5})
        df = load_from_url(source) if source_type == "url" else load_from_path(source)

        self.update_state(state="PROGRESS", meta={"step": "detecting_target", "progress": 12})
        target_result = detect_target_column(df, explicit_target=target_column)
        if target_result["status"] != "selected":
            raise ValueError(target_result["message"])
        target = target_result["selected"]

        self.update_state(state="PROGRESS", meta={"step": "preprocessing", "progress": 40})
        X, y, target_ops = prepare_xy_for_training(df, target)
        preprocessor, prep_ops = build_preprocessor(X)

        self.update_state(state="PROGRESS", meta={"step": "task_detection", "progress": 45})
        target_is_categorical = is_categorical_target_schema(df[target])

        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 55})
        training = train_and_evaluate(
            X,
            y,
            preprocessor,
            "auto",
            target_is_categorical=target_is_categorical,
            target_name=target,
        )
        task_type = training["task_type"]
        source_memory = estimate_dataframe_memory(df)
        memory_summary = dict(training.get("memory_summary") or {})
        memory_summary.update({
            "source_dataset": source_memory,
            "dataset_warning_threshold_mb": int(settings.DATASET_MEMORY_WARNING_MB),
            "dataset_warning": (
                source_memory["bytes"] > settings.DATASET_MEMORY_WARNING_MB * 1024 * 1024
            ),
            "full_memory_loading": True,
        })
        training["memory_summary"] = memory_summary
        target_encoder = training.get("target_encoder")
        all_ops = (
            target_ops + prep_ops + training.get("preprocessing_operations", [])
        )

        # Full-dataset EDA is descriptive only and runs after evaluation, once
        # Task 07 has resolved the task type from the training partition.
        self.update_state(state="PROGRESS", meta={"step": "eda", "progress": 75})
        eda = run_eda(df, target, task_type=task_type)
        eda["figures"]["model_comparison"] = training["comparison_chart"]

        self.update_state(state="PROGRESS", meta={"step": "saving_model", "progress": 85})
        model_path = settings.MODEL_DIR / f"{self.request.id}.pkl"
        joblib.dump(
            {
                "artifact_format_version": 1,
                "pipeline": training["best_model"],
                "target_encoder": target_encoder,
                "task_type": task_type,
                "input_columns": list(X.columns),
            },
            model_path,
        )

        self.update_state(state="PROGRESS", meta={"step": "generating_report", "progress": 92})
        report_result = generate_report_safely(
            report_id=str(self.request.id),
            analysis_id=str(self.request.id),
            data=describe(df),
            eda=eda,
            operations=all_ops,
            training=training,
            task_type=task_type,
            target_column=target,
        )

        report_artifact_exists = False
        for artifact in (report_result.get("pdf_path"), report_result.get("html_path")):
            if not artifact:
                continue
            try:
                resolved = Path(str(artifact)).resolve()
                resolved.relative_to(settings.REPORT_DIR.resolve())
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                report_artifact_exists = True
                break
        report_status = (
            "success"
            if report_result.get("status") == "success" and report_artifact_exists
            else "failed"
        )
        if report_result.get("status") == "success" and not report_artifact_exists:
            report_result = {
                **report_result,
                "status": "failed",
                "error_code": "report_artifact_missing",
                "message": "The analysis completed, but no report artifact was found.",
            }

        return {
            "status": "success",
            "analysis_status": "success",
            "report_status": report_status,
            "report": report_result,
            "report_path": report_result.get("pdf_path"),
            "model_path": str(model_path),
            "best_model": training["best_model_name"],
            "task_type": task_type,
            "target_column": target,
            "memory_summary": memory_summary,
            "loading_metadata": dict(df.attrs.get("loading_metadata", {})),
        }
    except Exception as exc:
        logger.exception("Analysis failed")
        raise
