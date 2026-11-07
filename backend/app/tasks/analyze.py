"""
Celery task: full analysis pipeline.
"""
from __future__ import annotations

import logging

import joblib

from app.config import settings
from app.core.data_loader import describe, load_from_path, load_from_url
from app.core.eda import run_eda
from app.core.model_selector import auto_detect_target, detect_task_type
from app.core.preprocessing import build_preprocessor, prepare_xy
from app.core.report_generator import generate_report
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
        target = target_column or auto_detect_target(df)
        if target is None or target not in df.columns:
            raise ValueError("Target column not found and could not be auto-detected.")

        self.update_state(state="PROGRESS", meta={"step": "eda", "progress": 25})
        eda = run_eda(df, target)

        self.update_state(state="PROGRESS", meta={"step": "preprocessing", "progress": 40})
        X, y, target_ops = prepare_xy(df, target)
        preprocessor, prep_ops = build_preprocessor(X)
        all_ops = target_ops + prep_ops

        self.update_state(state="PROGRESS", meta={"step": "task_detection", "progress": 45})
        task_type = detect_task_type(y, df[target])

        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 55})
        training = train_and_evaluate(X, y, preprocessor, task_type)
        eda["figures"]["model_comparison"] = training["comparison_chart"]

        self.update_state(state="PROGRESS", meta={"step": "saving_model", "progress": 85})
        model_path = settings.MODEL_DIR / f"{self.request.id}.pkl"
        joblib.dump(training["best_model"], model_path)

        self.update_state(state="PROGRESS", meta={"step": "generating_report", "progress": 92})
        report_path = generate_report(
            task_id=self.request.id,
            data=describe(df),
            eda=eda,
            operations=all_ops,
            training=training,
            task_type=task_type,
            target_column=target,
        )

        return {
            "status": "success",
            "report_path": str(report_path),
            "model_path": str(model_path),
            "best_model": training["best_model_name"],
            "task_type": task_type,
            "target_column": target,
        }
    except Exception as exc:
        logger.exception("Analysis failed")
        return {"status": "failed", "error": str(exc)}