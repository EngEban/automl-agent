"""
Celery task: full analysis pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from app.config.settings import settings
from app.agents.report_agent.service import generate_report_safely
from app.core.data_loader import describe, load_from_path, load_from_url
from app.core.eda import run_eda
from app.core.model_selector import detect_target_column, is_categorical_target_schema
from app.core.memory import estimate_dataframe_memory
from app.core.preprocessing import build_preprocessor, prepare_xy_for_training
from app.core.trainer import train_and_evaluate
from app.db import crud
from app.db.models import Artifact
from app.db.session import SessionLocal
from app.services.session_service import persist_legacy_analysis_dataset
from app.worker.worker import celery_app

logger = logging.getLogger(__name__)


def _claim_legacy_analysis(task_id: str) -> tuple[str, str, str]:
    """Require a persisted owner/session record before doing legacy work."""
    if not task_id:
        raise RuntimeError("The analysis task has no ownership record.")
    db = SessionLocal()
    try:
        run = crud.get_run_by_task_id(db, task_id)
        if run is None:
            raise RuntimeError("The analysis task has no ownership record.")
        session = crud.get_session(db, run.session_id)
        if session is None or str(session.user_id) != str(run.requested_by):
            raise RuntimeError("The analysis task ownership record is invalid.")
        if run.status == "queued":
            crud.update_run(
                db,
                run.id,
                status="running",
                current_step="loading",
                progress_percent=0,
                message="Analysis started.",
            )
        elif run.status != "running":
            raise RuntimeError("The analysis task is no longer active.")
        session.status = "running"
        db.commit()
        return session.id, str(session.user_id), str(run.id)
    finally:
        db.close()


def _update_legacy_progress(task_id: str, step: str, progress: int) -> None:
    db = SessionLocal()
    try:
        run = crud.get_run_by_task_id(db, task_id)
        if run is None or run.status != "running":
            return
        run.current_step = step
        run.progress_percent = max(int(run.progress_percent or 0), min(100, progress))
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Could not persist legacy analysis progress")
    finally:
        db.close()


def _finish_legacy_analysis(
    task_id: str,
    *,
    status: str,
    session_status: str,
    target_column: str | None = None,
    task_type: str | None = None,
) -> None:
    db = SessionLocal()
    try:
        run = crud.get_run_by_task_id(db, task_id)
        if run is None:
            return
        session = crud.get_session(db, run.session_id)
        if run.status in {"queued", "running"}:
            crud.update_run(
                db,
                run.id,
                status=status,
                current_step="finished" if status == "succeeded" else "failed",
                progress_percent=100 if status == "succeeded" else run.progress_percent,
                message=(
                    "Analysis completed."
                    if status == "succeeded"
                    else "Analysis failed. Check server logs for details."
                ),
                error_code=None if status == "succeeded" else "analysis_failed",
                error_message=None if status == "succeeded" else "Analysis task failed.",
            )
        if session is not None:
            session.status = session_status
            if target_column is not None:
                session.target_column = target_column
            if task_type is not None:
                session.task_type = task_type
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("Could not persist legacy analysis completion state")
    finally:
        db.close()


def _record_legacy_artifact(task_id: str, artifact_type: str, file_path: Path) -> None:
    """Persist a generated artifact against the run's owned session."""
    allowed_root = settings.MODEL_DIR if artifact_type == "model" else settings.REPORT_DIR
    try:
        root = allowed_root.resolve()
        resolved = file_path.resolve()
        resolved.relative_to(root)
        if not resolved.is_file():
            return
        db = SessionLocal()
        try:
            run = crud.get_run_by_task_id(db, task_id)
            if run is None:
                return
            session = crud.get_session(db, run.session_id)
            if session is None or str(session.user_id) != str(run.requested_by):
                return
            existing = [
                artifact
                for artifact in crud.list_artifacts(db, session.id)
                if artifact.artifact_type == artifact_type
                and Path(artifact.file_path).resolve() == resolved
            ]
            if not existing:
                db.add(Artifact(
                    session_id=session.id,
                    artifact_type=artifact_type,
                    file_path=str(resolved),
                    meta={"legacy_analysis": True},
                ))
                db.commit()
        finally:
            db.close()
    except (OSError, RuntimeError, ValueError):
        logger.warning("Legacy %s artifact was outside its storage directory", artifact_type)
    except Exception as exc:
        logger.warning(
            "Could not register legacy %s artifact (error_type=%s)",
            artifact_type,
            type(exc).__name__,
        )


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
    task_id = str(self.request.id or "")
    session_id: str | None = None
    try:
        session_id, owner_id, run_id = _claim_legacy_analysis(task_id)
        self.update_state(state="PROGRESS", meta={"step": "loading", "progress": 5})
        _update_legacy_progress(task_id, "loading", 5)
        df = load_from_url(source) if source_type == "url" else load_from_path(source)
        with SessionLocal() as db:
            persist_legacy_analysis_dataset(db, session_id=session_id, df=df)

        self.update_state(state="PROGRESS", meta={"step": "detecting_target", "progress": 12})
        _update_legacy_progress(task_id, "detecting_target", 12)
        target_result = detect_target_column(df, explicit_target=target_column)
        if target_result["status"] != "selected":
            raise ValueError(target_result["message"])
        target = target_result["selected"]
        with SessionLocal() as db:
            session = crud.get_session(db, session_id)
            if session is not None:
                session.target_column = target
                db.commit()

        self.update_state(state="PROGRESS", meta={"step": "preprocessing", "progress": 40})
        _update_legacy_progress(task_id, "preprocessing", 40)
        X, y, target_ops = prepare_xy_for_training(df, target)
        preprocessor, prep_ops = build_preprocessor(X)

        self.update_state(state="PROGRESS", meta={"step": "task_detection", "progress": 45})
        _update_legacy_progress(task_id, "task_detection", 45)
        target_is_categorical = is_categorical_target_schema(df[target])

        self.update_state(state="PROGRESS", meta={"step": "training", "progress": 55})
        _update_legacy_progress(task_id, "training", 55)
        training = train_and_evaluate(
            X,
            y,
            preprocessor,
            "auto",
            target_is_categorical=target_is_categorical,
            target_name=target,
        )
        task_type = training["task_type"]
        with SessionLocal() as db:
            session = crud.get_session(db, session_id)
            if session is not None:
                session.task_type = task_type
                db.commit()
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
        _update_legacy_progress(task_id, "eda", 75)
        eda = run_eda(df, target, task_type=task_type)
        eda["figures"]["model_comparison"] = training["comparison_chart"]

        self.update_state(state="PROGRESS", meta={"step": "saving_model", "progress": 85})
        _update_legacy_progress(task_id, "saving_model", 85)
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
        _record_legacy_artifact(task_id, "model", model_path)

        self.update_state(state="PROGRESS", meta={"step": "generating_report", "progress": 92})
        _update_legacy_progress(task_id, "generating_report", 92)
        report_result = generate_report_safely(
            report_id=str(self.request.id),
            analysis_id=str(self.request.id),
            session_id=session_id,
            expected_user_id=owner_id,
            run_id=run_id,
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
        report_path = report_result.get("pdf_path")
        if report_path and report_artifact_exists:
            _record_legacy_artifact(task_id, "report", Path(str(report_path)))

        _finish_legacy_analysis(
            task_id,
            status="succeeded",
            session_status="created",
            target_column=target,
            task_type=task_type,
        )
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
        if session_id is not None:
            _finish_legacy_analysis(
                task_id,
                status="failed",
                session_status="failed",
            )
        logger.exception("Analysis failed")
        raise
