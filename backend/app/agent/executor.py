"""
Executes tool calls from the agent by running the core pipeline steps.
All results are JSON-serialisable dicts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.core import eda as eda_mod
from app.core import preprocessing as prep_mod
from app.core import trainer as trainer_mod
from app.core.code_generator import generate_notebook
from app.core.data_loader import describe, load_from_path, load_from_url
from app.core.data_profiler import profile_dataset
from app.core.model_selector import (
    auto_detect_target,
    is_categorical_target_schema,
    validate_target_column,
)
from app.core.report_generator import generate_report as render_report
from app.db.crud import (
    create_artifact, create_operation, get_run, get_session, list_artifacts, list_operations,
    update_session, create_experiment, create_model_version, create_plan,
    get_latest_run_for_session, update_run, create_dataset_source,
)
from app.services.notification_service import send_completion_email

logger = logging.getLogger(__name__)

SESSIONS_DIR = settings.STORAGE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class ModelRegistryPersistenceError(RuntimeError):
    """Raised when a trained model cannot be registered atomically."""


def _require_training_run(db: DBSession, session_id: str, run_id: str | None):
    if not run_id:
        raise ValueError(
            "Training requires an explicit AnalysisRun context; no experiment "
            "was registered."
        )

    run = get_run(db, run_id)
    if run is None:
        raise ValueError(
            f"AnalysisRun '{run_id}' was not found; no experiment was registered."
        )
    if str(run.session_id) != str(session_id):
        raise ValueError(
            f"AnalysisRun '{run_id}' does not belong to session '{session_id}'; "
            "no experiment was registered."
        )
    return run


# ---------- path helpers ----------
def _session_dir(session_id: str) -> Path:
    path = SESSIONS_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(session_id: str) -> Path:
    return _session_dir(session_id) / "pipeline_state.json"


def _load_state(session_id: str) -> dict[str, Any]:
    path = _state_path(session_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(session_id: str, state: dict[str, Any]) -> None:
    _state_path(session_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_raw(db: DBSession, session_id: str) -> pd.DataFrame:
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")
    cache = _session_dir(session_id) / "raw.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = load_from_url(session.source_path) if session.source_type == "url" else load_from_path(session.source_path)
    df.to_pickle(cache)
    return df


# ---------- tool implementations ----------
def _tool_get_session_status(session_id: str, args: dict, db: DBSession) -> dict:
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")
    state = _load_state(session_id)
    artifacts = [{"type": artifact.artifact_type, "path": artifact.file_path} for artifact in list_artifacts(db, session_id)]
    session_files = [path.name for path in _session_dir(session_id).iterdir()]
    return {
        "session_id": session_id,
        "name": session.name,
        "status": session.status,
        "target_column": session.target_column,
        "task_type": session.task_type,
        "steps_run": {
            "eda": "eda_summary.json" in session_files,
            "preprocessing": "preprocessor.pkl" in session_files,
            "training": "training_results.json" in session_files,
            "report": any(item["type"] == "report" for item in artifacts),
        },
        "best_model": state.get("best_model"),
        "artifacts": artifacts,
    }


def _tool_inspect_dataset(session_id: str, args: dict, db: DBSession) -> dict:
    """Return structural view and data quality profile."""
    df = _load_raw(db, session_id)
    info = describe(df)
    session = get_session(db, session_id)
    target = session.target_column if session else None
    task_type = session.task_type if session else None
    profile = profile_dataset(df, target=target, task_type=task_type)
    
    profile_path = _session_dir(session_id) / "dataset_profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    result = {
        "shape": info["n_rows"],
        "columns": info["columns"],
        "dtypes": info["dtypes"],
        "missing_total": info["missing_total"],
        "duplicate_rows": info["duplicate_rows"],
        "quality_score": profile.quality_score,
        "warnings": profile.warnings[:20],
        "warning_count": len(profile.warnings),
        "profile_summary": {
            "numeric_columns": profile.numeric_count,
            "categorical_columns": profile.categorical_count,
            "boolean_columns": profile.boolean_count,
            "date_columns": profile.date_count,
            "missing_count": profile.missing_count,
            "missing_percentage": profile.missing_percentage,
            "duplicate_rows": profile.duplicate_row_count,
            "memory_usage_bytes": profile.memory_usage_bytes,
            "target": profile.target_info,
        },
        "columns_to_review": [
            {
                "name": column.name,
                "detected_type": column.detected_type,
                "missing_percentage": column.missing_percentage,
                "unique_count": column.unique_count,
                "cardinality_percentage": column.cardinality_percentage,
                "constant": column.is_constant,
                "near_constant": column.is_near_constant,
                "likely_identifier": column.likely_id,
                "high_cardinality": column.is_high_cardinality,
                "date_like": column.is_date_like,
                "issues": column.potential_issues,
            }
            for column in profile.columns if column.potential_issues
        ][:20],
        "target_suggestions": auto_detect_target(df),
    }

    # Handle explicit opt-in for sample rows (privacy)
    if args.get("include_sample_rows"):
        result["sample_rows"] = df.head(5).to_dict(orient="records")

    return result


def _tool_run_eda(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    session = get_session(db, session_id)
    target = args.get("target_column") or (session.target_column if session else None)
    task_type = session.task_type if session else None
    result = eda_mod.run_eda(df, target, task_type=task_type)
    session_dir = _session_dir(session_id)
    (session_dir / "eda_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (session_dir / "eda_figures.json").write_text(json.dumps(result["figures"], ensure_ascii=False), encoding="utf-8")
    update_session(db, session_id, status="eda_done")
    create_operation(db, session_id=session_id, op_type="eda", details={"target": target})
    return {
        "shape": result["summary"]["shape"],
        "readiness": result["summary"]["readiness"],
        "important_findings": result["summary"].get("important_findings", []),
        "target_profile": result["summary"].get("profile", {}).get("target_info"),
        "task_type": task_type,
        "n_figures": len(result["figures"]),
    }


def _tool_run_preprocessing(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    target = args.get("target_column")
    if not target:
        raise ValueError("target_column is required for preprocessing.")
    X, y, target_ops = prep_mod.prepare_xy_for_training(df, target)
    
    # Export rows with nonlearning cleanup only. Missing-value statistics are
    # fitted later inside each training/CV pipeline, never on this full dataset.
    cleaned_features = X.copy()
    for column in cleaned_features.columns:
        values = cleaned_features[column]
        if pd.api.types.is_numeric_dtype(values.dtype):
            cleaned_features[column] = values.replace([np.inf, -np.inf], np.nan)
    cleaned_data = cleaned_features.reset_index(drop=True)
    cleaned_data[target] = y.reset_index(drop=True).to_numpy()
    cleaned_path = _session_dir(session_id) / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    preprocessor, prep_ops = prep_mod.build_preprocessor(X)
    session_dir = _session_dir(session_id)
    y.to_frame(name="__target__").to_pickle(session_dir / "target.pkl")
    # Keep only the schema blueprint here. The model trainer creates fresh
    # copies and fits them inside each CV fold and the final training split.
    joblib.dump(preprocessor, session_dir / "preprocessor.pkl")

    state = _load_state(session_id)
    state["target_column"] = target
    # Task type is resolved once by the trainer using its training-only probe.
    state["task_type"] = None
    _save_state(session_id, state)

    all_ops = target_ops + prep_ops
    for operation in all_ops:
        create_operation(db, session_id=session_id, op_type=operation.get("type", "unknown"), details=operation)

    create_artifact(db, session_id=session_id, artifact_type="cleaned_data", file_path=str(cleaned_path), meta={"target_column": target, "imputed": False, "imputation_fit": "inside training folds and production training only"})
    update_session(db, session_id, target_column=target, task_type=None, status="preprocessed")
    return {
        "operations": all_ops,
        "task_type": None,
        "task_type_status": "pending_training_inference",
        "cleaned_data_path": str(cleaned_path),
    }


def _tool_train_models(
    session_id: str,
    args: dict,
    db: DBSession,
    run_id: str | None = None,
) -> dict:
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")
    run = _require_training_run(db, session_id, run_id)

    session_dir = _session_dir(session_id)
    df = _load_raw(db, session_id)
    target = session.target_column
    if not target:
        raise ValueError("Target column is not set in the session.")

    X, y, _ = prep_mod.prepare_xy_for_training(df, target)
    preprocessor, _ = prep_mod.build_preprocessor(X)
    state = _load_state(session_id)
    task_type = args.get("task_type", "auto") if isinstance(args, dict) else "auto"
    if task_type is None:
        task_type = "auto"
    target_is_categorical = is_categorical_target_schema(df[target])
    training = trainer_mod.train_and_evaluate(
        X,
        y,
        preprocessor,
        task_type,
        target_is_categorical=target_is_categorical,
        target_name=target,
    )
    task_type = training.get("task_type", task_type)
    target_encoder = training.get("target_encoder")
    preprocessing_operations = training.get("preprocessing_operations", [])

    state["preprocessing_operations"] = preprocessing_operations
    for operation in preprocessing_operations:
        create_operation(
            db,
            session_id=session_id,
            op_type=operation.get("type", "preprocessing_decision"),
            details=operation,
        )

    profile = profile_dataset(df, target=target, task_type=task_type)
    profile_path = session_dir / "dataset_profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    eda_summary_path = session_dir / "eda_summary.json"
    if eda_summary_path.exists():
        eda_summary = json.loads(eda_summary_path.read_text(encoding="utf-8"))
        eda_summary.update({
            "profile": profile.model_dump(mode="json"),
            "important_findings": profile.warnings[:12],
            "target": target,
            "task_type": task_type,
        })
        eda_summary_path.write_text(
            json.dumps(eda_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    state["task_type"] = task_type
    state["best_model"] = training["best_model_name"]
    state["best_score"] = training["best_score"]
    state["metric"] = training["metric"]
    state["metric_names"] = training["metric_names"]
    state["results"] = training["results"]
    state["cv_selection_score"] = training.get("cv_selection_score", training["best_score"])
    state["selection_metric"] = training.get("selection_metric", training["metric"])
    state["evaluation"] = training.get("evaluation")
    state["final_test_metrics"] = training.get("final_test_metrics")
    if training.get("tuning_params"):
        state["tuning_params"] = training["tuning_params"]
    _save_state(session_id, state)

    if training.get("comparison_chart"):
        (session_dir / "training_chart.json").write_text(json.dumps(training["comparison_chart"], ensure_ascii=False), encoding="utf-8")
    if training.get("importance_chart"):
        (session_dir / "importance_chart.json").write_text(json.dumps(training["importance_chart"], ensure_ascii=False), encoding="utf-8")
    if training.get("extra_charts"):
        (session_dir / "extra_charts.json").write_text(json.dumps(training["extra_charts"], ensure_ascii=False), encoding="utf-8")

    model_artifact_path = session_dir / "best_model.pkl"
    joblib.dump(
        {
            "artifact_format_version": 1,
            "pipeline": training["best_model"],
            "target_encoder": target_encoder,
            "task_type": task_type,
            "input_columns": list(X.columns),
        },
        model_artifact_path,
    )
    (session_dir / "training_results.json").write_text(
        json.dumps(
            {
                "results": training["results"],
                "best": training["best_model_name"],
                "score": training["best_score"],
                "cv_selection_score": training.get(
                    "cv_selection_score", training["best_score"]
                ),
                "selection_metric": training.get(
                    "selection_metric", training["metric"]
                ),
                "metric": training["metric"],
                "evaluation": training.get("evaluation"),
                "final_test_metrics": training.get("final_test_metrics"),
                "preprocessing_operations": preprocessing_operations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    update_session(db, session_id, task_type=task_type, status="trained")

    # Register in Model Registry as one atomic unit. The explicit run context
    # is validated above; never infer it from the session's latest run.
    try:
        artifact_uri = str(model_artifact_path)
        input_schema = {col: str(dtype) for col, dtype in X.dtypes.items()}
        metrics = {
            "best_score": training["best_score"],
            "metric": training["metric"],
            "metric_names": training.get("metric_names", []),
            "results": training.get("results", []),
            "cv_selection_score": training.get(
                "cv_selection_score", training["best_score"]
            ),
            "selection_metric": training.get(
                "selection_metric", training["metric"]
            ),
            "evaluation": training.get("evaluation"),
            "final_test_metrics": training.get("final_test_metrics"),
        }
        experiment = create_experiment(
            db,
            run_id=run.id,
            session_id=session.id,
            name=f"Experiment_{training['best_model_name']}",
            config_json={"target": target, "task_type": task_type},
            commit=False,
        )
        create_model_version(
            db,
            experiment_id=experiment.id,
            model_name=training["best_model_name"],
            artifact_uri=artifact_uri,
            schema_json=input_schema,
            metrics_json=metrics,
            parameters_json=training.get("tuning_params", {}),
            lifecycle_status="candidate",
            commit=False,
        )
        db.commit()
        logger.info("Registered model version in DB.")
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Failed to register model version for run %s and session %s",
            run.id,
            session.id,
        )
        raise ModelRegistryPersistenceError(
            f"Model Registry persistence failed for AnalysisRun '{run.id}' "
            f"and session '{session.id}'."
        ) from exc
        
    return {
        "best_model": training["best_model_name"],
        "task_type": task_type,
        "best_score": round(training["best_score"], 4),
        "metric": training["metric"],
        "results": training["results"],
        "cv_selection_score": training.get(
            "cv_selection_score", training["best_score"]
        ),
        "selection_metric": training.get(
            "selection_metric", training["metric"]
        ),
        "evaluation": training.get("evaluation"),
        "final_test_metrics": training.get("final_test_metrics"),
        "preprocessing_operations": preprocessing_operations,
    }


def _tool_generate_report(session_id: str, args: dict, db: DBSession) -> dict:
    session_dir = _session_dir(session_id)
    state = _load_state(session_id)
    if not state.get("results"):
        raise ValueError("Train models before generating the report.")

    figures = {}
    figures_path = session_dir / "eda_figures.json"
    if figures_path.exists():
        figures = json.loads(figures_path.read_text(encoding="utf-8"))
    chart_path = session_dir / "training_chart.json"
    if chart_path.exists():
        figures["model_comparison"] = json.loads(chart_path.read_text(encoding="utf-8"))
    importance_chart_path = session_dir / "importance_chart.json"
    if importance_chart_path.exists():
        figures["feature_importance"] = json.loads(importance_chart_path.read_text(encoding="utf-8"))
    extra_charts_path = session_dir / "extra_charts.json"
    if extra_charts_path.exists():
        figures.update(json.loads(extra_charts_path.read_text(encoding="utf-8")))

    df = _load_raw(db, session_id)
    data_desc = describe(df)
    eda_summary = {"summary": {"readiness": {}}, "figures": figures}
    summary_path = session_dir / "eda_summary.json"
    if summary_path.exists():
        eda_summary["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_path = session_dir / "dataset_profile.json"
    if profile_path.exists():
        data_desc["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))

    seen_ops = set()
    operations = []
    for operation in list_operations(db, session_id):
        operation_key = f"{operation.op_type}_{json.dumps(operation.details, sort_keys=True)}"
        if operation_key not in seen_ops:
            seen_ops.add(operation_key)
            operations.append({"type": operation.op_type, "details": operation.details})

    training_payload = {
        "results": state["results"], "best_model_name": state["best_model"],
        "best_score": state["best_score"], "metric": state["metric"], "metric_names": state["metric_names"],
        "cv_selection_score": state.get("cv_selection_score", state["best_score"]),
        "selection_metric": state.get("selection_metric", state["metric"]),
        "evaluation": state.get("evaluation"),
        "final_test_metrics": state.get("final_test_metrics"),
        "preprocessing_operations": state.get("preprocessing_operations", []),
    }

    pdf_path = render_report(
        task_id=session_id, data=data_desc, eda=eda_summary, operations=operations,
        training=training_payload, task_type=state.get("task_type", "classification"),
        target_column=state.get("target_column", ""),
    )
    create_artifact(db, session_id=session_id, artifact_type="report", file_path=str(pdf_path), meta={})

    session = get_session(db, session_id)
    notebook_path = None
    if session:
        metadata = session.dataset_meta if isinstance(session.dataset_meta, dict) else {}
        dataset_filename = metadata.get("original_filename", session.name)
        dataset_filename = Path(str(dataset_filename or "")).name
        if not dataset_filename or dataset_filename in {".", ".."}:
            dataset_filename = Path(str(session.source_path or "")).name

        notebook_path = generate_notebook(
            session_id=session_id, dataset_filename=dataset_filename,
            target_column=state.get("target_column", ""), task_type=state.get("task_type", "classification"),
            best_model_name=state.get("best_model", ""), tuning_params=state.get("tuning_params"),
        )
        create_artifact(db, session_id=session_id, artifact_type="notebook", file_path=str(notebook_path), meta={})

    update_session(db, session_id, status="reported")
    return {"report_path": str(pdf_path), "notebook_path": str(notebook_path) if notebook_path else None}


def _tool_get_operations_log(session_id: str, args: dict, db: DBSession) -> dict:
    operations = list_operations(db, session_id)
    return {"count": len(operations), "operations": [{"type": o.op_type, "details": o.details, "status": o.status, "at": o.created_at.isoformat()} for o in operations]}


def _tool_set_target_column(session_id: str, args: dict, db: DBSession) -> dict:
    column = args.get("column")
    if column is None:
        raise ValueError("column is required.")
    df = _load_raw(db, session_id)
    validate_target_column(df, column)
    update_session(db, session_id, target_column=column)
    state = _load_state(session_id)
    state["target_column"] = column
    _save_state(session_id, state)
    return {"target_column": column, "note": "rerun the pipeline to apply."}


def _tool_propose_plan(session_id: str, args: dict, db: DBSession) -> dict:
    """Save the plan and return a message asking for approval (Phase 4)."""
    run = get_latest_run_for_session(db, session_id)
    if not run:
        raise ValueError("No active run found for this session.")
    df = _load_raw(db, session_id)
    validate_target_column(df, args.get("target_column"))
    if args.get("task_type") not in {"classification", "regression"}:
        raise ValueError("A plan must specify classification or regression.")
        
    plan = create_plan(db, run_id=run.id, plan_json=args, approval_status="pending")
    update_run(db, run.id, status="awaiting_approval")
    return {
        "status": "awaiting_approval",
        "plan_id": plan.id,
        "message": f"I have proposed a plan to predict '{args['target_column']}' using {args['task_type']}. Please approve it to continue."
    }


def _tool_load_from_kaggle(session_id: str, args: dict, db: DBSession) -> dict:
    """Download a dataset from Kaggle into the current session (Phase 5)."""
    kaggle_ref = args.get("kaggle_ref")
    if not kaggle_ref:
        raise ValueError("kaggle_ref is required.")
    
    from app.core.connectors.kaggle import KaggleConnector
    
    session = get_session(db, session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found.")
    
    # Download into session folder
    dest_dir = _session_dir(session_id) / "kaggle_download"
    connector = KaggleConnector()
    if not connector.validate_reference(kaggle_ref):
        raise ValueError("Invalid Kaggle reference. Use 'owner/dataset-name'.")
    
    provenance = connector.download_to_staging(kaggle_ref, dest_dir)
    file_path = Path(provenance["file_path"])
    
    # Re-cache raw dataset for this session
    df = load_from_path(file_path)
    df.to_pickle(_session_dir(session_id) / "raw.pkl")
    
    # Update session to reflect new source
    update_session(
        db, session_id, 
        source_type="kaggle",
        source_path=str(file_path),
        name=f"{session.name} (Kaggle: {kaggle_ref})",
    )
    
    # Save provenance
    try:
        create_dataset_source(
            db,
            session_id=session_id,
            provider=provenance["provider"],
            source_id=provenance["source_id"],
            license=provenance.get("license"),
        )
    except Exception as exc:
        logger.warning(f"Failed to save dataset source: {exc}")
    
    return {
        "status": "success",
        "message": f"Dataset '{kaggle_ref}' downloaded and cached ({len(df)} rows).",
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


def _tool_rerun_pipeline(
    session_id: str,
    args: dict,
    db: DBSession,
    run_id: str | None = None,
) -> dict:
    requested_task_type = args.get("task_type")
    if requested_task_type is not None and requested_task_type not in {
        "classification", "regression", "auto"
    }:
        raise ValueError(
            f"Unsupported task type '{requested_task_type}'. Choose classification, regression, or auto."
        )

    approved_plan = None
    plan_run = get_run(db, run_id) if run_id else get_latest_run_for_session(db, session_id)
    if plan_run is not None:
        from app.db.crud import get_latest_plan_for_run

        candidate_plan = get_latest_plan_for_run(db, plan_run.id)
        if candidate_plan is not None and candidate_plan.approval_status == "approved":
            approved_plan = candidate_plan.plan_json

    if args.get("target_column") is not None:
        target_override = args["target_column"]
    else:
        target_override = (approved_plan or {}).get("target_column")
    if target_override is not None:
        _tool_set_target_column(session_id, {"column": target_override}, db)

    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")

    target = session.target_column
    if not target:
        df = _load_raw(db, session_id)
        target_info = auto_detect_target(df)
        target = target_info["best"]
        if not target:
            raise ValueError(target_info["message"])
        _tool_set_target_column(session_id, {"column": target}, db)

    eda_result = _tool_run_eda(session_id, {"target_column": target}, db)
    if "error" in eda_result: return {"step": "eda", "error": eda_result["error"]}
    prep_result = _tool_run_preprocessing(session_id, {"target_column": target}, db)
    if "error" in prep_result: return {"step": "preprocessing", "error": prep_result["error"]}
    train_result = _tool_train_models(
        session_id,
        {
            "task_type": (
                requested_task_type
                if requested_task_type in {"classification", "regression"}
                else (approved_plan or {}).get("task_type", "auto")
            )
        },
        db,
        run_id=run_id,
    )
    if "error" in train_result: return {"step": "training", "error": train_result["error"]}
    report_result = _tool_generate_report(session_id, {}, db)
    if "error" in report_result: return {"step": "report", "error": report_result["error"]}

    try:
        from app.db.models import User
        session = get_session(db, session_id)
        if session:
            user = db.get(User, session.user_id)
            if user and user.email:
                send_completion_email(email=user.email, session_id=session_id, best_model=str(train_result.get("best_model")))
    except Exception as exc:
        logger.error("Email notification failed: %s", exc)

    return {
        "status": "success",
        "target": target,
        "task_type": train_result.get("task_type"),
        "best_model": train_result.get("best_model"),
        "report_path": report_result.get("report_path"),
        "notebook_path": report_result.get("notebook_path"),
    }


# ---------- dispatcher ----------
_DISPATCH = {
    "get_session_status": _tool_get_session_status,
    "inspect_dataset": _tool_inspect_dataset,
    "run_eda": _tool_run_eda,
    "run_preprocessing": _tool_run_preprocessing,
    "train_models": _tool_train_models,
    "generate_report": _tool_generate_report,
    "get_operations_log": _tool_get_operations_log,
    "set_target_column": _tool_set_target_column,
    "rerun_pipeline": _tool_rerun_pipeline,
    "propose_plan": _tool_propose_plan,
    "load_from_kaggle": _tool_load_from_kaggle,
}


def execute_tool(
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    db: DBSession,
    run_id: str | None = None,
) -> dict[str, Any]:
    function = _DISPATCH.get(tool_name)
    if function is None:
        return {"ok": False, "tool": tool_name, "error": f"Unknown tool '{tool_name}'.", "error_type": "UnknownToolError"}

    try:
        if tool_name in {"train_models", "rerun_pipeline"}:
            result = function(session_id, arguments, db, run_id=run_id)
        else:
            result = function(session_id, arguments, db)
        if not isinstance(result, dict):
            raise TypeError(f"Tool returned {type(result).__name__}; expected a dictionary.")
        if result.get("error"):
            return {"ok": False, "tool": tool_name, "error": str(result["error"]), "error_type": result.get("error_type", "ToolError"), **{k: v for k, v in result.items() if k != "error"}}
        return {"ok": True, "tool": tool_name, **result}
    except Exception as exc:
        logger.exception("Tool '%s' failed", tool_name)
        return {"ok": False, "tool": tool_name, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}
