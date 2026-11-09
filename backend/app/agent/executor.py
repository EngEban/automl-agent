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
from app.core.model_selector import auto_detect_target, detect_task_type
from app.core.report_generator import generate_report as render_report
from app.db.crud import (
    create_artifact, create_operation, get_session, list_artifacts, list_operations,
    update_session, create_experiment, create_model_version, create_plan,
    get_latest_run_for_session, update_run, create_dataset_source,
)
from app.services.notification_service import send_completion_email

logger = logging.getLogger(__name__)

SESSIONS_DIR = settings.STORAGE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


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
    profile = profile_dataset(df)
    
    profile_path = _session_dir(session_id) / "dataset_profile.json"
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    result = {
        "shape": info["n_rows"],
        "columns": info["columns"],
        "dtypes": info["dtypes"],
        "missing_total": info["missing_total"],
        "duplicate_rows": info["duplicate_rows"],
        "quality_score": profile.quality_score,
        "warnings": profile.warnings,
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
    result = eda_mod.run_eda(df, target)
    session_dir = _session_dir(session_id)
    (session_dir / "eda_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    (session_dir / "eda_figures.json").write_text(json.dumps(result["figures"], ensure_ascii=False), encoding="utf-8")
    update_session(db, session_id, status="eda_done")
    create_operation(db, session_id=session_id, op_type="eda", details={"target": target})
    return {"shape": result["summary"]["shape"], "readiness": result["summary"]["readiness"], "n_figures": len(result["figures"])}


def _tool_run_preprocessing(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    target = args.get("target_column")
    if not target:
        raise ValueError("target_column is required for preprocessing.")
    X, y, target_ops = prep_mod.prepare_xy(df, target)
    
    # Export cleaned dataset
    cleaned_features = X.copy()
    for column in cleaned_features.columns:
        values = cleaned_features[column]
        if pd.api.types.is_numeric_dtype(values.dtype):
            fill_value = values.median()
            if pd.notna(fill_value):
                cleaned_features[column] = values.astype("float64").fillna(float(fill_value))
        else:
            modes = values.mode(dropna=True)
            if not modes.empty:
                cleaned_features[column] = values.fillna(modes.iloc[0])

    retained_rows = df.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    cleaned_data = cleaned_features.reset_index(drop=True)
    cleaned_data[target] = retained_rows[target].to_numpy()
    cleaned_path = _session_dir(session_id) / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    preprocessor, prep_ops = prep_mod.build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    Xdf = pd.DataFrame(X_processed)
    session_dir = _session_dir(session_id)
    Xdf.to_pickle(session_dir / "processed.pkl")
    y.to_frame(name="__target__").to_pickle(session_dir / "target.pkl")
    joblib.dump(preprocessor, session_dir / "preprocessor.pkl")

    state = _load_state(session_id)
    state["target_column"] = target
    state["task_type"] = detect_task_type(y, df[target])
    _save_state(session_id, state)

    all_ops = target_ops + prep_ops
    for operation in all_ops:
        create_operation(db, session_id=session_id, op_type=operation.get("type", "unknown"), details=operation)

    create_artifact(db, session_id=session_id, artifact_type="cleaned_data", file_path=str(cleaned_path), meta={"target_column": target, "imputed": True})
    update_session(db, session_id, target_column=target, task_type=state["task_type"], status="preprocessed")
    return {"operations": all_ops, "task_type": state["task_type"], "cleaned_data_path": str(cleaned_path)}


def _tool_train_models(session_id: str, args: dict, db: DBSession) -> dict:
    session_dir = _session_dir(session_id)
    df = _load_raw(db, session_id)
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")
    target = session.target_column
    if not target:
        raise ValueError("Target column is not set in the session.")

    X, y, _ = prep_mod.prepare_xy(df, target)
    preprocessor, _ = prep_mod.build_preprocessor(X)
    state = _load_state(session_id)
    task_type = args.get("task_type", "auto") if isinstance(args, dict) else "auto"
    if task_type == "auto":
        task_type = state.get("task_type") or detect_task_type(y, df[target])

    training = trainer_mod.train_and_evaluate(X, y, preprocessor, task_type)

    state["best_model"] = training["best_model_name"]
    state["best_score"] = training["best_score"]
    state["metric"] = training["metric"]
    state["metric_names"] = training["metric_names"]
    state["results"] = training["results"]
    if training.get("tuning_params"):
        state["tuning_params"] = training["tuning_params"]
    _save_state(session_id, state)

    if training.get("comparison_chart"):
        (session_dir / "training_chart.json").write_text(json.dumps(training["comparison_chart"], ensure_ascii=False), encoding="utf-8")
    if training.get("importance_chart"):
        (session_dir / "importance_chart.json").write_text(json.dumps(training["importance_chart"], ensure_ascii=False), encoding="utf-8")
    if training.get("extra_charts"):
        (session_dir / "extra_charts.json").write_text(json.dumps(training["extra_charts"], ensure_ascii=False), encoding="utf-8")

    joblib.dump(training["best_model"], session_dir / "best_model.pkl")
    (session_dir / "training_results.json").write_text(
        json.dumps({"results": training["results"], "best": training["best_model_name"], "score": training["best_score"], "metric": training["metric"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_session(db, session_id, status="trained")

    # Register in Model Registry
    try:
        experiment = create_experiment(db, run_id=session_id, session_id=session_id, name=f"Experiment_{training['best_model_name']}", config_json={"target": target, "task_type": task_type})
        artifact_uri = str(session_dir / "best_model.pkl")
        input_schema = {col: str(dtype) for col, dtype in X.dtypes.items()}
        create_model_version(
            db, experiment_id=experiment.id, model_name=training["best_model_name"], artifact_uri=artifact_uri,
            schema_json=input_schema, metrics_json={k: v for k, v in training.items() if k in ["best_score", "metric"]},
            parameters_json=training.get("tuning_params", {}), lifecycle_status="candidate",
        )
        logger.info("Registered model version in DB.")
    except Exception as exc:
        logger.error("Failed to register model version: %s", exc)
        
    return {"best_model": training["best_model_name"], "best_score": round(training["best_score"], 4), "metric": training["metric"], "results": training["results"]}


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
    if not column:
        raise ValueError("column is required.")
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


def _tool_rerun_pipeline(session_id: str, args: dict, db: DBSession) -> dict:
    if args.get("target_column"):
        _tool_set_target_column(session_id, {"column": args["target_column"]}, db)

    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")

    target = session.target_column
    if not target:
        df = _load_raw(db, session_id)
        target_info = auto_detect_target(df)
        target = target_info["best"]
        if not target:
            raise ValueError("A target column is required to run the pipeline.")
        _tool_set_target_column(session_id, {"column": target}, db)

    eda_result = _tool_run_eda(session_id, {"target_column": target}, db)
    if "error" in eda_result: return {"step": "eda", "error": eda_result["error"]}
    prep_result = _tool_run_preprocessing(session_id, {"target_column": target}, db)
    if "error" in prep_result: return {"step": "preprocessing", "error": prep_result["error"]}
    train_result = _tool_train_models(session_id, {"task_type": "auto"}, db)
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

    return {"status": "success", "target": target, "best_model": train_result.get("best_model"), "report_path": report_result.get("report_path"), "notebook_path": report_result.get("notebook_path")}


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


def execute_tool(session_id: str, tool_name: str, arguments: dict[str, Any], db: DBSession) -> dict[str, Any]:
    function = _DISPATCH.get(tool_name)
    if function is None:
        return {"ok": False, "tool": tool_name, "error": f"Unknown tool '{tool_name}'.", "error_type": "UnknownToolError"}

    try:
        result = function(session_id, arguments, db)
        if not isinstance(result, dict):
            raise TypeError(f"Tool returned {type(result).__name__}; expected a dictionary.")
        if result.get("error"):
            return {"ok": False, "tool": tool_name, "error": str(result["error"]), "error_type": result.get("error_type", "ToolError"), **{k: v for k, v in result.items() if k != "error"}}
        return {"ok": True, "tool": tool_name, **result}
    except Exception as exc:
        logger.exception("Tool '%s' failed", tool_name)
        return {"ok": False, "tool": tool_name, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}