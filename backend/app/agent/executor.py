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
from app.core.model_selector import auto_detect_target, detect_task_type
from app.core.report_generator import generate_report as render_report
from app.db.crud import (
    create_artifact,
    create_operation,
    get_session,
    list_artifacts,
    list_operations,
    update_session,
)
# استيراد دالة إرسال الإيميل الجديدة
from app.services.notification_service import send_completion_email

logger = logging.getLogger(__name__)

SESSIONS_DIR = settings.STORAGE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- path helpers ----------
def _session_dir(session_id: str) -> Path:
    p = SESSIONS_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path(session_id: str) -> Path:
    return _session_dir(session_id) / "pipeline_state.json"


def _load_state(session_id: str) -> dict[str, Any]:
    p = _state_path(session_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(session_id: str, state: dict[str, Any]) -> None:
    _state_path(session_id).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_raw(db: DBSession, session_id: str) -> pd.DataFrame:
    sess = get_session(db, session_id)
    if sess is None:
        raise ValueError(f"Session '{session_id}' not found.")
    cache = _session_dir(session_id) / "raw.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = load_from_url(sess.source_path) if sess.source_type == "url" \
        else load_from_path(sess.source_path)
    df.to_pickle(cache)
    return df


def _usable_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop features that become entirely empty after target-row filtering."""
    X = X.replace([np.inf, -np.inf], np.nan)
    all_null_columns = [column for column in X.columns if X[column].isna().all()]
    if all_null_columns:
        X = X.drop(columns=all_null_columns)
    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing all-null columns."
        )

    # The pipeline builder treats booleans as categorical features.
    boolean_columns = X.select_dtypes(include="bool").columns
    if len(boolean_columns):
        X = X.copy()
        for column in boolean_columns:
            X[column] = X[column].astype("object")
    return X, [str(column) for column in all_null_columns]


# ---------- tool implementations ----------
def _tool_get_session_status(session_id: str, args: dict, db: DBSession) -> dict:
    sess = get_session(db, session_id)
    if sess is None:
        raise ValueError(f"Session '{session_id}' not found.")
    state = _load_state(session_id)
    artifacts = [{"type": a.artifact_type, "path": a.file_path}
                 for a in list_artifacts(db, session_id)]
    return {
        "session_id": session_id,
        "name": sess.name,
        "status": sess.status,
        "target_column": sess.target_column,
        "task_type": sess.task_type,
        "steps_run": {
            "eda": "eda_summary.json" in [p.name for p in _session_dir(session_id).iterdir()],
            "preprocessing": "preprocessor.pkl" in [p.name for p in _session_dir(session_id).iterdir()],
            "training": "training_results.json" in [p.name for p in _session_dir(session_id).iterdir()],
            "report": any(a["type"] == "report" for a in artifacts),
        },
        "best_model": state.get("best_model"),
        "artifacts": artifacts,
    }


def _tool_inspect_dataset(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    info = describe(df)
    info["sample_rows"] = df.head(5).to_dict(orient="records")
    return info


def _tool_run_eda(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    target = args.get("target_column") or get_session(db, session_id).target_column
    result = eda_mod.run_eda(df, target)
    summary_path = _session_dir(session_id) / "eda_summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Save figures (base64 pngs) for the report generator
    (_session_dir(session_id) / "eda_figures.json").write_text(
        json.dumps(result["figures"], ensure_ascii=False),
        encoding="utf-8",
    )
    update_session(db, session_id, status="eda_done")
    create_operation(db, session_id=session_id, op_type="eda",
                     details={"target": target})
    return {
        "shape": result["summary"]["shape"],
        "readiness": result["summary"]["readiness"],
        "n_figures": len(result["figures"]),
    }


def _tool_run_preprocessing(session_id: str, args: dict, db: DBSession) -> dict:
    df = _load_raw(db, session_id)
    target = args.get("target_column")
    if not target:
        raise ValueError("target_column is required for preprocessing.")

    X, y, target_ops = prep_mod.prepare_xy(df, target)

    # Remove unusable all-null features before fitting the preprocessing pipeline.
    X, all_null_columns = _usable_features(X)
    if all_null_columns:
        target_ops.append({
            "type": "drop_all_missing_features",
            "columns": all_null_columns,
        })

    # Export a human-readable copy: impute features, but do not scale or encode.
    cleaned_features = X.copy()
    for column in cleaned_features.columns:
        values = cleaned_features[column]
        if pd.api.types.is_numeric_dtype(values.dtype):
            fill_value = values.median()
            if pd.notna(fill_value):
                cleaned_features[column] = (
                    values.astype("float64").fillna(float(fill_value))
                )
        else:
            modes = values.mode(dropna=True)
            if not modes.empty:
                cleaned_features[column] = values.fillna(modes.iloc[0])

    # prepare_xy label-encodes categorical targets for training. Preserve the
    # original target values in the downloadable, human-readable CSV.
    retained_rows = (
        df.dropna(subset=[target])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cleaned_data = cleaned_features.reset_index(drop=True)
    cleaned_data[target] = retained_rows[target].to_numpy()
    cleaned_columns = [
        column for column in retained_rows.columns if column in cleaned_data.columns
    ]
    cleaned_data = cleaned_data.loc[:, cleaned_columns]

    cleaned_path = _session_dir(session_id) / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    preprocessor, prep_ops = prep_mod.build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    Xdf = pd.DataFrame(X_processed)
    
    Xdf.to_pickle(_session_dir(session_id) / "processed.pkl")
    y.to_frame(name="__target__").to_pickle(
        _session_dir(session_id) / "target.pkl"
    )
    joblib.dump(preprocessor, _session_dir(session_id) / "preprocessor.pkl")

    state = _load_state(session_id)
    state["target_column"] = target
    state["task_type"] = detect_task_type(y, df[target])
    _save_state(session_id, state)

    all_ops = target_ops + prep_ops
    for op in all_ops:
        create_operation(db, session_id=session_id,
                         op_type=op.get("type", "unknown"), details=op)

    create_artifact(
        db,
        session_id=session_id,
        artifact_type="cleaned_data",
        file_path=str(cleaned_path),
        meta={"target_column": target, "imputed": True},
    )

    update_session(db, session_id, target_column=target,
                   task_type=state["task_type"], status="preprocessed")
    return {
        "operations": all_ops,
        "task_type": state["task_type"],
        "cleaned_data_path": str(cleaned_path),
    }


def _tool_train_models(session_id: str, args: dict, db: DBSession) -> dict:
    """Train on raw features; the trainer owns fitting the preprocessor."""
    sdir = _session_dir(session_id)

    df = _load_raw(db, session_id)
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found.")
    target = session.target_column
    if not target:
        raise ValueError("Target column is not set in the session.")

    X, y, _ = prep_mod.prepare_xy(df, target)
    X, _ = _usable_features(X)
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
    _save_state(session_id, state)

    if training.get("comparison_chart"):
        (sdir / "training_chart.json").write_text(
            json.dumps(training["comparison_chart"], ensure_ascii=False),
            encoding="utf-8",
        )

    joblib.dump(training["best_model"], sdir / "best_model.pkl")
    (sdir / "training_results.json").write_text(
        json.dumps(
            {"results": training["results"],
             "best": training["best_model_name"],
             "score": training["best_score"],
             "metric": training["metric"]},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_session(db, session_id, status="trained")
    return {
        "best_model": training["best_model_name"],
        "best_score": round(training["best_score"], 4),
        "metric": training["metric"],
        "results": training["results"],
    }

def _tool_generate_report(session_id: str, args: dict, db: DBSession) -> dict:
    sdir = _session_dir(session_id)
    state = _load_state(session_id)
    if not state.get("results"):
        raise ValueError("Train models before generating the report.")

    # Load EDA figures (if EDA ran)
    figures = {}
    if (sdir / "eda_figures.json").exists():
        figures = json.loads((sdir / "eda_figures.json").read_text(encoding="utf-8"))
    if (sdir / "training_chart.json").exists():
        figures["model_comparison"] = json.loads(
            (sdir / "training_chart.json").read_text(encoding="utf-8")
        )

    df = _load_raw(db, session_id)
    data_desc = describe(df)
    eda_summary = {"summary": {"readiness": {}}, "figures": figures}
    if (sdir / "eda_summary.json").exists():
        eda_summary["summary"] = json.loads(
            (sdir / "eda_summary.json").read_text(encoding="utf-8")
        )

    # Deduplicate operations for the report
    seen_ops = set()
    ops = []
    for o in list_operations(db, session_id):
        op_str = f"{o.op_type}_{json.dumps(o.details, sort_keys=True)}"
        if op_str not in seen_ops:
            seen_ops.add(op_str)
            ops.append({"type": o.op_type, "details": o.details})

    training_payload = {
        "results": state["results"],
        "best_model_name": state["best_model"],
        "best_score": state["best_score"],
        "metric": state["metric"],
        "metric_names": state["metric_names"],
    }

    # 1. Generate PDF Report
    pdf_path = render_report(
        task_id=session_id,
        data=data_desc,
        eda=eda_summary,
        operations=ops,
        training=training_payload,
        task_type=state.get("task_type", "classification"),
        target_column=state.get("target_column", ""),
    )

    create_artifact(db, session_id=session_id, artifact_type="report",
                    file_path=str(pdf_path), meta={})
    
    # 2. Generate Jupyter Notebook
    session = get_session(db, session_id)
    nb_path = None
    if session:
        metadata = session.dataset_meta if isinstance(session.dataset_meta, dict) else {}
        dataset_filename = metadata.get("original_filename", session.name)
        dataset_filename = Path(str(dataset_filename or "")).name
        if not dataset_filename or dataset_filename in {".", ".."}:
            dataset_filename = Path(str(session.source_path or "")).name

        nb_path = generate_notebook(
            session_id=session_id,
            dataset_filename=dataset_filename,
            target_column=state.get("target_column", ""),
            task_type=state.get("task_type", "classification"),
            best_model_name=state.get("best_model", ""),
        )
        create_artifact(db, session_id=session_id, artifact_type="notebook",
                        file_path=str(nb_path), meta={})

    update_session(db, session_id, status="reported")
    return {
        "report_path": str(pdf_path),
        "notebook_path": str(nb_path) if nb_path else None
    }


def _tool_get_operations_log(session_id: str, args: dict, db: DBSession) -> dict:
    ops = list_operations(db, session_id)
    return {
        "count": len(ops),
        "operations": [
            {"type": o.op_type, "details": o.details,
             "status": o.status, "at": o.created_at.isoformat()}
            for o in ops
        ],
    }


def _tool_set_target_column(session_id: str, args: dict, db: DBSession) -> dict:
    column = args.get("column")
    if not column:
        raise ValueError("column is required.")
    update_session(db, session_id, target_column=column)
    state = _load_state(session_id)
    state["target_column"] = column
    _save_state(session_id, state)
    return {"target_column": column, "note": "rerun the pipeline to apply."}


def _tool_rerun_pipeline(session_id: str, args: dict, db: DBSession) -> dict:
    """Run the full pipeline carefully with early exit on failure."""
    if args.get("target_column"):
        _tool_set_target_column(session_id, {"column": args["target_column"]}, db)
    
    target = get_session(db, session_id).target_column
    if not target:
        df = _load_raw(db, session_id)
        target = auto_detect_target(df)
        _tool_set_target_column(session_id, {"column": target}, db)

    # 1. EDA
    eda_result = _tool_run_eda(session_id, {"target_column": target}, db)
    if "error" in eda_result:
        return {"step": "eda", "error": eda_result["error"]}
    
    # 2. Preprocessing
    prep_result = _tool_run_preprocessing(session_id, {"target_column": target}, db)
    if "error" in prep_result:
        return {"step": "preprocessing", "error": prep_result["error"]}
    
    # 3. Training
    train_result = _tool_train_models(session_id, {"task_type": "auto"}, db)
    if "error" in train_result:
        return {"step": "training", "error": train_result["error"]}
    
    # 4. Report & Notebook
    report_result = _tool_generate_report(session_id, {}, db)
    if "error" in report_result:
        return {"step": "report", "error": report_result["error"]}

    # 5. Send success email notification
    try:
        from app.db.models import User
        session = get_session(db, session_id)
        if session:
            user = db.get(User, session.user_id)
            if user and user.email:
                send_completion_email(
                    email=user.email,
                    session_id=session_id,
                    best_model=str(train_result.get('best_model'))
                )
    except Exception as e:
        logger.error(f"Email notification failed: {e}")

    return {
        "status": "success",
        "target": target,
        "best_model": train_result.get("best_model"),
        "report_path": report_result.get("report_path"),
        "notebook_path": report_result.get("notebook_path")
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
}


def execute_tool(
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    db: DBSession,
) -> dict[str, Any]:
    """Dispatch a tool and return a stable, machine-readable result."""
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"Unknown tool '{tool_name}'.",
            "error_type": "UnknownToolError",
        }
    try:
        result = fn(session_id, arguments, db)
        if not isinstance(result, dict):
            raise TypeError(
                f"Tool returned {type(result).__name__}; expected a dictionary."
            )
        if result.get("error"):
            return {
                "ok": False,
                "tool": tool_name,
                "error": str(result["error"]),
                "error_type": result.get("error_type", "ToolError"),
                **{key: value for key, value in result.items() if key != "error"},
            }
        return {"ok": True, "tool": tool_name, **result}
    except Exception as exc:
        logger.exception("Tool '%s' failed", tool_name)
        return {
            "ok": False,
            "tool": tool_name,
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
        }
