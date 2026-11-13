"""Focused tests for AnalysisRun-aware model registry persistence."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from app.agent import executor


def _training_result() -> dict:
    return {
        "best_model": object(),
        "target_encoder": LabelEncoder().fit(["class-a", "class-b"]),
        "best_model_name": "RandomForest",
        "best_score": 0.91,
        "metric": "F1_Weighted",
        "selection_metric": "F1_Weighted",
        "cv_selection_score": 0.91,
        "metric_names": ["Accuracy", "F1_Weighted"],
        "results": [{"model": "RandomForest", "F1_Weighted": 0.91}],
        "evaluation": {
            "available": True,
            "method": "stratified_holdout",
            "test_size": 0.2,
            "test_rows": 10,
            "random_state": 42,
            "metrics": {"F1_Weighted": 0.8},
        },
        "final_test_metrics": {"F1_Weighted": 0.8},
        "tuning_params": {"n_estimators": 20},
    }


def _patch_training(monkeypatch, tmp_path, *, run):
    session = SimpleNamespace(id="session-1", target_column="target")
    db = Mock()
    db.saved_artifacts = []
    db.training_result = _training_result()
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, 1], name="target")
    target_encoder = LabelEncoder().fit(["class-a", "class-b"])

    monkeypatch.setattr(executor, "_session_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        executor,
        "_load_raw",
        lambda *_: pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
    )
    monkeypatch.setattr(executor, "get_session", lambda *_: session)
    monkeypatch.setattr(executor, "get_run", lambda *_: run)
    monkeypatch.setattr(executor, "_check_run_cancellation", lambda *_: None)
    monkeypatch.setattr(executor, "emit_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor.prep_mod,
        "prepare_xy_for_training",
        lambda *_: (X, y, []),
    )
    monkeypatch.setattr(executor.prep_mod, "build_preprocessor", lambda *_: (object(), []))
    monkeypatch.setattr(
        executor.trainer_mod,
        "train_and_evaluate",
        lambda *_args, **_kwargs: db.training_result,
    )
    monkeypatch.setattr(executor, "_load_state", lambda _: {})
    monkeypatch.setattr(executor, "_save_state", lambda *_: None)
    monkeypatch.setattr(executor, "update_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor.joblib,
        "dump",
        lambda artifact, path: db.saved_artifacts.append((artifact, path)),
    )
    return db, session


def test_training_registers_against_actual_run_and_session(monkeypatch, tmp_path) -> None:
    run = SimpleNamespace(id="run-actual", session_id="session-1")
    db, session = _patch_training(monkeypatch, tmp_path, run=run)
    experiment = SimpleNamespace(id="experiment-1")
    experiment_call = {}
    model_version_call = {}

    def fake_create_experiment(_db, **kwargs):
        experiment_call.update(kwargs)
        return experiment

    def fake_create_model_version(_db, **kwargs):
        model_version_call.update(kwargs)
        return SimpleNamespace(id="model-version-1")

    monkeypatch.setattr(executor, "create_experiment", fake_create_experiment)
    monkeypatch.setattr(executor, "create_model_version", fake_create_model_version)

    result = executor._tool_train_models(
        session.id,
        {"task_type": "classification"},
        db,
        run_id=run.id,
    )

    assert result["best_model"] == "RandomForest"
    assert experiment_call["run_id"] == run.id
    assert experiment_call["session_id"] == session.id
    assert experiment_call["commit"] is False
    assert model_version_call["experiment_id"] == experiment.id
    assert model_version_call["commit"] is False
    assert model_version_call["artifact_uri"].endswith("best_model.pkl")
    assert model_version_call["schema_json"] == {"feature": "float64"}
    assert model_version_call["metrics_json"]["results"]
    assert model_version_call["metrics_json"]["cv_selection_score"] == 0.91
    assert model_version_call["metrics_json"]["selection_metric"] == "F1_Weighted"
    assert model_version_call["metrics_json"]["evaluation"]["metrics"]["F1_Weighted"] == 0.8
    assert model_version_call["metrics_json"]["final_test_metrics"]["F1_Weighted"] == 0.8
    assert result["cv_selection_score"] == 0.91
    assert result["selection_metric"] == "F1_Weighted"
    assert result["evaluation"]["method"] == "stratified_holdout"
    assert result["final_test_metrics"]["F1_Weighted"] == 0.8
    assert model_version_call["parameters_json"] == {"n_estimators": 20}
    saved_bundle, saved_path = next(
        (artifact, path)
        for artifact, path in db.saved_artifacts
        if str(path).endswith("best_model.pkl")
    )
    assert saved_path == tmp_path / "best_model.pkl"
    assert saved_bundle["artifact_format_version"] == 1
    assert saved_bundle["pipeline"] is db.training_result["best_model"]
    assert list(saved_bundle["target_encoder"].classes_) == ["class-a", "class-b"]
    assert saved_bundle["task_type"] == "classification"
    assert saved_bundle["input_columns"] == ["feature"]
    assert model_version_call["artifact_uri"] == str(saved_path)
    db.commit.assert_called_once_with()


@pytest.mark.parametrize(
    ("run_id", "run", "message"),
    [
        (None, None, "explicit AnalysisRun context"),
        ("missing-run", None, "was not found"),
        (
            "other-run",
            SimpleNamespace(id="other-run", session_id="other-session"),
            "does not belong",
        ),
    ],
)
def test_invalid_run_context_does_not_register_experiment(
    monkeypatch,
    tmp_path,
    run_id,
    run,
    message,
) -> None:
    db, session = _patch_training(monkeypatch, tmp_path, run=run)
    create_experiment = Mock()
    monkeypatch.setattr(executor, "create_experiment", create_experiment)

    with pytest.raises(ValueError, match=message):
        executor._tool_train_models(session.id, {}, db, run_id=run_id)

    create_experiment.assert_not_called()


def test_registry_persistence_failure_is_returned_as_tool_error(monkeypatch, tmp_path) -> None:
    run = SimpleNamespace(id="run-actual", session_id="session-1")
    db, session = _patch_training(monkeypatch, tmp_path, run=run)
    monkeypatch.setattr(
        executor,
        "create_experiment",
        lambda *_args, **_kwargs: SimpleNamespace(id="experiment-1"),
    )
    monkeypatch.setattr(
        executor,
        "create_model_version",
        Mock(side_effect=RuntimeError("registry insert failed")),
    )

    result = executor.execute_tool(
        session.id,
        "train_models",
        {"task_type": "classification"},
        db,
        run_id=run.id,
    )

    assert result["ok"] is False
    assert result["error_type"] == "ModelRegistryPersistenceError"
    assert "Model Registry persistence failed" in result["error"]
    db.rollback.assert_called_once_with()


def test_execute_tool_passes_run_context_without_changing_tool_arguments(monkeypatch) -> None:
    captured = {}

    def fake_train(session_id, arguments, db, run_id=None):
        captured.update(
            session_id=session_id,
            arguments=arguments,
            db=db,
            run_id=run_id,
        )
        return {"status": "trained"}

    monkeypatch.setitem(executor._DISPATCH, "train_models", fake_train)
    db = object()
    arguments = {"task_type": "classification"}

    result = executor.execute_tool(
        "session-1",
        "train_models",
        arguments,
        db,
        run_id="run-actual",
    )

    assert result == {"ok": True, "tool": "train_models", "status": "trained"}
    assert captured == {
        "session_id": "session-1",
        "arguments": arguments,
        "db": db,
        "run_id": "run-actual",
    }
