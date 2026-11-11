"""Focused ownership and artifact validation tests for prediction endpoints."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import predict as predict_routes
from app.config import settings
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import inference_service


def _user(user_id: str = "owner-1"):
    return SimpleNamespace(id=user_id)


def _model_version(**overrides):
    values = {
        "id": "model-1",
        "experiment_id": "experiment-1",
        "artifact_uri": "unused",
        "schema_json": {"feature": "float64"},
        "model_name": "Classifier",
        "version": 1,
        "metrics_json": {"accuracy": 0.9},
        "lifecycle_status": "candidate",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_owner_can_list_models_without_internal_artifact_fields(monkeypatch) -> None:
    model = _model_version(artifact_uri="C:/private/storage/model.pkl")
    monkeypatch.setattr(
        predict_routes,
        "get_session",
        lambda _db, _session_id: SimpleNamespace(id="session-1", user_id="owner-1"),
    )
    monkeypatch.setattr(predict_routes, "list_model_versions", lambda *_: [model])

    result = predict_routes.list_models_for_session(
        "session-1", object(), _user()
    )

    assert result == [
        {
            "id": "model-1",
            "model_name": "Classifier",
            "version": 1,
            "metrics": {"accuracy": 0.9},
            "status": "candidate",
        }
    ]
    assert "artifact_uri" not in result[0]


@pytest.mark.parametrize("session", [None, SimpleNamespace(id="session-1", user_id="other-user")])
def test_user_cannot_list_missing_or_other_users_models(monkeypatch, session) -> None:
    list_models = Mock(return_value=[])
    monkeypatch.setattr(predict_routes, "get_session", lambda *_: session)
    monkeypatch.setattr(predict_routes, "list_model_versions", list_models)

    with pytest.raises(HTTPException) as exc_info:
        predict_routes.list_models_for_session("session-1", object(), _user())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session not found."
    list_models.assert_not_called()


def test_user_cannot_predict_with_another_users_model_before_loading(monkeypatch) -> None:
    model = _model_version()
    monkeypatch.setattr(predict_routes, "get_model_version", lambda *_: model)
    monkeypatch.setattr(
        predict_routes,
        "get_experiment",
        lambda *_: SimpleNamespace(id="experiment-1", session_id="session-1"),
    )
    monkeypatch.setattr(
        predict_routes,
        "get_session",
        lambda *_: SimpleNamespace(id="session-1", user_id="other-user"),
    )
    run_inference = Mock()
    monkeypatch.setattr(predict_routes.inference_service, "run_inference", run_inference)
    load = Mock()
    monkeypatch.setattr(inference_service.joblib, "load", load)

    with pytest.raises(HTTPException) as exc_info:
        predict_routes.predict(
            "model-1",
            PredictRequest(records=[{"feature": 2.0}]),
            object(),
            _user(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model version not found."
    run_inference.assert_not_called()
    load.assert_not_called()


def test_missing_model_version_returns_404(monkeypatch) -> None:
    monkeypatch.setattr(predict_routes, "get_model_version", lambda *_: None)
    get_experiment = Mock()
    monkeypatch.setattr(predict_routes, "get_experiment", get_experiment)

    with pytest.raises(HTTPException) as exc_info:
        predict_routes.predict(
            "missing-model",
            PredictRequest(records=[{"feature": 2.0}]),
            object(),
            _user(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model version not found."
    get_experiment.assert_not_called()


def test_owned_user_prediction_response_remains_compatible(monkeypatch) -> None:
    expected = PredictResponse(
        predictions=[1],
        probabilities=[[0.1, 0.9]],
        model_version_id="model-1",
        model_name="Classifier",
    )
    monkeypatch.setattr(predict_routes, "get_model_version", lambda *_: _model_version())
    monkeypatch.setattr(
        predict_routes,
        "get_experiment",
        lambda *_: SimpleNamespace(id="experiment-1", session_id="session-1"),
    )
    monkeypatch.setattr(
        predict_routes,
        "get_session",
        lambda *_: SimpleNamespace(id="session-1", user_id="owner-1"),
    )
    inference = Mock(return_value=expected)
    monkeypatch.setattr(predict_routes.inference_service, "run_inference", inference)

    response = predict_routes.predict(
        "model-1",
        PredictRequest(records=[{"feature": 2.0}]),
        object(),
        _user(),
    )

    assert response == expected
    inference.assert_called_once()


def _patch_inference(monkeypatch, storage_root: Path, artifact_path: Path):
    monkeypatch.setattr(settings, "STORAGE_DIR", storage_root)
    monkeypatch.setattr(
        inference_service,
        "get_model_version",
        lambda *_: _model_version(artifact_uri=str(artifact_path)),
    )


def test_artifact_inside_storage_is_validated_before_loading(monkeypatch, tmp_path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    artifact = storage_root / "models" / "model.pkl"
    artifact.parent.mkdir()
    artifact.write_bytes(b"mock artifact")
    _patch_inference(monkeypatch, storage_root, artifact)

    class FakePipeline:
        named_steps = {"model": object()}

        def predict(self, features):
            assert list(features.columns) == ["feature"]
            return np.array([1])

    load = Mock(return_value=FakePipeline())
    monkeypatch.setattr(inference_service.joblib, "load", load)

    response = inference_service.run_inference(
        object(), "model-1", [{"feature": 3.0}]
    )

    assert response.predictions == [1]
    assert load.call_args.args[0] == artifact.resolve()


def test_artifact_outside_storage_is_rejected_before_loading(monkeypatch, tmp_path) -> None:
    storage_root = tmp_path / "storage"
    outside_artifact = tmp_path / "outside.pkl"
    storage_root.mkdir()
    outside_artifact.write_bytes(b"mock artifact")
    _patch_inference(monkeypatch, storage_root, outside_artifact)
    load = Mock()
    monkeypatch.setattr(inference_service.joblib, "load", load)

    with pytest.raises(RuntimeError, match="Model artifact could not be loaded") as exc_info:
        inference_service.run_inference(object(), "model-1", [{"feature": 1.0}])

    assert str(outside_artifact) not in str(exc_info.value)
    load.assert_not_called()


def test_symlink_to_artifact_outside_storage_is_rejected(monkeypatch, tmp_path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    outside_artifact = tmp_path / "outside.pkl"
    outside_artifact.write_bytes(b"mock artifact")
    linked_artifact = storage_root / "linked-model.pkl"
    try:
        linked_artifact.symlink_to(outside_artifact)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is unavailable in this environment: {exc}")

    _patch_inference(monkeypatch, storage_root, linked_artifact)
    load = Mock()
    monkeypatch.setattr(inference_service.joblib, "load", load)

    with pytest.raises(RuntimeError, match="Model artifact could not be loaded"):
        inference_service.run_inference(object(), "model-1", [{"feature": 1.0}])

    load.assert_not_called()


def test_missing_artifact_is_rejected_safely(monkeypatch, tmp_path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    missing_artifact = storage_root / "missing.pkl"
    _patch_inference(monkeypatch, storage_root, missing_artifact)
    load = Mock()
    monkeypatch.setattr(inference_service.joblib, "load", load)

    with pytest.raises(RuntimeError, match="Model artifact could not be loaded") as exc_info:
        inference_service.run_inference(object(), "model-1", [{"feature": 1.0}])

    assert str(missing_artifact) not in str(exc_info.value)
    load.assert_not_called()


def test_missing_input_columns_return_422(monkeypatch, tmp_path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    artifact = storage_root / "model.pkl"
    artifact.write_bytes(b"mock artifact")
    _patch_inference(monkeypatch, storage_root, artifact)

    class FakePipeline:
        named_steps = {"model": object()}

    monkeypatch.setattr(inference_service.joblib, "load", Mock(return_value=FakePipeline()))
    monkeypatch.setattr(predict_routes, "get_model_version", lambda *_: _model_version())
    monkeypatch.setattr(
        predict_routes,
        "get_experiment",
        lambda *_: SimpleNamespace(id="experiment-1", session_id="session-1"),
    )
    monkeypatch.setattr(
        predict_routes,
        "get_session",
        lambda *_: SimpleNamespace(id="session-1", user_id="owner-1"),
    )

    with pytest.raises(HTTPException) as exc_info:
        predict_routes.predict(
            "model-1", PredictRequest(records=[{"other": 1}]), object(), _user()
        )

    assert exc_info.value.status_code == 422
