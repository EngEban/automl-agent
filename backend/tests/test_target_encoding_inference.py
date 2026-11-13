"""Target encoding persistence and inverse prediction tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.core.preprocessing import IntelligentPreprocessor, prepare_xy_with_target_encoder
from app.services import inference_service


class _FakeEstimator:
    def predict_proba(self, features):
        return np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])[: len(features)]


class _FakePipeline:
    def __init__(self, predictions=(0, 2)):
        self.predictions = np.asarray(predictions)
        self.named_steps = {"model": _FakeEstimator()}

    def predict(self, features):
        return self.predictions

    def predict_proba(self, features):
        return self.named_steps["model"].predict_proba(features)


def _persist_artifact(tmp_path: Path, monkeypatch, artifact):
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    artifact_path = storage_root / "models" / "model.pkl"
    artifact_path.parent.mkdir()
    joblib.dump(artifact, artifact_path)
    monkeypatch.setattr(settings, "STORAGE_DIR", storage_root)
    model_version = SimpleNamespace(
        id="model-version-1",
        artifact_uri=str(artifact_path),
        schema_json={"feature": "float64"},
        model_name="TestClassifier",
    )
    monkeypatch.setattr(
        inference_service,
        "get_model_version",
        lambda *_: model_version,
    )
    return artifact_path


def test_training_target_encoder_maps_string_classes_in_sorted_order() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["Low", "Medium", "High", "Low"],
        }
    )

    _X, encoded_target, _operations, encoder = prepare_xy_with_target_encoder(
        frame, "target"
    )

    assert encoder is not None
    assert list(encoder.classes_) == ["High", "Low", "Medium"]
    assert encoded_target.tolist() == [1, 2, 0, 1]


def test_bundle_prediction_restores_labels_and_probability_class_order(
    monkeypatch, tmp_path
) -> None:
    target_encoder = LabelEncoder().fit(["Low", "Medium", "High"])
    expected_class_order = ["High", "Low", "Medium"]
    artifact = {
        "artifact_format_version": 1,
        "pipeline": _FakePipeline(),
        "target_encoder": target_encoder,
        "task_type": "classification",
        "input_columns": ["feature"],
    }
    artifact_path = _persist_artifact(tmp_path, monkeypatch, artifact)

    response = inference_service.run_inference(
        object(),
        "model-version-1",
        [{"feature": 10.0}, {"feature": 20.0}],
    )

    assert response.predictions == ["High", "Medium"]
    assert all(isinstance(label, str) for label in response.predictions)
    assert response.probabilities == [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
    # LabelEncoder assigns integer class IDs in classes_ order, so probability
    # columns remain aligned as High, Low, Medium.
    assert expected_class_order == list(target_encoder.classes_)
    assert artifact_path.is_file()


def test_fitted_sklearn_pipeline_round_trips_with_original_labels(
    monkeypatch, tmp_path
) -> None:
    frame = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
            "target": ["Low", "Low", "Medium", "High"],
        }
    )
    X, y, _operations, target_encoder = prepare_xy_with_target_encoder(
        frame, "target"
    )
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))])
    pipeline.fit(X, y)
    _persist_artifact(
        tmp_path,
        monkeypatch,
        {
            "artifact_format_version": 1,
            "pipeline": pipeline,
            "target_encoder": target_encoder,
            "task_type": "classification",
            "input_columns": list(X.columns),
        },
    )

    response = inference_service.run_inference(
        object(), "model-version-1", [{"feature": 5.0}]
    )

    assert response.predictions == ["Low"]
    assert response.probabilities == [[0.25, 0.5, 0.25]]


def test_regression_bundle_without_encoder_preserves_numeric_predictions(
    monkeypatch, tmp_path
) -> None:
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0], "target": [2.5, 4.0, 7.5]})
    X, y, _operations, target_encoder = prepare_xy_with_target_encoder(
        frame, "target"
    )
    pipeline = Pipeline([("model", DummyRegressor(strategy="constant", constant=12.5))])
    pipeline.fit(X, y)
    _persist_artifact(
        tmp_path,
        monkeypatch,
        {
            "artifact_format_version": 1,
            "pipeline": pipeline,
            "target_encoder": target_encoder,
            "task_type": "regression",
            "input_columns": list(X.columns),
        },
    )

    response = inference_service.run_inference(
        object(), "model-version-1", [{"feature": 5.0}]
    )

    assert response.predictions == [12.5]
    assert response.probabilities is None


def test_legacy_pipeline_artifact_still_returns_its_predictions(monkeypatch, tmp_path) -> None:
    _persist_artifact(tmp_path, monkeypatch, _FakePipeline())

    response = inference_service.run_inference(
        object(), "model-version-1", [{"feature": 10.0}, {"feature": 20.0}]
    )

    assert response.predictions == [0, 2]
    assert response.probabilities == [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]


def test_invalid_encoded_prediction_fails_safely(monkeypatch, tmp_path) -> None:
    target_encoder = LabelEncoder().fit(["High", "Low", "Medium"])
    _persist_artifact(
        tmp_path,
        monkeypatch,
        {
            "artifact_format_version": 1,
            "pipeline": _FakePipeline(predictions=(3,)),
            "target_encoder": target_encoder,
            "task_type": "classification",
            "input_columns": ["feature"],
        },
    )

    with pytest.raises(RuntimeError, match="invalid encoded predictions"):
        inference_service.run_inference(
            object(), "model-version-1", [{"feature": 10.0}]
        )


def test_prediction_batches_preserve_order_labels_and_transform_once_per_batch(
    monkeypatch, tmp_path
) -> None:
    X = pd.DataFrame({
        "number": np.arange(12, dtype=float),
        "group": ["north", "south"] * 6,
    })
    labels = LabelEncoder().fit(["cat", "dog"])
    y = np.array([0, 1] * 6)
    pipeline = Pipeline([
        ("prep", IntelligentPreprocessor(list(X.columns))),
        ("model_input", "passthrough"),
        ("model", DummyClassifier(strategy="prior")),
    ])
    pipeline.fit(X, y)
    records = [
        {"number": 12.0, "group": "north"},
        {"number": 13.0, "group": "unseen"},
        {"number": 14.0},
        {"number": 15.0, "group": "south"},
        {"number": 16.0, "group": "north"},
    ]
    expected_input = pd.DataFrame(records).reindex(columns=list(X.columns))
    expected_predictions = labels.inverse_transform(pipeline.predict(expected_input)).tolist()
    expected_probabilities = pipeline.predict_proba(expected_input)
    _persist_artifact(
        tmp_path,
        monkeypatch,
        {
            "artifact_format_version": 1,
            "pipeline": pipeline,
            "target_encoder": labels,
            "task_type": "classification",
            "input_columns": list(X.columns),
        },
    )
    monkeypatch.setattr(settings, "PREDICTION_BATCH_SIZE", 2)

    transformed_batch_sizes = []
    original_transform = IntelligentPreprocessor.transform

    def count_transform(self, values):
        transformed_batch_sizes.append(len(values))
        return original_transform(self, values)

    def reject_refit(self, *_args, **_kwargs):
        raise AssertionError("Inference must reuse fitted preprocessing.")

    monkeypatch.setattr(IntelligentPreprocessor, "transform", count_transform)
    monkeypatch.setattr(IntelligentPreprocessor, "fit", reject_refit)
    monkeypatch.setattr(
        IntelligentPreprocessor,
        "fit_transform",
        reject_refit,
    )

    response = inference_service.run_inference(
        object(), "model-version-1", records
    )

    assert response.predictions == expected_predictions
    assert len(response.probabilities) == len(records)
    np.testing.assert_allclose(response.probabilities, expected_probabilities)
    assert transformed_batch_sizes == [2, 2, 1]
