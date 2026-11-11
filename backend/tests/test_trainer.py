"""Leakage-boundary regression tests for model training and reporting."""
from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from app.agent import executor
from app.core import trainer
from app.core import code_generator
from app.core.preprocessing import build_preprocessor
from app.core.report_generator import _normalize_training


_EVENTS = []


class _RecordingPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.fit_rows_ = tuple(X["row_id"].astype(int))
        _EVENTS.append(("preprocessor_fit", self.fit_rows_))
        return self

    def transform(self, X):
        return X[["row_id"]].to_numpy(dtype=float)


class _RecordingEstimator(BaseEstimator):
    def __init__(self, task_type="classification"):
        self.task_type = task_type

    def fit(self, X, y):
        row_ids = tuple(np.asarray(X)[:, 0].astype(int))
        self.fit_rows_ = row_ids
        self.value_ = (
            pd.Series(y).mode().iloc[0]
            if self.task_type == "classification"
            else float(np.mean(y))
        )
        _EVENTS.append(("model_fit", row_ids))
        return self

    def predict(self, X):
        row_ids = tuple(np.asarray(X)[:, 0].astype(int))
        _EVENTS.append(("predict", row_ids))
        return np.full(len(X), self.value_)


def _dataset(rows=80, task_type="classification"):
    X = pd.DataFrame(
        {"row_id": np.arange(rows), "feature": np.arange(rows, dtype=float) / 10}
    )
    if task_type == "classification":
        y = pd.Series(np.arange(rows) % 2, name="target")
    else:
        y = pd.Series(np.arange(rows, dtype=float) * 1.5, name="target")
    return X, y


def _quiet_training(monkeypatch):
    monkeypatch.setattr(trainer, "_plot_confusion_matrix", lambda *_: "confusion")
    monkeypatch.setattr(trainer, "_plot_regression_eval", lambda *_: ("actual", "residual"))
    monkeypatch.setattr(trainer, "_importance_chart", lambda *_: "importance")
    monkeypatch.setattr(trainer, "_comparison_chart", lambda *_: "comparison")

    def fake_importance(estimator, X, y, **_kwargs):
        test_rows = tuple(X["row_id"].astype(int))
        fit_rows = estimator.named_steps["prep"].fit_rows_
        _EVENTS.append(("importance", test_rows, fit_rows))
        return SimpleNamespace(importances_mean=np.zeros(X.shape[1]))

    monkeypatch.setattr(trainer, "permutation_importance", fake_importance)


def test_holdout_precedes_selection_and_training_boundaries_are_enforced(monkeypatch):
    _EVENTS.clear()
    X, y = _dataset()
    original_split = trainer._split_data

    def recording_split(*args):
        result = original_split(*args)
        train_ids = tuple(result[0]["row_id"].astype(int))
        test_ids = tuple(result[1]["row_id"].astype(int))
        _EVENTS.append(("split", train_ids, test_ids))
        return result

    def candidates(task_type, n_samples, n_classes):
        split_event = next(event for event in _EVENTS if event[0] == "split")
        assert n_samples == len(split_event[1])
        assert n_samples < len(X)
        _EVENTS.append(("selection", n_samples, n_classes))
        return [("Recording", _RecordingEstimator(task_type))]

    monkeypatch.setattr(trainer, "_split_data", recording_split)
    monkeypatch.setattr(trainer, "get_candidate_models", candidates)
    _quiet_training(monkeypatch)

    original_metrics = trainer._calculate_metrics

    def recording_metrics(y_true, predictions, task_type):
        test_ids = next(event[2] for event in _EVENTS if event[0] == "split")
        if tuple(y_true.index) == test_ids:
            _EVENTS.append(("holdout_metrics", tuple(y_true.index)))
        return original_metrics(y_true, predictions, task_type)

    monkeypatch.setattr(trainer, "_calculate_metrics", recording_metrics)
    result = trainer.train_and_evaluate(X, y, _RecordingPreprocessor(), "classification")

    split_event = next(event for event in _EVENTS if event[0] == "split")
    train_ids, test_ids = set(split_event[1]), set(split_event[2])
    selection_index = next(i for i, event in enumerate(_EVENTS) if event[0] == "selection")
    split_index = next(i for i, event in enumerate(_EVENTS) if event[0] == "split")
    assert split_index < selection_index

    model_fits = [event[1] for event in _EVENTS if event[0] == "model_fit"]
    assert any(set(rows) < train_ids for rows in model_fits)  # CV fold fits
    assert train_ids in [set(rows) for rows in model_fits]  # holdout evaluator
    assert set(range(len(X))) in [set(rows) for rows in model_fits]  # production fit
    assert all(set(rows).isdisjoint(test_ids) for rows in model_fits[:-1])
    assert [event[1] for event in _EVENTS if event[0] == "preprocessor_fit"] == model_fits

    prediction_rows = [event[1] for event in _EVENTS if event[0] == "predict"]
    assert prediction_rows.count(split_event[2]) == 1
    assert all(
        set(rows).issubset(train_ids)
        for rows in prediction_rows
        if rows != split_event[2]
    )
    importance_event = next(event for event in _EVENTS if event[0] == "importance")
    assert set(importance_event[1]) == test_ids
    assert set(importance_event[2]) == train_ids
    assert _EVENTS.index(importance_event) < max(
        i
        for i, event in enumerate(_EVENTS)
        if event[0] == "model_fit" and set(event[1]) == set(range(len(X)))
    )
    assert ("holdout_metrics", split_event[2]) in _EVENTS
    assert _EVENTS.index(("holdout_metrics", split_event[2])) < _EVENTS.index(
        importance_event
    )

    assert result["best_score"] == result["cv_selection_score"]
    assert result["best_score"] == result["results"][0]["F1_Weighted"]
    assert result["evaluation"]["available"] is True
    assert result["evaluation"]["metrics"] == result["final_test_metrics"]
    assert result["evaluation"]["test_rows"] == len(test_ids)
    assert result["evaluation"]["random_state"] == trainer.settings.RANDOM_STATE
    assert result["final_test_metrics"]["Accuracy"] == pytest.approx(
        np.mean(y.iloc[list(split_event[2])].to_numpy() == 0)
    )
    assert set(result["best_model"].named_steps["prep"].fit_rows_) == set(range(len(X)))


def test_optuna_trials_only_receive_training_partition(monkeypatch):
    _EVENTS.clear()
    X, y = _dataset(rows=100)
    train_ids_seen = []
    cv_models = []

    def fake_cv(_prep, model, X_cv, _y_cv, _task, _splitter):
        rows = tuple(X_cv["row_id"].astype(int))
        train_ids_seen.append(rows)
        cv_models.append(model)
        score = 0.8 if getattr(model, "n_estimators", 0) == 50 else 0.7
        return {"Accuracy": score, "F1_Weighted": score, "F1_Macro": score}

    class Trial:
        def suggest_int(self, _name, low, _high):
            return low

        def suggest_float(self, _name, low, _high, **_kwargs):
            return low

    class Study:
        best_params = {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2}

        def optimize(self, objective, **_kwargs):
            self.best_value = objective(Trial())

    monkeypatch.setattr(trainer, "_cross_validation_metrics", fake_cv)
    monkeypatch.setattr(trainer.optuna, "create_study", lambda **_kwargs: Study())
    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [("RandomForest", RandomForestClassifier(n_estimators=10))],
    )
    _quiet_training(monkeypatch)

    result = trainer.train_and_evaluate(X, y, _RecordingPreprocessor(), "classification")
    training_ids = set(train_ids_seen[0])
    holdout_ids = set(range(len(X))) - training_ids
    assert train_ids_seen
    assert all(set(rows).isdisjoint(holdout_ids) for rows in train_ids_seen)
    assert cv_models[0].get_params()["n_estimators"] == 10
    assert any(model.get_params()["n_estimators"] == 50 for model in cv_models)
    assert result["tuning_params"]["n_estimators"] == 50
    assert result["cv_selection_score"] == result["results"][0]["F1_Weighted"]


def test_automatic_task_inference_uses_only_a_training_probe(monkeypatch):
    _EVENTS.clear()
    X, y = _dataset(rows=100)
    inference_rows = []
    final_split = {}
    original_inference = trainer._infer_task_type_from_training_data
    original_auto_split = trainer._split_data_after_training_inference

    def recording_inference(
        y_train, target_is_categorical=None, target_name=None
    ):
        inference_rows.append(tuple(y_train.index))
        return original_inference(y_train, target_is_categorical, target_name)

    def recording_auto_split(*args, **kwargs):
        X_train, X_test, y_train, y_test, task_type = original_auto_split(
            *args, **kwargs
        )
        final_split.update(
            train=tuple(X_train["row_id"].astype(int)),
            test=tuple(X_test["row_id"].astype(int)),
        )
        return X_train, X_test, y_train, y_test, task_type

    def candidates(task_type, n_samples, n_classes):
        assert inference_rows
        assert task_type == "classification"
        assert n_samples < len(X)
        return [("Recording", _RecordingEstimator(task_type))]

    monkeypatch.setattr(
        trainer, "_infer_task_type_from_training_data", recording_inference
    )
    monkeypatch.setattr(
        trainer, "_split_data_after_training_inference", recording_auto_split
    )
    monkeypatch.setattr(trainer, "get_candidate_models", candidates)
    _quiet_training(monkeypatch)

    result = trainer.train_and_evaluate(
        X,
        y,
        _RecordingPreprocessor(),
        "auto",
        target_is_categorical=False,
    )

    assert result["task_type"] == "classification"
    assert len(inference_rows[0]) < len(X)
    assert set(inference_rows[0]).isdisjoint(set(final_split["test"]))
    assert set(inference_rows[0]).issubset(set(final_split["train"]))


def test_classification_holdout_is_stratified_and_small_classes_fail_explicitly():
    X, y = _dataset(rows=60)
    X_train, X_test, y_train, y_test = trainer._split_data(X, y, "classification")
    assert set(y_train.unique()) == set(y_test.unique()) == {0, 1}
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert trainer._cv_configuration(y_train, "classification") is not None

    rare_y = pd.Series([0] * 20 + [1] * 2)
    with pytest.raises(ValueError, match="at least 3 examples in every class"):
        trainer._split_data(
            pd.DataFrame({"x": range(len(rare_y))}), rare_y, "classification"
        )

    too_small_for_cv = pd.Series([0] * 3 + [1] * 3)
    split_X = pd.DataFrame({"x": range(len(too_small_for_cv))})
    train_X, _test_X, train_y, _test_y = trainer._split_data(
        split_X, too_small_for_cv, "classification"
    )
    assert len(train_X) < len(split_X)
    assert trainer._cv_configuration(train_y, "classification") is None


def test_model_registry_scaling_and_training_only_class_weight_decisions():
    X = pd.DataFrame({"measurement": [1.0, 2.0, 3.0, 4.0]})
    preprocessor, _ = build_preprocessor(X)
    linear_pipe = trainer._pipeline(preprocessor, LogisticRegression())
    tree_pipe = trainer._pipeline(preprocessor, DecisionTreeClassifier())

    assert linear_pipe.named_steps["prep"].scale_numeric is True
    assert tree_pipe.named_steps["prep"].scale_numeric is False

    candidates, operation = trainer._apply_training_class_weights(
        [
            ("LogisticRegression", LogisticRegression()),
            ("DecisionTree", DecisionTreeClassifier()),
            ("HistGradientBoosting", HistGradientBoostingClassifier()),
        ],
        pd.Series([0] * 18 + [1] * 2),
        "classification",
    )

    assert candidates[0][1].class_weight == "balanced"
    assert candidates[1][1].class_weight == "balanced"
    assert candidates[2][1].class_weight == "balanced"
    assert operation["models_adjusted"] == [
        "LogisticRegression", "DecisionTree", "HistGradientBoosting"
    ]

    regressors, regression_operation = trainer._apply_training_class_weights(
        [("LinearRegression", LinearRegression())],
        pd.Series([0.0, 1.0, 2.0]),
        "regression",
    )
    assert regressors[0][1].get_params() == LinearRegression().get_params()
    assert regression_operation is None


def test_regression_holdout_is_random_and_metrics_are_holdout_metrics(monkeypatch):
    _EVENTS.clear()
    X, y = _dataset(rows=50, task_type="regression")
    X_train, X_test, y_train, y_test = trainer._split_data(X, y, "regression")
    assert len(X_test) == 10
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert isinstance(trainer._cv_configuration(y_train, "regression"), trainer.KFold)

    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [("Recording", _RecordingEstimator("regression"))],
    )
    _quiet_training(monkeypatch)
    result = trainer.train_and_evaluate(X, y, _RecordingPreprocessor(), "regression")
    assert result["evaluation"]["method"] == "random_holdout"
    assert result["evaluation"]["test_rows"] == len(y_test)
    assert result["final_test_metrics"]["RMSE"] >= 0
    assert result["best_score"] == result["cv_selection_score"]


def test_categorical_target_encoder_is_fitted_on_training_labels_only(monkeypatch):
    _EVENTS.clear()
    X = pd.DataFrame({"row_id": np.arange(60), "feature": np.arange(60, dtype=float)})
    y = pd.Series(np.where(np.arange(60) % 2, "positive", "negative"), name="outcome")
    _X_train, _X_test, y_train, _y_test = trainer._split_data(
        X, y, "classification"
    )
    encoder_fit_rows = []
    original_encoder = trainer.LabelEncoder

    class RecordingLabelEncoder(original_encoder):
        def fit(self, labels):
            encoder_fit_rows.append(tuple(labels.index))
            return super().fit(labels)

    monkeypatch.setattr(trainer, "LabelEncoder", RecordingLabelEncoder)
    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [
            ("Recording", _RecordingEstimator("classification"))
        ],
    )
    _quiet_training(monkeypatch)

    result = trainer.train_and_evaluate(
        X, y, _RecordingPreprocessor(), "classification", target_name="outcome"
    )

    assert result["target_encoder"] is not None
    assert len(encoder_fit_rows) == 1
    assert set(encoder_fit_rows[0]) == set(y_train.index)
    assert list(result["target_encoder"].classes_) == sorted(y_train.unique())
    assert result["target_encoder"].classes_.tolist() == ["negative", "positive"]


def test_report_normalization_keeps_cv_and_holdout_metadata_separate():
    training = _normalize_training(
        {
            "best_score": 0.84,
            "metric": "F1_Weighted",
            "results": [{"model": "Tree", "F1_Weighted": 0.84}],
            "metric_names": ["F1_Weighted"],
            "evaluation": {
                "available": True,
                "method": "stratified_holdout",
                "test_rows": 12,
                "test_size": 0.2,
                "random_state": 42,
                "metrics": {"F1_Weighted": 0.75},
            },
            "final_test_metrics": {"F1_Weighted": 0.75},
        }
    )
    assert training["cv_selection_score"] == 0.84
    assert training["selection_metric"] == "F1_Weighted"
    assert training["results"][0]["F1_Weighted"] == 0.84
    assert training["evaluation"]["metrics"]["F1_Weighted"] == 0.75


def test_generated_notebook_splits_before_target_encoding_and_compiles(monkeypatch, tmp_path):
    monkeypatch.setattr(code_generator.settings, "STORAGE_DIR", tmp_path)
    notebook_path = code_generator.generate_notebook(
        session_id="notebook-test",
        dataset_filename="sample.csv",
        target_column="label",
        task_type="classification",
        best_model_name="DecisionTree",
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    prepare_cell = code_cells[2]
    preprocessing_cell = code_cells[3]

    for index, source in enumerate(code_cells):
        compile(source, f"generated_notebook_cell_{index}", "exec")
    assert prepare_cell.index("train_test_split(") < prepare_cell.index(
        "target_encoder.fit_transform"
    )
    assert "dropna(axis=1, how='all')" not in prepare_cell
    assert "keep_empty_features=True" in preprocessing_cell
    assert "max_categories=50" in preprocessing_cell


def test_agent_preprocessing_step_does_not_fit_a_full_dataset_transformer(
    monkeypatch, tmp_path
):
    frame = pd.DataFrame({"feature": [1.0, np.nan, 3.0], "target": [0, 1, 0]})
    X, y = frame[["feature"]], frame["target"]
    blueprint = object()
    saved = []
    artifact_metadata = []
    monkeypatch.setattr(executor, "_load_raw", lambda *_args: frame)
    monkeypatch.setattr(executor, "_session_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(
        executor.prep_mod, "prepare_xy_for_training", lambda *_args: (X, y, [])
    )
    monkeypatch.setattr(
        executor.prep_mod, "build_preprocessor", lambda *_args: (blueprint, [])
    )
    monkeypatch.setattr(executor, "_load_state", lambda *_args: {})
    monkeypatch.setattr(executor, "_save_state", lambda *_args: None)
    monkeypatch.setattr(executor, "create_operation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor,
        "create_artifact",
        lambda *_args, **kwargs: artifact_metadata.append(kwargs.get("meta")),
    )
    monkeypatch.setattr(executor, "update_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor.joblib, "dump", lambda artifact, path: saved.append((artifact, path))
    )

    result = executor._tool_run_preprocessing(
        "session-1", {"target_column": "target"}, object()
    )

    assert result["task_type"] is None
    assert result["task_type_status"] == "pending_training_inference"
    assert (blueprint, tmp_path / "preprocessor.pkl") in saved
    assert not (tmp_path / "processed.pkl").exists()
    exported = pd.read_csv(tmp_path / "cleaned_data.csv")
    assert pd.isna(exported.loc[1, "feature"])
    assert artifact_metadata[0]["imputed"] is False
