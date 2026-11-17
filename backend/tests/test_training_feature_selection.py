from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.agents.pipeline_agent import agent as pipeline_agent
from app.agents.pipeline_agent import executor
from app.agents.pipeline_agent.planning import RequestPlan
from app.core import feature_selection, trainer
from app.core.preprocessing import IntelligentPreprocessor, build_preprocessor
from app.schemas.chat import AnalysisOptions, ChatRequest
from app.services import inference_service


SELECTOR_CONFIGS = {
    "variance_filter": {"method": "variance_filter", "threshold": 0.0},
    "correlation_filter": {"method": "correlation_filter", "threshold": 0.95, "top_k": 3},
    "mutual_information": {"method": "mutual_information", "top_k": 3},
    "univariate_test": {"method": "univariate_test", "top_k": 3},
    "extra_trees": {"method": "extra_trees", "top_k": 3, "n_estimators": 10},
}


def _dataset(rows: int = 60, task_type: str = "classification"):
    rng = np.random.default_rng(2026)
    ids = np.arange(rows)
    signal = ids.astype(float) + rng.normal(0, 0.05, rows)
    values = rng.normal(size=rows)
    values[::11] = np.nan
    X = pd.DataFrame({
        "row_id": ids,
        "signal": signal,
        "near_signal": signal * 2.0,
        "measurement": values,
        "category": np.where(ids % 3 == 0, "north", np.where(ids % 3 == 1, "south", "west")),
    })
    if task_type == "classification":
        y = pd.Series(ids % 2, name="target")
    else:
        y = pd.Series(signal * 0.4 + rng.normal(0, 0.1, rows), name="target")
    return X, y


def _quiet_training(monkeypatch):
    monkeypatch.setattr(trainer, "_plot_confusion_matrix", lambda *_: "confusion")
    monkeypatch.setattr(trainer, "_plot_regression_eval", lambda *_: ("actual", "residual"))
    monkeypatch.setattr(trainer, "_importance_chart", lambda *_: "importance")
    monkeypatch.setattr(trainer, "_comparison_chart", lambda *_: "comparison")
    monkeypatch.setattr(
        trainer,
        "permutation_importance",
        lambda _model, X, *_args, **_kwargs: SimpleNamespace(
            importances_mean=np.zeros(X.shape[1])
        ),
    )


def test_analysis_options_validate_and_persist_selection_configuration():
    config = {
        "method": "mutual_information",
        "top_k": 4,
        "min_features": 2,
    }
    request = ChatRequest(
        mode="advanced",
        message="Train with feature selection",
        analysis_options={"feature_selection": config},
    )
    saved = request.analysis_options.model_dump(exclude_none=True)
    assert saved["feature_selection"] == config
    assert AnalysisOptions.model_validate(saved).feature_selection.method == "mutual_information"

    with pytest.raises(ValueError):
        AnalysisOptions.model_validate({"feature_selection": {"method": "unknown"}})
    with pytest.raises(ValueError):
        AnalysisOptions.model_validate({
            "feature_selection": {"method": "mutual_information", "top_k": 2, "min_features": 3}
        })
    with pytest.raises(ValueError):
        AnalysisOptions.model_validate({
            "feature_selection": {"method": "extra_trees", "top_k": 2, "n_estimators": 500}
        })
    with pytest.raises(ValueError):
        AnalysisOptions.model_validate({
            "feature_selection": {"method": "univariate_test", "top_k": 2, "threshold": 0.2}
        })

    normalized = pipeline_agent._validate_mode_options("advanced", {"feature_selection": config})
    plan = RequestPlan(
        intent="train", mode="advanced", session_id="session-1",
        target_column="target", task_type="classification",
        allowed_tools=frozenset({"train_models"}), execution_order=("train_models",),
        missing_information=(), requires_expensive_execution=True,
    )
    trusted = pipeline_agent._apply_trusted_options("train_models", {
        "feature_selection": {"method": "extra_trees", "top_k": 3, "n_estimators": 10},
    }, normalized, plan)
    assert trusted["feature_selection"] == config
    assert "feature_selection" not in pipeline_agent._apply_trusted_options(
        "train_models", {"feature_selection": config}, {}, plan
    )


def test_approved_analysis_plan_persists_feature_selection(monkeypatch):
    captured = {}
    run = SimpleNamespace(id="run-1", session_id="session-1", progress_percent=10)
    monkeypatch.setattr(executor, "get_run", lambda *_: run)
    monkeypatch.setattr(
        executor, "_load_raw",
        lambda *_: pd.DataFrame({"signal": [0, 1, 2, 3], "target": [0, 1, 0, 1]}),
    )
    monkeypatch.setattr(executor, "create_plan", lambda _db, **kwargs: captured.update(kwargs) or SimpleNamespace(id="plan-1"))
    monkeypatch.setattr(executor, "emit_event", lambda *_args, **_kwargs: None)

    result = executor._tool_propose_plan(
        "session-1",
        {
            "target_column": "target",
            "task_type": "classification",
            "feature_selection": {"method": "variance_filter"},
        },
        object(),
        run_id="run-1",
    )

    assert captured["plan_json"]["feature_selection"] == {
        "method": "variance_filter", "threshold": 0.0, "min_features": 1,
    }
    assert result["feature_selection"] == captured["plan_json"]["feature_selection"]


def test_approved_plan_passes_feature_selection_to_rerun_training(monkeypatch):
    from app.db import crud

    run = SimpleNamespace(id="run-1", session_id="session-1")
    plan = SimpleNamespace(
        approval_status="approved",
        plan_json={
            "target_column": "target",
            "task_type": "classification",
            "feature_selection": {
                "method": "univariate_test", "top_k": 2, "min_features": 1,
            },
        },
    )
    session = SimpleNamespace(id="session-1", target_column="target", user_id="user-1")
    session_reads = iter([session, None])
    captured = {}
    monkeypatch.setattr(executor, "_check_run_cancellation", lambda *_: None)
    monkeypatch.setattr(executor, "get_run", lambda *_: run)
    monkeypatch.setattr(crud, "get_latest_plan_for_run", lambda *_: plan)
    monkeypatch.setattr(executor, "get_session", lambda *_: next(session_reads))
    monkeypatch.setattr(executor, "_tool_set_target_column", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_tool_run_eda", lambda *_args, **_kwargs: {"status": "success"})
    monkeypatch.setattr(executor, "_tool_run_preprocessing", lambda *_args, **_kwargs: {"status": "success"})
    monkeypatch.setattr(
        executor,
        "_tool_train_models",
        lambda _session, args, *_args, **_kwargs: captured.update(args) or {
            "status": "success", "best_model": "DecisionTree", "task_type": "classification",
        },
    )
    monkeypatch.setattr(executor, "_tool_generate_report", lambda *_args, **_kwargs: {"status": "success"})
    monkeypatch.setattr(executor, "_load_state", lambda *_: {})
    monkeypatch.setattr(executor, "_save_state", lambda *_: None)
    monkeypatch.setattr(executor, "emit_event", lambda *_args, **_kwargs: None)

    result = executor._tool_rerun_pipeline("session-1", {}, object(), run_id="run-1")

    assert result["status"] == "success"
    assert captured["feature_selection"] == plan.plan_json["feature_selection"]


@pytest.mark.parametrize("method", tuple(SELECTOR_CONFIGS))
def test_every_selector_trains_inside_the_actual_classifier_pipeline(monkeypatch, method):
    X, y = _dataset()
    config = SELECTOR_CONFIGS[method]
    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [("DecisionTree", DecisionTreeClassifier(random_state=7))],
    )
    _quiet_training(monkeypatch)

    result = trainer.train_and_evaluate(
        X,
        y,
        build_preprocessor(X)[0],
        "classification",
        feature_selection_config=config,
    )

    selection = result["feature_selection"]
    assert result["best_model"].named_steps["feature_selection"].method == method
    assert selection["enabled"] is True
    assert selection["retained_features"] >= 1
    assert selection["input_features"] >= selection["retained_features"]
    assert selection["fit_partition"] == "training_partition_excluding_holdout"
    assert len(selection["cross_validation_folds"]) >= 2
    assert all(
        fold["fit_rows"] < selection["cross_validation_fit_rows"]
        for fold in selection["cross_validation_folds"]
    )


@pytest.mark.parametrize("method", tuple(SELECTOR_CONFIGS))
def test_every_selector_accepts_numeric_regression_targets(method):
    X, y = _dataset(task_type="regression")
    preprocessor = build_preprocessor(X)[0]
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("feature_selection", feature_selection.TrainingFeatureSelector(
            **SELECTOR_CONFIGS[method],
            task_type="regression",
        )),
        ("model_input", "passthrough"),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X, y)
    assert pipeline.named_steps["feature_selection"].n_features_selected_ >= 1
    assert len(pipeline.predict(X)) == len(X)


def test_selector_fits_are_independent_and_never_receive_holdout_rows(monkeypatch):
    X, y = _dataset()
    original_split = trainer._split_data
    X_train, X_holdout, y_train, _ = original_split(X, y, "classification")
    training_ids = set(X_train["row_id"].astype(int))
    holdout_ids = set(X_holdout["row_id"].astype(int))
    events = {"latest_rows": (), "selector_fits": []}

    original_preprocessor_fit = IntelligentPreprocessor._fit_transform_state

    def record_preprocessor_fit(self, values, target=None, **kwargs):
        events["latest_rows"] = tuple(values["row_id"].astype(int))
        return original_preprocessor_fit(self, values, target, **kwargs)

    original_selector_fit = feature_selection.TrainingFeatureSelector.fit

    def record_selector_fit(self, values, target=None):
        events["selector_fits"].append({
            "selector": self,
            "rows": events["latest_rows"],
            "target": tuple(np.asarray(target).tolist()),
        })
        return original_selector_fit(self, values, target)

    monkeypatch.setattr(IntelligentPreprocessor, "_fit_transform_state", record_preprocessor_fit)
    monkeypatch.setattr(feature_selection.TrainingFeatureSelector, "fit", record_selector_fit)
    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [("DecisionTree", DecisionTreeClassifier(random_state=7))],
    )
    _quiet_training(monkeypatch)

    result = trainer.train_and_evaluate(
        X, y, build_preprocessor(X)[0], "classification",
        feature_selection_config=SELECTOR_CONFIGS["mutual_information"],
    )

    assert len(events["selector_fits"]) == len(result["feature_selection"]["cross_validation_folds"]) + 1
    assert len({id(event["selector"]) for event in events["selector_fits"]}) == len(events["selector_fits"])
    assert all(set(event["rows"]).issubset(training_ids) for event in events["selector_fits"])
    assert all(set(event["rows"]).isdisjoint(holdout_ids) for event in events["selector_fits"])
    for event in events["selector_fits"]:
        expected = tuple(y.loc[list(event["rows"])].tolist())
        assert event["target"] == expected
    assert set(events["selector_fits"][-1]["rows"]) == training_ids
    assert result["best_model"].named_steps["feature_selection"].n_samples_fit_ == len(X_train)
    assert result["feature_selection"]["fit_rows"] == len(X_train)
    assert not set(X_holdout["row_id"]).intersection(
        value for fold in result["feature_selection"]["cross_validation_folds"]
        for value in fold.get("row_ids", [])
    )


def test_optuna_evaluates_the_selector_inside_each_training_fold(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    X, y = _dataset(rows=80)
    X_train, X_holdout, y_train, _ = trainer._split_data(X, y, "classification")
    holdout_ids = set(X_holdout["row_id"].astype(int))
    events = {"latest_rows": (), "selector_rows": []}
    original_preprocessor_fit = IntelligentPreprocessor._fit_transform_state
    original_selector_fit = feature_selection.TrainingFeatureSelector.fit

    def capture_preprocessor(self, values, target=None, **kwargs):
        events["latest_rows"] = tuple(values["row_id"].astype(int))
        return original_preprocessor_fit(self, values, target, **kwargs)

    def capture_selector(self, values, target=None):
        events["selector_rows"].append((self, events["latest_rows"]))
        return original_selector_fit(self, values, target)

    class Trial:
        def suggest_int(self, _name, low, _high):
            return low

        def suggest_float(self, _name, low, _high, **_kwargs):
            return low

    class Study:
        best_params = {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2}

        def optimize(self, objective, **_kwargs):
            objective(Trial())

    monkeypatch.setattr(IntelligentPreprocessor, "_fit_transform_state", capture_preprocessor)
    monkeypatch.setattr(feature_selection.TrainingFeatureSelector, "fit", capture_selector)
    monkeypatch.setattr(trainer.optuna, "create_study", lambda **_kwargs: Study())

    tuned, params = trainer._tune_with_optuna(
        build_preprocessor(X)[0],
        RandomForestClassifier(n_estimators=10, random_state=4),
        X_train,
        y_train,
        "classification",
        "F1_Weighted",
        feature_selection_config=SELECTOR_CONFIGS["variance_filter"],
    )

    assert params["n_estimators"] == 50
    assert tuned.n_estimators == 50
    assert len(events["selector_rows"]) == 5
    assert all(set(rows).isdisjoint(holdout_ids) for _, rows in events["selector_rows"])
    assert len({id(selector) for selector, _ in events["selector_rows"]}) == 5


def test_fitted_selector_round_trips_and_prediction_reuses_it(monkeypatch, tmp_path):
    X, y = _dataset()
    preprocessor = build_preprocessor(X)[0]
    pipeline = trainer._pipeline(
        preprocessor,
        DecisionTreeClassifier(random_state=7),
        feature_selection_config=feature_selection.selector_config_dict(
            SELECTOR_CONFIGS["mutual_information"]
        ),
        task_type="classification",
    )
    pipeline.fit(X, y)
    selection_metadata = trainer._feature_selection_summary(
        pipeline, fit_rows=len(X), include_scores=True
    )
    expected = pipeline.predict(X.iloc[:3])
    artifact_path = tmp_path / "selected-model.pkl"
    joblib.dump({
        "artifact_format_version": 1,
        "pipeline": pipeline,
        "target_encoder": None,
        "task_type": "classification",
        "input_columns": list(X.columns),
        "feature_selection": selection_metadata,
    }, artifact_path)
    model_version = SimpleNamespace(
        id="model-1",
        model_name="DecisionTree",
        artifact_uri=str(artifact_path),
        schema_json={column: str(dtype) for column, dtype in X.dtypes.items()},
    )
    monkeypatch.setattr(inference_service, "get_model_version", lambda *_: model_version)
    monkeypatch.setattr(inference_service, "_validated_artifact_path", lambda path: tmp_path / "selected-model.pkl")

    records = X.iloc[:3].loc[:, list(reversed(X.columns))].to_dict(orient="records")
    result = inference_service.run_inference(object(), "model-1", records)

    reloaded = joblib.load(artifact_path)
    assert reloaded["feature_selection"]["selected_feature_names"] == selection_metadata["selected_feature_names"]
    assert result.predictions == expected.tolist()
    assert reloaded["pipeline"].named_steps["feature_selection"].n_features_selected_ == selection_metadata["retained_features"]


def test_disabled_configuration_leaves_pipeline_without_selector(monkeypatch):
    X, _ = _dataset()
    pipeline = trainer._pipeline(build_preprocessor(X)[0], DecisionTreeClassifier())
    assert "feature_selection" not in pipeline.named_steps


def test_explicit_selector_failure_does_not_fall_back_to_unselected_training(monkeypatch):
    X, y = _dataset()
    monkeypatch.setattr(
        trainer,
        "get_candidate_models",
        lambda *_args, **_kwargs: [("DecisionTree", DecisionTreeClassifier(random_state=7))],
    )
    _quiet_training(monkeypatch)

    with pytest.raises(feature_selection.FeatureSelectionError, match="below min_features"):
        trainer.train_and_evaluate(
            X,
            y,
            build_preprocessor(X)[0],
            "classification",
            feature_selection_config={
                "method": "variance_filter",
                "threshold": 1_000_000,
                "min_features": 1,
            },
        )
