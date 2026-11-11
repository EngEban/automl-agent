"""Focused target-column and task-type inference tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.model_selector import (
    auto_detect_target,
    detect_target_column,
    detect_task_type,
    infer_task_type,
)


def test_explicit_target_is_respected_and_invalid_name_is_actionable():
    frame = pd.DataFrame({"target": [0, 1, 0], "feature": [4, 5, 6]})

    result = detect_target_column(frame, explicit_target="feature")

    assert result["status"] == "selected"
    assert result["selected"] == "feature"
    assert result["confidence"] == 100
    with pytest.raises(ValueError, match="does not exist.*Choose one of"):
        detect_target_column(frame, explicit_target="missing")


def test_strong_name_selects_candidate_without_last_column_bias():
    frame = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13],
            "price": [100, 120, 110, 130],
            "record_number": [1, 2, 3, 4],
        }
    )

    result = auto_detect_target(frame)

    assert result["status"] == "selected"
    assert result["best"] == "price"
    assert result["best"] != frame.columns[-1]


def test_identifier_columns_are_not_selected_automatically():
    frame = pd.DataFrame(
        {
            "customer_id": range(1, 21),
            "transaction_key": [f"key-{i}" for i in range(20)],
            "feature": np.arange(20, dtype=float),
        }
    )

    result = detect_target_column(frame)

    assert result["status"] == "not_found"
    assert result["best"] is None
    assert all(candidate["column"] not in {"customer_id", "transaction_key"}
               for candidate in result["candidates"])
    assert "Set the target column explicitly" in result["message"]


def test_multiple_strong_target_names_are_reported_as_ambiguous():
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "outcome": [1, 0, 1, 0],
            "feature": [2, 3, 4, 5],
        }
    )

    result = detect_target_column(frame)

    assert result["status"] == "ambiguous"
    assert result["best"] is None
    assert {item["column"] for item in result["candidates"]} == {"label", "outcome"}
    assert "Choose the column" in result["message"]


def test_weak_schema_evidence_does_not_select_a_target():
    frame = pd.DataFrame(
        {"kind": ["a", "b", "a"], "region": ["north", "south", "west"]}
    )

    result = detect_target_column(frame)

    assert result["status"] == "ambiguous"
    assert result["best"] is None
    assert len(result["candidates"]) == 2


@pytest.mark.parametrize(
    ("values", "dtype", "target_name", "categorical", "expected"),
    [
        ([0, 1, 0, 1, 0, 1], "int64", "target", False, "classification"),
        (["no", "yes", "yes", "no"], "object", "target", True, "classification"),
        (["low", "medium", "high", "low"], "object", "target", True, "classification"),
        ([0, 1, 2, 0, 1, 2], "int64", "target", False, "classification"),
        ([10.1, 20.3, 10.1, 35.0], "float64", "target", False, "regression"),
        ([100, 200, 150, 200, 100], "int64", "house_price", False, "regression"),
        ([1, 2, 3, 1, 2, 3], "int64", "count", False, "regression"),
        (list(range(40)), "int64", "target", False, "regression"),
    ],
)
def test_task_type_inference_uses_schema_name_and_multiple_value_indicators(
    values, dtype, target_name, categorical, expected
):
    labels = pd.Series(values, dtype=dtype, name=target_name)
    result = infer_task_type(
        labels,
        target_name=target_name,
        target_is_categorical=categorical,
    )

    assert result["status"] == "resolved"
    assert result["task_type"] == expected


def test_ambiguous_numeric_target_returns_an_explicit_decision_prompt():
    result = infer_task_type(
        pd.Series([10, 20, 30, 10, 20, 30], name="target"),
        target_name="target",
    )

    assert result["status"] == "ambiguous"
    assert result["task_type"] is None
    assert result["candidates"] == ["classification", "regression"]
    assert "Specify 'classification' or 'regression'" in result["message"]
    with pytest.raises(ValueError, match="Could not reliably infer"):
        detect_task_type(
            pd.Series([10, 20, 30, 10, 20, 30], name="target"),
            target_name="target",
        )


def test_explicit_task_type_takes_precedence_over_automatic_evidence():
    result = infer_task_type(
        pd.Series([10.5, 11.2, 12.9], name="price"),
        target_name="price",
        explicit_task_type="classification",
    )

    assert result["task_type"] == "classification"
    assert result["reason"] == "The task type was explicitly selected."
    with pytest.raises(ValueError, match="Unsupported task type"):
        infer_task_type(pd.Series([1, 2]), explicit_task_type="clustering")


@pytest.mark.parametrize(
    ("values", "error"),
    [([1, 1, 1], "only one usable value"), ([np.nan, np.nan], "no usable values")],
)
def test_invalid_explicit_target_values_fail_with_actionable_error(values, error):
    frame = pd.DataFrame({"target": values, "feature": range(len(values))})

    with pytest.raises(ValueError, match=error):
        detect_target_column(frame, explicit_target="target")


def test_agent_rerun_surfaces_ambiguous_target_candidates(monkeypatch):
    from app.agent import executor

    session = type("Session", (), {"target_column": None})()
    frame = pd.DataFrame(
        {"label": [0, 1, 0, 1], "outcome": [1, 0, 1, 0]}
    )
    monkeypatch.setattr(executor, "get_session", lambda *_args: session)
    monkeypatch.setattr(executor, "get_latest_run_for_session", lambda *_args: None)
    monkeypatch.setattr(executor, "_load_raw", lambda *_args: frame)

    result = executor.execute_tool(
        "session-1", "rerun_pipeline", {}, object()
    )

    assert result["ok"] is False
    assert "similarly plausible" in result["error"]
    assert "label, outcome" in result["error"]


def test_agent_training_surfaces_ambiguous_numeric_task_type(monkeypatch):
    from app.agent import executor

    session = type("Session", (), {"id": "session-1", "target_column": "target"})()
    frame = pd.DataFrame(
        {"feature": range(60), "target": [10, 20, 30] * 20}
    )
    monkeypatch.setattr(executor, "get_session", lambda *_args: session)
    monkeypatch.setattr(
        executor, "_require_training_run", lambda *_args: object()
    )
    monkeypatch.setattr(executor, "_session_dir", lambda *_args: __import__("pathlib").Path("."))
    monkeypatch.setattr(executor, "_load_raw", lambda *_args: frame)
    monkeypatch.setattr(executor, "_load_state", lambda *_args: {})
    monkeypatch.setattr(executor, "_save_state", lambda *_args: None)
    monkeypatch.setattr(executor, "update_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        executor.prep_mod,
        "prepare_xy_for_training",
        lambda *_args: (frame[["feature"]], frame["target"], []),
    )

    result = executor.execute_tool(
        "session-1", "train_models", {"task_type": "auto"}, object(), run_id="run-1"
    )

    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert "Could not reliably infer a task type" in result["error"]
    assert "classification" in result["error"] and "regression" in result["error"]


def test_agent_rerun_rejects_an_unsupported_explicit_task_type():
    from app.agent import executor

    result = executor.execute_tool(
        "session-1", "rerun_pipeline", {"task_type": "clustering"}, object()
    )

    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert "Choose classification, regression, or auto" in result["error"]
