from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.data_profiler import profile_dataset


def _classification_frame() -> pd.DataFrame:
    rows = 100
    frame = pd.DataFrame({
        "amount": [*map(float, range(99)), 10000.0],
        "amount_copy": [*map(float, range(99)), 10000.0],
        "category": [f"category-{index}" for index in range(rows)],
        "record_id": range(rows),
        "almost_constant": ["common"] * 99 + ["rare"],
        "constant": [7] * rows,
        "active": [bool(index % 2) for index in range(rows)],
        "event_date": [
            *pd.date_range("2024-01-01", periods=99).strftime("%Y-%m-%d").tolist(),
            "invalid-date",
        ],
        "label": ["major"] * 90 + ["minor"] * 10,
    })
    frame.loc[3, "amount"] = np.nan
    frame["amount_copy"] = frame["amount"]
    frame.loc[4, "label"] = None
    return pd.concat([frame, frame.iloc[[2]]], ignore_index=True)


def test_profile_reports_general_and_column_quality_details():
    profile = profile_dataset(_classification_frame(), "label", "classification")
    columns = {column.name: column for column in profile.columns}

    assert (profile.n_rows, profile.n_columns) == (101, 9)
    assert profile.numeric_count == 4
    assert profile.categorical_count == 3
    assert profile.boolean_count == 1
    assert profile.date_count == 1
    assert profile.missing_count == 3
    assert profile.missing_percentage > 0
    assert profile.duplicate_row_count == 1
    assert profile.memory_usage_bytes and profile.memory_usage_bytes > 0
    assert columns["amount"].detected_type == "numeric"
    assert columns["amount"].missing_count == 1
    assert columns["amount"].numeric_statistics["maximum"] == 10000.0
    assert columns["amount"].outlier_count >= 1
    assert columns["amount_copy"].duplicate_of == "amount"
    assert columns["constant"].is_constant
    assert columns["almost_constant"].is_near_constant
    assert columns["category"].is_high_cardinality
    assert columns["category"].unique_count == 100
    assert columns["record_id"].likely_id
    assert columns["event_date"].is_date_like
    assert columns["event_date"].detected_type == "date"
    assert columns["category"].category_frequency[0]["count"] == 2
    assert columns["category"].rare_category_count == 100


def test_profile_reports_class_distribution_and_imbalance_from_resolved_task():
    profile = profile_dataset(_classification_frame(), "label", "classification")

    target = profile.target_info
    assert target["task_type"] == "classification"
    assert target["missing_count"] == 1
    assert target["class_count"] == 2
    assert target["is_imbalanced"] is True
    assert target["majority_class_percentage"] > 80
    assert target["minority_class_percentage"] < 20
    assert {item["label"] for item in target["class_distribution"]} == {"major", "minor"}


def test_profile_reports_regression_distribution_and_target_outliers():
    frame = pd.DataFrame({
        "feature": np.arange(100, dtype=float),
        "measurement": [*np.arange(99, dtype=float), 10000.0],
    })
    frame.loc[1, "measurement"] = np.nan

    target = profile_dataset(frame, "measurement", "regression").target_info

    assert target["task_type"] == "regression"
    assert target["missing_count"] == 1
    assert target["distribution_statistics"]["minimum"] == 0.0
    assert target["distribution_statistics"]["maximum"] == 10000.0
    assert target["outlier_count"] >= 1


def test_numeric_measurement_is_not_misclassified_as_a_date():
    frame = pd.DataFrame({"year_code": [20200101, 20200102, 20200103]})

    column = profile_dataset(frame).columns[0]

    assert column.detected_type == "numeric"
    assert column.is_date_like is False
