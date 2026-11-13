import pandas as pd
import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from app.core.preprocessing import (
    IntelligentPreprocessor,
    build_preprocessor,
    prepare_xy,
    prepare_xy_for_training,
    prepare_xy_with_target_encoder,
)


def test_prepare_xy_with_encoder_returns_mapping_and_legacy_api_matches() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["Low", "Medium", "High", "Low"],
        }
    )

    X, y, operations, encoder = prepare_xy_with_target_encoder(frame, "target")
    old_X, old_y, old_operations = prepare_xy(frame, "target")

    assert isinstance(encoder, LabelEncoder)
    assert list(encoder.classes_) == ["High", "Low", "Medium"]
    assert y.tolist() == [1, 2, 0, 1]
    assert X.equals(old_X)
    assert y.equals(old_y)
    assert operations == old_operations


def test_prepare_xy_with_encoder_keeps_regression_target_unencoded() -> None:
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": [2.5, 4.0, 7.5]})

    X, y, operations, encoder = prepare_xy_with_target_encoder(frame, "target")

    assert X.columns.tolist() == ["feature"]
    assert y.tolist() == [2.5, 4.0, 7.5]
    assert encoder is None
    assert all(operation.get("type") != "target_encoding" for operation in operations)


def test_training_data_preparation_keeps_target_labels_unencoded_until_split():
    frame = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "target": ["class-b", "class-a", "class-b", "class-a"]}
    )

    X, y, operations = prepare_xy_for_training(frame, "target")

    assert X.columns.tolist() == ["feature"]
    assert y.tolist() == ["class-b", "class-a", "class-b", "class-a"]
    assert all(operation.get("type") != "target_encoding" for operation in operations)


def test_preprocessor_construction_uses_schema_without_inspecting_feature_values():
    first = pd.DataFrame(
        {
            "number": [np.nan, np.nan, np.nan],
            "category": ["a", "b", "c"],
        }
    )
    second = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0],
            "category": ["same", "same", "same"],
        }
    )

    _first_preprocessor, first_ops = build_preprocessor(first)
    _second_preprocessor, second_ops = build_preprocessor(second)

    assert first_ops == second_ops
    assert not any("drop" in str(operation).lower() for operation in first_ops)
    assert first_ops[0]["max_categories"] == 50


def test_preprocessor_learns_imputation_and_categories_from_training_rows_only():
    train = pd.DataFrame(
        {
            "numeric_with_gaps": [np.nan if index % 7 == 0 else float(index) for index in range(60)],
            "high_cardinality": [f"category-{index}" for index in range(60)],
        }
    )
    heldout = pd.DataFrame(
        {"numeric_with_gaps": [999.0], "high_cardinality": ["heldout-only-category"]}
    )
    complete = pd.concat([train, heldout], ignore_index=True)
    preprocessor, operations = build_preprocessor(complete)
    pipeline = Pipeline([("preprocessor", preprocessor)])

    pipeline.fit(train)
    transformed = pipeline.transform(heldout)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    numeric_imputer = fitted_preprocessor.named_transformers_["num"].named_steps["imputer"]
    numeric_scaler = fitted_preprocessor.named_transformers_["num"].named_steps["scaler"]
    categorical_encoder = (
        fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    )
    learned_categories = set(categorical_encoder.categories_[0])

    assert numeric_imputer.statistics_[0] != 999.0
    expected_imputed_mean = train["numeric_with_gaps"].fillna(
        numeric_imputer.statistics_[0]
    ).mean()
    assert numeric_scaler.mean_[0] == pytest.approx(expected_imputed_mean)
    assert "heldout-only-category" not in learned_categories
    assert transformed.shape[0] == 1
    assert transformed.shape[1] <= 1 + 50
    assert any(operation.get("max_categories") == 50 for operation in operations)


def test_all_missing_feature_is_safely_removed_when_other_features_are_usable():
    frame = pd.DataFrame({"empty": [np.nan] * 8, "signal": range(8)})
    preprocessor, _operations = build_preprocessor(frame)

    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape == (8, 1)
    assert preprocessor.removed_constant_columns_ == ["empty"]
    assert any(operation["type"] == "drop_constant_features" for operation in preprocessor.operations_)


def test_numeric_and_categorical_imputation_and_unseen_categories_are_safe():
    train = pd.DataFrame({
        "number": [1.0, np.nan, 3.0, np.inf, -np.inf, 5.0],
        "group": ["a", "a", None, "b", "b", "a"],
    })
    inference = pd.DataFrame({"number": [999.0], "group": ["unseen"]})
    preprocessor, _ = build_preprocessor(pd.concat([train, inference], ignore_index=True))

    transformed_train = preprocessor.fit_transform(train)
    transformed_inference = preprocessor.transform(inference)
    fitted = preprocessor.named_transformers_
    numeric = fitted["num"].named_steps["imputer"]
    categories = fitted["cat"].named_steps["encoder"].categories_[0]

    assert np.isfinite(transformed_train.data if sparse.issparse(transformed_train) else transformed_train).all()
    assert np.isfinite(transformed_inference.data if sparse.issparse(transformed_inference) else transformed_inference).all()
    assert sparse.issparse(transformed_train)
    assert sparse.issparse(transformed_inference)
    assert numeric.statistics_[0] == pytest.approx(3.0)
    assert set(categories) == {"a", "b"}
    assert transformed_inference.shape[1] == transformed_train.shape[1]


def test_high_cardinality_categories_are_capped_and_reported():
    frame = pd.DataFrame({"product_family": [f"family-{i}" for i in range(120)]})
    preprocessor, _ = build_preprocessor(frame)

    transformed = preprocessor.fit_transform(frame)
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    decisions = preprocessor.operations_

    assert transformed.shape[1] <= 50
    assert sparse.issparse(transformed) is preprocessor.column_transformer_.sparse_output_
    assert encoder.max_categories == 50
    assert any(item["type"] == "high_cardinality_handling" for item in decisions)
    decision = next(item for item in decisions if item["type"] == "high_cardinality_handling")
    assert decision["columns"][0]["name"] == "product_family"


def test_constant_and_duplicate_features_are_removed_without_dropping_target():
    frame = pd.DataFrame({
        "signal": [1, 2, 3, 4, 5, 6],
        "same_signal": [1, 2, 3, 4, 5, 6],
        "constant": [7] * 6,
        "target": ["a", "b", "a", "b", "a", "b"],
    })
    X, y, _ = prepare_xy_for_training(frame, "target")
    preprocessor, _ = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert y.tolist() == ["a", "b", "a", "b", "a", "b"]
    assert "target" not in X.columns
    assert preprocessor.removed_constant_columns_ == ["constant"]
    assert preprocessor.removed_duplicate_columns_ == [
        {"column": "same_signal", "duplicate_of": "signal"}
    ]
    assert transformed.shape == (6, 1)


def test_date_features_extract_calendar_parts_and_invalid_values_are_imputed():
    train = pd.DataFrame({
        "event_date": [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-06", "2024-01-07", "invalid",
        ]
    })
    inference = pd.DataFrame({"event_date": ["not a date", "2024-02-01"]})
    preprocessor, _ = build_preprocessor(pd.concat([train, inference], ignore_index=True))

    transformed_train = preprocessor.fit_transform(train)
    transformed_inference = preprocessor.transform(inference)

    assert preprocessor.date_columns_ == ["event_date"]
    assert transformed_train.shape[1] == 4
    assert np.isfinite(transformed_train.data if sparse.issparse(transformed_train) else transformed_train).all()
    assert np.isfinite(transformed_inference.data if sparse.issparse(transformed_inference) else transformed_inference).all()
    date_op = next(op for op in preprocessor.operations_ if op["type"] == "date_feature_extraction")
    assert date_op["features"] == ["year", "month", "day", "day_of_week"]


def test_date_detection_is_fitted_on_training_rows_not_holdout_values():
    train = pd.DataFrame({"activity": ["north", "south", "north", "south"] * 3})
    heldout = pd.DataFrame({"activity": ["2024-01-01", "2024-01-02"]})
    preprocessor, _ = build_preprocessor(pd.concat([train, heldout], ignore_index=True))

    preprocessor.fit(train)

    assert preprocessor.date_columns_ == []
    assert set(preprocessor.named_transformers_["cat"].named_steps["encoder"].categories_[0]) == {
        "north", "south"
    }


def test_holdout_mixed_dtype_cannot_change_training_feature_type():
    train_values = pd.DataFrame({"measurement": [1, 2, 3, 4, 5, 6]})
    heldout_values = pd.DataFrame({"measurement": ["not-numeric"]})
    complete = pd.concat([train_values, heldout_values], ignore_index=True)
    train = complete.iloc[:6]
    heldout = complete.iloc[6:]
    preprocessor, _ = build_preprocessor(complete)

    preprocessor.fit(train)
    transformed = preprocessor.transform(heldout)

    assert train["measurement"].dtype == object
    assert preprocessor.numeric_string_columns_ == ["measurement"]
    assert preprocessor.categorical_features_ == []
    assert np.isfinite(transformed.data if sparse.issparse(transformed) else transformed).all()


def test_preprocessing_is_compatible_with_classification_and_regression_models():
    frame = pd.DataFrame({
        "number": np.arange(20, dtype=float),
        "segment": ["low", "high"] * 10,
    })
    inference = pd.DataFrame({"number": [21.0], "segment": ["new-segment"]})
    cases = [
        (np.array([0, 1] * 10), LogisticRegression(max_iter=1000)),
        (np.arange(20, dtype=float) * 2.5, LinearRegression()),
    ]

    for target, estimator in cases:
        preprocessor, _ = build_preprocessor(frame)
        model = Pipeline([("prep", preprocessor), ("model", estimator)])
        model.fit(frame, target)
        prediction = model.predict(inference)
        assert len(prediction) == 1


def test_numeric_only_preprocessing_stays_dense_and_reports_output_format():
    frame = pd.DataFrame({"left": [1.0, 2.0, 3.0, 4.0], "right": [4.0, 2.0, 3.0, 1.0]})
    preprocessor, _ = build_preprocessor(frame)

    transformed = preprocessor.fit_transform(frame)

    assert isinstance(transformed, np.ndarray)
    output_operation = next(
        operation
        for operation in preprocessor.operations_
        if operation["type"] == "encoded_matrix_format"
    )
    assert output_operation["format"] == "dense"
    assert preprocessor.get_feature_names_out().tolist() == ["num__left", "num__right"]


def test_cv_fits_preprocessing_independently_inside_each_fold(monkeypatch):
    from sklearn.base import BaseEstimator

    from app.core import preprocessing
    from app.core.trainer import _cross_validation_metrics

    X = pd.DataFrame({
        "position": np.arange(30),
        "category": [f"category-{row}" for row in range(30)],
    })
    y = pd.Series([0, 1] * 15)
    preprocessor, _ = build_preprocessor(X)
    fit_records = []
    original_fit_transform_state = IntelligentPreprocessor._fit_transform_state

    def recording_fit_transform_state(self, values, target=None, **fit_params):
        result = original_fit_transform_state(self, values, target, **fit_params)
        encoder = self.named_transformers_["cat"].named_steps["encoder"]
        fit_records.append({
            "row_ids": set(values["position"].astype(int)),
            "categories": set(encoder.categories_[0]),
        })
        return result

    class TinyClassifier(BaseEstimator):
        def fit(self, values, labels):
            self.label_ = int(pd.Series(labels).mode().iloc[0])
            return self

        def predict(self, values):
            return np.full(values.shape[0], self.label_)

    monkeypatch.setattr(
        preprocessing.IntelligentPreprocessor,
        "_fit_transform_state",
        recording_fit_transform_state,
    )
    _cross_validation_metrics(
        preprocessor,
        TinyClassifier(),
        X,
        y,
        "classification",
        StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    )

    assert len(fit_records) == 3
    assert all(
        record["categories"] == {f"category-{row}" for row in record["row_ids"]}
        for record in fit_records
    )
    assert len({tuple(sorted(record["categories"])) for record in fit_records}) == 3
