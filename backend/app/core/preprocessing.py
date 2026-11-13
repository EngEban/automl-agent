"""Leakage-safe feature preprocessing and operation logging."""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

from app.core.data_profiler import is_date_like_series
from app.core.model_selector import validate_target_column

logger = logging.getLogger(__name__)

MAX_CARDINALITY = 50
RARE_THRESHOLD = 0.05
_DATE_PARTS = ("year", "month", "day", "day_of_week")


def _replace_infinite_values(values):
    """Replace infinities without learning statistics from the input rows."""
    if isinstance(values, pd.DataFrame):
        return values.replace([np.inf, -np.inf], np.nan)
    array = np.asarray(values, dtype=float).copy()
    array[~np.isfinite(array)] = np.nan
    return array


def _categorical_strings(values):
    """Normalize heterogeneous category values while preserving missingness."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    return frame.apply(
        lambda column: column.map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
    )


def prepare_xy(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """Legacy helper returning X/y and cleaning operations."""
    X, y, operations, _ = prepare_xy_with_target_encoder(df, target)
    return X, y, operations


def prepare_xy_with_target_encoder(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]], LabelEncoder | None]:
    """Legacy eager-encoding helper; model training uses the split-safe helper."""
    X, y, operations = prepare_xy_for_training(df, target)
    target_encoder: LabelEncoder | None = None
    if y.dtype == "object" or y.dtype.name == "category" or pd.api.types.is_string_dtype(y.dtype):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target)
        operations.append({
            "type": "target_encoding", "method": "LabelEncoder",
            "classes": [str(value) for value in target_encoder.classes_],
        })
    return X, y, operations, target_encoder


def prepare_xy_for_training(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """Prepare rows without fitting a target-label mapping before the split."""
    validate_target_column(df, target)
    operations: list[dict[str, Any]] = []
    before = len(df)
    cleaned = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped:
        operations.append({"type": "drop_missing_target", "count": dropped})
    duplicates = int(cleaned.duplicated().sum())
    if duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        operations.append({"type": "drop_duplicates", "count": duplicates})
    return cleaned.drop(columns=[target]), cleaned[target], operations


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _looks_like_identifier(column: Any) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(column).casefold()))
    return bool(tokens & {"id", "uuid", "guid", "key", "index", "identifier"})


def _numeric_from_training_values(series: pd.Series, column: Any) -> bool:
    """Recognize numeric text using only values in the current fit partition."""
    if (
        pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or _looks_like_identifier(column)
    ):
        return False
    values = series.dropna()
    if values.empty:
        return False
    text = values.astype("string").str.strip()
    if text.str.match(r"^0\d+$").any():
        return False
    parsed = pd.to_numeric(values, errors="coerce")
    return bool(parsed.notna().mean() >= 0.95)


class IntelligentPreprocessor(BaseEstimator, TransformerMixin):
    """Fit feature decisions on each training partition and retain them for inference."""

    def __init__(
        self,
        feature_columns: list[Any],
        max_categories: int = MAX_CARDINALITY,
        rare_threshold: float = RARE_THRESHOLD,
        scale_numeric: bool = True,
    ):
        self.feature_columns = feature_columns
        self.max_categories = max_categories
        self.rare_threshold = rare_threshold
        self.scale_numeric = scale_numeric

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if list(X.columns) == list(self.feature_columns):
                return X
            return X.reindex(columns=self.feature_columns)
        else:
            frame = pd.DataFrame(X, columns=self.feature_columns)
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame.loc[:, self.feature_columns]

    def _expand_dates(self, frame: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, Any] = {}
        for index, column in enumerate(self.kept_columns_):
            values = frame[column]
            if column in self.date_columns_:
                parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
                for part in _DATE_PARTS:
                    key = f"__automl_date_{index}_{part}"
                    while key in self.feature_columns:
                        key = f"_{key}"
                    if part == "day_of_week":
                        output[key] = parsed.dt.dayofweek.astype(float)
                    else:
                        output[key] = getattr(parsed.dt, part).astype(float)
            elif column in self.numeric_source_columns_:
                output[str(column)] = pd.to_numeric(values, errors="coerce")
            else:
                output[str(column)] = values
        return pd.DataFrame(output, index=frame.index)

    def _fit_transform_state(self, X, y=None, **fit_params):
        frame = self._as_frame(X)
        self.n_features_in_ = len(self.feature_columns)
        self.feature_names_in_ = np.asarray(self.feature_columns, dtype=object)
        self.removed_constant_columns_ = []
        self.removed_duplicate_columns_ = []
        self.near_constant_columns_ = []
        self.kept_columns_ = []
        self.date_columns_ = []
        self.numeric_string_columns_ = []
        self.numeric_identifier_columns_ = []
        seen_series: list[tuple[Any, pd.Series]] = []

        for column in self.feature_columns:
            values = frame[column]
            if int(values.nunique(dropna=False)) <= 1:
                self.removed_constant_columns_.append(column)
                continue
            duplicate_of = next(
                (prior for prior, prior_values in seen_series if values.equals(prior_values)),
                None,
            )
            if duplicate_of is not None:
                self.removed_duplicate_columns_.append({"column": column, "duplicate_of": duplicate_of})
                continue
            seen_series.append((column, values))
            self.kept_columns_.append(column)
            counts = values.value_counts(dropna=True)
            if len(counts) > 1 and counts.iloc[0] / max(1, counts.sum()) >= 0.99:
                self.near_constant_columns_.append(column)
            if is_date_like_series(values):
                self.date_columns_.append(column)
            elif pd.api.types.is_numeric_dtype(values.dtype) and _looks_like_identifier(column):
                self.numeric_identifier_columns_.append(column)
            elif _numeric_from_training_values(values, column):
                self.numeric_string_columns_.append(column)

        if not self.kept_columns_:
            raise ValueError(
                "No usable feature columns remain after constant and duplicate columns are removed."
            )

        self.numeric_source_columns_ = [
            column for column in self.kept_columns_
            if column not in self.date_columns_
            and column not in self.numeric_identifier_columns_
            and (
                pd.api.types.is_numeric_dtype(frame[column].dtype)
                or column in self.numeric_string_columns_
            )
        ]
        expanded = self._expand_dates(frame)
        self.numeric_features_ = [
            column for column in expanded.columns
            if column.startswith("__automl_date_")
            or (
                column not in {str(value) for value in self.numeric_identifier_columns_}
                and not _is_categorical(expanded[column])
            )
        ]
        self.categorical_features_ = [
            column for column in expanded.columns if column not in self.numeric_features_
        ]

        transformers = []
        if self.numeric_features_:
            numeric_pipe = Pipeline([
                ("replace_infinity", FunctionTransformer(
                    _replace_infinite_values,
                    feature_names_out="one-to-one",
                    validate=False,
                )),
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler() if self.scale_numeric else "passthrough"),
            ])
            transformers.append(("num", numeric_pipe, self.numeric_features_))
        if self.categorical_features_:
            categorical_pipe = Pipeline([
                ("categorical_strings", FunctionTransformer(
                    _categorical_strings,
                    feature_names_out="one-to-one",
                    validate=False,
                )),
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encoder", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    sparse_output=True,
                    min_frequency=self.rare_threshold,
                    max_categories=self.max_categories,
                )),
            ])
            transformers.append(("cat", categorical_pipe, self.categorical_features_))
        if not transformers:
            raise ValueError("No supported feature columns remain after preprocessing decisions.")

        self.column_transformer_ = ColumnTransformer(
            transformers, remainder="drop", sparse_threshold=1.0
        )
        transformed = self.column_transformer_.fit_transform(expanded, y, **fit_params)
        self.operations_ = self._fitted_operations(frame)
        self.operations_.append({
            "type": "encoded_matrix_format",
            "format": (
                f"{transformed.format.upper()} sparse"
                if sparse.issparse(transformed)
                else "dense"
            ),
            "reason": "Sparse output is retained when it reduces preprocessing memory.",
        })
        return transformed

    def fit(self, X, y=None):
        self._fit_transform_state(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self._fit_transform_state(X, y, **fit_params)

    def transform(self, X):
        if not hasattr(self, "column_transformer_"):
            raise ValueError("IntelligentPreprocessor must be fitted before transform.")
        frame = self._as_frame(X)
        return self.column_transformer_.transform(self._expand_dates(frame))

    @property
    def named_transformers_(self):
        return self.column_transformer_.named_transformers_

    def get_feature_names_out(self, input_features=None):
        return self.column_transformer_.get_feature_names_out()

    def _fitted_operations(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = [{
            "type": "numeric_scaling",
            "columns": [str(column) for column in self.numeric_features_],
            "method": "StandardScaler" if self.scale_numeric else "None (model is scale-insensitive)",
        }]
        source_numeric = [
            str(column) for column in self.kept_columns_
            if column not in self.date_columns_
            and column not in self.numeric_identifier_columns_
            and (column in self.numeric_string_columns_ or not _is_categorical(frame[column]))
        ]
        if self.numeric_features_:
            operations.append({
                "type": "numeric_imputation",
                "columns": source_numeric + [str(column) for column in self.date_columns_],
                "method": "Median fitted on each training fold; empty learned features use the sklearn safe fill.",
            })
        source_categorical = [
            str(column) for column in self.kept_columns_
            if column not in self.date_columns_
            and (column in self.numeric_identifier_columns_ or _is_categorical(frame[column]))
        ]
        if self.categorical_features_:
            operations.append({
                "type": "categorical_imputation_encoding",
                "columns": source_categorical,
                "imputation": "Most frequent value fitted on each training fold.",
                "encoding": "One-hot; rare categories are grouped, unknown values are ignored safely.",
                "max_categories": self.max_categories,
            })
        if self.removed_constant_columns_:
            operations.append({
                "type": "drop_constant_features",
                "columns": [str(column) for column in self.removed_constant_columns_],
                "reason": "Constant within the training rows; the feature has no learned signal.",
            })
        if self.removed_duplicate_columns_:
            operations.append({
                "type": "drop_duplicate_features",
                "columns": [
                    {"name": str(item["column"]), "duplicate_of": str(item["duplicate_of"])}
                    for item in self.removed_duplicate_columns_
                ],
                "reason": "Exact duplicate feature values within the training rows.",
            })
        if self.near_constant_columns_:
            operations.append({
                "type": "review_near_constant_features",
                "columns": [str(column) for column in self.near_constant_columns_],
                "reason": "Retained because low-frequency values can still be informative.",
            })
        if self.date_columns_:
            operations.append({
                "type": "date_feature_extraction",
                "columns": [str(column) for column in self.date_columns_],
                "features": list(_DATE_PARTS),
                "reason": "Reliable date parsing was confirmed using this fitted training partition.",
            })

        high_cardinality = []
        for column in self.categorical_features_:
            source_column = next(
                (candidate for candidate in self.kept_columns_
                 if candidate == column and candidate not in self.date_columns_),
                None,
            )
            if source_column is None:
                continue
            values = frame[source_column]
            unique_count = int(values.nunique(dropna=True))
            if unique_count > self.max_categories:
                likely_id = _looks_like_identifier(source_column) or (
                    len(values) >= 20 and unique_count / len(values) >= 0.95
                )
                high_cardinality.append({
                    "name": str(source_column),
                    "unique_count": unique_count,
                    "likely_identifier": bool(likely_id),
                    "strategy": "Rare values are grouped and one-hot output is capped at "
                    f"{self.max_categories}; feature retained.",
                })
        if high_cardinality:
            operations.append({
                "type": "high_cardinality_handling",
                "columns": high_cardinality[:20],
                "additional_columns": max(0, len(high_cardinality) - 20),
                "rare_threshold": self.rare_threshold,
                "max_categories": self.max_categories,
            })
        return operations


def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[IntelligentPreprocessor, list[dict[str, Any]]]:
    """Build a schema-only blueprint; all value-based decisions happen in fit."""
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    operations = [{
        "type": "preprocessing_blueprint",
        "columns": [str(column) for column in X.columns],
        "method": "Feature types and value-based decisions are determined during each training-fold fit.",
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "max_categories": MAX_CARDINALITY,
        "rare_threshold": RARE_THRESHOLD,
        "date_features": list(_DATE_PARTS),
    }]
    preprocessor = IntelligentPreprocessor(list(X.columns))
    return preprocessor, operations
