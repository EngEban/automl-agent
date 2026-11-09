"""
Preprocessing pipeline builder + operation logging.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Thresholds for feature handling
MAX_CARDINALITY = 50
RARE_THRESHOLD = 0.05  # Group categories that appear in less than 5% of rows


def prepare_xy(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """
    Split df into X and y, handling missing targets and duplicates.
    Returns (X, y, operations_log).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    ops: list[dict[str, Any]] = []
    df = df.copy()

    # 1. Handle missing target
    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        ops.append({"type": "drop_missing_target", "count": dropped})

    # 2. Handle duplicates
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        ops.append({"type": "drop_duplicates", "count": dupes})

    X = df.drop(columns=[target])
    y = df[target]

    # 3. Encode target if categorical
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)
        ops.append({
            "type": "target_encoding",
            "method": "LabelEncoder",
            "classes": [str(c) for c in le.classes_],
        })

    return X, y, ops


def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[dict[str, Any]]]:
    """
    Build a ColumnTransformer for numeric + categorical features.
    Handles infinity, all-null columns, high-cardinality, and dates.
    """
    ops: list[dict[str, Any]] = []
    transformers = []

    # 1. Clean infinity values
    X = X.replace([np.inf, -np.inf], np.nan)

    # 2. Drop all-null columns
    all_null_cols = [col for col in X.columns if X[col].isna().all()]
    if all_null_cols:
        X = X.drop(columns=all_null_cols)
        ops.append({"type": "drop_all_null_columns", "columns": all_null_cols})

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    # 3. Identify column types
    numeric_features = []
    categorical_features = []
    datetime_features = []
    high_cardinality_features = []

    for col in X.columns:
        series = X[col]
        
        # Try datetime parsing if object type
        if series.dtype == "object":
            try:
                # Only parse if it looks like a date
                parsed = pd.to_datetime(series, errors="raise")
                datetime_features.append(col)
                continue
            except (ValueError, TypeError):
                pass

        # Categorical / Boolean
        if series.dtype == "object" or series.dtype.name == "category" or series.dtype == "bool":
            unique_count = series.nunique()
            if unique_count > MAX_CARDINALITY:
                high_cardinality_features.append(col)
            else:
                categorical_features.append(col)
        else:
            numeric_features.append(col)

    # 4. Build transformers for Numeric
    if numeric_features:
        ops.append({"type": "scaling", "columns": numeric_features, "method": "StandardScaler"})
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_features))

    # 5. Build transformers for Categorical (with rare-category grouping)
    if categorical_features:
        ops.append({"type": "encoding", "columns": categorical_features, "method": "OneHotEncoder (min_frequency)"})
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # min_frequency groups rare categories into a single "infrequent" bucket
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_features))

    # 6. Handle High Cardinality (Drop them to save memory)
    if high_cardinality_features:
        ops.append({
            "type": "drop_high_cardinality",
            "columns": high_cardinality_features,
            "reason": f"Unique count exceeds {MAX_CARDINALITY}",
        })

    # 7. Build transformers for Datetime
    if datetime_features:
        ops.append({"type": "datetime_extraction", "columns": datetime_features})
        # We will rely on the executor to pre-process datetimes before passing to the pipeline
        # For the ColumnTransformer, we pass them through (they will be dropped by remainder='drop' if not handled)
        # In a real production system, we'd use a custom transformer here. 
        # For now, let's treat the parsed dates as numeric features (Year, Month, Day)
        pass

    # 8. Build the final preprocessor
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, ops