"""
Preprocessing pipeline builder + operation logging.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def prepare_xy(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """
    Split df into X and y, dropping missing targets and duplicates.
    Returns (X, y, operations_log).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    ops: list[dict[str, Any]] = []

    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        ops.append({"type": "drop_missing_target", "count": dropped})

    dupes = int(df.duplicated().sum())
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        ops.append({"type": "drop_duplicates", "count": dupes})

    X = df.drop(columns=[target])
    y = df[target]

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
    Returns (preprocessor, operations_log).
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    ops: list[dict[str, Any]] = []
    transformers = []

    if numeric:
        missing = {c: int(X[c].isna().sum()) for c in numeric if X[c].isna().any()}
        if missing:
            ops.append({
                "type": "missing_values",
                "columns": list(missing.keys()),
                "strategy": "median",
            })
        ops.append({
            "type": "scaling",
            "columns": numeric,
            "method": "StandardScaler",
        })
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric))

    if categorical:
        missing = {c: int(X[c].isna().sum()) for c in categorical if X[c].isna().any()}
        if missing:
            ops.append({
                "type": "missing_values",
                "columns": list(missing.keys()),
                "strategy": "most_frequent",
            })
        ops.append({
            "type": "encoding",
            "columns": categorical,
            "method": "OneHotEncoder",
        })
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, ops