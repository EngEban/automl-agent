"""
Exploratory Data Analysis (EDA).
Generates analytical figures (base64) and a summary dict.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.core.data_profiler import is_date_like_series, profile_dataset

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.bbox"] = "tight"


def _to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------- Figure generators ----------
def _plot_missing(df: pd.DataFrame) -> str | None:
    m = df.isna().sum()
    m = m[m > 0].sort_values(ascending=False)
    if m.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, len(m) * 0.35)))
    sns.barplot(x=m.values, y=m.index, ax=ax, color="#e74c3c")
    ax.set_title("Missing values per column", fontsize=13)
    ax.set_xlabel("Count")
    return _to_b64(fig)


def _plot_numeric(df: pd.DataFrame, max_cols: int = 12) -> str | None:
    cols = df.select_dtypes(include="number").columns[:max_cols]
    if len(cols) == 0:
        return None
    n, cols_per_row = len(cols), 3
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            sns.histplot(values, kde=True, ax=ax, color="#3498db")
        ax.set_title(f"Distribution of {col}")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Numeric variable distributions", fontsize=14)
    fig.tight_layout()
    return _to_b64(fig)


def _plot_categorical(df: pd.DataFrame, max_cols: int = 9) -> str | None:
    cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns[:max_cols]
    if len(cols) == 0:
        return None
    n, cols_per_row = len(cols), 3
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts().head(10)
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="#9b59b6")
        ax.set_title(f"Distribution of {col}")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Categorical variable distributions", fontsize=14)
    fig.tight_layout()
    return _to_b64(fig)


def _plot_correlation(df: pd.DataFrame) -> str | None:
    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    size = min(12, 2 + corr.shape[0] * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, annot=corr.shape[0] <= 15, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix", fontsize=13)
    return _to_b64(fig)


def _plot_outliers(df: pd.DataFrame, max_cols: int = 9) -> str | None:
    cols = df.select_dtypes(include="number").columns[:max_cols]
    if len(cols) == 0:
        return None
    n, cols_per_row = len(cols), 3
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            sns.boxplot(x=values, ax=ax, color="#f39c12")
        ax.set_title(col)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Outlier detection (Boxplots)", fontsize=14)
    fig.tight_layout()
    return _to_b64(fig)


def _plot_target(
    df: pd.DataFrame, target: str | None, task_type: str | None = None
) -> str | None:
    if not target or target not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    is_classification = task_type == "classification" or (
        task_type is None
        and (
            pd.api.types.is_object_dtype(df[target].dtype)
            or isinstance(df[target].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[target].dtype)
        )
    )
    if is_classification:
        vc = df[target].value_counts().head(20)
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="#2ecc71")
        ax.set_title(f"Target distribution: {target}", fontsize=13)
        plt.xticks(rotation=45, ha="right")
    else:
        values = df[target]
        if pd.api.types.is_numeric_dtype(values.dtype):
            values = values.replace([np.inf, -np.inf], np.nan)
        sns.histplot(values.dropna(), kde=True, ax=ax, color="#2ecc71")
        ax.set_title(f"Target distribution: {target}", fontsize=13)
    return _to_b64(fig)


# ---------- Public API ----------
def run_eda(
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Run EDA and return {summary, figures}."""
    logger.info("EDA start: shape=%s", df.shape)

    figures = {
        "missing": _plot_missing(df),
        "numeric_dist": _plot_numeric(df),
        "categorical_dist": _plot_categorical(df),
        "correlation": _plot_correlation(df),
        "outliers": _plot_outliers(df),
        "target": _plot_target(df, target, task_type),
    }
    figures = {k: v for k, v in figures.items() if v}

    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    profile = profile_dataset(df, target=target, task_type=task_type)
    summary: dict[str, Any] = {
        "shape": list(df.shape),
        "numeric_count": int(numeric.shape[1]),
        "categorical_count": profile.categorical_count,
        "boolean_count": profile.boolean_count,
        "date_count": profile.date_count,
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_column": {
            c: int(v) for c, v in df.isna().sum().items() if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_stats": numeric.describe().round(3).to_dict() if not numeric.empty else {},
        "readiness": _assess_readiness(df),
        "target": target,
        "task_type": task_type,
        "profile": profile.model_dump(mode="json"),
        "important_findings": profile.warnings[:12],
    }
    logger.info("EDA done: %d figures", len(figures))
    return {"summary": summary, "figures": figures}


def _assess_readiness(df: pd.DataFrame) -> dict[str, Any]:
    total = df.shape[0] * df.shape[1]
    missing_pct = (df.isna().sum().sum() / total * 100) if total else 0
    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    categorical = df.select_dtypes(include=["object", "category", "bool", "string"])
    date_count = sum(1 for col in df.columns if is_date_like_series(df[col]))

    needs_scaling = False
    if not numeric.empty:
        ranges = (numeric.max() - numeric.min()).dropna()
        needs_scaling = bool((ranges > 100).any() or (numeric.std() > 10).any())

    return {
        "needs_cleaning": bool(missing_pct > 1 or df.duplicated().sum() > 0),
        "missing_percentage": round(float(missing_pct), 2),
        "needs_encoding": bool(not categorical.empty),
        "date_columns_present": date_count > 0,
        "needs_scaling": needs_scaling,
    }
