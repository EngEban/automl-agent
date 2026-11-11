from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.profile import ColumnProfile, DatasetProfile

logger = logging.getLogger(__name__)

RARE_CATEGORY_FRACTION = 0.05
NEAR_CONSTANT_FRACTION = 0.99
_DATE_NAME_PARTS = {"date", "datetime", "time", "timestamp", "created", "updated", "birth"}
_ID_NAME_PARTS = {"id", "uuid", "guid", "key", "index", "identifier"}


def is_date_like_series(series: pd.Series) -> bool:
    """Detect reliably parseable text dates; numeric measurements stay numeric."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        return False
    values = series.dropna()
    if len(values) < 3:
        return False
    text = values.astype("string").str.strip()
    text = text[text.notna() & text.ne("")]
    if len(text) < 3:
        return False

    name_parts = set(re.findall(r"[a-z0-9]+", str(series.name).casefold()))
    named_date = bool(name_parts & _DATE_NAME_PARTS)
    has_date_marker = text.str.contains(r"[-/:T]|[A-Za-z]{3,}", regex=True).mean()
    # Reject numeric strings such as "1", "2", "3" even if pandas can parse
    # them as timestamps. A date-like name is useful evidence, not a bypass for
    # successful parsing on the training rows.
    if has_date_marker < (0.5 if named_date else 0.8):
        return False
    try:
        parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(parsed.notna().mean() >= (0.6 if named_date else 0.8))


def _numeric_summary(series: pd.Series) -> tuple[dict[str, Any], int]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {}, 0
    q1, q3 = numeric.quantile([0.25, 0.75]).tolist()
    iqr = q3 - q1
    outlier_count = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
    skew = numeric.skew() if len(numeric) >= 3 else np.nan
    values = {
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "standard_deviation": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "skewness": float(skew) if pd.notna(skew) and np.isfinite(skew) else None,
    }
    return values, outlier_count


def _name_looks_like_id(name: Any) -> bool:
    parts = set(re.findall(r"[a-z0-9]+", str(name).casefold()))
    return bool(parts & _ID_NAME_PARTS)


def _profile_target(df: pd.DataFrame, target: str, task_type: str | None) -> dict[str, Any]:
    series = df[target]
    missing = int(series.isna().sum())
    result: dict[str, Any] = {
        "column": str(target),
        "task_type": task_type,
        "missing_count": missing,
        "missing_percentage": round(missing / max(1, len(df)) * 100, 2),
    }
    if task_type == "classification":
        counts = series.dropna().value_counts(dropna=False)
        total = int(counts.sum())
        ordered = counts.sort_values(ascending=False)
        distribution = [
            {
                "label": str(label),
                "count": int(count),
                "percentage": round(float(count) / max(1, total) * 100, 2),
            }
            for label, count in ordered.head(20).items()
        ]
        if len(ordered) > 20:
            remainder = int(ordered.iloc[20:].sum())
            distribution.append({"label": "Other classes", "count": remainder,
                                 "percentage": round(remainder / max(1, total) * 100, 2)})
        majority = float(ordered.iloc[0] / total * 100) if total else 0.0
        minority = float(ordered.iloc[-1] / total * 100) if total else 0.0
        ratio = float(ordered.iloc[0] / ordered.iloc[-1]) if len(ordered) and ordered.iloc[-1] else None
        result.update({
            "class_count": int(len(counts)),
            "class_distribution": distribution,
            "majority_class_percentage": round(majority, 2),
            "minority_class_percentage": round(minority, 2),
            "imbalance_ratio": round(ratio, 3) if ratio is not None else None,
            "is_imbalanced": bool(len(ordered) > 1 and (ratio >= 3 or minority < 20)),
        })
    elif task_type == "regression":
        statistics, outlier_count = _numeric_summary(series)
        result.update({"distribution_statistics": statistics, "outlier_count": outlier_count})
    return result


def profile_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str | None = None,
) -> DatasetProfile:
    """Build a descriptive, full-dataset profile without changing training data."""
    n_rows, n_cols = len(df), len(df.columns)
    if n_rows == 0:
        raise ValueError("Cannot profile an empty dataset.")

    columns: list[ColumnProfile] = []
    warnings: list[str] = []
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    missing_columns: list[str] = []
    high_cardinality_count = constant_count = 0
    numeric_count = categorical_count = boolean_count = date_count = 0
    seen: list[Any] = []

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = missing / n_rows * 100
        unique_count = int(series.nunique(dropna=True))
        cardinality_pct = unique_count / n_rows * 100
        is_bool = pd.api.types.is_bool_dtype(series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype) and not is_bool
        is_date = is_date_like_series(series)
        is_categorical = (
            pd.api.types.is_object_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ) and not is_date

        if is_bool:
            detected_type = "boolean"
            boolean_count += 1
        elif is_date:
            detected_type = "date"
            date_count += 1
        elif is_numeric:
            detected_type = "numeric"
            numeric_count += 1
        elif is_categorical:
            detected_type = "categorical"
            categorical_count += 1
        else:
            detected_type = "other"

        counts = series.value_counts(dropna=True)
        dominant_fraction = float(counts.iloc[0] / max(1, counts.sum())) if len(counts) else 0.0
        is_constant = unique_count <= 1
        near_constant = bool(unique_count > 1 and dominant_fraction >= NEAR_CONSTANT_FRACTION)
        likely_id = bool(
            _name_looks_like_id(col)
            or (n_rows >= 20 and cardinality_pct >= 98 and not is_date)
        )
        is_high_cardinality = bool(
            unique_count > 50 and (is_categorical or likely_id)
        )
        duplicate_of = next((prior for prior in seen if series.equals(df[prior])), None)
        seen.append(col)
        issues: list[str] = []
        if missing_pct >= 30:
            issues.append(f"High missingness ({missing_pct:.1f}%).")
            missing_columns.append(str(col))
        if is_constant:
            issues.append("Constant or entirely missing feature.")
            constant_count += 1
        if near_constant:
            issues.append("Near-constant values; review for low information.")
        if likely_id:
            issues.append("May be an identifier; retained unless training data shows it is redundant.")
        if is_high_cardinality:
            issues.append("High cardinality; rare values are grouped by the fitted encoder.")
            high_cardinality_count += 1
        if is_date:
            issues.append("Date-like values; training pipeline extracts calendar features.")
        if duplicate_of is not None:
            issues.append(f"Duplicates feature column '{duplicate_of}'.")

        numeric_stats = outlier_count = None
        if is_numeric:
            numeric_stats, outlier_count = _numeric_summary(series)
            if outlier_count:
                issues.append(f"{outlier_count} potential numeric outlier(s) by the 1.5×IQR rule.")

        category_frequency: list[dict[str, Any]] = []
        top_categories: list[dict[str, Any]] = []
        rare_category_count = None
        if is_categorical or is_bool:
            nonmissing_count = int(series.notna().sum())
            rare_category_count = int((counts < max(1, int(np.ceil(nonmissing_count * RARE_CATEGORY_FRACTION)))).sum())
            for value, count in counts.head(10).items():
                item = {
                    "category": str(value),
                    "count": int(count),
                    "percentage": round(float(count) / max(1, nonmissing_count) * 100, 2),
                }
                category_frequency.append(item)
                top_categories.append(dict(item))
            if rare_category_count:
                issues.append(f"{rare_category_count} category value(s) occur in under 5% of nonmissing rows.")

        columns.append(ColumnProfile(
            name=str(col), dtype=str(series.dtype), missing_count=missing,
            missing_percentage=round(missing_pct, 2), unique_count=unique_count,
            is_constant=is_constant, is_high_cardinality=is_high_cardinality,
            likely_id=likely_id, detected_type=detected_type,
            cardinality_percentage=round(cardinality_pct, 2),
            is_near_constant=near_constant, is_date_like=is_date,
            duplicate_of=str(duplicate_of) if duplicate_of is not None else None,
            potential_issues=issues, numeric_statistics=numeric_stats,
            outlier_count=outlier_count, top_categories=top_categories,
            category_frequency=category_frequency,
            rare_category_count=rare_category_count,
        ))

        if is_constant:
            warnings.append(f"Column '{col}' is constant or entirely missing.")
        elif near_constant:
            warnings.append(f"Column '{col}' is near-constant and may carry little information.")
        if likely_id:
            warnings.append(f"Column '{col}' may be an identifier.")
        if missing_pct >= 30:
            warnings.append(f"Column '{col}' has high missingness ({missing_pct:.1f}%).")
        if is_high_cardinality:
            warnings.append(f"Column '{col}' has high cardinality ({unique_count} values).")
        if duplicate_of is not None:
            warnings.append(f"Column '{col}' duplicates '{duplicate_of}'.")
        if is_date:
            warnings.append(f"Column '{col}' contains reliably parseable date values.")
        if outlier_count:
            warnings.append(
                f"Column '{col}' has {outlier_count} potential numeric outlier(s) by the 1.5×IQR rule."
            )
        if rare_category_count:
            warnings.append(
                f"Column '{col}' has {rare_category_count} rare category value(s) below 5% frequency."
            )

    if duplicate_rows:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate row(s).")
    missing_pct = total_missing / max(1, n_rows * n_cols) * 100
    quality_score = max(
        0.0,
        100.0 - missing_pct - (constant_count / max(1, n_cols) * 100)
        - (high_cardinality_count / max(1, n_cols) * 50),
    )
    try:
        memory_usage = int(df.memory_usage(index=True, deep=True).sum())
    except (TypeError, ValueError, OverflowError):
        memory_usage = None

    target_info = None
    if target is not None and target in df.columns:
        target_info = _profile_target(df, target, task_type)
        if target_info.get("is_imbalanced"):
            warnings.insert(0,
                f"Target '{target}' is imbalanced: majority {target_info['majority_class_percentage']:.1f}%, "
                f"minority {target_info['minority_class_percentage']:.1f}%."
            )

    return DatasetProfile(
        n_rows=n_rows, n_columns=n_cols, columns=columns,
        quality_score=round(quality_score, 1),
        quality_details={
            "missing_penalty": round(missing_pct, 1),
            "constant_penalty": round(constant_count / max(1, n_cols) * 100, 1),
            "cardinality_penalty": round(high_cardinality_count / max(1, n_cols) * 50, 1),
        },
        warnings=warnings, numeric_count=numeric_count,
        categorical_count=categorical_count, boolean_count=boolean_count,
        date_count=date_count, missing_count=total_missing,
        missing_percentage=round(missing_pct, 2),
        duplicate_row_count=duplicate_rows, memory_usage_bytes=memory_usage,
        target_info=target_info,
    )
