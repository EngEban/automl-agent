from __future__ import annotations
import logging
from typing import Any

import pandas as pd

from app.schemas.profile import DatasetProfile, ColumnProfile

logger = logging.getLogger(__name__)

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Analyze the dataset for quality, leakage, and structural issues."""
    n_rows = len(df)
    n_cols = len(df.columns)
    if n_rows == 0:
        raise ValueError("Cannot profile an empty dataset.")

    columns = []
    warnings = []
    
    # Metrics for quality score
    total_missing = 0
    high_cardinality_count = 0
    constant_count = 0

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        total_missing += missing
        missing_pct = (missing / n_rows) * 100
        unique_count = int(series.nunique(dropna=True))
        
        is_constant = unique_count <= 1
        is_high_card = unique_count > max(50, n_rows * 0.5)  # Heuristic
        # If unique count equals row count and it's not the target, it's likely an ID
        likely_id = (unique_count == n_rows) and (col.lower() not in ["target", "price", "species"])

        if is_constant:
            constant_count += 1
            warnings.append(f"Column '{col}' is constant (only {unique_count} unique value). Consider dropping it.")
        if likely_id:
            warnings.append(f"Column '{col}' looks like an ID column. It might cause overfitting.")
        if missing_pct > 30:
            warnings.append(f"Column '{col}' has a high missing rate ({missing_pct:.1f}%).")
        if is_high_card:
            high_cardinality_count += 1

        columns.append(ColumnProfile(
            name=str(col),
            dtype=str(series.dtype),
            missing_count=missing,
            missing_percentage=round(missing_pct, 2),
            unique_count=unique_count,
            is_constant=is_constant,
            is_high_cardinality=is_high_card,
            likely_id=likely_id,
        ))

    # Calculate Quality Score (0-100)
    missing_penalty = (total_missing / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    const_penalty = (constant_count / n_cols) * 100 if n_cols > 0 else 0
    card_penalty = (high_cardinality_count / n_cols) * 50 if n_cols > 0 else 0
    
    score = max(0.0, 100.0 - missing_penalty - const_penalty - card_penalty)

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_cols,
        columns=columns,
        quality_score=round(score, 1),
        quality_details={
            "missing_penalty": round(missing_penalty, 1),
            "constant_penalty": round(const_penalty, 1),
            "cardinality_penalty": round(card_penalty, 1),
        },
        warnings=warnings,
    )