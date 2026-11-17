"""Bounded dataset exploration and training-partition-only selector previews."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from app.core.feature_selection import TrainingFeatureSelector

from app.core.data_profiler import profile_dataset

MAX_PROFILE_ROWS = 5000
MAX_PROFILE_COLUMNS = 100
MAX_RELATIONSHIP_FEATURES = 30
MAX_SELECTION_ROWS = 5000
MAX_SELECTION_COLUMNS = 50
MAX_TRANSFORMED_SELECTION_FEATURES = 300
CORRELATION_THRESHOLD = 0.95


def _bounded_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=42).sort_index()


def _correlations(frame: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    numeric = [
        column for column in frame.columns
        if column != target_column
        if pd.api.types.is_numeric_dtype(frame[column].dtype)
        and not pd.api.types.is_bool_dtype(frame[column].dtype)
    ]
    omitted = max(0, len(numeric) - MAX_RELATIONSHIP_FEATURES)
    numeric = numeric[:MAX_RELATIONSHIP_FEATURES]
    if len(numeric) < 2:
        return {
            "features": [str(column) for column in numeric],
            "pearson": [],
            "spearman": [],
            "strong_pairs": [],
            "omitted_numeric_feature_count": omitted,
            "status": "insufficient_numeric_features",
        }

    values = frame[numeric].replace([np.inf, -np.inf], np.nan)
    matrices: dict[str, pd.DataFrame] = {}
    for method in ("pearson", "spearman"):
        matrices[method] = values.corr(method=method, min_periods=3)

    pairs = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1:]:
            pearson = matrices["pearson"].loc[left, right]
            spearman = matrices["spearman"].loc[left, right]
            coefficients = [float(value) for value in (pearson, spearman) if pd.notna(value)]
            strongest = max((abs(value) for value in coefficients), default=0.0)
            if strongest >= 0.70:
                pairs.append({
                    "feature_a": str(left),
                    "feature_b": str(right),
                    "pearson": float(pearson) if pd.notna(pearson) else None,
                    "spearman": float(spearman) if pd.notna(spearman) else None,
                    "strength": round(strongest, 4),
                })
    pairs.sort(key=lambda item: (-item["strength"], item["feature_a"], item["feature_b"]))

    def serialize(matrix: pd.DataFrame) -> list[list[float | None]]:
        return [
            [round(float(value), 4) if pd.notna(value) else None for value in row]
            for row in matrix.to_numpy()
        ]

    return {
        "features": [str(column) for column in numeric],
        "pearson": serialize(matrices["pearson"]),
        "spearman": serialize(matrices["spearman"]),
        "strong_pairs": pairs[:50],
        "omitted_numeric_feature_count": omitted,
        "status": "available",
        "notes": [
            "Pearson measures linear association; Spearman measures monotonic rank association.",
            "Pairwise complete numeric rows are used. Correlation does not imply causation.",
            "Categorical association statistics are not implemented; categories are summarized in the feature profile.",
        ],
    }


def _feature_for_transformed_name(transformed_name: str, source_columns: list[str]) -> str | None:
    name = transformed_name.split("__", 1)[-1]
    matches = [
        column for column in source_columns
        if name == column or name.startswith(f"{column}_") or name.startswith(f"{column}-")
    ]
    return max(matches, key=len) if matches else None


def _aggregate_scores(
    scores: np.ndarray,
    transformed_names: list[str],
    source_columns: list[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {column: [] for column in source_columns}
    for name, raw_score in zip(transformed_names, scores):
        column = _feature_for_transformed_name(str(name), source_columns)
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            continue
        if column is not None and np.isfinite(score):
            grouped[column].append(score)
    return {column: max(values) if values else 0.0 for column, values in grouped.items()}


def _ranked_features(scores: dict[str, float], order: list[str]) -> list[str]:
    positions = {feature: index for index, feature in enumerate(order)}
    return sorted(scores, key=lambda feature: (-scores[feature], positions[feature]))


def _method_result(
    *, name: str, method_type: str, description: str, requirements: str,
    selected: list[str], scores: dict[str, float], order: list[str],
    excluded: list[dict[str, str]] | None = None,
    status: str = "success", message: str | None = None,
    started: float | None = None, limitations: list[str] | None = None,
) -> dict[str, Any]:
    ranked = _ranked_features(scores, order)
    return {
        "name": name,
        "type": method_type,
        "description": description,
        "requirements": requirements,
        "status": status,
        "message": message,
        "selected_features": selected,
        "selected_count": len(selected),
        "ranking": [
            {"feature": feature, "score": round(float(scores[feature]), 6)}
            for feature in ranked
        ],
        "excluded_features": excluded or [],
        "duration_ms": round((time.perf_counter() - started) * 1000, 2) if started else None,
        "limitations": limitations or [],
    }


def _split_training_rows(
    frame: pd.DataFrame, target: str, task_type: str,
) -> tuple[pd.DataFrame, pd.Series, int]:
    from app.core.preprocessing import prepare_xy_for_training
    from app.core.trainer import _split_data

    X, y, _ = prepare_xy_for_training(frame, target)
    X_train, X_holdout, y_train, _ = _split_data(X, y, task_type)
    holdout_rows = int(len(X_holdout))
    if len(X_train) > MAX_SELECTION_ROWS:
        sampled = X_train.sample(n=MAX_SELECTION_ROWS, random_state=42)
        X_train = sampled
        y_train = y_train.loc[sampled.index]
    return X_train, y_train, holdout_rows


def _selection_comparison(
    frame: pd.DataFrame,
    target: str,
    task_type: str,
    max_features: int,
) -> dict[str, Any]:
    from app.config.settings import settings

    try:
        frame_memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    except (TypeError, ValueError, OverflowError):
        frame_memory_bytes = 0
    memory_limit_bytes = int(settings.DATASET_MEMORY_WARNING_MB) * 1024 * 1024
    if frame_memory_bytes > memory_limit_bytes:
        limit_mb = int(settings.DATASET_MEMORY_WARNING_MB)
        return {
            "status": "unavailable",
            "target_column": target,
            "task_type": task_type,
            "message": (
                "Selector comparison is limited to datasets whose estimated memory "
                f"does not exceed the configured {limit_mb} MiB warning threshold."
            ),
            "methods": [],
            "agreement": [],
            "holdout_rows_used_for_selection": 0,
        }

    try:
        X_train, y_train, holdout_rows = _split_training_rows(frame, target, task_type)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "target_column": target,
            "task_type": task_type,
            "message": str(exc),
            "methods": [],
            "agreement": [],
            "holdout_rows_used_for_selection": 0,
        }

    input_feature_labels = list(X_train.columns)
    input_features = [str(column) for column in input_feature_labels]
    omitted_input = input_features[MAX_SELECTION_COLUMNS:]
    X_train = X_train.iloc[:, :MAX_SELECTION_COLUMNS]
    input_feature_labels = list(X_train.columns)
    input_features = [str(column) for column in input_feature_labels]
    omitted_reasons = [
        {"feature": feature, "reason": "Feature-selection analysis is limited to the first 50 source features."}
        for feature in omitted_input
    ]

    try:
        from app.core.preprocessing import IntelligentPreprocessor

        preprocessor = IntelligentPreprocessor(input_feature_labels, scale_numeric=False)
        transformed = preprocessor.fit_transform(X_train, y_train)
        transformed_names = [str(value) for value in preprocessor.get_feature_names_out()]
    except (TypeError, ValueError, FloatingPointError) as exc:
        return {
            "status": "unavailable",
            "target_column": target,
            "task_type": task_type,
            "message": f"Training-only preprocessing could not prepare features: {type(exc).__name__}.",
            "methods": [],
            "agreement": [],
            "holdout_rows_used_for_selection": 0,
        }

    kept = [str(column) for column in getattr(preprocessor, "kept_columns_", [])]
    removed_constant = [str(column) for column in getattr(preprocessor, "removed_constant_columns_", [])]
    removed_duplicates = getattr(preprocessor, "removed_duplicate_columns_", [])
    excluded_base = [
        {"feature": column, "reason": "Constant or entirely missing in the training partition."}
        for column in removed_constant
    ] + [
        {
            "feature": str(item["column"]),
            "reason": f"Duplicate of '{item['duplicate_of']}' in the training partition.",
        }
        for item in removed_duplicates
    ] + omitted_reasons

    dimension_count = int(transformed.shape[1])
    method_rows: list[dict[str, Any]] = []

    def append_shared_result(
        *, name: str, method: str, method_type: str, description: str,
        requirements: str, matrix: Any, names: list[str], config: dict[str, Any],
        source_order: list[str], limitations: list[str],
    ) -> None:
        started = time.perf_counter()
        try:
            selector = TrainingFeatureSelector(method=method, task_type=task_type, **config)
            selector.fit(matrix, y_train)
            scores = _aggregate_scores(selector.scores_, names, source_order)
            selected_sources = {
                source for index in selector.selected_indices_
                if (source := _feature_for_transformed_name(names[int(index)], source_order))
            }
            ranked = _ranked_features(scores, source_order)
            selected = [feature for feature in ranked if feature in selected_sources][:max_features]
            selected_set = set(selected)
            ranked_excluded = [
                {"feature": feature, "reason": "No transformed dimension was retained in this training-only selector preview."}
                for feature in ranked if feature not in selected_set
            ]
            method_rows.append(_method_result(
                name=name, method_type=method_type, description=description,
                requirements=requirements, selected=selected, scores=scores,
                order=source_order, excluded=[*excluded_base, *ranked_excluded],
                started=started, limitations=limitations,
            ))
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            method_rows.append(_method_result(
                name=name, method_type=method_type, description=description,
                requirements=requirements, selected=[], scores={}, order=source_order,
                excluded=excluded_base, status="failed",
                message=f"Could not calculate this selector ({type(exc).__name__}).",
                started=started,
            ))

    append_shared_result(
        name="Constant / zero-variance filter", method="variance_filter",
        method_type="filter",
        description="Removes transformed dimensions below the configured training-partition variance threshold.",
        requirements="Training-partition feature values only.",
        matrix=transformed, names=transformed_names,
        config={"threshold": 1e-12, "min_features": 1}, source_order=kept,
        limitations=["This method is a basic quality filter and does not use the target."],
    )

    numeric_labels = [
        column for column in X_train.columns
        if str(column) in kept
        and pd.api.types.is_numeric_dtype(X_train[column].dtype)
        and not pd.api.types.is_bool_dtype(X_train[column].dtype)
    ]
    numeric = X_train[numeric_labels].replace([np.inf, -np.inf], np.nan)
    numeric.columns = [str(column) for column in numeric.columns]
    numeric = numeric.fillna(numeric.median()).dropna(axis=1, how="all")
    if numeric.shape[1]:
        append_shared_result(
            name="Correlation filter", method="correlation_filter",
            method_type="filter",
            description=f"Greedily removes redundant numeric source features at |Pearson r| ≥ {CORRELATION_THRESHOLD:.2f}.",
            requirements="At least one numeric source feature; correlation is computed on training rows only.",
            matrix=numeric.to_numpy(dtype=float), names=list(numeric.columns),
            config={"threshold": CORRELATION_THRESHOLD, "top_k": max_features, "min_features": 1},
            source_order=list(numeric.columns),
            limitations=["Categorical features are excluded; correlation is not applied to category codes.", "The ranking value is raw feature variance, so its magnitude depends on feature units."],
        )
    else:
        method_rows.append(_method_result(
            name="Correlation filter", method_type="filter",
            description="Greedily removes redundant numeric source features.",
            requirements="At least one numeric source feature.", selected=[], scores={},
            order=kept, excluded=excluded_base, status="skipped",
            message="No numeric source features are available.",
        ))

    scored_specs = (
        ("Mutual information", "mutual_information", "filter", "Ranks features by mutual information with the target."),
        ("Univariate statistical test", "univariate_test", "filter", "Uses ANOVA F for classification or F regression for regression."),
        ("Extra Trees importance", "extra_trees", "embedded", "Fits a bounded Extra Trees estimator and ranks impurity importance."),
    )
    if dimension_count > MAX_TRANSFORMED_SELECTION_FEATURES:
        for name, _method, method_type, description in scored_specs:
            method_rows.append(_method_result(
                name=name, method_type=method_type, description=description,
                requirements="At most 300 transformed dimensions; classification or numeric regression target.",
                selected=[], scores={}, order=kept, excluded=excluded_base,
                status="skipped",
                message=f"Preprocessing produced {dimension_count} encoded features; score-based methods are capped at {MAX_TRANSFORMED_SELECTION_FEATURES}.",
                limitations=["No score was computed."],
            ))
    else:
        for name, method, method_type, description in scored_specs:
            append_shared_result(
                name=name, method=method, method_type=method_type,
                description=description,
                requirements="Classification or numeric regression target.",
                matrix=transformed, names=transformed_names,
                config={
                    "top_k": min(max_features, max(1, dimension_count)),
                    "min_features": 1,
                    **({"n_estimators": 100} if method == "extra_trees" else {}),
                },
                source_order=kept,
                limitations=["Encoded category scores are combined by taking the maximum dimension score per source feature."],
            )

    frequencies: dict[str, int] = {}
    successful = [method for method in method_rows if method["status"] == "success"]
    for method in successful:
        for feature in method["selected_features"]:
            frequencies[feature] = frequencies.get(feature, 0) + 1
    agreement = [
        {"feature": feature, "method_count": count, "method_total": len(successful)}
        for feature, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]
    return {
        "status": "success" if successful else "unavailable",
        "target_column": target,
        "task_type": task_type,
        "holdout_rows_reserved": holdout_rows,
        "holdout_rows_used_for_selection": 0,
        "training_rows_used": int(len(X_train)),
        "max_features_per_ranked_method": max_features,
        "methods": method_rows,
        "agreement": agreement,
        "notes": [
            "This is an exploratory preview and does not change a training run or the source dataset.",
            "The holdout partition is reserved before scoring; every learned selector is fitted using training rows only.",
            "Scores have method-specific meanings and must not be compared as if they shared one scale.",
            "Cross-validation training integration remains separate; this preview does not alter existing CV behavior.",
        ],
    }


def explore_features(
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    task_type: str | None = None,
    max_features: int = 10,
) -> dict[str, Any]:
    """Return bounded descriptive data and an optional leakage-safe comparison."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Cannot explore an empty dataset.")
    if target_column is not None and target_column not in frame.columns:
        raise ValueError("The selected target column is not present in this dataset.")
    if task_type not in {None, "classification", "regression"}:
        raise ValueError("Task type must be classification or regression.")
    if not 1 <= max_features <= 50:
        raise ValueError("max_features must be between 1 and 50.")

    profiled_frame = _bounded_sample(frame, MAX_PROFILE_ROWS)
    profile = profile_dataset(profiled_frame, target=target_column, task_type=task_type)
    all_profile_columns = profile.columns
    profile.columns = all_profile_columns[:MAX_PROFILE_COLUMNS]
    correlation = _correlations(profiled_frame, target_column)
    if len(frame.columns) > MAX_PROFILE_COLUMNS:
        profile.warnings.append(
            f"The overview is capped at {MAX_PROFILE_COLUMNS} features; {len(frame.columns) - MAX_PROFILE_COLUMNS} additional features are omitted from this response."
        )
    if len(frame) > MAX_PROFILE_ROWS:
        profile.warnings.append(
            f"Descriptive statistics use a deterministic sample of {MAX_PROFILE_ROWS:,} of {len(frame):,} rows."
        )

    selection: dict[str, Any]
    if target_column is None:
        selection = {
            "status": "requires_target",
            "message": "Select a target and task type to compare feature-selection methods.",
            "methods": [],
            "agreement": [],
            "holdout_rows_used_for_selection": 0,
        }
    elif task_type is None:
        selection = {
            "status": "requires_task_type",
            "target_column": target_column,
            "message": "Choose classification or regression to compare feature-selection methods.",
            "methods": [],
            "agreement": [],
            "holdout_rows_used_for_selection": 0,
        }
    else:
        selection = _selection_comparison(frame, target_column, task_type, max_features)

    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "profiled_rows": int(len(profiled_frame)),
        "sampled": bool(len(frame) > MAX_PROFILE_ROWS),
        "profile": profile,
        "relationships": correlation,
        "selection_comparison": selection,
    }
