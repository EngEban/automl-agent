"""Training-fold feature selectors shared with exploratory feature analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression


METHODS = {
    "variance_filter",
    "correlation_filter",
    "mutual_information",
    "univariate_test",
    "extra_trees",
}
MAX_SCORED_FEATURES = 300


class FeatureSelectionError(ValueError):
    """A requested selector could not be applied safely to training data."""


def _as_score_array(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(scores).all():
        finite = scores[np.isfinite(scores)]
        replacement = float(finite.max()) if finite.size else 0.0
        scores = np.nan_to_num(scores, nan=0.0, posinf=replacement, neginf=0.0)
    return scores


def _moments(X: Any) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        second = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    else:
        values = np.asarray(X, dtype=float)
        means = np.mean(values, axis=0)
        second = np.mean(values * values, axis=0)
    return means, np.maximum(0.0, second - means * means)


class TrainingFeatureSelector(BaseEstimator, TransformerMixin):
    """Select preprocessed numeric dimensions using only the rows passed to fit."""

    def __init__(
        self,
        method: str,
        task_type: str,
        threshold: float | None = None,
        top_k: int | None = None,
        min_features: int = 1,
        n_estimators: int = 50,
        memory_limit_mb: float | None = None,
    ) -> None:
        self.method = method
        self.task_type = task_type
        self.threshold = threshold
        self.top_k = top_k
        self.min_features = min_features
        self.n_estimators = n_estimators
        self.memory_limit_mb = memory_limit_mb

    def fit(self, X, y=None):
        if self.method not in METHODS:
            raise FeatureSelectionError(f"Unsupported feature-selection method '{self.method}'.")
        if self.task_type not in {"classification", "regression"}:
            raise FeatureSelectionError("Feature selection requires a resolved classification or regression task.")
        if X is None or len(getattr(X, "shape", ())) != 2 or X.shape[1] == 0:
            raise FeatureSelectionError("Feature selection received no usable transformed features.")
        if y is None or len(y) != X.shape[0]:
            raise FeatureSelectionError("Feature selection requires one training target per training row.")

        self.n_features_in_ = int(X.shape[1])
        self.n_samples_fit_ = int(X.shape[0])
        if self.min_features < 1:
            raise FeatureSelectionError("min_features must be at least 1.")
        if self.method in {"mutual_information", "univariate_test", "extra_trees"} and self.n_features_in_ > MAX_SCORED_FEATURES:
            raise FeatureSelectionError(
                f"This method is limited to {MAX_SCORED_FEATURES} transformed features; "
                f"preprocessing produced {self.n_features_in_}. Reduce source features or category cardinality."
            )
        if self.method == "correlation_filter" and self.n_features_in_ > MAX_SCORED_FEATURES:
            raise FeatureSelectionError(
                f"Correlation filtering is limited to {MAX_SCORED_FEATURES} transformed features; "
                f"preprocessing produced {self.n_features_in_}. Reduce source features or category cardinality."
            )

        _, variances = _moments(X)
        active = variances > 1e-12
        if not np.any(active):
            raise FeatureSelectionError("All transformed features are constant in this training fold.")
        self.scores_ = np.zeros(self.n_features_in_, dtype=float)
        self.scores_[active] = variances[active]
        self.excluded_reasons_ = {
            int(index): "Constant or zero-variance in this training partition."
            for index in np.flatnonzero(~active)
        }

        try:
            if self.method == "variance_filter":
                threshold = 0.0 if self.threshold is None else float(self.threshold)
                selected = active & (variances > threshold)
                for index in np.flatnonzero(active & ~selected):
                    self.excluded_reasons_[int(index)] = (
                        f"Training-partition variance did not exceed threshold {threshold:g}."
                    )
            elif self.method == "correlation_filter":
                selected = self._correlation_mask(X, variances, active)
            else:
                active_indices = np.flatnonzero(active)
                X_active = X[:, active_indices]
                y_values = np.asarray(y)
                if self.method == "mutual_information":
                    selected_scores = self._mutual_information_scores(X_active, y_values)
                elif self.method == "univariate_test":
                    if self.task_type == "classification":
                        selected_scores = f_classif(X_active, y_values)[0]
                    else:
                        selected_scores = f_regression(X_active, y_values.astype(float))[0]
                    selected_scores = _as_score_array(selected_scores)
                else:
                    selected_scores = self._extra_trees_scores(X_active, y_values)
                self.scores_[active_indices] = selected_scores
                selected = self._top_k_mask(active_indices, selected_scores)
        except FeatureSelectionError:
            raise
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise FeatureSelectionError(
                f"The {self.method} selector could not safely process this training fold: {exc}"
            ) from exc

        selected_indices = np.flatnonzero(selected)
        if self.top_k is not None and self.method in {"variance_filter", "correlation_filter"}:
            order = self._ranked_indices(selected_indices)
            selected_indices = order[: int(self.top_k)]
        if len(selected_indices) < self.min_features:
            raise FeatureSelectionError(
                f"The {self.method} selector retained {len(selected_indices)} features, below "
                f"min_features={self.min_features}. Relax the threshold or increase the available features."
            )
        if not len(selected_indices):
            raise FeatureSelectionError("The selected configuration removed every transformed feature.")

        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[selected_indices] = True
        self.selected_indices_ = np.sort(selected_indices)
        self.n_features_selected_ = int(len(selected_indices))
        self.requested_top_k_ = self.top_k
        self.effective_top_k_ = self.n_features_selected_ if self.top_k is not None else None
        for index in np.flatnonzero(~self.support_):
            self.excluded_reasons_.setdefault(
                int(index), f"Ranked below the requested configuration for {self.method}."
            )
        return self

    def _ranked_indices(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(sorted(indices.tolist(), key=lambda index: (-self.scores_[index], index)), dtype=int)

    def _top_k_mask(self, active_indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
        order = np.asarray(
            sorted(range(len(active_indices)), key=lambda pos: (-scores[pos], int(active_indices[pos]))),
            dtype=int,
        )
        count = min(int(self.top_k), len(order)) if self.top_k is not None else len(order)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[active_indices[order[:count]]] = True
        return mask

    def _correlation_mask(self, X, variances: np.ndarray, active: np.ndarray) -> np.ndarray:
        threshold = 0.95 if self.threshold is None else float(self.threshold)
        indices = np.flatnonzero(active)
        if len(indices) > MAX_SCORED_FEATURES:
            raise FeatureSelectionError(
                f"Correlation filtering is limited to {MAX_SCORED_FEATURES} nonconstant features; "
                f"found {len(indices)}."
            )
        if sparse.issparse(X):
            gram = X[:, indices].T @ X[:, indices]
            gram = gram.toarray() if sparse.issparse(gram) else np.asarray(gram)
        else:
            values = np.asarray(X, dtype=float)[:, indices]
            gram = values.T @ values
        means, _ = _moments(X[:, indices])
        covariance = gram / max(1, self.n_samples_fit_) - np.outer(means, means)
        active_variances = variances[indices]
        denominator = np.sqrt(np.outer(active_variances, active_variances))
        correlation = np.divide(
            np.abs(covariance), denominator,
            out=np.zeros_like(covariance, dtype=float), where=denominator > 0,
        )
        ranked_local = sorted(range(len(indices)), key=lambda local: (-active_variances[local], int(indices[local])))
        kept_local: list[int] = []
        for local in ranked_local:
            if any(correlation[local, prior] >= threshold for prior in kept_local):
                self.excluded_reasons_[int(indices[local])] = (
                    f"Absolute training-partition correlation is at least {threshold:.3f} "
                    "with a higher-ranked feature."
                )
            else:
                kept_local.append(local)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices[kept_local]] = True
        return mask

    def _mutual_information_scores(self, X, y: np.ndarray) -> np.ndarray:
        from app.config.settings import settings

        itemsize = np.dtype(np.float64).itemsize
        required_bytes = int(X.shape[0]) * int(X.shape[1]) * itemsize
        limit_mb = (
            float(self.memory_limit_mb)
            if self.memory_limit_mb is not None
            else float(settings.MAX_DENSE_CONVERSION_MB)
        )
        if required_bytes > limit_mb * 1024 * 1024:
            raise FeatureSelectionError(
                "Mutual-information selection needs a dense training-fold matrix of "
                f"{required_bytes / (1024 ** 2):.1f} MiB, above the configured {limit_mb:g} MiB "
                "dense-conversion limit. Choose another selector or reduce rows/features."
            )
        values = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        discrete = self._discrete_columns(X)
        from app.config.settings import settings

        if self.task_type == "classification":
            scores = mutual_info_classif(values, y, discrete_features=discrete, random_state=settings.RANDOM_STATE)
        else:
            scores = mutual_info_regression(values, y.astype(float), discrete_features=discrete, random_state=settings.RANDOM_STATE)
        return _as_score_array(scores)

    @staticmethod
    def _discrete_columns(X) -> np.ndarray:
        """Infer low-cardinality dimensions from this training fold only."""
        discrete = np.zeros(X.shape[1], dtype=bool)
        for index in range(X.shape[1]):
            if sparse.issparse(X):
                column = X.getcol(index).tocoo()
                values = column.data[column.data != 0]
                includes_zero = column.nnz < X.shape[0]
            else:
                values = np.asarray(X)[:, index]
                includes_zero = bool(np.any(values == 0))
            discrete[index] = len(np.unique(values)) + int(includes_zero) <= 20
        return discrete

    def _extra_trees_scores(self, X, y: np.ndarray) -> np.ndarray:
        from app.config.settings import settings

        if self.task_type == "classification":
            estimator = ExtraTreesClassifier(
                n_estimators=int(self.n_estimators), max_depth=12,
                random_state=settings.RANDOM_STATE, n_jobs=1,
                class_weight="balanced",
            )
        else:
            estimator = ExtraTreesRegressor(
                n_estimators=int(self.n_estimators), max_depth=12,
                random_state=settings.RANDOM_STATE, n_jobs=1,
            )
        estimator.fit(X, y)
        return _as_score_array(estimator.feature_importances_)

    def transform(self, X):
        if not hasattr(self, "support_"):
            raise FeatureSelectionError("The feature selector must be fitted before transformation.")
        if X.shape[1] != self.n_features_in_:
            raise FeatureSelectionError(
                "The transformed feature count changed after selector fitting; use the fitted preprocessing pipeline."
            )
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        if not hasattr(self, "support_"):
            raise FeatureSelectionError("The feature selector has not been fitted.")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        names = input_features
        if names is None:
            names = getattr(self, "feature_names_in_", None)
        if names is None:
            raise FeatureSelectionError("Fitted transformed feature names are not available.")
        names = np.asarray(names, dtype=object)
        if len(names) != self.n_features_in_:
            raise FeatureSelectionError("The transformed feature names do not match the fitted selector.")
        return names[self.support_]


def selector_config_dict(config: Any) -> dict[str, Any] | None:
    """Return validated, serialization-safe selector settings or None."""
    if config is None:
        return None
    from app.schemas.feature_selection import FeatureSelectionConfig

    try:
        parsed = FeatureSelectionConfig.model_validate(config)
    except Exception as exc:
        raise FeatureSelectionError(f"Invalid feature-selection configuration: {exc}") from exc
    return parsed.model_dump(exclude_none=True)
