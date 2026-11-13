"""Small memory estimators and a guarded sparse-to-dense pipeline step."""
from __future__ import annotations

import sys
import math
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from app.config import settings

_BYTES_PER_MB = 1024 * 1024


class DenseConversionLimitError(RuntimeError):
    """A sparse-to-dense conversion would exceed the configured memory limit."""


def _dimension(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _itemsize(dtype: Any) -> int:
    try:
        size = int(np.dtype(dtype).itemsize)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported numeric dtype: {dtype!r}.") from exc
    if size <= 0:
        raise ValueError("dtype must have a positive item size.")
    return size


def _checked_sum(parts: list[int]) -> int:
    total = sum(int(part) for part in parts)
    if total > sys.maxsize:
        raise OverflowError("Estimated memory size exceeds the platform addressable range.")
    return total


def estimate_dense_matrix_bytes(rows: int, columns: int, dtype: Any = np.float64) -> int:
    """Estimate dense array storage, rejecting arithmetic beyond sys.maxsize."""
    row_count = _dimension("rows", rows)
    column_count = _dimension("columns", columns)
    itemsize = _itemsize(dtype)
    cells = row_count * column_count
    if cells > sys.maxsize // itemsize:
        raise OverflowError("Estimated dense matrix size exceeds the platform addressable range.")
    return cells * itemsize


def estimate_sparse_matrix_bytes(
    nonzero_count: int,
    shape: tuple[int, int],
    dtype: Any = np.float64,
    index_dtype: Any = np.int32,
    format: str = "csr",
) -> int:
    """Estimate CSR or CSC data, index, and pointer-array storage."""
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape must be a (rows, columns) tuple.")
    rows = _dimension("rows", shape[0])
    columns = _dimension("columns", shape[1])
    nnz = _dimension("nonzero_count", nonzero_count)
    if nnz > rows * columns:
        raise ValueError("nonzero_count cannot exceed the number of matrix cells.")
    fmt = str(format).lower()
    if fmt not in {"csr", "csc"}:
        raise ValueError("Only CSR and CSC sparse formats are supported.")
    value_size = _itemsize(dtype)
    index_size = _itemsize(index_dtype)
    pointer_count = rows + 1 if fmt == "csr" else columns + 1
    return _checked_sum([
        nnz * value_size,
        nnz * index_size,
        pointer_count * index_size,
    ])


def estimate_sparse_storage_bytes(matrix: Any) -> int:
    """Estimate actual CSR/CSC array storage without materializing dense values."""
    if not sparse.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix.")
    fmt = getattr(matrix, "format", "")
    if fmt not in {"csr", "csc"}:
        raise ValueError(f"Unsupported sparse format {fmt!r}; expected CSR or CSC.")
    return _checked_sum([
        int(matrix.data.nbytes),
        int(matrix.indices.nbytes),
        int(matrix.indptr.nbytes),
    ])


def estimate_dataframe_memory(df: pd.DataFrame) -> dict[str, int | float]:
    """Estimate DataFrame storage, including object values and the index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    usage = df.memory_usage(index=True, deep=True)
    byte_count = _checked_sum([int(value) for value in usage.tolist()])
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "bytes": byte_count,
        "megabytes": round(byte_count / _BYTES_PER_MB, 3),
    }


def dense_conversion_limit_bytes(max_memory_mb: int | float | None = None) -> int:
    limit_mb = settings.MAX_DENSE_CONVERSION_MB if max_memory_mb is None else max_memory_mb
    if (
        isinstance(limit_mb, bool)
        or not isinstance(limit_mb, (int, float))
        or not math.isfinite(limit_mb)
        or limit_mb <= 0
    ):
        raise ValueError("Dense conversion memory limit must be a positive number of MB.")
    limit_bytes = int(limit_mb * _BYTES_PER_MB)
    if limit_bytes > sys.maxsize:
        raise OverflowError("Dense conversion memory limit exceeds the platform addressable range.")
    return limit_bytes


def _format_megabytes(byte_count: int) -> str:
    return f"{byte_count / _BYTES_PER_MB:.1f} MiB"


def sparse_to_dense_checked(matrix: Any, max_memory_mb: int | float | None = None):
    """Convert sparse input only after checking the full dense allocation size."""
    if not sparse.issparse(matrix):
        return matrix
    rows, columns = matrix.shape
    required = estimate_dense_matrix_bytes(rows, columns, matrix.dtype)
    limit = dense_conversion_limit_bytes(max_memory_mb)
    if required > limit:
        raise DenseConversionLimitError(
            f"This estimator requires dense input, estimated at {_format_megabytes(required)} "
            f"for {rows:,} rows × {columns:,} encoded features. The configured limit is "
            f"{_format_megabytes(limit)} (MAX_DENSE_CONVERSION_MB). Use a sparse-compatible "
            "candidate, reduce dataset rows or categorical features, or raise the limit "
            "after checking available memory."
        )
    return matrix.toarray()


class SparseToDenseGuard(BaseEstimator, TransformerMixin):
    """Pipeline adapter that preflights dense allocations before model fitting."""

    def __init__(self, max_memory_mb: int | float | None = None, projected_rows: int | None = None):
        self.max_memory_mb = max_memory_mb
        self.projected_rows = projected_rows

    def fit(self, X, y=None):
        if sparse.issparse(X):
            rows, columns = X.shape
            projected_rows = rows if self.projected_rows is None else max(
                rows, _dimension("projected_rows", self.projected_rows)
            )
            required = estimate_dense_matrix_bytes(projected_rows, columns, X.dtype)
            limit = dense_conversion_limit_bytes(self.max_memory_mb)
            if required > limit:
                raise DenseConversionLimitError(
                    f"This estimator needs a dense matrix estimated at {_format_megabytes(required)} "
                    f"for up to {projected_rows:,} rows × {columns:,} encoded features; the "
                    f"configured limit is {_format_megabytes(limit)} "
                    "(MAX_DENSE_CONVERSION_MB). Use a sparse-compatible candidate or reduce "
                    "the dataset size or categorical feature count."
                )
            self.projected_dense_bytes_ = required
        else:
            self.projected_dense_bytes_ = 0
        return self

    def transform(self, X):
        return sparse_to_dense_checked(X, self.max_memory_mb)
