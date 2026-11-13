from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from app.core.memory import (
    DenseConversionLimitError,
    SparseToDenseGuard,
    dense_conversion_limit_bytes,
    estimate_dataframe_memory,
    estimate_dense_matrix_bytes,
    estimate_sparse_matrix_bytes,
    estimate_sparse_storage_bytes,
    sparse_to_dense_checked,
)


def test_dense_matrix_estimate_respects_dtype_and_zero_dimensions():
    assert estimate_dense_matrix_bytes(5, 4, np.float32) == 80
    assert estimate_dense_matrix_bytes(5, 4, np.float64) == 160
    assert estimate_dense_matrix_bytes(0, 4, np.float64) == 0


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_matrix_estimate_matches_data_index_and_pointer_storage(fmt):
    matrix = sparse.csr_matrix([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
    if fmt == "csc":
        matrix = matrix.tocsc()
    expected = (
        3 * np.dtype(np.float32).itemsize
        + 3 * matrix.indices.dtype.itemsize
        + (matrix.shape[0 if fmt == "csr" else 1] + 1) * matrix.indptr.dtype.itemsize
    )
    assert estimate_sparse_matrix_bytes(
        matrix.nnz,
        matrix.shape,
        matrix.dtype,
        matrix.indices.dtype,
        format=fmt,
    ) == expected
    assert estimate_sparse_storage_bytes(matrix) == expected


def test_empty_sparse_storage_and_large_dimension_overflow_are_safe():
    empty = sparse.csr_matrix((0, 12), dtype=np.float64)
    assert estimate_sparse_storage_bytes(empty) == empty.data.nbytes + empty.indices.nbytes + empty.indptr.nbytes
    assert estimate_sparse_matrix_bytes(0, (0, 12), np.float64, np.int32) > 0
    with pytest.raises(OverflowError):
        estimate_dense_matrix_bytes(sys.maxsize, 2, np.float64)
    with pytest.raises(OverflowError):
        estimate_sparse_matrix_bytes(
            sys.maxsize, (sys.maxsize, 1), np.float64, np.int64
        )


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda: estimate_dense_matrix_bytes(-1, 2), ValueError),
        (lambda: estimate_dense_matrix_bytes(True, 2), TypeError),
        (lambda: estimate_sparse_matrix_bytes(2, (1, 1)), ValueError),
        (lambda: estimate_sparse_matrix_bytes(0, (1, 1), format="coo"), ValueError),
        (lambda: dense_conversion_limit_bytes(float("inf")), ValueError),
    ],
)
def test_memory_estimators_reject_invalid_inputs(call, error):
    with pytest.raises(error):
        call()


def test_dataframe_estimate_includes_deep_object_memory_and_shape():
    frame = pd.DataFrame({"text": ["long value " * 10, "another value " * 10], "x": [1, 2]})
    estimate = estimate_dataframe_memory(frame)
    shallow = int(frame.memory_usage(index=True, deep=False).sum())
    assert estimate["rows"] == 2
    assert estimate["columns"] == 2
    assert estimate["bytes"] == int(frame.memory_usage(index=True, deep=True).sum())
    assert estimate["bytes"] > shallow
    assert estimate["megabytes"] >= 0


def test_dense_conversion_is_rejected_before_allocating(monkeypatch):
    matrix = sparse.csr_matrix((2_000, 2_000), dtype=np.float64)
    monkeypatch.setattr("app.core.memory.settings.MAX_DENSE_CONVERSION_MB", 1)
    with pytest.raises(DenseConversionLimitError, match="MAX_DENSE_CONVERSION_MB"):
        sparse_to_dense_checked(matrix)


def test_dense_guard_preflights_projected_training_rows(monkeypatch):
    matrix = sparse.csr_matrix((10, 100), dtype=np.float64)
    guard = SparseToDenseGuard(max_memory_mb=0.01, projected_rows=1_000)
    with pytest.raises(DenseConversionLimitError, match="1,000 rows"):
        guard.fit(matrix)

