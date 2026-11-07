"""
Data loading utilities.
Supports local files (CSV, JSON, Excel) and direct URLs.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}


class DataLoadError(Exception):
    """Raised when data cannot be loaded."""


def _detect_format(source: str) -> str:
    """Detect file format from extension or URL path."""
    path = urlparse(source).path if source.startswith(("http://", "https://")) else source
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    lower = source.lower()
    if "csv" in lower:
        return ".csv"
    if "json" in lower:
        return ".json"
    if "xls" in lower:
        return ".xlsx"
    return ".csv"


def _read_bytes(content: bytes, fmt: str) -> pd.DataFrame:
    """Read a DataFrame from raw bytes based on format."""
    buffer = io.BytesIO(content)
    try:
        if fmt == ".csv":
            return pd.read_csv(buffer)
        if fmt == ".tsv":
            return pd.read_csv(buffer, sep="\t")
        if fmt in (".xlsx", ".xls"):
            return pd.read_excel(buffer)
        if fmt == ".json":
            return pd.read_json(buffer)
        if fmt == ".jsonl":
            return pd.read_json(buffer, lines=True)
    except Exception as exc:
        raise DataLoadError(f"Failed to read data ({fmt}): {exc}") from exc
    raise DataLoadError(f"Unsupported format: {fmt}")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def _post_process(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Light cleaning after loading."""
    if df.empty:
        raise DataLoadError("Dataset is empty.")

    df = _clean_columns(df)
    df = df.dropna(axis=1, how="all")

    if len(df) > settings.MAX_ROWS_SAMPLE:
        logger.info("Sampling %d rows from %d", settings.MAX_ROWS_SAMPLE, len(df))
        df = df.sample(
            n=settings.MAX_ROWS_SAMPLE,
            random_state=settings.RANDOM_STATE,
        ).reset_index(drop=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None})

    logger.info("Loaded %s: shape=%s", source, df.shape)
    return df


def load_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Load a DataFrame from uploaded bytes."""
    fmt = Path(filename).suffix.lower()
    if fmt not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(f"Unsupported extension: {fmt}")
    df = _read_bytes(content, fmt)
    return _post_process(df, source=f"upload:{filename}")


def load_from_url(url: str) -> pd.DataFrame:
    """Load a DataFrame from a direct URL."""
    if not url.startswith(("http://", "https://")):
        raise DataLoadError("URL must start with http:// or https://")

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DataLoadError(f"Failed to download URL: {exc}") from exc

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(resp.content) > max_bytes:
        raise DataLoadError(f"File exceeds {settings.MAX_FILE_SIZE_MB} MB.")

    fmt = _detect_format(url)
    df = _read_bytes(resp.content, fmt)
    return _post_process(df, source=url)


def load_from_path(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a local path (used in tests)."""
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {p}")
    if p.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DataLoadError("File too large.")
    df = _read_bytes(p.read_bytes(), p.suffix.lower())
    return _post_process(df, source=str(p))


def describe(df: pd.DataFrame) -> dict[str, Any]:
    """Structural description of the loaded data."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_column": {
            c: int(v) for c, v in df.isna().sum().items() if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }