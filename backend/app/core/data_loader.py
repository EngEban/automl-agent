"""
Data loading utilities.
Supports local files (CSV, JSON, Excel) and direct URLs.
"""
from __future__ import annotations

import io
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls"}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_REDIRECTS = 5
BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata.google.internal.",
}


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
    """Load a DataFrame from a URL after SSRF and size checks."""
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    current_url = url
    content = bytearray()

    for redirect_number in range(MAX_REDIRECTS + 1):
        _validate_remote_url(current_url)
        response = None
        try:
            # Redirects are handled manually so every destination is checked.
            response = requests.get(
                current_url,
                timeout=60,
                stream=True,
                allow_redirects=False,
            )

            if response.status_code in {301, 302, 303, 307, 308}:
                location = (
                    response.headers.get("Location")
                    or response.headers.get("location")
                )
                if not location:
                    raise DataLoadError("URL redirect did not provide a destination.")
                if redirect_number >= MAX_REDIRECTS:
                    raise DataLoadError("Too many URL redirects.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_length = (
                response.headers.get("Content-Length")
                or response.headers.get("content-length")
            )
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise DataLoadError(
                            f"File exceeds {settings.MAX_FILE_SIZE_MB} MB."
                        )
                except ValueError:
                    logger.warning("Ignoring invalid Content-Length header from %s", current_url)

            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise DataLoadError(
                        f"File exceeds {settings.MAX_FILE_SIZE_MB} MB."
                    )
            break
        except DataLoadError:
            raise
        except requests.RequestException as exc:
            raise DataLoadError(f"Failed to download URL: {exc}") from exc
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    fmt = _detect_format(current_url)
    df = _read_bytes(bytes(content), fmt)
    return _post_process(df, source=url)


def _validate_remote_url(url: str) -> None:
    """Reject non-web URLs and destinations that resolve to local networks."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise DataLoadError("URL must use http:// or https://")
    if not parsed.hostname:
        raise DataLoadError("URL must include a hostname.")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise DataLoadError("URL hostname is not allowed.")

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise DataLoadError("URL contains an invalid port.") from exc

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise DataLoadError(f"Could not resolve URL hostname: {hostname}") from exc

    resolved_ips = {
        ipaddress.ip_address(result[4][0])
        for result in addresses
        if result[4]
    }
    if not resolved_ips:
        raise DataLoadError("URL hostname did not resolve to an address.")

    blocked_ip = next(
        (address for address in resolved_ips if _is_blocked_ip(address)),
        None,
    )
    if blocked_ip is not None:
        raise DataLoadError(f"URL resolves to a blocked address: {blocked_ip}")


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether an address is local, private, link-local, or special-use."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    metadata_ip = ipaddress.ip_address("169.254.169.254")
    return (
        address == metadata_ip
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )


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