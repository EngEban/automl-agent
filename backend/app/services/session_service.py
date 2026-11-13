"""
Session orchestration: create a session, load the dataset, persist it.
"""
from __future__ import annotations

import logging
import math
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.core.data_loader import describe, load_from_bytes, load_from_path, load_from_url
from app.db.crud import create_session
from app.db.models import Session as SessionModel
from app.core.connectors.kaggle import KaggleConnector

logger = logging.getLogger(__name__)

SESSIONS_DIR = settings.STORAGE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str) -> Path:
    p = SESSIONS_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _clean_json(obj: Any) -> Any:
    """Recursively convert NaN and Inf to None so PostgreSQL can store it as JSON."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def create_session_from_url(
    db: DBSession, *, user_id: str, name: str, url: str,
    target_column: str | None = None, notes: str | None = None,
) -> SessionModel:
    """Load from URL, cache raw pickle, persist the session row."""
    df = load_from_url(url)
    original_filename = Path(unquote(urlsplit(url).path)).name or name or "data.csv"
    return _finish_create(
        db, user_id=user_id, name=name, source_type="url", source_path=url,
        df=df, target_column=target_column, notes=notes,
        original_filename=original_filename,
    )


def create_session_from_upload(
    db: DBSession,
    *,
    user_id: str,
    name: str,
    content: bytes | None = None,
    original_filename: str | None = None,
    filename: str | None = None,
    temporary_path: str | Path | None = None,
    target_column: str | None = None,
    notes: str | None = None,
) -> SessionModel:
    """Load from raw bytes, cache raw pickle, persist the session row."""
    # ``filename`` remains as a backwards-compatible alias for existing callers.
    supplied_filename = original_filename or filename or "data.csv"
    safe_filename = Path(str(supplied_filename).replace("\\", "/")).name
    safe_filename = "".join(
        "_"
        if ord(character) < 32 or ord(character) == 127 or character in '<>:"|?*'
        else character
        for character in safe_filename
    ).rstrip(" .")
    if safe_filename in {"", ".", ".."}:
        safe_filename = "data.csv"

    if (content is None) == (temporary_path is None):
        raise ValueError("Provide exactly one of content or temporary_path.")
    if temporary_path is not None:
        df = load_from_path(temporary_path, format_hint=Path(safe_filename).suffix)
    else:
        df = load_from_bytes(content, safe_filename)
    session_id = str(uuid.uuid4())
    stored = settings.UPLOAD_DIR / f"{session_id}{Path(safe_filename).suffix.lower()}"
    if temporary_path is not None:
        shutil.copyfile(temporary_path, stored)
    else:
        stored.write_bytes(content)
    return _finish_create(
        db, user_id=user_id, name=name, source_type="upload", source_path=str(stored),
        df=df, target_column=target_column, notes=notes, forced_id=session_id,
        original_filename=safe_filename,
    )


def _finish_create(
    db: DBSession, *, user_id: str, name: str, source_type: str, source_path: str,
    df: Any, target_column: str | None, notes: str | None,
    original_filename: str,
    forced_id: str | None = None,
) -> SessionModel:
    """Common finishing logic: cache pickle + create DB row."""
    session_id = forced_id or str(uuid.uuid4())
    sdir = _session_dir(session_id)
    
    # Save the raw dataset
    df.to_pickle(sdir / "raw.pkl")

    meta = describe(df)
    meta["original_filename"] = original_filename
    # Clean the sample rows to remove NaN values before saving to PostgreSQL
    sample_data = df.head(5).to_dict(orient="records")
    meta["sample_rows"] = _clean_json(sample_data)

    obj = create_session(
        db,
        id=session_id,
        user_id=user_id,
        name=name,
        source_type=source_type,
        source_path=source_path,
        target_column=target_column,
        status="created",
        dataset_meta=meta,
        notes=notes,
    )
    logger.info("Session created: %s by user %s (%d rows)", session_id, user_id, len(df))
    return obj
def create_session_from_kaggle(
    db: DBSession, *, user_id: str, name: str, kaggle_ref: str,
    target_column: str | None = None, notes: str | None = None,
) -> SessionModel:
    """Create a session by downloading a dataset from Kaggle."""
    session_id = str(uuid.uuid4())
    dest_dir = _session_dir(session_id) / "kaggle_download"
    
    try:
        connector = KaggleConnector()
        if not connector.validate_reference(kaggle_ref):
            raise ValueError("Invalid Kaggle reference format. Use 'owner/dataset-name'.")
            
        provenance = connector.download_to_staging(kaggle_ref, dest_dir)
        file_path = Path(provenance["file_path"])
        
        # Read the downloaded file
        df = load_from_path(file_path)
        
        # Create the session
        obj = _finish_create(
            db, user_id=user_id, name=name, source_type="kaggle", source_path=str(file_path),
            df=df, target_column=target_column, notes=notes, forced_id=session_id,
            original_filename=file_path.name
        )
        
        # Save provenance
        from app.db.models import DatasetSource
        from datetime import datetime
        ds = DatasetSource(
            session_id=session_id,
            provider=provenance["provider"],
            source_id=provenance["source_id"],
            license=provenance["license"],
        )
        db.add(ds)
        db.commit()
        
        return obj
    except Exception as exc:
        logger.error(f"Kaggle download failed: {exc}")
        raise
