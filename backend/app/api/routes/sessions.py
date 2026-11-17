"""
Session management endpoints.
"""
from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response
import pandas as pd

from app.api.deps import DB, CurrentUser
from app.config.settings import settings
from app.db import crud
from app.schemas.session import (
    SessionCreateURL,
    SessionListItem,
    SessionRead,
    SessionCreateKaggle,
)
from app.schemas.feature_analysis import FeatureAnalysisRead, FeatureAnalysisRequest
from app.services import session_service

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("", response_model=SessionRead, status_code=201)
def create_from_url(payload: SessionCreateURL, db: DB, current_user: CurrentUser) -> SessionRead:
    """Create a session from a direct URL."""
    try:
        obj = session_service.create_session_from_url(
            db, user_id=current_user.id, name=payload.name, url=payload.url,
            target_column=payload.target_column, notes=payload.notes,
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to create session: {exc}") from exc
    return SessionRead.model_validate(obj)


@router.post("/upload", response_model=SessionRead, status_code=201)
async def create_from_upload(
    db: DB,
    current_user: CurrentUser,
    name: str = Form(...),
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
) -> SessionRead:
    """Create a session from an uploaded file with a hard size limit."""
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    partial_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload-",
            suffix=".part",
            dir=settings.UPLOAD_DIR,
            delete=False,
        ) as partial_file:
            partial_path = Path(partial_file.name)
            total_bytes = 0
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB.",
                    )
                partial_file.write(chunk)

        obj = session_service.create_session_from_upload(
            db, user_id=current_user.id, name=name,
            original_filename=file.filename or "data.csv",
            temporary_path=partial_path,
            target_column=target_column, notes=notes,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to create session: {exc}") from exc
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        await file.close()
    return SessionRead.model_validate(obj)


@router.post("/kaggle", response_model=SessionRead, status_code=201)
def create_from_kaggle(
    payload: SessionCreateKaggle, db: DB, current_user: CurrentUser
) -> SessionRead:
    """Create a session by downloading a dataset from Kaggle."""
    try:
        obj = session_service.create_session_from_kaggle(
            db, user_id=current_user.id, name=payload.name, kaggle_ref=payload.kaggle_ref,
            target_column=payload.target_column, notes=payload.notes,
        )
    except Exception as exc:
        raise HTTPException(400, f"Failed to create session: {exc}") from exc
    return SessionRead.model_validate(obj)


@router.get("", response_model=list[SessionListItem])
def list_sessions(db: DB, current_user: CurrentUser, limit: int = 50, offset: int = 0) -> list[SessionListItem]:
    """List sessions for the current user (newest first)."""
    items = crud.list_sessions(db, user_id=current_user.id, limit=limit, offset=offset)
    return [SessionListItem.model_validate(x) for x in items]


@router.get("/{session_id}", response_model=SessionRead)
def get_session(session_id: str, db: DB, current_user: CurrentUser) -> SessionRead:
    """Fetch a single session (owned by the current user)."""
    obj = crud.get_session(db, session_id)
    if obj is None or obj.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
    return SessionRead.model_validate(obj)


@router.post(
    "/{session_id}/feature-analysis",
    response_model=FeatureAnalysisRead,
)
def analyze_session_features(
    session_id: str,
    payload: FeatureAnalysisRequest,
    db: DB,
    current_user: CurrentUser,
) -> FeatureAnalysisRead:
    """Return a bounded profile and optional training-only selector comparison."""
    try:
        canonical_id = str(uuid.UUID(session_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc

    session = crud.get_session(db, canonical_id)
    if session is None or str(session.user_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")

    sessions_root = (settings.STORAGE_DIR / "sessions").resolve()
    try:
        session_dir = (sessions_root / canonical_id).resolve(strict=True)
        session_dir.relative_to(sessions_root)
        dataset_path = session_dir / "raw.pkl"
        resolved_path = dataset_path.resolve(strict=True)
        resolved_path.relative_to(session_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="The saved dataset is unavailable for feature analysis.",
        ) from exc
    if not resolved_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="The saved dataset is unavailable for feature analysis.",
        )

    try:
        frame = pd.read_pickle(resolved_path)
        from app.core.feature_explorer import explore_features

        target_column = (
            payload.target_column
            if "target_column" in payload.model_fields_set
            else session.target_column
        )
        task_type = (
            payload.task_type
            if "task_type" in payload.model_fields_set
            else session.task_type
        )
        result = explore_features(
            frame,
            target_column=target_column,
            task_type=task_type,
            max_features=payload.max_features,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "Feature analysis failed for an owned session (error_type=%s)",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Feature analysis could not be completed for this dataset.",
        ) from exc
    return FeatureAnalysisRead.model_validate(result)


@router.delete("/{session_id}", status_code=204, response_class=Response)
def delete_session(session_id: str, db: DB, current_user: CurrentUser) -> Response:
    """Delete a session (owned by the current user)."""
    obj = crud.get_session(db, session_id)
    if obj is None or obj.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
        
    crud.delete_session(db, session_id)
    return Response(status_code=204)
