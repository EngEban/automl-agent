"""
Session management endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.schemas.session import (
    SessionCreateURL,
    SessionListItem,
    SessionRead,
)
from app.services import session_service

logger = logging.getLogger(__name__)
router = APIRouter()


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
    """Create a session from an uploaded file."""
    content = await file.read()
    try:
        obj = session_service.create_session_from_upload(
            db, user_id=current_user.id, name=name,
            original_filename=file.filename or "data.csv",
            content=content, target_column=target_column, notes=notes,
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


@router.delete("/{session_id}", status_code=204, response_class=Response)
def delete_session(session_id: str, db: DB, current_user: CurrentUser) -> Response:
    """Delete a session (owned by the current user)."""
    obj = crud.get_session(db, session_id)
    if obj is None or obj.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
        
    crud.delete_session(db, session_id)
    return Response(status_code=204)
