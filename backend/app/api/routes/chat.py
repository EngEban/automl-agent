"""
Chat endpoint: send a message to the agent.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.schemas.chat import ChatRequest, ChatResponse, MessageRead
from app.services import analysis_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{session_id}", response_model=ChatResponse)
async def send_message(session_id: str, payload: ChatRequest, db: DB, current_user: CurrentUser) -> ChatResponse:
    """Run one agent turn for a session."""
    session = crud.get_session(db, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
        
    try:
        return await analysis_service.handle_user_message(
            db, session_id, payload.message,
        )
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(500, f"Agent error: {exc}") from exc


@router.get("/{session_id}/messages", response_model=list[MessageRead])
def list_messages(session_id: str, db: DB, current_user: CurrentUser, limit: int = 200) -> list[MessageRead]:
    """Return the persisted chat transcript."""
    session = crud.get_session(db, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
        
    rows = crud.list_messages(db, session_id, limit=limit)
    return [MessageRead.model_validate(m) for m in rows]