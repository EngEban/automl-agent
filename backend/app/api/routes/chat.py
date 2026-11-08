from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.db.models import AnalysisRun
from app.schemas.chat import ChatRequest, MessageRead
from app.tasks.agent_tasks import run_agent_task

logger = logging.getLogger(__name__)
router = APIRouter()

class TaskAcceptedResponse(BaseModel):
    run_id: str
    status: str = "queued"
    status_url: str

@router.post("/{session_id}", response_model=TaskAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_message(session_id: str, payload: ChatRequest, db: DB, current_user: CurrentUser):
    """Enqueue the agent run and return 202 immediately."""
    session = crud.get_session(db, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(404, "Session not found")

    # Create Run record
    run = crud.create_run(
        db,
        session_id=session_id,
        requested_by=current_user.id,
        status="queued",
        config_json={"message": payload.message},
    )

    # Enqueue Celery task
    run_agent_task.delay(run.id, session_id, payload.message)
    run.task_id = run.id
    db.commit()

    return TaskAcceptedResponse(
        run_id=run.id,
        status_url=f"/api/runs/{run.id}"
    )

@router.get("/{session_id}/messages", response_model=list[MessageRead])
def list_messages(session_id: str, db: DB, current_user: CurrentUser, limit: int = 200):
    session = crud.get_session(db, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(404, "Session not found")
    rows = crud.list_messages(db, session_id, limit=limit)
    return [MessageRead.model_validate(m) for m in rows]