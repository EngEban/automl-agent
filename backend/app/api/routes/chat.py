from __future__ import annotations
import logging
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.db.models import AnalysisRun
from app.schemas.chat import ChatRequest, MessageRead
from app.services.run_service import emit_event
from app.tasks.agent_tasks import run_agent_task

logger = logging.getLogger(__name__)
router = APIRouter()

class TaskAcceptedResponse(BaseModel):
    run_id: str
    status: str = "queued"
    status_url: str

@router.post("/{session_id}", response_model=TaskAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_message(
    session_id: str,
    payload: ChatRequest,
    db: DB,
    current_user: CurrentUser,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=255),
):
    """Enqueue the agent run and return 202 immediately."""
    session = crud.get_session_for_update(db, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(404, "Session not found")

    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()
        if not idempotency_key:
            raise HTTPException(400, "Idempotency-Key cannot be blank")
        existing = crud.get_run_by_idempotency_key(
            db,
            str(current_user.id),
            idempotency_key,
        )
        if existing is not None:
            return TaskAcceptedResponse(
                run_id=existing.id,
                status=existing.status,
                status_url=f"/api/runs/{existing.id}",
            )

    active_run = crud.get_active_run_for_session(
        db,
        session_id,
        str(current_user.id),
    )
    if active_run is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "An analysis is already active for this session.",
                "run_id": active_run.id,
                "status_url": f"/api/runs/{active_run.id}",
            },
        )

    try:
        run = crud.create_run(
            db,
            session_id=session_id,
            requested_by=current_user.id,
            status="queued",
            idempotency_key=idempotency_key,
            task_id=None,
            config_json={
                "message": payload.message,
                "mode": payload.mode,
                "analysis_options": (
                    payload.analysis_options.model_dump(exclude_none=True)
                    if payload.analysis_options
                    else {}
                ),
            },
        )
    except IntegrityError:
        db.rollback()
        if idempotency_key is not None:
            existing = crud.get_run_by_idempotency_key(
                db,
                str(current_user.id),
                idempotency_key,
            )
            if existing is not None:
                return TaskAcceptedResponse(
                    run_id=existing.id,
                    status=existing.status,
                    status_url=f"/api/runs/{existing.id}",
                )
        raise HTTPException(status_code=409, detail="Duplicate run request") from None

    try:
        emit_event(
            db,
            run.id,
            "run.created",
            step="created",
            progress=0,
            message="Analysis run created.",
            payload={"mode": payload.mode},
        )
        emit_event(
            db,
            run.id,
            "run.queued",
            step="queued",
            progress=0,
            message="Analysis run queued.",
            payload={"mode": payload.mode},
        )
        # Persist the deterministic task ID before dispatch so a fast worker
        # cannot finish before its Celery identifier is recorded.
        crud.update_run(db, run.id, task_id=run.id)
        run_agent_task.apply_async(
            args=[
                run.id,
                session_id,
                payload.message,
                payload.mode,
                payload.analysis_options.model_dump(exclude_none=True)
                if payload.analysis_options
                else {},
            ],
            task_id=run.id,
        )
    except Exception:
        logger.exception("Analysis task could not be queued (run_id=%s)", run.id)
        current = crud.get_run(db, run.id)
        if current is not None and current.status == "queued":
            emit_event(
                db,
                run.id,
                "run.failed",
                step="dispatch",
                progress=0,
                message="The analysis could not be queued. Retry the request.",
                payload={"error_code": "task_dispatch_failed"},
            )
        raise HTTPException(status_code=503, detail="The analysis could not be queued") from None

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
