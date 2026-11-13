from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.deps import CurrentUser, DB
from app.core.run_lifecycle import InvalidRunTransition, TERMINAL_RUN_STATUSES
from app.db import crud
from app.db.session import SessionLocal
from app.schemas.run import AnalysisRunRead, RunEventRead
from app.services.run_service import emit_event, request_cancellation
from app.tasks.agent_tasks import run_agent_task

logger = logging.getLogger(__name__)
router = APIRouter()
_EVENT_PAGE_SIZE = 100
_SSE_POLL_SECONDS = 0.75


class PlanApprovalRequest(BaseModel):
    plan_id: str = Field(min_length=1, max_length=64)


def _owned_run(run_id: str, db, current_user):
    run = crud.get_run(db, run_id)
    if run is None or str(run.requested_by) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def _uuid_cursor(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid event cursor") from None


@router.get("/{run_id}", response_model=AnalysisRunRead)
def get_run_status(run_id: str, db: DB, current_user: CurrentUser):
    return AnalysisRunRead.model_validate(_owned_run(run_id, db, current_user))


@router.post("/{run_id}/approve")
def approve_plan(run_id: str, payload: PlanApprovalRequest, db: DB, current_user: CurrentUser):
    run = _owned_run(run_id, db, current_user)
    if run.status != "awaiting_approval":
        raise HTTPException(status_code=409, detail="Run is not waiting for approval")

    plan = crud.get_plan(db, payload.plan_id)
    if plan is None or str(plan.run_id) != str(run_id):
        raise HTTPException(status_code=404, detail="Plan not found for this run")
    if plan.approval_status != "pending":
        raise HTTPException(status_code=409, detail="Plan is not pending approval")

    config = run.config_json if isinstance(run.config_json, dict) else {}
    options = config.get("analysis_options", {})
    options = options if isinstance(options, dict) else {}
    mode = config.get("mode", "simple")
    session_id = run.session_id
    task_id = str(uuid.uuid4())

    crud.update_plan(db, plan.id, approval_status="approved", approved_by=current_user.id)
    try:
        crud.update_run(db, run_id, status="queued", task_id=task_id)
        emit_event(
            db,
            run_id,
            "run.queued",
            step="queued",
            progress=int(run.progress_percent or 0),
            message="Approved plan queued for execution.",
            payload={"plan_id": plan.id},
        )
        run_agent_task.apply_async(
            args=[run_id, session_id, "Execute the approved analysis plan.", mode, options],
            task_id=task_id,
        )
    except Exception:
        logger.exception("Approved run could not be queued (run_id=%s)", run_id)
        latest = crud.get_run(db, run_id)
        if latest is not None and latest.status == "queued":
            emit_event(
                db,
                run_id,
                "run.failed",
                step="dispatch",
                progress=int(latest.progress_percent or 0),
                message="The approved run could not be queued. Retry from the session.",
                payload={"error_code": "task_dispatch_failed"},
            )
        raise HTTPException(status_code=503, detail="The approved run could not be queued") from None

    return {"run_id": run_id, "status": "queued"}


@router.post("/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
def cancel_run(run_id: str, db: DB, current_user: CurrentUser):
    _owned_run(run_id, db, current_user)
    try:
        run, requested = request_cancellation(db, run_id)
    except InvalidRunTransition:
        db.rollback()
        run = _owned_run(run_id, db, current_user)
        requested = False
    return {
        "run_id": run_id,
        "status": run.status,
        "cancellation_requested": requested,
        "message": (
            "Cancellation is waiting for a safe execution checkpoint."
            if run.status == "cancelling"
            else "Run is already in a terminal state."
            if run.status in TERMINAL_RUN_STATUSES
            else "Run cancellation was recorded."
        ),
    }


@router.get("/{run_id}/events/history")
def get_run_event_history(
    run_id: str,
    db: DB,
    current_user: CurrentUser,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=36),
):
    _owned_run(run_id, db, current_user)
    event_cursor = _uuid_cursor(cursor)
    if event_cursor is not None and crud.get_event_for_run(db, run_id, event_cursor) is None:
        raise HTTPException(status_code=400, detail="Invalid event cursor")
    try:
        rows = crud.list_event_page(db, run_id, cursor=event_cursor, limit=limit + 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid event cursor") from None
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [RunEventRead.model_validate(row).model_dump(mode="json") for row in rows]
    return {
        "items": items,
        "next_cursor": items[-1]["id"] if has_more and items else None,
        "has_more": has_more,
    }


def _read_stream_batch(run_id: str, cursor: str | None) -> tuple[list[dict[str, Any]], str]:
    """Read one short-lived DB snapshot for SSE without retaining a session."""
    db = SessionLocal()
    try:
        run = crud.get_run(db, run_id)
        if run is None:
            return [], "failed"
        events = crud.list_events_after(
            db,
            run_id,
            after_event_id=cursor,
            limit=_EVENT_PAGE_SIZE,
        )
        serialized = [
            {
                "id": str(event.id),
                "event": event.event_type,
                "data": json.dumps(
                    RunEventRead.model_validate(event).model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            for event in events
        ]
        return serialized, run.status
    finally:
        db.close()


@router.get("/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    db: DB,
    current_user: CurrentUser,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    _owned_run(run_id, db, current_user)
    cursor = _uuid_cursor(last_event_id)
    if cursor is not None and crud.get_event_for_run(db, run_id, cursor) is None:
        raise HTTPException(status_code=400, detail="Invalid event cursor")
    # FastAPI may keep yield-dependency sessions alive through streaming.
    # Release this request session before returning the long-lived response.
    db.close()

    async def event_generator():
        nonlocal cursor
        try:
            while not await request.is_disconnected():
                events, run_status = await asyncio.to_thread(
                    _read_stream_batch, run_id, cursor
                )
                for event in events:
                    if await request.is_disconnected():
                        return
                    cursor = event["id"]
                    yield event
                if run_status in TERMINAL_RUN_STATUSES and not events:
                    return
                if not events:
                    await asyncio.sleep(_SSE_POLL_SECONDS)
        except asyncio.CancelledError:
            logger.debug("Run event stream disconnected (run_id=%s)", run_id)
            raise

    return EventSourceResponse(event_generator(), ping=15)
