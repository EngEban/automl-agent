from __future__ import annotations
import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.schemas.run import AnalysisRunRead

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{run_id}", response_model=AnalysisRunRead)
def get_run_status(run_id: str, db: DB, current_user: CurrentUser):
    run = crud.get_run(db, run_id)
    if run is None or run.requested_by != current_user.id:
        raise HTTPException(404, "Run not found")
    return AnalysisRunRead.model_validate(run)

@router.get("/{run_id}/events")
async def stream_run_events(run_id: str, request: Request, db: DB, current_user: CurrentUser):
    run = crud.get_run(db, run_id)
    if run is None or run.requested_by != current_user.id:
        raise HTTPException(404, "Run not found")

    async def event_generator():
        last_event_id = 0
        while True:
            if await request.is_disconnected():
                break
            
            # Fetch new events
            events = crud.list_events_for_run(db, run_id)
            for event in events[last_event_id:]:
                last_event_id += 1
                yield {
                    "id": str(last_event_id),
                    "event": event.event_type,
                    "data": json.dumps({
                        "step": event.step,
                        "progress": event.progress_percent,
                        "message": event.message,
                    })
                }
            
            # If run is finished, stop streaming
            refreshed_run = crud.get_run(db, run_id)
            if refreshed_run.status in ["succeeded", "failed", "cancelled"]:
                break
                
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())