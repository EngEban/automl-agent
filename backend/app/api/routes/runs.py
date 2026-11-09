from __future__ import annotations
import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.api.deps import DB, CurrentUser
from app.db import crud
from app.schemas.run import AnalysisRunRead

logger = logging.getLogger(__name__)
router = APIRouter()


class PlanApprovalRequest(BaseModel):
    plan_id: str


@router.get("/{run_id}", response_model=AnalysisRunRead)
def get_run_status(run_id: str, db: DB, current_user: CurrentUser):
    run = crud.get_run(db, run_id)
    if run is None or run.requested_by != current_user.id:
        raise HTTPException(404, "Run not found")
    return AnalysisRunRead.model_validate(run)


@router.post("/{run_id}/approve")
def approve_plan(run_id: str, payload: PlanApprovalRequest, db: DB, current_user: CurrentUser):
    run = crud.get_run(db, run_id)
    if run is None or run.requested_by != current_user.id:
        raise HTTPException(404, "Run not found")
        
    plan = crud.get_plan(db, payload.plan_id)
    if not plan or plan.run_id != run_id:
        raise HTTPException(404, "Plan not found for this run")
        
    if plan.approval_status != "pending":
        raise HTTPException(400, "Plan is not pending approval")
        
    crud.update_plan(db, plan.id, approval_status="approved", approved_by=current_user.id)
    crud.update_run(db, run_id, status="running")
    
    # Note: In a full production system, you would resume the Celery worker here.
    return {"message": "Plan approved successfully. You can now rerun the pipeline."}


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
            
            refreshed_run = crud.get_run(db, run_id)
            if refreshed_run.status in ["succeeded", "failed", "cancelled"]:
                break
                
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())