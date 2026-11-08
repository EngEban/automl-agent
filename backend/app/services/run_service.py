from __future__ import annotations
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session as DBSession

from app.db.crud import create_event, update_run
from app.db.models import RunEvent

logger = logging.getLogger(__name__)

def emit_event(
    db: DBSession,
    run_id: str,
    event_type: str,
    step: str | None = None,
    progress: int = 0,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    """Write an event to the DB and update the parent Run's status/progress."""
    
    # 1. Create the event record
    event = create_event(
        db,
        run_id=run_id,
        event_type=event_type,
        step=step,
        progress_percent=progress,
        message=message,
        payload=payload or {},
    )
    
    # 2. Update the AnalysisRun snapshot
    updates = {
        "current_step": step,
        "progress_percent": progress,
        "message": message,
    }
    
    # Handle status transitions based on event type
    if event_type == "run.started":
        updates["status"] = "running"
    elif event_type == "run.succeeded":
        updates["status"] = "succeeded"
    elif event_type == "run.failed":
        updates["status"] = "failed"
    elif event_type == "run.cancelled":
        updates["status"] = "cancelled"
        
    update_run(db, run_id, **updates)
    
    logger.debug(f"Emitted {event_type} for run {run_id} ({progress}%)")
    return event