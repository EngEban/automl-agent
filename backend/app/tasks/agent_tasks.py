from __future__ import annotations
import asyncio
import logging

from app.agent.agent import run_agent
from app.db.crud import get_run, get_session
from app.db.session import SessionLocal
from app.services.run_service import emit_event
from app.worker import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="run_agent_task")
def run_agent_task(run_id: str, session_id: str, user_message: str):
    """Celery task to run the agent asynchronously."""
    db = SessionLocal()
    try:
        run = get_run(db, run_id)
        if not run:
            return {"status": "error", "message": "Run not found"}

        emit_event(db, run_id, "run.started", step="started", progress=5, message="Agent started working.")
        
        # Run the async agent loop inside the sync Celery task
        result = asyncio.run(run_agent(session_id, user_message, db, run_id=run_id))
        
        emit_event(db, run_id, "run.succeeded", step="done", progress=100, message="Success.")
        return {"status": "success", "content": result["content"]}
    except Exception as exc:
        logger.exception("Agent task failed")
        emit_event(db, run_id, "run.failed", step="error", progress=100, message=str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        db.close()