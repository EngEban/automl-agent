from __future__ import annotations

import asyncio
import logging

from app.agent.agent import run_agent
from app.core.run_lifecycle import TERMINAL_RUN_STATUSES
from app.db.crud import get_run, get_session
from app.db.session import SessionLocal
from app.services.run_service import claim_run, emit_event
from app.worker import celery_app

logger = logging.getLogger(__name__)


class AgentExecutionFailed(RuntimeError):
    """The Pipeline Agent completed with an application-level failure."""


@celery_app.task(name="run_agent_task")
def run_agent_task(
    run_id: str,
    session_id: str,
    user_message: str,
    mode: str = "simple",
    analysis_options: dict | None = None,
):
    """Run one queued AnalysisRun; duplicate Celery deliveries do no work."""
    db = SessionLocal()
    try:
        run = get_run(db, run_id)
        if run is None:
            raise AgentExecutionFailed("Analysis run not found.")
        session = get_session(db, session_id)
        if (
            session is None
            or str(run.session_id) != str(session_id)
            or str(run.requested_by) != str(session.user_id)
        ):
            raise PermissionError("Analysis run does not belong to the selected session owner.")

        run, claimed = claim_run(db, run_id)
        if not claimed:
            if run.status == "failed":
                raise AgentExecutionFailed("Analysis run had already failed.")
            if run.status == "succeeded":
                return {"status": "completed", "run_id": run_id}
            if run.status == "partially_completed":
                return {"status": "partial_success", "run_id": run_id}
            if run.status == "cancelled":
                return {"status": "cancelled", "run_id": run_id}
            return {"status": "already_running", "run_id": run_id}

        result = asyncio.run(
            run_agent(
                session_id,
                user_message,
                db,
                run_id=run_id,
                mode=mode,
                analysis_options=analysis_options,
                user_id=str(run.requested_by),
            )
        )
        current_run = get_run(db, run_id)
        if current_run is not None and current_run.status == "awaiting_approval":
            return {
                "status": "awaiting_approval",
                "content": result.get("content", ""),
                "plan_id": next(
                    (
                        item.get("result", {}).get("plan_id")
                        for item in result.get("tool_results", [])
                        if isinstance(item, dict)
                        and isinstance(item.get("result"), dict)
                        and item["result"].get("plan_id")
                    ),
                    None,
                ),
            }
        agent_status = result.get("status")
        if agent_status in {"failed", "incomplete"}:
            emit_event(
                db,
                run_id,
                "run.failed",
                step="agent",
                progress=int(get_run(db, run_id).progress_percent or 0),
                message="The Agent could not complete this request. Confirmed results remain available.",
                payload={
                    "error_code": "agent_incomplete",
                    "agent_status": agent_status,
                    "analysis_status": result.get("analysis_status", "not_run"),
                },
            )
            raise AgentExecutionFailed("The Pipeline Agent did not complete this run.")

        if agent_status == "cancelled":
            event_type = "run.cancelled"
            message = "Run stopped at a safe cancellation checkpoint."
        elif agent_status == "partial_success":
            event_type = "run.partially_completed"
            message = "Analysis work completed partially; see the recorded results."
        elif agent_status == "needs_input":
            event_type = "run.awaiting_input"
            message = "The run needs additional user input."
        else:
            event_type = "run.succeeded"
            message = "The requested Agent operation completed."

        emit_event(
            db,
            run_id,
            event_type,
            step="finished" if agent_status != "needs_input" else "needs_input",
            progress=100 if agent_status in {"completed", "partial_success"} else int(run.progress_percent or 0),
            message=message,
            payload={
                "agent_status": agent_status,
                "analysis_status": result.get("analysis_status", "not_run"),
            },
        )
        return {
            "status": agent_status,
            "analysis_status": result.get("analysis_status", "not_run"),
            "content": result.get("content", ""),
            "tool_results": result.get("tool_results", []),
        }
    except Exception as exc:
        logger.exception("Agent task failed (run_id=%s error_type=%s)", run_id, type(exc).__name__)
        run = get_run(db, run_id)
        if run is not None and run.status not in TERMINAL_RUN_STATUSES:
            message = (
                "The run or session is not available to the authenticated owner."
                if isinstance(exc, PermissionError)
                else "The Agent task failed. Check server logs for details."
            )
            try:
                emit_event(
                    db,
                    run_id,
                    "run.failed",
                    step="error",
                    progress=int(run.progress_percent or 0),
                    message=message,
                    payload={"error_code": "agent_task_failed"},
                )
            except Exception:
                logger.exception("Could not persist Agent task failure (run_id=%s)", run_id)
        raise
    finally:
        db.close()
