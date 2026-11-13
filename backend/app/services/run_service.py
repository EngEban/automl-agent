"""Transactional run lifecycle and append-only progress events."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as DBSession

from app.core.run_lifecycle import InvalidRunTransition, transition_run_status
from app.db import crud
from app.db.models import AnalysisRun, RunEvent
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

_EVENT_STATUS = {
    "run.started": "running",
    "run.awaiting_input": "awaiting_input",
    "run.awaiting_approval": "awaiting_approval",
    "run.cancellation_requested": "cancelling",
    "run.succeeded": "succeeded",
    "run.partially_completed": "partially_completed",
    "run.failed": "failed",
    "run.cancelled": "cancelled",
}
_SENSITIVE_KEYS = re.compile(
    r"(api.?key|authorization|password|secret|token|dataset|records?|rows?|samples?|"
    r"source.?path|artifact.?uri|file.?path|absolute.?path)",
    re.IGNORECASE,
)
_MAX_EVENT_BYTES = 16_384


def _clean_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in list(value.items())[:100]:
            safe_key = str(key)[:100]
            if _SENSITIVE_KEYS.search(safe_key):
                continue
            cleaned[safe_key] = _clean_value(item, depth=depth + 1)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_clean_value(item, depth=depth + 1) for item in value[:100]]
    return str(value)[:1000]


def _safe_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _clean_value(payload or {})
    if not isinstance(cleaned, dict):
        cleaned = {}
    encoded = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_EVENT_BYTES:
        return {"truncated": True}
    return cleaned


def _lock_run(db: DBSession, run_id: str) -> AnalysisRun | None:
    statement = select(AnalysisRun).where(AnalysisRun.id == run_id).with_for_update()
    return db.scalars(statement).first()


def _same_as_latest(
    db: DBSession,
    run_id: str,
    event_type: str,
    step: str | None,
    progress: int,
    message: str | None,
    payload: dict[str, Any],
) -> RunEvent | None:
    latest = db.scalars(
        select(RunEvent)
        .where(RunEvent.run_id == run_id)
        .order_by(RunEvent.sequence.desc())
        .limit(1)
    ).first()
    if (
        latest is not None
        and latest.event_type == event_type
        and latest.step == step
        and latest.progress_percent == progress
        and latest.message == message
        and (latest.payload or {}) == payload
    ):
        return latest
    return None


def _append_locked(
    db: DBSession,
    run: AnalysisRun,
    event_type: str,
    *,
    step: str | None,
    progress: int,
    message: str | None,
    payload: dict[str, Any],
) -> RunEvent:
    duplicate = _same_as_latest(
        db, run.id, event_type, step, progress, message, payload
    )
    if duplicate is not None:
        return duplicate

    next_status = _EVENT_STATUS.get(event_type)
    if next_status is not None:
        transition_run_status(run, next_status)
    run.current_step = step
    run.progress_percent = max(
        int(run.progress_percent or 0), max(0, min(100, progress))
    )
    run.message = (message or "")[:1000] or None
    if event_type == "run.failed":
        run.error_code = str(payload.get("error_code") or "run_failed")[:50]
        run.error_message = run.message

    event = RunEvent(
        run_id=run.id,
        sequence=int(
            db.scalar(
                select(func.max(RunEvent.sequence)).where(RunEvent.run_id == run.id)
            )
            or 0
        ) + 1,
        event_type=event_type,
        step=step,
        progress_percent=run.progress_percent,
        message=run.message,
        payload=payload,
    )
    db.add(event)
    db.flush()
    return event


def emit_event(
    db: DBSession,
    run_id: str,
    event_type: str,
    step: str | None = None,
    progress: int = 0,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    """Persist an event and its run snapshot update in one short transaction."""
    safe_payload = _safe_payload(payload)
    safe_message = (message or "")[:1000] or None
    try:
        run = _lock_run(db, run_id)
        if run is None:
            raise ValueError("Analysis run not found.")
        event = _append_locked(
            db,
            run,
            event_type,
            step=step,
            progress=progress,
            message=safe_message,
            payload=safe_payload,
        )
        db.commit()
        db.refresh(event)
        return event
    except Exception:
        db.rollback()
        raise


def claim_run(db: DBSession, run_id: str) -> tuple[AnalysisRun | None, bool]:
    """Atomically claim a queued run so duplicate Celery deliveries do no work."""
    try:
        run = _lock_run(db, run_id)
        if run is None or run.status != "queued":
            return run, False
        event = _append_locked(
            db,
            run,
            "run.started",
            step="started",
            progress=max(1, int(run.progress_percent or 0)),
            message="Analysis run started.",
            payload={},
        )
        db.commit()
        db.refresh(run)
        db.refresh(event)
        return run, True
    except Exception:
        db.rollback()
        raise


def request_cancellation(
    db: DBSession,
    run_id: str,
) -> tuple[AnalysisRun | None, bool]:
    """Request cooperative cancellation; queued/waiting work is safe to stop now."""
    try:
        run = _lock_run(db, run_id)
        if run is None:
            return None, False
        if run.status in {"succeeded", "partially_completed", "failed", "cancelled", "cancelling"}:
            return run, False
        if run.status in {"queued", "awaiting_input", "awaiting_approval"}:
            event_type = "run.cancelled"
            message = "Run cancelled before another operation started."
        else:
            event_type = "run.cancellation_requested"
            message = "Cancellation requested; waiting for a safe execution checkpoint."
        event = _append_locked(
            db,
            run,
            event_type,
            step="cancellation",
            progress=int(run.progress_percent or 0),
            message=message,
            payload={"requested": True},
        )
        db.commit()
        db.refresh(run)
        db.refresh(event)
        return run, True
    except Exception:
        db.rollback()
        raise


def is_cancellation_requested(run_id: str | None) -> bool:
    """Read cancellation state with a short-lived, independent DB session."""
    if not run_id:
        return False
    db = SessionLocal()
    try:
        # Read directly instead of using crud.get_run, which deliberately
        # collapses lookup errors to None for API convenience. A DB outage is
        # an execution failure, not a cancellation request.
        run = db.get(AnalysisRun, run_id)
        return run is None or run.status in {"cancelling", "cancelled"}
    finally:
        db.close()
