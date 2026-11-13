"""Run status vocabulary and transition rules shared by API and workers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class InvalidRunTransition(ValueError):
    """Raised when a run is asked to move to an invalid lifecycle state."""


TERMINAL_RUN_STATUSES = frozenset({
    "succeeded",
    "partially_completed",
    "failed",
    "cancelled",
})

ACTIVE_RUN_STATUSES = frozenset({"queued", "running", "cancelling"})

_ALLOWED_TRANSITIONS = {
    "queued": {"running", "cancelling", "cancelled", "failed"},
    "running": {
        "awaiting_input", "awaiting_approval", "cancelling", "succeeded",
        "partially_completed", "failed", "cancelled",
    },
    "awaiting_input": {"queued", "cancelling", "cancelled", "failed"},
    "awaiting_approval": {"queued", "cancelling", "cancelled", "failed"},
    "cancelling": {"succeeded", "partially_completed", "failed", "cancelled"},
    "succeeded": set(),
    "partially_completed": set(),
    "failed": set(),
    "cancelled": set(),
}


def transition_run_status(run: Any, new_status: str) -> bool:
    """Apply a legal state transition and set lifecycle timestamps.

    Returns False for an idempotent same-state update and raises for an
    invalid transition. Terminal records cannot be reopened.
    """
    current = str(run.status)
    if new_status == current:
        return False
    if new_status not in _ALLOWED_TRANSITIONS:
        raise InvalidRunTransition(f"Unknown run status '{new_status}'.")
    if new_status not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidRunTransition(
            f"Run cannot transition from '{current}' to '{new_status}'."
        )

    now = datetime.now(timezone.utc)
    run.status = new_status
    if new_status == "running" and run.started_at is None:
        run.started_at = now
    if new_status in TERMINAL_RUN_STATUSES:
        run.finished_at = now
    return True
