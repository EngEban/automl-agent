"""
High-level pipeline runners (synchronous) for the REST API.
The agent calls the same tools through executor.py.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session as DBSession

from app.agent.executor import execute_tool

logger = logging.getLogger(__name__)


def run_full_pipeline(
    db: DBSession, session_id: str, target_column: str | None = None,
) -> dict[str, Any]:
    """
    Run EDA -> preprocessing -> training -> report for a session.
    Returns {'ok': bool, 'steps': {...}, 'error': str | None}.
    """
    args = {"target_column": target_column} if target_column else {}
    result = execute_tool(session_id, "rerun_pipeline", args, db)
    if "error" in result:
        return {"ok": False, "error": result["error"], "steps": {}}
    return {"ok": True, "steps": result, "error": None}


def get_operations(db: DBSession, session_id: str) -> dict[str, Any]:
    """Return the full operation log for a session."""
    return execute_tool(session_id, "get_operations_log", {}, db)