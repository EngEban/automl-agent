"""
The agent loop: alternate between LLM decisions and tool execution.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session as DBSession
from starlette.concurrency import run_in_threadpool

from app.agent.executor import execute_tool
from app.agent.llm import chat, extract_message
from app.agent.memory import format_tool_result, load_history
from app.agent.prompts import build_system_prompt
from app.agent.tools import get_tool_schemas
from app.config import settings
from app.db.crud import create_message
from app.services.run_service import emit_event

logger = logging.getLogger(__name__)


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Decode a tool-call argument payload and require a JSON object."""
    if raw_arguments is None or raw_arguments == "":
        return {}
    arguments = (
        json.loads(raw_arguments)
        if isinstance(raw_arguments, str)
        else raw_arguments
    )
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return arguments


def _error_result(tool_name: str, error: str, error_type: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "error": error,
        "error_type": error_type,
    }


async def run_agent(
    session_id: str,
    user_message: str,
    db: DBSession,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Run one agent turn and persist the complete user/assistant/tool transcript.
    Emits progress events if run_id is provided.
    """
    create_message(db, session_id=session_id, role="user", content=user_message)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        *load_history(db, session_id),
    ]
    tools = get_tool_schemas()

    all_tool_calls: list[dict[str, Any]] = []
    final_content = ""
    iterations = 0
    terminated_on_error = False
    tool_call_history: list[str] = []

    for iteration in range(1, settings.AGENT_MAX_ITERATIONS + 1):
        iterations = iteration
        logger.info(
            "Agent iteration %d/%d",
            iteration,
            settings.AGENT_MAX_ITERATIONS,
        )

        if run_id:
            progress = min(85, 10 + (iteration * 10))
            emit_event(
                db, run_id, "step.progress", 
                step="thinking", 
                progress=progress, 
                message=f"Agent reasoning (iteration {iteration})..."
            )

        response = await chat(messages=messages, tools=tools)
        model_message = extract_message(response)
        content = model_message.get("content") or ""
        tool_calls = model_message.get("tool_calls") or []

        if not tool_calls:
            final_content = content
            messages.append({"role": "assistant", "content": content})
            create_message(
                db,
                session_id=session_id,
                role="assistant",
                content=content,
            )
            break

        all_tool_calls.extend(tool_calls)
        create_message(
            db,
            session_id=session_id,
            role="assistant",
            content=content,
            tool_calls={"calls": tool_calls},
        )
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "")
            tool_call_id = str(tool_call.get("id") or "")

            # Loop breaker
            tool_call_history.append(tool_name)
            if len(tool_call_history) >= 3 and tool_call_history[-1] == tool_call_history[-2] == tool_call_history[-3]:
                final_content = f"I stopped because I was stuck in a loop trying to run '{tool_name}'."
                terminated_on_error = True
                break

            if run_id:
                emit_event(
                    db, run_id, "step.started", 
                    step=tool_name, 
                    progress=50, 
                    message=f"Executing tool: {tool_name}"
                )

            try:
                arguments = _parse_tool_arguments(function.get("arguments"))
                logger.info("Executing tool: %s(%s)", tool_name, arguments)
                
                # Run tool in threadpool to avoid Playwright sync/async clash.
                # Preserve the four-argument call for agent turns that do not
                # have an AnalysisRun context.
                if run_id is None:
                    result = await run_in_threadpool(
                        execute_tool, session_id, tool_name, arguments, db
                    )
                else:
                    result = await run_in_threadpool(
                        execute_tool,
                        session_id,
                        tool_name,
                        arguments,
                        db,
                        run_id,
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                result = _error_result(
                    tool_name,
                    f"Invalid arguments for '{tool_name}': {exc}",
                    type(exc).__name__,
                )

            if not isinstance(result, dict):
                result = _error_result(
                    tool_name,
                    "Tool returned a non-dictionary result.",
                    "TypeError",
                )

            result_str = format_tool_result(tool_name, result)
            create_message(
                db,
                session_id=session_id,
                role="tool",
                content=result_str,
                tool_results={
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                },
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_str,
            })

            if "error" in result or result.get("ok") is False:
                error = str(result.get("error") or "The tool failed without an error message.")
                final_content = f"Tool '{tool_name}' failed: {error}"
                logger.error(final_content)
                messages.append({"role": "assistant", "content": final_content})
                create_message(db, session_id=session_id, role="assistant", content=final_content)
                terminated_on_error = True
                break

        if terminated_on_error:
            break
    else:
        final_content = "I reached the maximum number of steps without finishing. Please refine the request and try again."
        logger.warning(final_content)
        create_message(db, session_id=session_id, role="assistant", content=final_content)

    if run_id:
        emit_event(
            db, run_id, "step.completed", 
            step="chat", 
            progress=100, 
            message="Analysis finished."
        )

    return {
        "content": final_content,
        "tool_calls": all_tool_calls,
        "iterations": iterations,
    }
