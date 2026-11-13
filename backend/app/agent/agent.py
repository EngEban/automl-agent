"""Controlled Pipeline Agent lifecycle: plan, execute, observe, and respond."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from sqlalchemy.orm import Session as DBSession
from starlette.concurrency import run_in_threadpool

from app.agent import executor
from app.agent.llm import (
    AgentConfigurationError,
    MalformedAgentResponse,
    chat,
    extract_message,
)
from app.agent.memory import format_tool_result, load_history
from app.agent.planning import RequestPlan, prepare_request_plan
from app.agent.prompts import build_system_prompt
from app.agent.tools import (
    ToolValidationError,
    UnknownToolError,
    get_tool_schemas,
    validate_tool_arguments,
)
from app.config import settings
from app.db.crud import create_message, get_run, get_session
from app.db.session import SessionLocal
from app.services.run_service import emit_event, is_cancellation_requested

logger = logging.getLogger(__name__)
execute_tool = executor.execute_tool
_EXPENSIVE_TOOLS = frozenset({
    "train_models", "rerun_pipeline", "generate_report", "load_from_kaggle",
})


class _PrerequisiteRequired(ToolValidationError):
    """A required low-cost planning step has not completed in this turn."""


def _execute_tool_in_fresh_session(
    session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    run_id: str | None,
    user_id: str | None,
) -> dict[str, Any]:
    """Keep SQLAlchemy sessions thread-confined for synchronous tool calls."""
    tool_db = SessionLocal()
    try:
        if run_id is None and user_id is None:
            return execute_tool(session_id, tool_name, arguments, tool_db)
        return execute_tool(
            session_id, tool_name, arguments, tool_db, run_id=run_id, user_id=user_id
        )
    finally:
        tool_db.close()


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Decode a structured dictionary or serialized JSON object."""
    if raw_arguments is None or raw_arguments == "":
        return {}
    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be a JSON object.")
    return arguments


def _error_result(tool_name: str, error: str, error_type: str) -> dict[str, Any]:
    return {"ok": False, "tool": tool_name, "error": error, "error_type": error_type}


def _close_db_if_open(db: DBSession) -> None:
    close = getattr(db, "close", None)
    if callable(close):
        close()


def _run_progress(db: DBSession, run_id: str) -> int:
    run = get_run(db, run_id)
    return int(getattr(run, "progress_percent", 0) or 0)


def _call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"tool": tool_name, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_mode_options(mode: str, analysis_options: dict[str, Any] | None) -> dict[str, Any]:
    if mode not in {"simple", "advanced"}:
        raise ValueError("Mode must be 'simple' or 'advanced'.")
    if analysis_options is None:
        return {}
    if not isinstance(analysis_options, dict):
        raise ValueError("Analysis options must be an object.")
    if mode != "advanced":
        raise ValueError("Analysis options are available only in Advanced Mode.")
    unsupported = set(analysis_options) - {"target_column", "task_type"}
    if unsupported:
        raise ValueError(
            "Unsupported advanced option(s): " + ", ".join(sorted(unsupported)) + "."
        )
    target = analysis_options.get("target_column")
    task_type = analysis_options.get("task_type")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("Advanced target_column must be a non-empty string.")
    if task_type is not None and task_type not in {"classification", "regression"}:
        raise ValueError("Advanced task_type must be classification or regression.")
    return {
        key: value.strip() if isinstance(value, str) else value
        for key, value in analysis_options.items()
    }


def _apply_trusted_options(
    tool_name: str,
    arguments: dict[str, Any],
    options: dict[str, Any],
    plan: RequestPlan,
) -> dict[str, Any]:
    result = dict(arguments)
    execution_tools = {"train_models", "rerun_pipeline"}
    if tool_name in execution_tools:
        if plan.target_column:
            result["target_column"] = plan.target_column
        else:
            # Do not let the model resolve an ambiguous target by guessing.
            result.pop("target_column", None)
        if options.get("task_type"):
            result["task_type"] = options["task_type"]
    elif tool_name == "set_target_column" and options.get("target_column"):
        # An explicit target named in the current message takes precedence.
        result.setdefault("column", options["target_column"])
    elif tool_name == "propose_plan":
        if plan.target_column:
            result["target_column"] = plan.target_column
        result["task_type"] = options.get("task_type") or "auto"
    return result


def _tool_success(result: dict[str, Any]) -> bool:
    if result.get("ok") is not True:
        return False
    if result.get("status") == "partial_success":
        return result.get("analysis_status") == "success"
    return result.get("status") not in {"failed", "error", "needs_input"}


def _confirmed_result_summary(results: list[dict[str, Any]]) -> str:
    summaries = []
    for item in results:
        payload = item.get("result", {})
        summary = {"tool": item.get("name"), "status": payload.get("status", "success")}
        for key in (
            "target", "target_column", "task_type", "best_model", "best_score",
            "metric", "results", "evaluation", "final_test_metrics",
            "report_status", "pdf_status", "report_path",
        ):
            if key in payload and payload[key] is not None:
                summary[key] = payload[key]
        summaries.append(summary)
    return json.dumps(summaries, ensure_ascii=False, separators=(",", ":"), default=str)[:5000]


def _clarification(result: dict[str, Any]) -> str:
    error = str(result.get("error") or "The target column needs confirmation.")
    candidates = result.get("target_candidates") or []
    if candidates:
        return error + " Candidate columns: " + ", ".join(map(str, candidates)) + "."
    return error


def _safe_failure(tool_name: str, result: dict[str, Any], prior: list[dict[str, Any]]) -> str:
    message = str(result.get("error") or "The operation did not complete.")
    succeeded = [item.get("name", "operation") for item in prior if item.get("success")]
    prefix = (
        f"The {tool_name} operation did not complete."
        if tool_name
        else "The requested operation did not complete."
    )
    if succeeded:
        prefix += " Earlier confirmed steps completed: " + ", ".join(succeeded) + "."
    return f"{prefix} {message}"


async def run_agent(
    session_id: str,
    user_message: str,
    db: DBSession,
    run_id: str | None = None,
    mode: str = "simple",
    analysis_options: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Run one bounded turn while keeping run and owner context server controlled."""
    options = _validate_mode_options(mode, analysis_options)
    trusted_user_id = user_id
    session = None
    if run_id is not None:
        analysis_run = get_run(db, run_id)
        session = get_session(db, session_id)
        if (
            analysis_run is None
            or session is None
            or str(analysis_run.session_id) != str(session_id)
            or str(analysis_run.requested_by) != str(session.user_id)
            or (trusted_user_id is not None and str(analysis_run.requested_by) != str(trusted_user_id))
        ):
            raise PermissionError("The analysis run is not available to this session owner.")
        trusted_user_id = str(analysis_run.requested_by)
    elif trusted_user_id is not None:
        session = get_session(db, session_id)
        if session is None or str(session.user_id) != str(trusted_user_id):
            raise PermissionError("The session is not available to the authenticated user.")

    create_message(db, session_id=session_id, role="user", content=user_message)
    history = load_history(db, session_id)
    target = getattr(session, "target_column", None) if session is not None else None
    if run_id:
        emit_event(
            db,
            run_id,
            "planning.started",
            step="planning",
            progress=_run_progress(db, run_id),
            message="Preparing the request plan.",
            payload={"mode": mode},
        )
    plan = prepare_request_plan(
        session_id=session_id,
        user_message=user_message,
        mode=mode,
        analysis_options=options,
        current_target=target,
    )
    if run_id:
        emit_event(
            db,
            run_id,
            "planning.completed",
            step="planning",
            progress=5,
            message="The request plan is ready.",
            payload={"intent": plan.intent, "mode": mode},
        )
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": build_system_prompt(
            mode=mode,
            request_plan=plan.to_prompt_dict(),
            analysis_options=options,
        ),
    }]
    # Recent history is already scoped to this session; only retain complete
    # tool exchanges accepted by memory.load_history.
    messages.extend(history)
    tools = get_tool_schemas(set(plan.allowed_tools))

    all_tool_calls: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    successful_tools: set[str] = set()
    final_content = ""
    iterations = 0
    tool_call_count = 0
    terminal_status = "completed"
    cancellation_observed = False

    for iteration in range(1, settings.AGENT_MAX_ITERATIONS + 1):
        iterations = iteration
        if is_cancellation_requested(run_id):
            terminal_status = "cancelled"
            final_content = "The run was cancelled at a safe checkpoint. Completed results remain available."
            create_message(db, session_id=session_id, role="assistant", content=final_content)
            break
        if run_id:
            emit_event(
                db,
                run_id,
                "step.progress",
                step="thinking",
                progress=_run_progress(db, run_id),
                message="Preparing the next response.",
            )
        _close_db_if_open(db)
        try:
            response = await asyncio.wait_for(
                chat(messages=messages, tools=tools),
                timeout=settings.AGENT_LLM_TIMEOUT_SECONDS,
            )
            model_message = extract_message(response)
        except (
            AgentConfigurationError,
            MalformedAgentResponse,
            RuntimeError,
            ValueError,
            asyncio.TimeoutError,
        ) as exc:
            logger.warning(
                "Pipeline Agent response failed (run_id=%s error_type=%s)",
                run_id,
                type(exc).__name__,
            )
            if completed:
                terminal_status = "partial_success"
                final_content = (
                    "The backend operation completed, but the Agent could not finish "
                    "the conversational summary. Confirmed results: "
                    + _confirmed_result_summary(completed)
                )
            else:
                terminal_status = "failed"
                final_content = str(exc)
            create_message(db, session_id=session_id, role="assistant", content=final_content)
            break

        if is_cancellation_requested(run_id):
            terminal_status = "cancelled"
            final_content = "The run was cancelled at a safe checkpoint. Completed results remain available."
            create_message(db, session_id=session_id, role="assistant", content=final_content)
            break

        content = model_message.get("content") or ""
        tool_calls = model_message.get("tool_calls") or []
        if not tool_calls:
            final_content = content.strip() or "I could not produce a response. Please rephrase your request."
            create_message(db, session_id=session_id, role="assistant", content=final_content)
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
        tool_results_for_turn: list[tuple[str, dict[str, Any], str]] = []
        turn_failed = False

        for tool_call in tool_calls:
            tool_call_count += 1
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "")
            tool_call_id = str(tool_call.get("id") or "")
            try:
                if tool_call_count > settings.AGENT_MAX_TOOL_CALLS:
                    raise ToolValidationError(
                        "The maximum number of tool calls for this request has been reached."
                    )
                arguments = _parse_tool_arguments(function.get("arguments"))
                arguments = _apply_trusted_options(tool_name, arguments, options, plan)
                arguments = validate_tool_arguments(tool_name, arguments)
                if tool_name not in plan.allowed_tools:
                    raise UnknownToolError(
                        f"Tool '{tool_name}' is not enabled for this request. Ask the user before changing the requested operation."
                    )
                fingerprint = _call_fingerprint(tool_name, arguments)
                if fingerprint in seen_calls:
                    raise ToolValidationError(
                        "This identical tool call already ran in this request. Use its recorded result instead of repeating it."
                    )
                if (
                    tool_name in plan.execution_order
                    and len(plan.execution_order) > 1
                    and tool_name != plan.execution_order[0]
                    and plan.execution_order[0] in {"inspect_dataset", "get_session_status"}
                    and plan.execution_order[0] not in successful_tools
                ):
                    raise _PrerequisiteRequired(
                        f"Call {plan.execution_order[0]} first so the plan uses current session data."
                    )
                if tool_name in _EXPENSIVE_TOOLS and any(
                    item.get("name") == tool_name and item.get("success")
                    for item in completed
                ):
                    raise ToolValidationError(
                        "This expensive operation already succeeded in this request and will not be repeated."
                    )
                seen_calls.add(fingerprint)
            except (json.JSONDecodeError, ToolValidationError, TypeError, ValueError) as exc:
                error_type = "PrerequisiteRequired" if isinstance(exc, _PrerequisiteRequired) else type(exc).__name__
                error = _error_result(tool_name or "unknown", str(exc), error_type)
                result_text = format_tool_result(tool_name or "unknown", error)
                create_message(
                    db,
                    session_id=session_id,
                    role="tool",
                    content=result_text,
                    tool_results={"tool_call_id": tool_call_id, "tool": tool_name or "unknown"},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                })
                tool_results_for_turn.append((tool_name, error, tool_call_id))
                completed.append({"name": tool_name or "unknown", "arguments": {}, "result": error, "success": False, "attempted": False})
                turn_failed = True
                continue

            if run_id and is_cancellation_requested(run_id):
                cancellation_observed = True
                result = _error_result(
                    tool_name,
                    "The operation was not started because cancellation was requested.",
                    "CancellationRequested",
                )
            elif turn_failed:
                result = _error_result(
                    tool_name,
                    "This call was not executed because an earlier call in the same tool batch failed.",
                    "PriorToolCallFailed",
                )
            else:
                if run_id:
                    emit_event(
                        db,
                        run_id,
                        "tool.started",
                        step=tool_name,
                        progress=_run_progress(db, run_id),
                        message=f"Executing {tool_name}.",
                    )
                try:
                    _close_db_if_open(db)
                    future = asyncio.create_task(
                        run_in_threadpool(
                            _execute_tool_in_fresh_session,
                            session_id,
                            tool_name,
                            arguments,
                            run_id=run_id,
                            user_id=trusted_user_id,
                        )
                    )
                    result = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=settings.AGENT_TOOL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if run_id:
                        emit_event(
                            db,
                            run_id,
                            "tool.waiting_for_completion",
                            step=tool_name,
                            progress=_run_progress(db, run_id),
                            message=(
                                f"{tool_name} is still running after its observation timeout; "
                                "the run remains active until the operation can be reconciled."
                            ),
                            payload={"operation_still_running": True},
                        )
                    logger.warning(
                        "Tool exceeded observation timeout and continues safely (tool=%s run_id=%s)",
                        tool_name,
                        run_id,
                    )
                    try:
                        result = await future
                    except Exception as exc:
                        logger.exception(
                            "Timed-out tool later failed (tool=%s run_id=%s error_type=%s)",
                            tool_name,
                            run_id,
                            type(exc).__name__,
                        )
                        result = _error_result(
                            tool_name,
                            "The backend operation failed after its observation timeout.",
                            type(exc).__name__,
                        )
                except Exception as exc:
                    logger.exception(
                        "Pipeline tool execution failed (tool=%s run_id=%s error_type=%s)",
                        tool_name,
                        run_id,
                        type(exc).__name__,
                    )
                    result = _error_result(
                        tool_name,
                        "The backend operation failed unexpectedly. Check the run status before requesting another attempt.",
                        type(exc).__name__,
                    )

                if run_id and is_cancellation_requested(run_id):
                    cancellation_observed = True

            if not isinstance(result, dict):
                result = _error_result(
                    tool_name,
                    "The backend returned an invalid result.",
                    "InvalidToolResult",
                )
            result_text = format_tool_result(tool_name, result)
            create_message(
                db,
                session_id=session_id,
                role="tool",
                content=result_text,
                tool_results={"tool_call_id": tool_call_id, "tool": tool_name},
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_text,
            })
            succeeded = _tool_success(result)
            completed.append({
                "name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": succeeded,
                "attempted": not turn_failed,
            })
            if succeeded:
                successful_tools.add(tool_name)
            tool_results_for_turn.append((tool_name, result, tool_call_id))
            if run_id:
                emit_event(
                    db,
                    run_id,
                    "tool.completed",
                    step=tool_name,
                    progress=_run_progress(db, run_id),
                    message=(
                        f"{tool_name} completed."
                        if succeeded
                        else f"{tool_name} did not complete."
                    ),
                    payload={"success": succeeded},
                )
            if not succeeded:
                turn_failed = True

        if cancellation_observed:
            terminal_status = "cancelled"
            final_content = "The run was cancelled at a safe checkpoint. Completed results remain available."
            create_message(db, session_id=session_id, role="assistant", content=final_content)
            break

        if turn_failed:
            if (
                len(tool_results_for_turn) == 1
                and tool_results_for_turn[0][1].get("error_type") == "PrerequisiteRequired"
                and iteration < settings.AGENT_MAX_ITERATIONS
            ):
                messages.append({
                    "role": "system",
                    "content": "Complete the required session planning step first, then continue this same request without repeating successful work.",
                })
                continue
            failed_name, failed_result, _ = next(
                (name, result, call_id)
                for name, result, call_id in tool_results_for_turn
                if not _tool_success(result)
            )
            if failed_result.get("error_type") == "TargetSelectionRequired" or failed_result.get("status") == "needs_input":
                terminal_status = "needs_input"
                final_content = _clarification(failed_result)
            else:
                terminal_status = "partial_success" if any(item.get("success") for item in completed) else "failed"
                final_content = _safe_failure(failed_name, failed_result, completed[:-1])
            create_message(db, session_id=session_id, role="assistant", content=final_content)
            break
        else:
            messages.append({
                "role": "system",
                "content": "Use the confirmed tool results above. Do not repeat successful work or claim any unreported outcome.",
            })
    else:
        terminal_status = "incomplete"
        final_content = (
            "The request is incomplete because the Agent reached its iteration limit. "
            "Confirmed earlier tool results were preserved: "
            + _confirmed_result_summary(completed)
        )
        create_message(db, session_id=session_id, role="assistant", content=final_content)

    if run_id:
        emit_event(
            db,
            run_id,
            (
                "step.cancelled"
                if terminal_status == "cancelled"
                else "step.completed"
                if terminal_status in {"completed", "needs_input", "partial_success"}
                else "step.failed"
            ),
            step="chat",
            progress=_run_progress(db, run_id),
            message=(
                "Agent response completed."
                if terminal_status in {"completed", "needs_input"}
                else "Run cancellation reached a safe checkpoint."
                if terminal_status == "cancelled"
                else "Agent response is incomplete; see confirmed results."
            ),
            payload={"agent_status": terminal_status},
        )

    return {
        "status": terminal_status,
        "analysis_status": (
            "success"
            if any(item.get("success") and item.get("name") in _EXPENSIVE_TOOLS for item in completed)
            else "not_run"
        ),
        "content": final_content,
        "tool_calls": all_tool_calls,
        "tool_results": completed,
        "iterations": iterations,
        "mode": mode,
        "intent": plan.intent,
        "request_plan": plan.to_prompt_dict(),
    }
