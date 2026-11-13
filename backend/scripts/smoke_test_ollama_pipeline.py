"""Live Ollama preflight and bounded synthetic Pipeline Agent smoke test.

Only request metadata and sanitized provider error bodies are printed. Endpoint
values, credentials, headers, full prompts, tool arguments, and synthetic
records are kept out of stdout and exception messages.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
ADVANCED_ONLY = "--advanced-only" in sys.argv
PIPELINE_ONLY = "--pipeline-only" in sys.argv or ADVANCED_ONLY


def report(stage: str, outcome: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"{stage}={outcome}{suffix}", flush=True)


def fail(stage: str, reason: str, code: int = 1) -> None:
    report(stage, "failed", reason)
    raise SystemExit(code)


try:
    from app.config import settings
except Exception as exc:
    fail("configuration", f"load_error_type={type(exc).__name__}")

if settings.LLM_PROVIDER.strip().lower() != "ollama":
    fail("configuration", "provider_is_not_ollama")
if settings.LLM_MODEL.strip() != "qwen3:14b":
    fail("configuration", "configured_model_mismatch_expected_qwen3_14b")
if not settings.LLM_BASE_URL or not settings.LLM_BASE_URL.strip():
    fail("configuration", "endpoint_not_configured")
if not settings.LLM_MODEL or not settings.LLM_MODEL.strip():
    fail("configuration", "model_not_configured")

parsed = urlsplit(settings.LLM_BASE_URL.strip())
if parsed.scheme not in {"http", "https"} or not parsed.netloc:
    fail("configuration", "endpoint_format_invalid")
report("configuration", "pass", "effective_model=qwen3:14b")
base_path = parsed.path.rstrip("/")
if base_path.endswith("/v1"):
    base_path = base_path[:-3]
native_root = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", "")).rstrip("/")

headers = {"Accept": "application/json"}
if settings.LLM_API_KEY:
    headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
if "ngrok" in parsed.netloc.lower():
    headers["ngrok-skip-browser-warning"] = "true"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, response_headers, newurl):
        return None


opener = build_opener(NoRedirect())


def get_json(path: str) -> dict:
    request = Request(native_root + path, headers=headers, method="GET")
    with opener.open(request, timeout=20) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"http_status_{response.status}")
        return json.loads(response.read().decode("utf-8"))


if PIPELINE_ONLY:
    report("endpoint_connectivity", "skipped", "previous_live_check_passed")
    report("model_query", "skipped", "previous_live_check_passed")
else:
    try:
        get_json("/api/version")
    except HTTPError as exc:
        fail("endpoint_connectivity", f"http_status={exc.code}")
    except (URLError, TimeoutError) as exc:
        fail("endpoint_connectivity", f"error_type={type(exc).__name__}")
    except Exception as exc:
        fail("endpoint_connectivity", f"error_type={type(exc).__name__}")
    report("endpoint_connectivity", "pass")

    try:
        model_payload = get_json("/api/tags")
        available_models = model_payload.get("models")
        if not isinstance(available_models, list):
            fail("model_query", "invalid_model_list")
        available_names = {
            str(item.get("name", "")).strip()
            for item in available_models
            if isinstance(item, dict)
        }
        configured_model = settings.LLM_MODEL.strip()
        normalized_model = configured_model.removeprefix("ollama_chat/").removeprefix("ollama/")
        if normalized_model not in available_names:
            fail(
                "model_availability",
                f"configured_model_present=false available_count={len(available_names)}",
            )
    except SystemExit:
        raise
    except HTTPError as exc:
        fail("model_query", f"http_status={exc.code}")
    except (URLError, TimeoutError) as exc:
        fail("model_query", f"error_type={type(exc).__name__}")
    except Exception as exc:
        fail("model_query", f"error_type={type(exc).__name__}")
    report("model_query", f"pass configured_model_present=true available_count={len(available_names)}")

try:
    from app.agent import llm as llm_module
except Exception as exc:
    fail("configuration", f"llm_import_error_type={type(exc).__name__}")

chat = llm_module.chat
extract_message = llm_module.extract_message
supports_native_tool_calling = llm_module.supports_native_tool_calling
if not supports_native_tool_calling(settings.LLM_PROVIDER, settings.LLM_MODEL):
    fail("native_tool_calling", "configured_model_family_not_supported")

_SAFE_ERROR_KEYS = {"error", "detail", "message", "code", "type", "status", "status_code"}
_SENSITIVE_KEYS = {"authorization", "api_key", "apikey", "token", "access_token", "secret", "password"}
_known_prompt_texts: set[str] = set()
_active_agent_iteration: int | None = None
_provider_attempts: dict[str, int] = {}


def _redact_text(value: object, limit: int = 400) -> str:
    text = str(value)
    if settings.LLM_BASE_URL:
        text = text.replace(settings.LLM_BASE_URL, "[configured-endpoint]")
    if settings.LLM_API_KEY:
        text = text.replace(settings.LLM_API_KEY, "[redacted]")
    for prompt in sorted(_known_prompt_texts, key=len, reverse=True):
        if prompt:
            text = text.replace(prompt, "[prompt]")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key|token|secret|password)\s*[:=]\s*[^,\s;]+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"https?://[^\s\"'<>]+", "[url]", text)
    return text[:limit]


def _safe_error_value(value: object, key: str = "") -> object:
    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in _SENSITIVE_KEYS or any(
        marker in normalized_key for marker in ("token", "secret", "password", "authorization")
    ):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(child_key): _safe_error_value(child_value, str(child_key))
            for child_key, child_value in value.items()
            if str(child_key).casefold().replace("-", "_") in _SAFE_ERROR_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_error_value(item) for item in value[:4]]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(value)


def _provider_error_details(exc: Exception) -> tuple[object, str, str]:
    response_obj = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response_obj is not None:
        status = getattr(response_obj, "status_code", None)

    headers_obj = getattr(response_obj, "headers", None)
    request_id_value = getattr(exc, "request_id", None)
    request_id = _redact_text(request_id_value, limit=128) if request_id_value else "unavailable"
    if headers_obj is not None:
        for key in ("x-request-id", "request-id", "x-correlation-id", "x-trace-id"):
            try:
                candidate = headers_obj.get(key)
            except Exception:
                candidate = None
            if candidate:
                request_id = _redact_text(candidate, limit=128)
                break

    body: object = _redact_text(getattr(exc, "body", "")) or "unavailable"
    if response_obj is not None:
        try:
            raw_body = response_obj.json()
        except Exception:
            raw_body = getattr(response_obj, "text", None)
        if isinstance(raw_body, dict):
            safe_body = {
                str(key): _safe_error_value(value, str(key))
                for key, value in raw_body.items()
                if str(key).casefold().replace("-", "_") in _SAFE_ERROR_KEYS
            }
            body = json.dumps(safe_body or {"body": "[error fields unavailable]"}, ensure_ascii=False)
        elif isinstance(raw_body, str):
            body = _redact_text(raw_body) or "[empty_body]"
        elif raw_body is not None:
            body = f"[body_type={type(raw_body).__name__}]"

    message = str(exc).casefold()
    if status in {401, 403} or "unauthorized" in message or "api key" in message:
        category = "authentication"
    elif status == 404 or "not found" in message:
        category = "route_or_model_not_found"
    elif isinstance(status, int) and status >= 500:
        category = "provider_http_5xx"
    elif "timeout" in type(exc).__name__.casefold() or "timed out" in message:
        category = "timeout"
    elif "connect" in type(exc).__name__.casefold() or "connection" in message:
        category = "connection"
    else:
        category = "provider_request_error"
    return status, category, f"request_id={request_id} response_body={body}"


original_acompletion = llm_module.acompletion


async def observe_provider_request(*args, **kwargs):
    started = perf_counter()
    phase_key = f"agent-{_active_agent_iteration}" if _active_agent_iteration is not None else "preflight"
    _provider_attempts[phase_key] = _provider_attempts.get(phase_key, 0) + 1
    attempt = _provider_attempts[phase_key]
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    model = kwargs.get("model", "unknown")
    timeout = kwargs.get("timeout", "configured")
    report(
        "provider_attempt",
        "started",
        f"iteration={_active_agent_iteration or 'preflight'} attempt={attempt} "
        f"started_utc={started_utc} model={model} timeout_s={timeout}",
    )
    try:
        response = await original_acompletion(*args, **kwargs)
        report(
            "provider_attempt",
            "completed",
            f"iteration={_active_agent_iteration or 'preflight'} attempt={attempt} "
            f"duration_ms={round((perf_counter() - started) * 1000)}",
        )
        return response
    except asyncio.CancelledError:
        report(
            "provider_attempt",
            "cancelled",
            f"iteration={_active_agent_iteration or 'preflight'} attempt={attempt} "
            f"duration_ms={round((perf_counter() - started) * 1000)}",
        )
        raise
    except Exception as exc:
        elapsed_ms = round((perf_counter() - started) * 1000)
        status, category, safe_details = _provider_error_details(exc)
        detail = (
            f"iteration={_active_agent_iteration or 'preflight'} attempt={attempt} "
            f"duration_ms={elapsed_ms} category={category} error_type={type(exc).__name__} "
            f"http_status={status if status is not None else 'unavailable'} {safe_details}"
        )
        report("provider_attempt", "failed", detail)
        raise


llm_module.acompletion = observe_provider_request


_agent_state = {
    "mode": "unset",
    "iteration": 0,
    "last_response_iteration": 0,
    "pending_tool_calls": deque(),
    "tool_results_forwarded": {},
    "tool_result_statuses": {},
    "tool_execution_statuses": {},
}


def _tool_call_summary(messages: list[dict]) -> list[dict[str, str]]:
    summary = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = llm_module.normalize_tool_calls(message.get("tool_calls")) or []
        for call in calls:
            function = call.get("function") or {}
            summary.append({
                "name": str(function.get("name") or "unknown"),
                "id": str(call.get("id") or "missing"),
            })
    return summary


def _tool_result_ids(messages: list[dict]) -> list[str]:
    return [
        str(message.get("tool_call_id") or "missing")
        for message in messages
        if message.get("role") == "tool"
    ]


def _safe_result_payload(content: object) -> dict:
    text = str(content or "")
    marker = "] "
    if marker not in text:
        return {}
    try:
        value = json.loads(text.split(marker, 1)[1])
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_tool_error_reason(payload: dict) -> str:
    """Classify known validation errors without printing model-supplied text."""
    error = str(payload.get("error", ""))
    error_type = str(payload.get("error_type", ""))
    if "Call get_session_status first" in error:
        return "prerequisite_get_session_status_first"
    if "Unsupported argument(s)" in error:
        return "unsupported_arguments"
    if "Missing required argument(s)" in error:
        return "missing_required_arguments"
    if "must be a JSON object" in error:
        return "arguments_not_object"
    if "must be one of:" in error:
        return "argument_enum_invalid"
    if "cannot be empty" in error:
        return "argument_empty"
    if "must be " in error:
        return "argument_type_invalid"
    if "maximum number of tool calls" in error:
        return "tool_call_limit"
    if "identical tool call already ran" in error:
        return "duplicate_tool_call"
    if error_type == "UnknownToolError":
        return "unknown_or_disallowed_tool"
    return "other_tool_error" if error else "none"


async def observe_agent_chat(original_chat, *, messages, tools, **kwargs):
    global _active_agent_iteration
    _agent_state["iteration"] += 1
    iteration = _agent_state["iteration"]
    _agent_state["last_response_iteration"] = iteration
    _active_agent_iteration = iteration
    _provider_attempts[f"agent-{iteration}"] = 0

    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    history_calls = _tool_call_summary(messages)
    result_ids = _tool_result_ids(messages)
    matched_ids = sorted({call["id"] for call in history_calls if call["id"] in result_ids})
    followup = bool(result_ids)
    for call_id in result_ids:
        _agent_state["tool_results_forwarded"][call_id] = call_id in matched_ids
    schema_names = [
        str(item.get("function", {}).get("name", "unknown"))
        for item in (tools or [])
        if isinstance(item, dict)
    ]
    model = llm_module._resolve_model()
    timeout = settings.AGENT_LLM_TIMEOUT_SECONDS
    roles = [str(message.get("role", "unknown")) for message in messages]
    report(
        "agent_iteration_request",
        "started",
        f"mode={_agent_state['mode']} iteration={iteration} started_utc={started_utc} model={model} timeout_s={timeout} "
        f"message_count={len(messages)} roles={json.dumps(roles)} "
        f"tools={json.dumps(schema_names)} prior_assistant_tool_calls={json.dumps(history_calls)} "
        f"tool_result_ids={json.dumps(result_ids)} matched_tool_result_ids={json.dumps(matched_ids)} "
        f"tool_results_appended={bool(result_ids) and len(matched_ids) == len(result_ids)} followup={followup}",
    )

    for message in messages:
        if isinstance(message.get("content"), str):
            _known_prompt_texts.add(message["content"])

    try:
        response = await original_chat(messages=messages, tools=tools, **kwargs)
        parsed = extract_message(response)
        response_calls = parsed.get("tool_calls") or []
        summary = []
        for call in response_calls:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments")
            try:
                parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                valid_argument_object = isinstance(parsed_arguments, dict)
                argument_key_count = len(parsed_arguments) if valid_argument_object else 0
            except (TypeError, ValueError):
                valid_argument_object = False
                argument_key_count = 0
            info = {
                "name": str(function.get("name") or "unknown"),
                "id": str(call.get("id") or "missing"),
                "valid_argument_object": valid_argument_object,
                "argument_key_count": argument_key_count,
            }
            summary.append(info)
            _agent_state["pending_tool_calls"].append(info)
        elapsed_ms = round((perf_counter() - started) * 1000)
        report(
            "agent_iteration_response",
            "pass",
            f"mode={_agent_state['mode']} iteration={iteration} duration_ms={elapsed_ms} followup={followup} "
            f"assistant_tool_calls={json.dumps(summary)} final_text_nonempty={bool(parsed.get('content', '').strip())}",
        )
        return response
    except asyncio.CancelledError:
        elapsed_ms = round((perf_counter() - started) * 1000)
        report(
            "agent_iteration_response",
            "cancelled",
            f"mode={_agent_state['mode']} iteration={iteration} duration_ms={elapsed_ms} followup={followup}",
        )
        raise
    except Exception as exc:
        elapsed_ms = round((perf_counter() - started) * 1000)
        report(
            "agent_iteration_response",
            "failed",
            f"mode={_agent_state['mode']} iteration={iteration} duration_ms={elapsed_ms} followup={followup} error_type={type(exc).__name__}",
        )
        raise
    finally:
        _active_agent_iteration = None


def observe_tool_execution(original_execute, session_id, tool_name, arguments, db, **context):
    pending = _agent_state["pending_tool_calls"]
    call_id = "unmatched"
    for index, call in enumerate(pending):
        if call["name"] == tool_name:
            call_id = call["id"]
            del pending[index]
            break
    iteration = _agent_state["last_response_iteration"]
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    started = perf_counter()
    report(
        "tool_execution",
        "started",
        f"mode={_agent_state['mode']} iteration={iteration} tool={tool_name} tool_call_id={call_id} started_utc={started_utc}",
    )
    try:
        result = original_execute(session_id, tool_name, arguments, db, **context)
    except Exception as exc:
        _agent_state["tool_execution_statuses"][call_id] = {
            "tool": tool_name,
            "ok": False,
            "status": "exception",
            "error_type": type(exc).__name__,
        }
        report(
            "tool_execution",
            "failed",
            f"mode={_agent_state['mode']} iteration={iteration} tool={tool_name} tool_call_id={call_id} "
            f"duration_ms={round((perf_counter() - started) * 1000)} error_type={type(exc).__name__}",
        )
        raise

    status = result.get("status", "field_absent") if isinstance(result, dict) else "invalid_result"
    ok = result.get("ok") is True if isinstance(result, dict) else False
    _agent_state["tool_execution_statuses"][call_id] = {
        "tool": tool_name,
        "ok": ok,
        "status": str(status),
        "error_type": str(result.get("error_type", "none")) if isinstance(result, dict) else "invalid_result",
    }
    report(
        "tool_execution",
        "completed" if ok else "failed",
        f"mode={_agent_state['mode']} iteration={iteration} tool={tool_name} tool_call_id={call_id} "
        f"duration_ms={round((perf_counter() - started) * 1000)} result_status={status} result_ok={ok} "
        f"error_type={result.get('error_type', 'none') if isinstance(result, dict) else 'invalid_result'}",
    )
    return result


def observe_create_message(original_create, db, **message):
    saved = original_create(db, **message)
    if message.get("role") == "tool":
        metadata = message.get("tool_results")
        metadata = metadata if isinstance(metadata, dict) else {}
        payload = _safe_result_payload(message.get("content"))
        call_id = str(metadata.get("tool_call_id") or "missing")
        execution = _agent_state["tool_execution_statuses"].get(call_id, {})
        result_status = payload.get("status", execution.get("status"))
        if result_status is None:
            result_status = "validation_error" if payload.get("error_type") else "field_absent"
        error_reason = _safe_tool_error_reason(payload)
        _agent_state["tool_result_statuses"][call_id] = {
            "tool": str(metadata.get("tool") or "unknown"),
            "status": str(result_status),
            "ok": payload.get(
                "ok",
                execution.get("ok", False),
            ) is True,
            "error_type": str(payload.get(
                "error_type",
                execution.get("error_type", "none"),
            )),
            "error_reason": error_reason,
        }
        report(
            "tool_result",
            "persisted",
            f"mode={_agent_state['mode']} iteration={_agent_state['last_response_iteration']} tool={metadata.get('tool', 'unknown')} "
            f"tool_call_id={call_id} result_status={_agent_state['tool_result_statuses'][call_id]['status']} "
            f"result_ok={_agent_state['tool_result_statuses'][call_id]['ok']} "
            f"error_type={_agent_state['tool_result_statuses'][call_id]['error_type']} "
            f"error_reason={error_reason}",
        )
    return saved

if PIPELINE_ONLY:
    report("simple_chat", "skipped", "previous_live_check_passed")
else:
    _known_prompt_texts.update({
        "Reply with exactly the single word READY.",
        "Confirm that the live chat endpoint is working.",
    })
    try:
        if not supports_native_tool_calling(settings.LLM_PROVIDER, settings.LLM_MODEL):
            fail("native_tool_calling", "configured_model_family_not_supported")
        response = asyncio.run(
            chat(
                messages=[
                    {"role": "system", "content": "Reply with exactly the single word READY."},
                    {"role": "user", "content": "Confirm that the live chat endpoint is working."},
                ],
                tools=None,
                max_retries=2,
            )
        )
        simple_message = extract_message(response)
        if not simple_message.get("content", "").strip():
            fail("simple_chat", "empty_completion")
    except SystemExit:
        raise
    except Exception as exc:
        fail("simple_chat", f"error_type={type(exc).__name__}")
    report("simple_chat", "pass nonempty_completion=true")

probe_tool = {
    "type": "function",
    "function": {
        "name": "ollama_integration_probe",
        "description": (
            "A harmless local test function. It only validates one fixed string "
            "and has no file, network, or data side effects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probe": {"type": "string", "enum": ["integration-ready"]}
            },
            "required": ["probe"],
            "additionalProperties": False,
        },
    },
}
if PIPELINE_ONLY:
    report("native_tool_calling", "skipped", "previous_live_check_passed")
else:
    _known_prompt_texts.update({
        "You are verifying native function calling. Call ollama_integration_probe exactly once with probe set to integration-ready. Do not answer in prose.",
        "Run the harmless integration probe now.",
    })
    try:
        response = asyncio.run(
            chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are verifying native function calling. Call "
                            "ollama_integration_probe exactly once with probe set to "
                            "integration-ready. Do not answer in prose."
                        ),
                    },
                    {"role": "user", "content": "Run the harmless integration probe now."},
                ],
                tools=[probe_tool],
                tool_choice="required",
                max_retries=1,
            )
        )
        tool_message = extract_message(response)
        calls = tool_message.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "ollama_integration_probe":
            fail("native_tool_calling", "expected_one_probe_tool_call")
        arguments = json.loads(calls[0]["function"]["arguments"])
        if arguments != {"probe": "integration-ready"}:
            fail("native_tool_calling", "probe_arguments_invalid")
        probe_result = {"ok": arguments["probe"] == "integration-ready"}
        if probe_result["ok"] is not True:
            fail("native_tool_calling", "probe_execution_failed")
    except SystemExit:
        raise
    except Exception as exc:
        fail("native_tool_calling", f"error_type={type(exc).__name__}")
    report("native_tool_calling", "pass probe_executed=true")

agent_module = None
executor = None
originals = None
try:
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.agent import agent as agent_module
    from app.agent import executor
    from app.agent.planning import prepare_request_plan
    from app.db.base import Base
    from app.db.models import Session as UserSession, User

    e2e_request = (
        "First call get_session_status, then call inspect_dataset. "
        "Using those results, tell me the row count and all column names."
    )
    pipeline_mode = "advanced" if ADVANCED_ONLY else "simple"
    pipeline_options = (
        {"target_column": "label", "task_type": "classification"}
        if pipeline_mode == "advanced"
        else None
    )
    plan = prepare_request_plan(
        session_id="synthetic-e2e",
        user_message=e2e_request,
        mode=pipeline_mode,
        analysis_options=pipeline_options,
    )
    bounded_tools = frozenset({"get_session_status", "inspect_dataset"})
    bounded_order = ("get_session_status", "inspect_dataset")
    if plan.execution_order != bounded_order:
        fail("pipeline_agent_e2e", "request_plan_order_unexpected")

    originals = {
        "agent_factory": agent_module.SessionLocal,
        "executor_factory": executor.SessionLocal,
        "sessions_dir": executor.SESSIONS_DIR,
        "max_iterations": settings.AGENT_MAX_ITERATIONS,
        "max_tools": settings.AGENT_MAX_TOOL_CALLS,
        "agent_chat": agent_module.chat,
        "agent_execute_tool": agent_module.execute_tool,
        "agent_create_message": agent_module.create_message,
        "agent_plan": agent_module.prepare_request_plan,
    }
    with tempfile.TemporaryDirectory(prefix="ollama-pipeline-e2e-") as temporary:
        root = Path(temporary)
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        try:
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            agent_module.SessionLocal = factory
            executor.SessionLocal = factory
            executor.SESSIONS_DIR = root / "sessions"
            settings.AGENT_MAX_ITERATIONS = min(originals["max_iterations"], 4)
            settings.AGENT_MAX_TOOL_CALLS = min(originals["max_tools"], 4)

            def bounded_request_plan(**kwargs):
                original_plan = originals["agent_plan"](**kwargs)
                return replace(
                    original_plan,
                    allowed_tools=bounded_tools,
                    execution_order=bounded_order,
                    requires_expensive_execution=False,
                )

            async def instrumented_agent_chat(*, messages, tools, **kwargs):
                return await observe_agent_chat(
                    originals["agent_chat"],
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )

            def instrumented_execute_tool(session_id, tool_name, arguments, db, **context):
                return observe_tool_execution(
                    originals["agent_execute_tool"],
                    session_id,
                    tool_name,
                    arguments,
                    db,
                    **context,
                )

            def instrumented_create_message(db, **message):
                return observe_create_message(
                    originals["agent_create_message"],
                    db,
                    **message,
                )

            agent_module.prepare_request_plan = bounded_request_plan
            agent_module.chat = instrumented_agent_chat
            agent_module.execute_tool = instrumented_execute_tool
            agent_module.create_message = instrumented_create_message

            synthetic = pd.DataFrame(
                {
                    "segment": ["A", "B", "A", "C", None, "A", "C", "B", "A", "C", "B", "A"],
                    "weekly_usage": [2, 5, None, 7, 4, 6, 8, 3, 5, 2, 7, 4],
                    "label": ["keep", "leave", "keep", "leave", "keep", "leave", "keep", "leave", "keep", "leave", "keep", "leave"],
                }
            )
            if len(synthetic) != 12 or not pd.api.types.is_numeric_dtype(synthetic["weekly_usage"]):
                fail("pipeline_agent_e2e", "synthetic_numeric_rows_invalid")
            if synthetic["segment"].nunique(dropna=True) < 2 or synthetic["label"].nunique() != 2:
                fail("pipeline_agent_e2e", "synthetic_categories_or_classes_invalid")
            if synthetic.isna().sum().sum() == 0:
                fail("pipeline_agent_e2e", "synthetic_missing_values_absent")
            report(
                "synthetic_dataset",
                "pass",
                "rows=12 numeric_column=true categorical_column=true missing_values=true classification_target=true",
            )
            db = factory()
            try:
                user = User(email="ollama-e2e@example.test", hashed_password="synthetic-only")
                db.add(user)
                db.flush()
                session = UserSession(
                    user_id=user.id,
                    name="Synthetic Ollama integration dataset",
                    source_type="upload",
                    source_path="synthetic-only.csv",
                    target_column="label",
                    dataset_meta={
                        "synthetic": True,
                        "loading_metadata": {
                            "original_row_count": len(synthetic),
                            "loaded_row_count": len(synthetic),
                            "sampling_applied": False,
                            "sampling_method": "none",
                            "sampling_seed": None,
                            "chunk_size": None,
                        },
                    },
                )
                db.add(session)
                db.commit()
                cache_dir = executor.SESSIONS_DIR / session.id
                cache_dir.mkdir(parents=True, exist_ok=True)
                synthetic.to_pickle(cache_dir / "raw.pkl")

                _known_prompt_texts.add(e2e_request)
                _agent_state["mode"] = pipeline_mode
                _agent_state["iteration"] = 0
                _agent_state["last_response_iteration"] = 0
                _agent_state["pending_tool_calls"].clear()
                _agent_state["tool_results_forwarded"].clear()
                _agent_state["tool_result_statuses"].clear()
                _agent_state["tool_execution_statuses"].clear()

                result = asyncio.run(
                    agent_module.run_agent(
                        session_id=session.id,
                        user_message=e2e_request,
                        db=db,
                        mode=pipeline_mode,
                        analysis_options=pipeline_options,
                    )
                )
                agent_status = result.get("status", "unknown")
                response_text = str(result.get("content", ""))
                final_response_nonempty = bool(response_text.strip())
                for call_id, tool_result in _agent_state["tool_result_statuses"].items():
                    execution = _agent_state["tool_execution_statuses"].get(call_id)
                    report(
                        "tool_final_summary",
                        "executed" if execution else "not_executed",
                        f"tool={tool_result['tool']} tool_call_id={call_id} "
                        f"execution_success={execution.get('ok') if execution else False} "
                        f"tool_result_status={tool_result['status']} tool_result_ok={tool_result['ok']} "
                        f"forwarded_to_followup={_agent_state['tool_results_forwarded'].get(call_id, False)} "
                        f"error_type={tool_result['error_type']} error_reason={tool_result['error_reason']}",
                    )
                for expected_tool in bounded_order:
                    if not any(
                        value["tool"] == expected_tool
                        for value in _agent_state["tool_result_statuses"].values()
                    ):
                        report("tool_final_summary", "not_called", f"tool={expected_tool}")
                    successful_tool_results = [
                        item for item in result.get("tool_results", [])
                        if item.get("name") == expected_tool and item.get("success") is True
                    ]
                    if not successful_tool_results:
                        fail("pipeline_agent_e2e", f"mode={pipeline_mode}_tool_not_successful_{expected_tool}")
                    result_ids = [
                        call_id for call_id, item in _agent_state["tool_result_statuses"].items()
                        if item["tool"] == expected_tool
                    ]
                    if not result_ids or not all(
                        _agent_state["tool_results_forwarded"].get(call_id, False)
                        for call_id in result_ids
                    ):
                        fail("pipeline_agent_e2e", f"mode={pipeline_mode}_tool_result_not_forwarded_{expected_tool}")
                inspected = [
                    item
                    for item in result.get("tool_results", [])
                    if item.get("name") == "inspect_dataset" and item.get("success") is True
                ]
                if not inspected:
                    fail("pipeline_agent_e2e", "inspect_dataset_tool_not_executed")
                inspection = inspected[0].get("result", {})
                if inspection.get("shape") != len(synthetic):
                    fail("pipeline_agent_e2e", "synthetic_row_count_mismatch")
                if set(inspection.get("columns", [])) != set(synthetic.columns):
                    fail("pipeline_agent_e2e", "synthetic_columns_mismatch")
                if synthetic.isna().sum().sum() == 0:
                    fail("pipeline_agent_e2e", "synthetic_missing_values_absent")
                response_folded = response_text.casefold()
                row_count_mentioned = "12" in response_folded or "twelve" in response_folded
                columns_mentioned = all(
                    str(column).casefold() in response_folded
                    for column in inspection.get("columns", [])
                )
                final_response_ok = (
                    agent_status == "completed"
                    and final_response_nonempty
                    and row_count_mentioned
                    and columns_mentioned
                )
                report(
                    "pipeline_agent_final_response",
                    "pass" if final_response_ok else "failed",
                    f"mode={pipeline_mode} agent_status={agent_status} nonempty={final_response_nonempty} "
                    f"grounded_in_tool_result={row_count_mentioned and columns_mentioned}",
                )
                if any(item.get("name") not in bounded_tools for item in result.get("tool_results", [])):
                    fail("pipeline_agent_e2e", "unexpected_expensive_tool_execution")
                tool_status = inspection.get("status", "field_absent")
                report(
                    "dataset_inspection_tool",
                    "pass",
                    f"tool_result_status={tool_status} success=true synthetic_rows=12 synthetic_columns=3",
                )
                if agent_status == "partial_success":
                    report("pipeline_agent_e2e", "partial_success", "dataset_inspection_tool=pass")
                    raise SystemExit(1)
                if agent_status != "completed":
                    fail("pipeline_agent_e2e", f"agent_status={agent_status}")
                if not final_response_ok:
                    fail("pipeline_agent_e2e", "final_response_not_grounded")
                report(
                    "pipeline_agent_e2e",
                    "pass",
                    f"mode={pipeline_mode} agent_status={agent_status} inspect_dataset_executed=true synthetic_rows=12 synthetic_columns=3",
                )

                if ADVANCED_ONLY:
                    raise SystemExit(0)

                advanced_options = {"target_column": "label", "task_type": "classification"}
                _agent_state["mode"] = "advanced"
                _agent_state["iteration"] = 0
                _agent_state["last_response_iteration"] = 0
                _agent_state["pending_tool_calls"].clear()
                _agent_state["tool_results_forwarded"].clear()
                _agent_state["tool_result_statuses"].clear()
                _agent_state["tool_execution_statuses"].clear()
                advanced_db = factory()
                try:
                    advanced_user = User(
                        email="ollama-e2e-advanced@example.test",
                        hashed_password="synthetic-only",
                    )
                    advanced_db.add(advanced_user)
                    advanced_db.flush()
                    advanced_session = UserSession(
                        user_id=advanced_user.id,
                        name="Synthetic Ollama integration dataset (advanced)",
                        source_type="upload",
                        source_path="synthetic-only.csv",
                        target_column="label",
                        dataset_meta={
                            "synthetic": True,
                            "loading_metadata": {
                                "original_row_count": len(synthetic),
                                "loaded_row_count": len(synthetic),
                                "sampling_applied": False,
                                "sampling_method": "none",
                                "sampling_seed": None,
                                "chunk_size": None,
                            },
                        },
                    )
                    advanced_db.add(advanced_session)
                    advanced_db.commit()
                    advanced_cache = executor.SESSIONS_DIR / advanced_session.id
                    advanced_cache.mkdir(parents=True, exist_ok=True)
                    synthetic.to_pickle(advanced_cache / "raw.pkl")
                    advanced_result = asyncio.run(
                        agent_module.run_agent(
                            session_id=advanced_session.id,
                            user_message=e2e_request,
                            db=advanced_db,
                            mode="advanced",
                            analysis_options=advanced_options,
                        )
                    )
                    advanced_status = advanced_result.get("status", "unknown")
                    advanced_text = str(advanced_result.get("content", ""))
                    advanced_inspections = [
                        item for item in advanced_result.get("tool_results", [])
                        if item.get("name") == "inspect_dataset" and item.get("success") is True
                    ]
                    advanced_profile = advanced_inspections[0].get("result", {}) if advanced_inspections else {}
                    advanced_rows = advanced_profile.get("shape")
                    advanced_columns = advanced_profile.get("columns", [])
                    advanced_text_folded = advanced_text.casefold()
                    advanced_grounded = (
                        (str(advanced_rows) in advanced_text_folded or (advanced_rows == 12 and "twelve" in advanced_text_folded))
                        and all(str(column).casefold() in advanced_text_folded for column in advanced_columns)
                    ) if advanced_rows is not None else False
                    advanced_final_ok = (
                        advanced_status == "completed"
                        and bool(advanced_text.strip())
                        and advanced_grounded
                    )
                    report(
                        "pipeline_agent_final_response",
                        "pass" if advanced_final_ok else "failed",
                        f"mode=advanced agent_status={advanced_status} nonempty={bool(advanced_text.strip())} "
                        f"grounded_in_tool_result={advanced_grounded}",
                    )
                    for call_id, tool_result in _agent_state["tool_result_statuses"].items():
                        execution = _agent_state["tool_execution_statuses"].get(call_id)
                        report(
                            "tool_final_summary",
                            "executed" if execution else "not_executed",
                            f"mode=advanced tool={tool_result['tool']} tool_call_id={call_id} "
                            f"execution_success={execution.get('ok') if execution else False} "
                            f"tool_result_status={tool_result['status']} tool_result_ok={tool_result['ok']} "
                            f"forwarded_to_followup={_agent_state['tool_results_forwarded'].get(call_id, False)} "
                            f"error_type={tool_result['error_type']} error_reason={tool_result['error_reason']}",
                        )
                    if any(item.get("name") not in bounded_tools for item in advanced_result.get("tool_results", [])):
                        fail("pipeline_agent_e2e", "mode=advanced unexpected_tool_executed")
                    for expected_tool in bounded_order:
                        if not any(
                            item.get("name") == expected_tool and item.get("success") is True
                            for item in advanced_result.get("tool_results", [])
                        ):
                            fail("pipeline_agent_e2e", f"mode=advanced tool_not_successful_{expected_tool}")
                        call_ids = [
                            call_id for call_id, item in _agent_state["tool_result_statuses"].items()
                            if item["tool"] == expected_tool
                        ]
                        if not call_ids or not all(
                            _agent_state["tool_results_forwarded"].get(call_id, False)
                            for call_id in call_ids
                        ):
                            fail("pipeline_agent_e2e", f"mode=advanced tool_result_not_forwarded_{expected_tool}")
                    if (
                        advanced_rows != len(synthetic)
                        or set(advanced_columns) != set(synthetic.columns)
                        or synthetic.isna().sum().sum() == 0
                    ):
                        fail("pipeline_agent_e2e", "mode=advanced synthetic_inspection_mismatch")
                    if advanced_status == "partial_success":
                        fail("pipeline_agent_e2e", "mode=advanced final_response_partial_success")
                    if advanced_status != "completed" or not advanced_final_ok:
                        fail("pipeline_agent_e2e", "mode=advanced final_response_not_grounded")
                    report(
                        "pipeline_agent_e2e",
                        "pass",
                        "mode=advanced agent_status=completed tools=2 synthetic_rows=12 synthetic_columns=3",
                    )
                finally:
                    advanced_db.close()
            finally:
                db.close()
        finally:
            engine.dispose()
except SystemExit:
    raise
except Exception as exc:
    fail("pipeline_agent_e2e", f"error_type={type(exc).__name__}")
finally:
    if originals is not None and agent_module is not None and executor is not None:
        agent_module.SessionLocal = originals["agent_factory"]
        executor.SessionLocal = originals["executor_factory"]
        executor.SESSIONS_DIR = originals["sessions_dir"]
        settings.AGENT_MAX_ITERATIONS = originals["max_iterations"]
        settings.AGENT_MAX_TOOL_CALLS = originals["max_tools"]
        agent_module.chat = originals["agent_chat"]
        agent_module.execute_tool = originals["agent_execute_tool"]
        agent_module.create_message = originals["agent_create_message"]
        agent_module.prepare_request_plan = originals["agent_plan"]
