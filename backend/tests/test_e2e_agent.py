"""End-to-end agent-loop tests with the provider and executor isolated."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from app.agent import agent as agent_module
from app.agent.llm import extract_message


def _response(*, content: str = "", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(arguments):
    function = SimpleNamespace(
        name="rerun_pipeline",
        arguments=arguments,
    )
    return SimpleNamespace(id="call_1", type="function", function=function)


def test_extract_message_serializes_dict_tool_arguments():
    response = _response(
        tool_calls=[_tool_call({"target_column": "price"})],
    )

    message = extract_message(response)
    arguments = message["tool_calls"][0]["function"]["arguments"]

    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"target_column": "price"}


def test_agent_executes_tool_then_finishes_without_argument_crash():
    responses = [
        _response(
            tool_calls=[
                _tool_call(json.dumps({"target_column": "price"}))
            ],
        ),
        _response(content="Pipeline finished successfully."),
    ]
    tool_arguments = []
    persisted = []
    llm_call_messages = []

    async def fake_chat(*, messages, tools):
        llm_call_messages.append(messages)
        return responses.pop(0)

    def fake_execute_tool(session_id, tool_name, arguments, db):
        tool_arguments.append((session_id, tool_name, arguments))
        return {
            "ok": True,
            "tool": tool_name,
            "status": "success",
        }

    def fake_create_message(db, **kwargs):
        persisted.append(kwargs)
        return SimpleNamespace(**kwargs)

    with (
        patch.object(agent_module, "chat", side_effect=fake_chat),
        patch.object(agent_module, "execute_tool", side_effect=fake_execute_tool),
        patch.object(agent_module, "create_message", side_effect=fake_create_message),
        patch.object(agent_module, "load_history", return_value=[]),
        patch.object(agent_module, "get_tool_schemas", return_value=[]),
    ):
        result = asyncio.run(
            agent_module.run_agent(
                session_id="session-1",
                user_message="Run the pipeline for price.",
                db=object(),
            )
        )

    assert result["content"] == "Pipeline finished successfully."
    assert result["iterations"] == 2
    assert tool_arguments == [
        ("session-1", "rerun_pipeline", {"target_column": "price"})
    ]
    assert len(llm_call_messages) == 2
    assistant_message = next(
        item
        for item in llm_call_messages[1]
        if item.get("role") == "assistant" and "tool_calls" in item
    )
    assistant_tool_calls = assistant_message["tool_calls"]
    assert isinstance(
        assistant_tool_calls[0]["function"]["arguments"],
        str,
    )
    assert any(
        item.get("role") == "assistant"
        and item.get("content") == "Pipeline finished successfully."
        for item in persisted
    )


def test_agent_stops_after_tool_error_without_retrying():
    persisted = []
    llm_calls = 0

    async def fake_chat(*, messages, tools):
        nonlocal llm_calls
        llm_calls += 1
        return _response(
            tool_calls=[_tool_call(json.dumps({"target_column": "price"}))],
        )

    def fake_create_message(db, **kwargs):
        persisted.append(kwargs)
        return SimpleNamespace(**kwargs)

    with (
        patch.object(agent_module, "chat", side_effect=fake_chat),
        patch.object(
            agent_module,
            "execute_tool",
            return_value={"ok": False, "error": "dataset is too small"},
        ),
        patch.object(agent_module, "create_message", side_effect=fake_create_message),
        patch.object(agent_module, "load_history", return_value=[]),
        patch.object(agent_module, "get_tool_schemas", return_value=[]),
    ):
        result = asyncio.run(
            agent_module.run_agent(
                session_id="session-1",
                user_message="Run the pipeline.",
                db=object(),
            )
        )

    assert llm_calls == 1
    assert result["iterations"] == 1
    assert result["content"] == (
        "Tool 'rerun_pipeline' failed: dataset is too small"
    )
    assert any(
        item.get("role") == "assistant"
        and item.get("content") == result["content"]
        for item in persisted
    )
