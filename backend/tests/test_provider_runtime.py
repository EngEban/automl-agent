from __future__ import annotations

import json
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.pipeline_agent import llm as pipeline_llm
from app.config.settings import settings
from app.db.base import Base
from app.db.models import AnalysisRun, Session, User
from app.services import provider_runtime
from app.services.provider_assignments import assign_profile
from app.services.provider_errors import ProviderManagementError
from app.services.provider_profiles import create_profile
from app.services.provider_registry import CAPABILITIES
from app.services.provider_runtime import (
    RuntimeProvider,
    ProviderRuntimeError,
    complete_async,
    complete_sync,
    resolve_assigned_provider,
)


SECRET_A = "runtime-user-a-secret"
SECRET_B = "runtime-user-b-secret"
_CHAT_RESPONSE = {"choices": [{"message": {"content": "OK"}}]}
_TOOL_RESPONSE = {
    "choices": [{"message": {"content": None, "tool_calls": [{
        "id": "probe-1",
        "type": "function",
        "function": {"name": "inspect_dataset", "arguments": "{}"},
    }]}}],
}


def _run_immediate(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    finally:
        coroutine.close()
    raise AssertionError("The mocked provider call unexpectedly performed asynchronous I/O.")


@pytest.fixture
def provider_store(monkeypatch):
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setattr(settings, "PROVIDER_CREDENTIAL_ENCRYPTION_KEY", key)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    users = [
        User(email="runtime-a@example.test", is_active=True),
        User(email="runtime-b@example.test", is_active=True),
    ]
    db.add_all(users)
    db.commit()
    for user in users:
        db.refresh(user)
    sessions = [
        Session(user_id=user.id, name=f"session-{index}", source_type="upload", source_path="unused.csv")
        for index, user in enumerate(users, start=1)
    ]
    db.add_all(sessions)
    db.commit()
    for session in sessions:
        db.refresh(session)
    runs = [
        AnalysisRun(session_id=session.id, requested_by=user.id, status="running")
        for session, user in zip(sessions, users)
    ]
    db.add_all(runs)
    db.commit()
    for run in runs:
        db.refresh(run)
    try:
        yield {"db": db, "users": users, "sessions": sessions, "runs": runs}
    finally:
        db.close()
        engine.dispose()


def _profile(db, user, *, provider_id, model, credential, configuration=None):
    profile = create_profile(
        db,
        user_id=str(user.id),
        display_name=f"{provider_id} profile",
        provider_id=provider_id,
        model_identifier=model,
        endpoint_url=("http://127.0.0.1:11434/v1" if provider_id == "ollama" else None),
        configuration=configuration or {"temperature": 0.15, "max_tokens": 256},
        credential=credential,
    )
    profile.connection_status = "connected"
    profile.capability_status_json = {name: "supported" for name in CAPABILITIES}
    db.commit()
    db.refresh(profile)
    return profile


def _assign(db, user, profile, role):
    return assign_profile(
        db, user_id=str(user.id), agent_role=role, profile_id=profile.id,
    )


def _resolve(db, user, session, run, role="pipeline_agent"):
    return resolve_assigned_provider(
        db,
        session_id=str(session.id),
        expected_user_id=str(user.id),
        agent_role=role,
        run_id=str(run.id),
    )


def test_pipeline_resolution_uses_each_authenticated_owner_and_role(provider_store):
    db, users, sessions, runs = (
        provider_store["db"], provider_store["users"],
        provider_store["sessions"], provider_store["runs"],
    )
    a_profile = _profile(
        db, users[0], provider_id="openai", model="gpt-test-a", credential=SECRET_A,
        configuration={"temperature": 0.1, "max_tokens": 111},
    )
    b_profile = _profile(
        db, users[1], provider_id="groq", model="openai/gpt-oss-20b", credential=SECRET_B,
        configuration={"temperature": 0.3, "max_tokens": 222},
    )
    _assign(db, users[0], a_profile, "pipeline_agent")
    _assign(db, users[1], b_profile, "pipeline_agent")
    report_profile = _profile(
        db, users[0], provider_id="anthropic", model="claude-test", credential="report-secret",
    )
    _assign(db, users[0], report_profile, "report_agent")

    pipeline_a = _resolve(db, users[0], sessions[0], runs[0])
    pipeline_b = _resolve(db, users[1], sessions[1], runs[1])
    report_a = _resolve(db, users[0], sessions[0], runs[0], "report_agent")

    assert (pipeline_a.provider_id, pipeline_a.model, pipeline_a.credential) == (
        "openai", "openai/gpt-test-a", SECRET_A,
    )
    assert pipeline_a.configuration["max_tokens"] == 111
    assert (pipeline_b.provider_id, pipeline_b.model, pipeline_b.credential) == (
        "groq", "groq/openai/gpt-oss-20b", SECRET_B,
    )
    assert pipeline_b.configuration["max_tokens"] == 222
    assert (report_a.provider_id, report_a.model) == ("anthropic", "anthropic/claude-test")
    assert report_a.credential == "report-secret"
    assert SECRET_A not in repr(pipeline_a)
    assert SECRET_B not in repr(pipeline_b)


def test_owner_run_role_and_missing_assignment_are_checked(provider_store):
    db, users, sessions, runs = (
        provider_store["db"], provider_store["users"],
        provider_store["sessions"], provider_store["runs"],
    )
    profile = _profile(
        db, users[0], provider_id="openai", model="test-model", credential=SECRET_A,
    )
    _assign(db, users[0], profile, "pipeline_agent")

    with pytest.raises(ProviderRuntimeError, match="authenticated owner"):
        _resolve(db, users[1], sessions[0], runs[0])
    with pytest.raises(ProviderRuntimeError, match="Report Agent") as missing:
        _resolve(db, users[0], sessions[0], runs[0], "report_agent")
    assert missing.value.code == "provider_assignment_required"
    with pytest.raises(ProviderRuntimeError, match="analysis run"):
        resolve_assigned_provider(
            db, session_id=sessions[0].id, expected_user_id=users[0].id,
            agent_role="pipeline_agent", run_id=runs[1].id,
        )


def test_invalid_saved_credential_fails_without_exposing_ciphertext(provider_store):
    db, user, session, run = (
        provider_store["db"], provider_store["users"][0],
        provider_store["sessions"][0], provider_store["runs"][0],
    )
    profile = _profile(
        db, user, provider_id="openai", model="test-model", credential=SECRET_A,
    )
    _assign(db, user, profile, "pipeline_agent")
    profile.credential_ciphertext = "enc:v1:invalid-ciphertext"
    db.commit()

    with pytest.raises(ProviderRuntimeError) as error:
        _resolve(db, user, session, run)
    assert error.value.code == "provider_credential_unavailable"
    assert SECRET_A not in str(error.value)
    assert "invalid-ciphertext" not in str(error.value)


def test_unsupported_required_capability_blocks_and_unknown_is_preserved(provider_store):
    db, user, session, run = (
        provider_store["db"], provider_store["users"][0],
        provider_store["sessions"][0], provider_store["runs"][0],
    )
    profile = _profile(
        db, user, provider_id="openai", model="test-model", credential=SECRET_A,
    )
    _assign(db, user, profile, "pipeline_agent")
    profile.capability_status_json = {**profile.capability_status_json, "tool_calling": "unsupported"}
    db.commit()
    with pytest.raises(ProviderRuntimeError) as unsupported:
        _resolve(db, user, session, run)
    assert unsupported.value.code == "incompatible_provider_capabilities"

    profile.capability_status_json = {**profile.capability_status_json, "tool_calling": "unknown"}
    db.commit()
    runtime = _resolve(db, user, session, run)
    assert runtime.capabilities["tool_calling"] == "unknown"
    assert runtime.unknown_required_capabilities == ("tool_calling",)

    report_profile = _profile(
        db, user, provider_id="openai", model="report-model", credential=SECRET_B,
    )
    _assign(db, user, report_profile, "report_agent")
    report_profile.capability_status_json = {
        **report_profile.capability_status_json,
        "json_schema": "unsupported",
    }
    db.commit()
    with pytest.raises(ProviderRuntimeError) as report_unsupported:
        _resolve(db, user, session, run, "report_agent")
    assert report_unsupported.value.code == "incompatible_provider_capabilities"


def test_provider_assignment_cannot_be_used_for_the_other_agent_role():
    pipeline_provider = _custom_runtime(role="pipeline_agent")
    with pytest.raises(ProviderRuntimeError) as mismatch:
        complete_sync(
            pipeline_provider,
            role="report_agent",
            messages=[{"role": "user", "content": "write report"}],
            timeout=5,
        )
    assert mismatch.value.code == "provider_role_mismatch"


def test_task_payload_and_report_graph_state_have_no_credential_fields():
    from app.agents.report_agent.state import ReportAgentState
    from app.tasks.agent_tasks import run_agent_task

    task_parameters = set(inspect.signature(run_agent_task.run).parameters)
    assert not task_parameters.intersection({"provider", "credential", "api_key"})
    state_fields = set(ReportAgentState.__annotations__)
    assert not any(
        any(secret_term in name.casefold() for secret_term in ("credential", "api_key", "secret"))
        for name in state_fields
    )


def test_celery_execution_resolves_provider_from_the_database_owner(provider_store, monkeypatch):
    from app.tasks import agent_tasks

    db, user, session, run = (
        provider_store["db"], provider_store["users"][0],
        provider_store["sessions"][0], provider_store["runs"][0],
    )
    profile = _profile(
        db, user, provider_id="openai", model="assigned-model", credential=SECRET_A,
    )
    _assign(db, user, profile, "pipeline_agent")
    captured = {}

    async def fake_run_agent(*_args, **kwargs):
        captured.update(kwargs)
        return {"status": "completed", "content": "done"}

    monkeypatch.setattr(agent_tasks, "SessionLocal", lambda: db)
    monkeypatch.setattr(agent_tasks, "get_run", lambda *_args: run)
    monkeypatch.setattr(agent_tasks, "get_session", lambda *_args: session)
    monkeypatch.setattr(agent_tasks, "claim_run", lambda *_args: (run, True))
    monkeypatch.setattr(agent_tasks, "emit_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_tasks, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        agent_tasks.asyncio,
        "run",
        lambda coroutine: _run_immediate(coroutine),
    )

    result = agent_tasks.run_agent_task.run(
        str(run.id), str(session.id), "Inspect this session.", "simple", None,
    )

    assert result["status"] == "completed"
    assert captured["user_id"] == str(user.id)
    assert captured["provider"].provider_id == "openai"
    assert captured["provider"].model == "openai/assigned-model"
    assert captured["provider"].credential == SECRET_A


def test_pipeline_tool_call_uses_selected_models_credentials_and_parameters(
    provider_store, monkeypatch,
):
    db, user, session, run = (
        provider_store["db"], provider_store["users"][0],
        provider_store["sessions"][0], provider_store["runs"][0],
    )
    profile = _profile(
        db, user, provider_id="openai", model="gpt-test", credential=SECRET_A,
        configuration={"temperature": 0.25, "max_tokens": 333, "frequency_penalty": 0.2},
    )
    _assign(db, user, profile, "pipeline_agent")
    runtime = _resolve(db, user, session, run)
    captured = {}
    monkeypatch.setenv("OPENAI_API_BASE", "https://developer.example/v1")

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _TOOL_RESPONSE

    import openai

    class InvocationAsyncClient:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    clients = []

    def async_client_factory(**kwargs):
        client = InvocationAsyncClient(**kwargs)
        clients.append(client)
        return client

    import litellm
    monkeypatch.setattr(openai, "AsyncOpenAI", async_client_factory)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = _run_immediate(pipeline_llm.chat(
        messages=[{"role": "user", "content": "Inspect the dataset."}],
        tools=[{"type": "function", "function": {"name": "inspect_dataset"}}],
        provider=runtime,
    ))

    assert result == _TOOL_RESPONSE
    assert captured["model"] == "openai/gpt-test"
    assert captured["api_base"] == "https://api.openai.com/v1"
    assert captured["api_key"] == SECRET_A
    assert captured["client"] is clients[0]
    assert clients[0].options["api_key"] == SECRET_A
    assert clients[0].options["base_url"] == "https://api.openai.com/v1"
    assert clients[0].options["max_retries"] == 0
    assert clients[0].closed
    assert captured["temperature"] == 0.25
    assert captured["max_tokens"] == 333
    assert captured["frequency_penalty"] == 0.2
    assert captured["tools"][0]["function"]["name"] == "inspect_dataset"
    assert captured["tool_choice"] == "auto"
    assert captured["stream"] is False
    assert captured["num_retries"] == 0


def test_concurrent_provider_calls_do_not_share_credentials(provider_store, monkeypatch):
    db, users = provider_store["db"], provider_store["users"]
    a = _profile(db, users[0], provider_id="openai", model="gpt-a", credential=SECRET_A)
    b = _profile(db, users[1], provider_id="openai", model="gpt-b", credential=SECRET_B)
    _assign(db, users[0], a, "pipeline_agent")
    _assign(db, users[1], b, "pipeline_agent")
    runtime_a = _resolve(db, users[0], provider_store["sessions"][0], provider_store["runs"][0])
    runtime_b = _resolve(db, users[1], provider_store["sessions"][1], provider_store["runs"][1])
    observed = []
    clients = []

    import openai

    class InvocationSyncClient:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False

        def close(self):
            self.closed = True

    def sync_client_factory(**kwargs):
        client = InvocationSyncClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", sync_client_factory)

    def fake_completion(**kwargs):
        observed.append((kwargs["model"], kwargs["api_key"], kwargs["client"]))
        return _CHAT_RESPONSE

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                complete_sync, runtime, role="pipeline_agent",
                messages=[{"role": "user", "content": "hello"}], timeout=5,
            )
            for runtime in (runtime_a, runtime_b)
        ]
        assert [future.result() for future in futures] == [_CHAT_RESPONSE, _CHAT_RESPONSE]

    assert {(model, key) for model, key, _client in observed} == {
        ("openai/gpt-a", SECRET_A),
        ("openai/gpt-b", SECRET_B),
    }
    assert all(key == client.options["api_key"] for _model, key, client in observed)
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert {client.options["api_key"] for client in clients} == {SECRET_A, SECRET_B}
    assert all(client.closed for client in clients)


@pytest.mark.parametrize(
    ("provider_id", "model", "expected"),
    [
        ("openai", "gpt-4.1", "openai/gpt-4.1"),
        ("anthropic", "claude-model", "anthropic/claude-model"),
        ("google_gemini", "gemini-model", "gemini/gemini-model"),
        ("groq", "openai/gpt-oss-20b", "groq/openai/gpt-oss-20b"),
        ("openrouter", "anthropic/claude-model", "openrouter/anthropic/claude-model"),
    ],
)
def test_litellm_provider_model_mapping(provider_id, model, expected):
    from app.services.provider_registry import resolve_litellm_model

    assert resolve_litellm_model(provider_id, model) == expected


def test_mismatched_litellm_prefix_is_rejected():
    from app.services.provider_registry import resolve_litellm_model

    with pytest.raises(ValueError, match="different LiteLLM provider prefix"):
        resolve_litellm_model("anthropic", "openai/gpt-model")


@pytest.mark.parametrize(
    ("provider_id", "expected"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("anthropic", "https://api.anthropic.com"),
        ("google_gemini", "https://generativelanguage.googleapis.com"),
        ("groq", "https://api.groq.com/openai/v1"),
        ("mistral", "https://api.mistral.ai/v1"),
        ("deepseek", "https://api.deepseek.com/beta"),
        ("openrouter", "https://openrouter.ai/api/v1"),
    ],
)
def test_managed_provider_uses_its_canonical_litellm_endpoint(provider_id, expected):
    from app.services.provider_registry import litellm_api_base_for

    assert litellm_api_base_for(provider_id) == expected


class _Response:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.is_redirect = 300 <= status_code < 400
        self._body = json.dumps(payload or _CHAT_RESPONSE).encode()
        self.headers = headers or {}
        self.closed = False

    async def aiter_bytes(self):
        yield self._body

    def iter_bytes(self):
        yield self._body

    async def aclose(self):
        self.closed = True

    def close(self):
        self.closed = True


class _MockClientBase:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.requests = []
        self.response = _Response()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def build_request(self, method, url, *, headers, json, extensions):
        import httpx

        return httpx.Request(
            method, url, headers=headers, json=json, extensions=extensions,
        )


class _AsyncMockClient(_MockClientBase):
    async def send(self, request, *, stream, follow_redirects):
        assert stream is True
        assert follow_redirects is False
        self.requests.append(request)
        return self.response


class _SyncMockClient(_MockClientBase):
    def send(self, request, *, stream, follow_redirects):
        assert stream is True
        assert follow_redirects is False
        self.requests.append(request)
        return self.response


def _custom_runtime(provider_id="custom_openai_compatible", role="pipeline_agent"):
    return RuntimeProvider(
        user_id="owner-1",
        agent_role=role,
        provider_id=provider_id,
        model_identifier="custom-model",
        model="custom-model",
        endpoint_url="https://custom.example/v1" if provider_id == "custom_openai_compatible" else "http://localhost:11434/v1",
        configuration={"temperature": 0.2, "max_tokens": 100},
        capabilities={name: "supported" for name in CAPABILITIES},
        unknown_required_capabilities=(),
        credential=SECRET_A,
    )


def test_custom_async_execution_uses_pinned_host_and_blocks_redirects(monkeypatch):
    endpoint = provider_runtime.PinnedEndpoint(
        base_url="https://93.184.216.34/v1",
        host_header="custom.example",
        sni_hostname="custom.example",
        scheme="https",
        pinned_ip="93.184.216.34",
    )

    async def resolve(_provider_id, _url):
        return endpoint

    clients = []
    redirect_response = []

    def client_factory(**kwargs):
        client = _AsyncMockClient(**kwargs)
        if redirect_response:
            client.response = redirect_response[0]
        clients.append(client)
        return client

    monkeypatch.setattr(provider_runtime, "resolve_and_pin_endpoint_async", resolve)
    monkeypatch.setattr(provider_runtime.httpx, "AsyncClient", client_factory)
    payload = _run_immediate(complete_async(
        _custom_runtime(), role="pipeline_agent",
        messages=[{"role": "user", "content": "inspect"}],
        tools=[{"type": "function", "function": {"name": "inspect_dataset"}}],
        timeout=7,
    ))
    request = clients[0].requests[0]
    body = json.loads(request.content)
    assert payload == _CHAT_RESPONSE
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "custom.example"
    assert request.extensions["sni_hostname"] == "custom.example"
    assert request.headers["authorization"] == f"Bearer {SECRET_A}"
    assert body["model"] == "custom-model"
    assert clients[0].options["trust_env"] is False
    assert clients[0].options["follow_redirects"] is False
    assert clients[0].response.closed

    redirect_response.append(_Response(
        status_code=302, payload={}, headers={"location": "http://169.254.169.254/"},
    ))
    with pytest.raises(ProviderRuntimeError) as redirect:
        _run_immediate(complete_async(
            _custom_runtime(), role="pipeline_agent",
            messages=[{"role": "user", "content": "inspect"}], timeout=7,
        ))
    assert redirect.value.code == "unsafe_provider_redirect"


def test_custom_sync_execution_and_unsafe_dns_use_same_security_boundary(monkeypatch):
    endpoint = provider_runtime.PinnedEndpoint(
        base_url="https://93.184.216.34/v1",
        host_header="custom.example",
        sni_hostname="custom.example",
        scheme="https",
        pinned_ip="93.184.216.34",
    )
    clients = []
    monkeypatch.setattr(provider_runtime, "resolve_and_pin_endpoint", lambda *_args: endpoint)
    def client_factory(**kwargs):
        client = _SyncMockClient(**kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(provider_runtime.httpx, "Client", client_factory)
    result = complete_sync(
        _custom_runtime(role="report_agent"), role="report_agent",
        messages=[{"role": "user", "content": "write JSON"}],
        response_format={"type": "json_schema", "json_schema": {"strict": True}},
        timeout=8,
    )
    client = clients[0]
    request = client.requests[0]
    assert result == _CHAT_RESPONSE
    assert request.url.host == "93.184.216.34"
    assert request.headers["host"] == "custom.example"
    assert json.loads(request.content)["response_format"]["json_schema"]["strict"] is True
    assert client.options["trust_env"] is False
    assert client.response.closed

    def reject_private(*_args):
        raise ProviderManagementError("private_destination", "private destination", 422)

    monkeypatch.setattr(provider_runtime, "resolve_and_pin_endpoint", reject_private)
    monkeypatch.setattr(
        provider_runtime.httpx, "Client",
        lambda **_kwargs: pytest.fail("Unsafe DNS results must not create an HTTP client."),
    )
    with pytest.raises(ProviderRuntimeError) as unsafe:
        complete_sync(
            _custom_runtime(role="report_agent"), role="report_agent",
            messages=[{"role": "user", "content": "write"}], timeout=8,
        )
    assert unsafe.value.code == "unsafe_provider_endpoint"
    assert SECRET_A not in str(unsafe.value)


def test_provider_failure_errors_and_logs_do_not_echo_credentials(provider_store, monkeypatch, caplog):
    db, user = provider_store["db"], provider_store["users"][0]
    profile = _profile(db, user, provider_id="openai", model="gpt-test", credential=SECRET_A)
    _assign(db, user, profile, "pipeline_agent")
    runtime = _resolve(db, user, provider_store["sessions"][0], provider_store["runs"][0])

    class AuthError(Exception):
        status_code = 401

    async def fail(**_kwargs):
        raise AuthError(f"invalid key {SECRET_A}")

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fail)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ProviderRuntimeError) as error:
            _run_immediate(complete_async(
                runtime, role="pipeline_agent",
                messages=[{"role": "user", "content": "hello"}], timeout=5,
            ))
    assert error.value.code == "provider_authentication_failed"
    assert SECRET_A not in str(error.value)
    assert SECRET_A not in caplog.text

