from __future__ import annotations

import json
import socket

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import ai_providers as provider_routes
from app.config.settings import settings
from app.db.base import Base
from app.db.models import (
    AIProviderProfile,
    User,
    UserAgentProviderAssignment,
)
from app.schemas.ai_provider import (
    ProviderAssignmentSet,
    ProviderCredentialReplace,
    ProviderProfileCreate,
    ProviderProfileUpdate,
)
from app.services import provider_validation
from app.services.provider_credentials import (
    ProviderCredentialConfigurationError,
    decrypt_provider_credential,
    encrypt_provider_credential,
)
from app.services.provider_endpoint_security import (
    resolve_and_pin_endpoint,
    validate_endpoint_syntax,
)
from app.services.provider_registry import CAPABILITIES


SECRET = "provider-secret-that-must-never-leak"
NEW_SECRET = "replacement-provider-secret"
_TOOL_CALL = {
    "choices": [{
        "message": {
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "provider_capability_probe",
                    "arguments": "{}",
                },
            }],
        },
    }],
}
_JSON_RESPONSE = {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
_CHAT_RESPONSE = {"choices": [{"message": {"content": "OK"}}]}


class _MockResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._body = json.dumps(payload).encode("utf-8")
        self.headers = headers or {}
        self.is_redirect = 300 <= status_code < 400
        self.closed = False

    async def aiter_bytes(self):
        yield self._body

    async def aclose(self):
        self.closed = True


class _MockHttpClient:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.is_closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.is_closed = True

    def build_request(self, method, url, *, headers, json, extensions):
        return httpx.Request(
            method,
            url,
            headers=headers,
            json=json,
            extensions=extensions,
        )

    async def send(self, request, *, stream, follow_redirects):
        assert stream is True
        assert follow_redirects is False
        self.requests.append(request)
        return self.handler(request)

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def provider_context(monkeypatch):
    test_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setattr(settings, "PROVIDER_CREDENTIAL_ENCRYPTION_KEY", test_key)

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False,
    )
    db = session_factory()
    users = [
        User(email="provider-one@example.test", is_active=True),
        User(email="provider-two@example.test", is_active=True),
    ]
    db.add_all(users)
    db.commit()
    for user in users:
        db.refresh(user)

    try:
        yield {
            "db": db,
            "users": users,
            "key": test_key,
        }
    finally:
        db.close()
        engine.dispose()


def _payload(**updates) -> ProviderProfileCreate:
    data = {
        "display_name": "My OpenAI profile",
        "provider_id": "openai",
        "model_identifier": "arbitrary-model-id",
        "configuration": {"temperature": 0.2, "max_tokens": 128},
    }
    data.update(updates)
    return ProviderProfileCreate.model_validate(data)


def _create_profile(context, user=None, **updates):
    return provider_routes.create_provider_profile(
        _payload(**updates),
        context["db"],
        user or context["users"][0],
    )


def _run_immediate(coroutine):
    """Run async code whose mocked awaits complete without an event loop."""
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    finally:
        coroutine.close()
    raise AssertionError("A mocked provider test attempted asynchronous I/O.")


async def _mock_litellm_completion(**kwargs):
    if kwargs.get("tools"):
        return _TOOL_CALL
    if kwargs.get("response_format"):
        return _JSON_RESPONSE
    return _CHAT_RESPONSE


async def _resolve_endpoint_immediately(provider_id, endpoint_url):
    """Keep custom endpoint tests synchronous while exercising DNS checks."""
    return resolve_and_pin_endpoint(provider_id, endpoint_url)


def _enable_mock_cloud_probes(monkeypatch):
    monkeypatch.setattr(
        provider_validation, "_litellm_acompletion", _mock_litellm_completion,
    )


def _assert_http_error(call, expected_status: int, expected_code: str | None = None):
    with pytest.raises(HTTPException) as captured:
        call()
    assert captured.value.status_code == expected_status
    if expected_code:
        assert captured.value.detail["code"] == expected_code
    return captured.value


def test_ai_provider_routes_are_authenticated_and_catalog_is_capability_aware(
    provider_context,
):
    # Use the mounted application schema to check that each new operation has
    # the existing HTTP bearer dependency without starting an ASGI event loop.
    from app.main import app

    paths = app.openapi()["paths"]
    operations = [
        operation
        for path, methods in paths.items()
        if path.startswith("/api/settings/ai-providers")
        for operation in methods.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert operations
    assert all(operation.get("security") for operation in operations)

    catalog = provider_routes.get_catalog(provider_context["users"][0])
    entries = {entry.provider_id: entry for entry in catalog.providers}
    assert {
        "openai", "anthropic", "google_gemini", "groq", "mistral",
        "deepseek", "openrouter", "ollama", "vllm", "lm_studio",
        "custom_openai_compatible",
    } <= set(entries)
    assert set(entries["openai"].capabilities.values()) == {"unknown"}
    assert entries["openai"].agent_requirements["pipeline_agent"] == [
        "chat_completion", "tool_calling",
    ]
    assert entries["openai"].agent_requirements["report_agent"] == [
        "chat_completion", "structured_output", "json_schema",
    ]
    assert entries["openai"].supported_configuration == [
        "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
    ]
    assert len(CAPABILITIES) == 5


def test_profile_crud_encrypts_credentials_and_never_serializes_them(provider_context):
    db, user = provider_context["db"], provider_context["users"][0]
    created = _create_profile(provider_context, credential=SECRET)
    profile_id = created.id
    response_body = created.model_dump(mode="json")
    assert created.credential_configured is True
    assert SECRET not in json.dumps(response_body)
    assert "credential_ciphertext" not in response_body
    assert "credential_key_version" not in response_body

    profile = db.get(AIProviderProfile, profile_id)
    ciphertext = profile.credential_ciphertext
    assert ciphertext.startswith("enc:v1:")
    assert SECRET not in ciphertext
    assert profile.credential_key_version == "v1"
    assert decrypt_provider_credential(ciphertext, "v1") == SECRET

    listing = provider_routes.list_provider_profiles(db, user)
    assert [item.id for item in listing] == [profile_id]
    read = provider_routes.get_provider_profile(profile_id, db, user)
    assert read.credential_configured

    updated = provider_routes.update_provider_profile(
        profile_id,
        ProviderProfileUpdate.model_validate({"display_name": "Renamed profile"}),
        db,
        user,
    )
    assert updated.display_name == "Renamed profile"
    assert updated.credential_configured
    assert db.get(AIProviderProfile, profile_id).credential_ciphertext == ciphertext

    replaced = provider_routes.replace_provider_credential(
        profile_id,
        ProviderCredentialReplace.model_validate({"credential": NEW_SECRET}),
        db,
        user,
    )
    assert replaced.credential_configured
    replaced_ciphertext = db.get(AIProviderProfile, profile_id).credential_ciphertext
    assert replaced_ciphertext != ciphertext
    assert decrypt_provider_credential(replaced_ciphertext, "v1") == NEW_SECRET
    assert NEW_SECRET not in json.dumps(replaced.model_dump(mode="json"))

    removed = provider_routes.remove_provider_credential(profile_id, db, user)
    assert removed.credential_configured is False
    assert db.get(AIProviderProfile, profile_id).credential_ciphertext is None
    assert db.get(AIProviderProfile, profile_id).credential_key_version is None
    result = provider_routes.delete_provider_profile(profile_id, db, user)
    assert result.status_code == 204
    assert db.get(AIProviderProfile, profile_id) is None


def test_missing_or_invalid_encryption_key_fails_closed_but_reads_work(
    provider_context, monkeypatch,
):
    monkeypatch.setattr(settings, "PROVIDER_CREDENTIAL_ENCRYPTION_KEY", None)
    assert provider_routes.get_catalog(provider_context["users"][0]).providers
    _assert_http_error(
        lambda: _create_profile(provider_context, credential=SECRET),
        503,
        "credential_encryption_unavailable",
    )
    assert provider_context["db"].scalar(select(func.count()).select_from(AIProviderProfile)) == 0

    monkeypatch.setattr(settings, "PROVIDER_CREDENTIAL_ENCRYPTION_KEY", "not-a-fernet-key")
    _assert_http_error(
        lambda: _create_profile(provider_context, credential=SECRET),
        503,
        "credential_encryption_unavailable",
    )


def test_encryption_key_errors_versions_and_authentication_are_safe(
    provider_context, monkeypatch,
):
    monkeypatch.setattr(settings, "PROVIDER_CREDENTIAL_ENCRYPTION_KEY", None)
    with pytest.raises(ProviderCredentialConfigurationError) as missing:
        encrypt_provider_credential(SECRET)
    assert SECRET not in str(missing.value)

    monkeypatch.setattr(
        settings,
        "PROVIDER_CREDENTIAL_ENCRYPTION_KEY",
        provider_context["key"],
    )
    ciphertext, version = encrypt_provider_credential(SECRET)
    with pytest.raises(ProviderCredentialConfigurationError) as wrong_version:
        decrypt_provider_credential(ciphertext, "v9")
    assert SECRET not in str(wrong_version.value)

    monkeypatch.setattr(
        settings,
        "PROVIDER_CREDENTIAL_ENCRYPTION_KEY",
        Fernet.generate_key().decode("ascii"),
    )
    with pytest.raises(ProviderCredentialConfigurationError) as wrong_key:
        decrypt_provider_credential(ciphertext, version)
    assert SECRET not in str(wrong_key.value)


def test_profiles_and_credentials_are_isolated_by_owner(provider_context):
    db, user_one, user_two = (
        provider_context["db"],
        provider_context["users"][0],
        provider_context["users"][1],
    )
    profile = _create_profile(provider_context, credential=SECRET)
    profile_id = profile.id
    assert provider_routes.list_provider_profiles(db, user_two) == []

    operations = (
        lambda: provider_routes.get_provider_profile(profile_id, db, user_two),
        lambda: provider_routes.update_provider_profile(
            profile_id,
            ProviderProfileUpdate.model_validate({"display_name": "attacker"}),
            db,
            user_two,
        ),
        lambda: provider_routes.delete_provider_profile(profile_id, db, user_two),
        lambda: provider_routes.replace_provider_credential(
            profile_id,
            ProviderCredentialReplace.model_validate({"credential": NEW_SECRET}),
            db,
            user_two,
        ),
        lambda: provider_routes.remove_provider_credential(profile_id, db, user_two),
    )
    for operation in operations:
        error = _assert_http_error(operation, 404, "provider_profile_not_found")
        assert SECRET not in str(error.detail)
    assert provider_routes.get_provider_profile(profile_id, db, user_one).id == profile_id


def test_profile_validation_rejects_missing_model_secret_options_and_bad_values(
    provider_context,
):
    db, user = provider_context["db"], provider_context["users"][0]
    with pytest.raises(ValueError):
        _payload(model_identifier="")
    for configuration in (
        {"api_key": SECRET},
        {"temperature": 99},
        {"max_tokens": True},
    ):
        _assert_http_error(
            lambda configuration=configuration: _create_profile(
                provider_context, configuration=configuration,
            ),
            422,
        )
    assert db.scalar(select(func.count()).select_from(AIProviderProfile)) == 0


@pytest.mark.parametrize(
    ("provider_id", "endpoint_url", "expected_code"),
    [
        ("custom_openai_compatible", "ftp://example.com/v1", "unsupported_endpoint_scheme"),
        ("custom_openai_compatible", "https://user:pass@example.com/v1", "invalid_endpoint"),
        ("custom_openai_compatible", "https://example.com/v1?api_key=hidden-value", "endpoint_query_not_allowed"),
        ("custom_openai_compatible", "https://169.254.169.254/latest", "unsafe_endpoint"),
        ("custom_openai_compatible", "https://10.0.0.5/v1", "private_destination"),
        ("custom_openai_compatible", "http://example.com/v1", "https_required"),
        ("ollama", "http://192.168.1.20:11434/v1", "private_destination"),
    ],
)
def test_unsafe_endpoint_configurations_are_rejected(
    provider_context, provider_id, endpoint_url, expected_code,
):
    error = _assert_http_error(
        lambda: _create_profile(
            provider_context,
            provider_id=provider_id,
            endpoint_url=endpoint_url,
            credential=SECRET if provider_id == "custom_openai_compatible" else None,
        ),
        422,
        expected_code,
    )
    assert "hidden-value" not in str(error.detail)
    assert SECRET not in str(error.detail)


def test_loopback_endpoint_is_supported_without_dns_during_profile_creation(
    provider_context, monkeypatch,
):
    def fail_dns(*_args, **_kwargs):
        raise AssertionError("Saving a profile must not trigger DNS resolution.")

    monkeypatch.setattr(socket, "getaddrinfo", fail_dns)
    profile = _create_profile(
        provider_context,
        provider_id="ollama",
        model_identifier="qwen-local",
        endpoint_url="http://127.0.0.1:11434/v1/",
    )
    assert profile.endpoint_url == "http://127.0.0.1:11434/v1"


def test_missing_required_api_key_skips_provider_request(provider_context, monkeypatch):
    profile = _create_profile(provider_context)

    def fail_if_called(**_kwargs):
        pytest.fail("A profile without a required key must not call LiteLLM.")

    monkeypatch.setattr(provider_validation, "_litellm_acompletion", fail_if_called)
    result = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert result.status == "failed"
    assert result.category == "missing_api_key"
    assert set(result.capabilities.values()) == {"unknown"}


def test_mocked_cloud_connection_checks_chat_tool_and_json_schema(
    provider_context, monkeypatch,
):
    _enable_mock_cloud_probes(monkeypatch)
    profile = _create_profile(provider_context, credential=SECRET)
    result = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert result.status == "connected"
    assert result.capabilities == {
        "chat_completion": "supported",
        "tool_calling": "supported",
        "structured_output": "supported",
        "json_schema": "supported",
        "streaming": "unknown",
    }
    assert SECRET not in result.message
    assert SECRET not in json.dumps(result.capabilities)
    saved = provider_context["db"].get(AIProviderProfile, profile.id)
    assert saved.connection_status == "connected"
    assert saved.credential_ciphertext != SECRET
    assert saved.capability_status_json == result.capabilities


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (401, "authentication_failed"),
        (404, "unsupported_model"),
        (503, "network_failure"),
    ],
)
def test_provider_errors_are_classified_without_echoing_exception_text(
    provider_context, monkeypatch, caplog, status_code, expected_category,
):
    profile = _create_profile(provider_context, credential=SECRET)

    class ProviderFailure(Exception):
        def __init__(self):
            super().__init__(f"upstream included {SECRET}")
            self.status_code = status_code

    async def raise_provider_failure(**_kwargs):
        raise ProviderFailure()

    monkeypatch.setattr(
        provider_validation, "_litellm_acompletion", raise_provider_failure,
    )
    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert outcome.status == "failed"
    assert outcome.category == expected_category
    assert SECRET not in outcome.message
    assert SECRET not in caplog.text


def test_capability_rejection_is_distinguished_from_connection_failure(
    provider_context, monkeypatch,
):
    profile = _create_profile(provider_context, credential=SECRET)

    class BadRequest(Exception):
        status_code = 400

    async def completion(**kwargs):
        if kwargs.get("tools"):
            raise BadRequest("secret text should not be exposed")
        if kwargs.get("response_format"):
            return _JSON_RESPONSE
        return _CHAT_RESPONSE

    monkeypatch.setattr(provider_validation, "_litellm_acompletion", completion)
    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert outcome.status == "connected"
    assert outcome.capabilities["tool_calling"] == "unsupported"
    assert outcome.capabilities["json_schema"] == "supported"


def test_assignment_switching_roles_clearing_and_constraints(
    provider_context, monkeypatch,
):
    _enable_mock_cloud_probes(monkeypatch)
    db, user = provider_context["db"], provider_context["users"][0]
    first = _create_profile(provider_context, display_name="First", credential=SECRET)
    second = _create_profile(provider_context, display_name="Second", credential=NEW_SECRET)
    for profile in (first, second):
        outcome = _run_immediate(provider_routes.test_provider_connection(
            profile.id, db, user,
        ))
        assert outcome.status == "connected"

    pipeline = provider_routes.set_provider_assignment(
        "pipeline_agent", ProviderAssignmentSet(profile_id=first.id), db, user,
    )
    report = provider_routes.set_provider_assignment(
        "report_agent", ProviderAssignmentSet(profile_id=first.id), db, user,
    )
    assert pipeline.profile_id == report.profile_id == first.id

    changed = provider_routes.set_provider_assignment(
        "pipeline_agent", ProviderAssignmentSet(profile_id=second.id), db, user,
    )
    assert changed.profile_id == second.id
    assignment_count = db.scalar(
        select(func.count()).select_from(UserAgentProviderAssignment).where(
            UserAgentProviderAssignment.user_id == user.id,
            UserAgentProviderAssignment.agent_role == "pipeline_agent",
        )
    )
    assert assignment_count == 1
    assert provider_routes.get_provider_assignment(
        "report_agent", db, user,
    ).profile_id == first.id
    assert len(provider_routes.get_provider_assignments(db, user)) == 2

    provider_routes.clear_provider_assignment("pipeline_agent", db, user)
    assert provider_routes.get_provider_assignment("pipeline_agent", db, user) is None
    assert provider_routes.get_provider_assignment("report_agent", db, user).profile_id == first.id


def test_cross_user_assignment_is_rejected_and_delete_clears_assignment(
    provider_context, monkeypatch,
):
    _enable_mock_cloud_probes(monkeypatch)
    db, user_one, user_two = (
        provider_context["db"],
        provider_context["users"][0],
        provider_context["users"][1],
    )
    profile = _create_profile(provider_context, credential=SECRET)
    _run_immediate(provider_routes.test_provider_connection(profile.id, db, user_one))
    error = _assert_http_error(
        lambda: provider_routes.set_provider_assignment(
            "pipeline_agent",
            ProviderAssignmentSet(profile_id=profile.id),
            db,
            user_two,
        ),
        404,
        "provider_profile_not_found",
    )
    assert SECRET not in str(error.detail)

    provider_routes.set_provider_assignment(
        "pipeline_agent",
        ProviderAssignmentSet(profile_id=profile.id),
        db,
        user_one,
    )
    provider_routes.delete_provider_profile(profile.id, db, user_one)
    assert provider_routes.get_provider_assignments(db, user_one) == []


def test_model_changes_and_credential_rotation_invalidate_assignment(
    provider_context, monkeypatch,
):
    _enable_mock_cloud_probes(monkeypatch)
    db, user = provider_context["db"], provider_context["users"][0]
    profile = _create_profile(provider_context, credential=SECRET)

    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, db, user,
    ))
    assert outcome.status == "connected"
    provider_routes.set_provider_assignment(
        "pipeline_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )
    provider_routes.set_provider_assignment(
        "report_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )

    renamed = provider_routes.update_provider_profile(
        profile.id,
        ProviderProfileUpdate.model_validate({"display_name": "Renamed only"}),
        db,
        user,
    )
    assert renamed.connection_status == "connected"
    assert provider_routes.get_provider_assignment("pipeline_agent", db, user)
    assert provider_routes.get_provider_assignment("report_agent", db, user)

    changed = provider_routes.update_provider_profile(
        profile.id,
        ProviderProfileUpdate.model_validate({"model_identifier": "different-model"}),
        db,
        user,
    )
    assert changed.connection_status is None
    assert set(changed.capability_status.values()) == {"unknown"}
    assert provider_routes.get_provider_assignment("pipeline_agent", db, user) is None
    assert provider_routes.get_provider_assignment("report_agent", db, user) is None

    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, db, user,
    ))
    assert outcome.status == "connected"
    provider_routes.set_provider_assignment(
        "pipeline_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )
    provider_routes.set_provider_assignment(
        "report_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )
    provider_routes.replace_provider_credential(
        profile.id,
        ProviderCredentialReplace.model_validate({"credential": NEW_SECRET}),
        db,
        user,
    )
    assert db.get(AIProviderProfile, profile.id).connection_status is None
    assert provider_routes.get_provider_assignment("pipeline_agent", db, user) is None
    assert provider_routes.get_provider_assignment("report_agent", db, user) is None

    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, db, user,
    ))
    assert outcome.status == "connected"
    provider_routes.set_provider_assignment(
        "pipeline_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )
    provider_routes.set_provider_assignment(
        "report_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
    )
    removed = provider_routes.remove_provider_credential(profile.id, db, user)
    assert removed.credential_configured is False
    assert removed.connection_status is None
    assert provider_routes.get_provider_assignment("pipeline_agent", db, user) is None
    assert provider_routes.get_provider_assignment("report_agent", db, user) is None


def test_endpoint_and_configuration_changes_clear_both_agent_assignments(provider_context):
    db, user = provider_context["db"], provider_context["users"][0]
    profile = _create_profile(
        provider_context,
        provider_id="ollama",
        model_identifier="local-model",
        endpoint_url="http://127.0.0.1:11434/v1",
        configuration={"temperature": 0.2},
    )

    profile_row = db.get(AIProviderProfile, profile.id)
    profile_row.connection_status = "connected"
    profile_row.capability_status_json = {
        name: "supported" for name in CAPABILITIES
    }
    db.commit()
    for role in ("pipeline_agent", "report_agent"):
        provider_routes.set_provider_assignment(
            role, ProviderAssignmentSet(profile_id=profile.id), db, user,
        )

    endpoint_changed = provider_routes.update_provider_profile(
        profile.id,
        ProviderProfileUpdate.model_validate({
            "endpoint_url": "http://127.0.0.1:11435/v1",
        }),
        db,
        user,
    )
    assert endpoint_changed.connection_status is None
    assert set(endpoint_changed.capability_status.values()) == {"unknown"}
    assert provider_routes.get_provider_assignments(db, user) == []

    profile_row = db.get(AIProviderProfile, profile.id)
    profile_row.connection_status = "connected"
    profile_row.capability_status_json = {
        name: "supported" for name in CAPABILITIES
    }
    db.commit()
    for role in ("pipeline_agent", "report_agent"):
        provider_routes.set_provider_assignment(
            role, ProviderAssignmentSet(profile_id=profile.id), db, user,
        )

    configuration_changed = provider_routes.update_provider_profile(
        profile.id,
        ProviderProfileUpdate.model_validate({
            "configuration": {"temperature": 0.7},
        }),
        db,
        user,
    )
    assert configuration_changed.connection_status is None
    assert set(configuration_changed.capability_status.values()) == {"unknown"}
    assert provider_routes.get_provider_assignments(db, user) == []


def test_unsupported_capability_blocks_assignment_unknown_is_returned(
    provider_context, monkeypatch,
):
    _enable_mock_cloud_probes(monkeypatch)
    db, user = provider_context["db"], provider_context["users"][0]
    profile = _create_profile(provider_context, credential=SECRET)
    _run_immediate(provider_routes.test_provider_connection(profile.id, db, user))

    saved = db.get(AIProviderProfile, profile.id)
    saved.capability_status_json = {
        **saved.capability_status_json,
        "tool_calling": "unsupported",
    }
    db.commit()
    _assert_http_error(
        lambda: provider_routes.set_provider_assignment(
            "pipeline_agent",
            ProviderAssignmentSet(profile_id=profile.id),
            db,
            user,
        ),
        409,
        "incompatible_capabilities",
    )

    saved.capability_status_json = {
        **saved.capability_status_json,
        "tool_calling": "unknown",
    }
    db.commit()
    assignment = provider_routes.set_provider_assignment(
        "pipeline_agent",
        __import__("app.schemas.ai_provider", fromlist=["ProviderAssignmentSet"]).ProviderAssignmentSet(
            profile_id=profile.id,
        ),
        db,
        user,
    )
    assert assignment.unverified_capabilities == ["tool_calling"]


def test_endpoint_dns_private_answer_is_rejected_before_http_client_creation(
    provider_context, monkeypatch,
):
    profile = _create_profile(
        provider_context,
        provider_id="custom_openai_compatible",
        model_identifier="remote/model",
        endpoint_url="https://custom.example/v1",
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443)),
        ],
    )
    monkeypatch.setattr(
        provider_validation,
        "_create_http_client",
        lambda: pytest.fail("Unsafe DNS answers must not create a client."),
    )
    monkeypatch.setattr(
        provider_validation,
        "resolve_and_pin_endpoint_async",
        _resolve_endpoint_immediately,
    )
    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert outcome.status == "failed"
    assert outcome.category == "private_destination"


def test_public_custom_dns_is_pinned_and_redirects_are_not_followed(
    provider_context, monkeypatch,
):
    profile = _create_profile(
        provider_context,
        provider_id="custom_openai_compatible",
        model_identifier="remote/model",
        endpoint_url="https://custom.example/v1",
    )
    dns_calls = []

    def public_dns(host, port, **_kwargs):
        dns_calls.append((host, port))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
        ]

    clients = []

    def success_handler(request):
        payload = json.loads(request.content)
        if payload.get("tools"):
            response = _TOOL_CALL
        elif payload.get("response_format"):
            response = _JSON_RESPONSE
        else:
            response = _CHAT_RESPONSE
        return _MockResponse(200, response)

    def client_factory():
        client = _MockHttpClient(success_handler)
        clients.append(client)
        return client

    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(
        provider_validation,
        "resolve_and_pin_endpoint_async",
        _resolve_endpoint_immediately,
    )
    monkeypatch.setattr(
        provider_validation, "_create_http_client", client_factory,
    )
    outcome = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert outcome.status == "connected"
    assert len(clients[0].requests) == 3
    assert dns_calls == [("custom.example", 443)]
    for request in clients[0].requests:
        assert request.url.host == "93.184.216.34"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["host"] == "custom.example"
        assert request.extensions["sni_hostname"] == "custom.example"

    def redirect_handler(_request):
        return _MockResponse(
            302,
            {},
            headers={"location": "http://169.254.169.254/latest"},
        )

    monkeypatch.setattr(
        provider_validation,
        "_create_http_client",
        lambda: _MockHttpClient(redirect_handler),
    )
    redirected = _run_immediate(provider_routes.test_provider_connection(
        profile.id, provider_context["db"], provider_context["users"][0],
    ))
    assert redirected.status == "failed"
    assert redirected.category == "unsafe_redirect"


def test_localhost_is_resolved_checked_and_pinned_to_loopback(provider_context, monkeypatch):
    endpoint = validate_endpoint_syntax("ollama", "http://localhost:11434/v1")
    assert endpoint == "http://localhost:11434/v1"
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 11434, 0, 0)),
        ],
    )
    pinned = resolve_and_pin_endpoint("ollama", endpoint)
    assert pinned.base_url in {
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
    }
    assert pinned.host_header == "localhost:11434"


def test_assignment_requires_key_and_a_successful_explicit_connection_test(
    provider_context,
):
    db, user = provider_context["db"], provider_context["users"][0]
    profile = _create_profile(provider_context)
    _assert_http_error(
        lambda: provider_routes.set_provider_assignment(
            "pipeline_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
        ),
        409,
        "credential_required",
    )

    saved = db.get(AIProviderProfile, profile.id)
    saved.credential_ciphertext = "test-ciphertext"
    saved.credential_key_version = "v1"
    db.commit()
    _assert_http_error(
        lambda: provider_routes.set_provider_assignment(
            "pipeline_agent", ProviderAssignmentSet(profile_id=profile.id), db, user,
        ),
        409,
        "connection_test_required",
    )


def test_database_constraint_prevents_duplicate_user_role_assignment(provider_context):
    db, user = provider_context["db"], provider_context["users"][0]
    profile = AIProviderProfile(
        user_id=user.id,
        display_name="constraint-test",
        provider_id="ollama",
        model_identifier="model",
        endpoint_url="http://127.0.0.1:11434/v1",
        configuration_json={},
        capability_status_json={name: "unknown" for name in CAPABILITIES},
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    db.add_all([
        UserAgentProviderAssignment(
            user_id=user.id, profile_id=profile.id, agent_role="pipeline_agent",
        ),
        UserAgentProviderAssignment(
            user_id=user.id, profile_id=profile.id, agent_role="pipeline_agent",
        ),
    ])
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
