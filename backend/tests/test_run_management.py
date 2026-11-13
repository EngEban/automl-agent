"""Focused local-database tests for run state, event history, SSE, and tasks."""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sse_starlette.sse import ensure_bytes

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from app.core.run_lifecycle import InvalidRunTransition, transition_run_status
from app.db.base import Base
from app.db import crud
from app.db.models import AnalysisRun, RunEvent, Session as UserSession, User
from app.services import run_service


@pytest.fixture
def run_store():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    user = User(email="run-owner@example.test", hashed_password="not-a-real-password")
    other_user = User(email="other-owner@example.test", hashed_password="not-a-real-password")
    db.add_all([user, other_user])
    db.flush()
    session = UserSession(
        user_id=user.id,
        name="test dataset",
        source_type="upload",
        source_path="local-test-only.csv",
    )
    other_session = UserSession(
        user_id=other_user.id,
        name="other dataset",
        source_type="upload",
        source_path="other-test-only.csv",
    )
    db.add_all([session, other_session])
    db.commit()
    store = SimpleNamespace(
        engine=engine,
        factory=factory,
        user_id=user.id,
        other_user_id=other_user.id,
        session_id=session.id,
        other_session_id=other_session.id,
    )
    db.close()
    try:
        yield store
    finally:
        engine.dispose()


def _new_run(store, *, status: str = "queued", other_owner: bool = False) -> str:
    db = store.factory()
    try:
        owner_id = store.other_user_id if other_owner else store.user_id
        session_id = store.other_session_id if other_owner else store.session_id
        run = AnalysisRun(
            session_id=session_id,
            requested_by=owner_id,
            status=status,
            config_json={},
        )
        db.add(run)
        db.commit()
        return run.id
    finally:
        db.close()


@pytest.fixture
def run_routes(run_store, monkeypatch):
    from app.api.routes import runs as routes

    monkeypatch.setattr(routes, "SessionLocal", run_store.factory)
    return routes


def test_run_transitions_set_timestamps_and_protect_terminal_state():
    run = SimpleNamespace(status="queued", started_at=None, finished_at=None)

    assert transition_run_status(run, "running") is True
    assert run.started_at is not None
    assert run.finished_at is None
    assert transition_run_status(run, "partially_completed") is True
    assert run.finished_at is not None

    with pytest.raises(InvalidRunTransition, match="cannot transition"):
        transition_run_status(run, "running")


def test_event_ids_order_payload_sanitization_and_cursor_pagination(run_store):
    run_id = _new_run(run_store)
    db = run_store.factory()
    try:
        first = run_service.emit_event(
            db,
            run_id,
            "step.progress",
            step="profiling",
            progress=10,
            message="Profile available.",
            payload={"rows": [{"private": "dataset row"}], "safe_count": 4},
        )
        repeated = run_service.emit_event(
            db,
            run_id,
            "step.progress",
            step="profiling",
            progress=10,
            message="Profile available.",
            payload={"rows": [{"private": "dataset row"}], "safe_count": 4},
        )
        third = run_service.emit_event(
            db,
            run_id,
            "training.started",
            step="training",
            progress=10,
            message="Training started.",
            payload={"candidate_count": 2},
        )
        all_events = crud.list_events_for_run(db, run_id)
        first_page = crud.list_event_page(db, run_id, limit=2)
        second_page = crud.list_event_page(
            db, run_id, cursor=first_page[-1].id, limit=2
        )

        assert first.id == repeated.id
        assert first.id != third.id
        assert [event.sequence for event in all_events] == [1, 2]
        assert all_events[0].payload == {"safe_count": 4}
        assert json.loads(json.dumps(all_events[0].payload)) == {"safe_count": 4}
        assert [event.id for event in first_page + second_page] == [first.id, third.id]
    finally:
        db.close()


def test_status_update_is_monotonic_and_failed_run_cannot_reopen(run_store):
    run_id = _new_run(run_store)
    db = run_store.factory()
    try:
        crud.update_run(db, run_id, status="running", progress_percent=45)
        run = crud.update_run(db, run_id, progress_percent=12)
        assert run.progress_percent == 45

        crud.update_run(db, run_id, status="failed")
        failed = crud.get_run(db, run_id)
        assert failed.finished_at is not None
        with pytest.raises(InvalidRunTransition):
            crud.update_run(db, run_id, status="running")
        db.rollback()
    finally:
        db.close()


def test_cancellation_waits_for_running_operation_and_cancels_queued_work(
    run_store, monkeypatch
):
    monkeypatch.setattr(run_service, "SessionLocal", run_store.factory)
    running_id = _new_run(run_store, status="running")
    queued_id = _new_run(run_store)
    db = run_store.factory()
    try:
        running, requested = run_service.request_cancellation(db, running_id)
        assert requested is True
        assert running.status == "cancelling"
        assert run_service.is_cancellation_requested(running_id) is True

        queued, requested = run_service.request_cancellation(db, queued_id)
        assert requested is True
        assert queued.status == "cancelled"
        assert queued.finished_at is not None
        assert run_service.is_cancellation_requested(queued_id) is True
    finally:
        db.close()


def test_database_lookup_failure_is_not_misreported_as_cancellation(monkeypatch):
    class BrokenDatabase:
        def get(self, *_args):
            raise RuntimeError("database unavailable")

        def close(self):
            pass

    monkeypatch.setattr(run_service, "SessionLocal", BrokenDatabase)
    with pytest.raises(RuntimeError, match="database unavailable"):
        run_service.is_cancellation_requested("run-id")


def test_duplicate_worker_delivery_does_not_start_a_second_agent(run_store, monkeypatch):
    from app.tasks import agent_tasks

    monkeypatch.setattr(agent_tasks, "SessionLocal", run_store.factory)
    run_id = _new_run(run_store)
    db = run_store.factory()
    try:
        claimed_run, claimed = run_service.claim_run(db, run_id)
        assert claimed is True
        assert claimed_run.status == "running"
    finally:
        db.close()

    agent = Mock()

    async def should_not_start(*_args, **_kwargs):
        agent()
        return {"status": "completed"}

    monkeypatch.setattr(agent_tasks, "run_agent", should_not_start)
    result = agent_tasks.run_agent_task.run(
        run_id, run_store.session_id, "Analyze the dataset."
    )

    assert result["status"] == "already_running"
    agent.assert_not_called()


def test_history_api_paginates_and_hides_other_users_runs(run_store, run_routes):
    run_id = _new_run(run_store, status="succeeded")
    other_run_id = _new_run(run_store, status="succeeded", other_owner=True)
    db = run_store.factory()
    try:
        for index in range(5):
            run_service.emit_event(
                db,
                run_id,
                "step.completed",
                step=f"step-{index}",
                progress=index,
                message=f"Completed step {index}.",
                payload={"index": index},
            )
    finally:
        db.close()

    pages = []
    cursor = None
    for _ in range(4):
        db = run_store.factory()
        try:
            page = run_routes.get_run_event_history(
                run_id,
                db,
                SimpleNamespace(id=run_store.user_id),
                limit=2,
                cursor=cursor,
            )
        finally:
            db.close()
        pages.extend(page["items"])
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]
    else:
        pytest.fail("Event history pagination did not reach its final page")

    assert [item["sequence"] for item in pages] == [1, 2, 3, 4, 5]
    assert len({item["id"] for item in pages}) == 5
    db = run_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            run_routes.get_run_event_history(
                other_run_id,
                db,
                SimpleNamespace(id=run_store.user_id),
            )
        assert error.value.status_code == 404
    finally:
        db.close()


def test_sse_formats_events_and_replays_after_last_event_id(run_store, run_routes):
    run_id = _new_run(run_store, status="succeeded")
    db = run_store.factory()
    try:
        events = [
            run_service.emit_event(
                db,
                run_id,
                "step.completed",
                step=f"step-{index}",
                progress=index,
                message=f"Event {index}.",
                payload={"index": index},
            )
            for index in range(3)
        ]
    finally:
        db.close()

    async def not_disconnected():
        return False

    request = SimpleNamespace(is_disconnected=not_disconnected)

    async def collect_sse(last_event_id=None):
        db = run_store.factory()
        try:
            response = await run_routes.stream_run_events(
                run_id,
                request,
                db,
                SimpleNamespace(id=run_store.user_id),
                last_event_id=last_event_id,
            )
            assert response.ping_interval == 15
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(ensure_bytes(chunk, response.sep).decode())
            return "".join(chunks)
        finally:
            db.close()

    initial = asyncio.run(asyncio.wait_for(collect_sse(), timeout=5))
    assert f"id: {events[0].id}" in initial
    assert "event: step.completed" in initial
    assert '"sequence":1' in initial
    assert f"id: {events[2].id}" in initial

    replay = asyncio.run(asyncio.wait_for(collect_sse(events[0].id), timeout=5))
    assert events[0].id not in replay
    assert f"id: {events[1].id}" in replay
    assert f"id: {events[2].id}" in replay

    db = run_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                run_routes.stream_run_events(
                    run_id,
                    request,
                    db,
                    SimpleNamespace(id=run_store.user_id),
                    last_event_id="not-a-uuid",
                )
            )
        assert error.value.status_code == 400
    finally:
        db.close()


def test_sse_reads_later_events_and_rejects_cross_run_cursor(run_store, run_routes):
    run_id = _new_run(run_store, status="running")
    other_run_id = _new_run(run_store, status="running", other_owner=True)
    db = run_store.factory()
    try:
        first = run_service.emit_event(
            db, run_id, "step.started", step="profiling", message="Started."
        )
        initial, _ = run_routes._read_stream_batch(run_id, None)
        late = run_service.emit_event(
            db, run_id, "step.completed", step="profiling", message="Complete."
        )
        tail, _ = run_routes._read_stream_batch(run_id, first.id)
        foreign = run_service.emit_event(
            db,
            other_run_id,
            "step.started",
            step="other",
            message="Other run.",
        )

        assert [item["id"] for item in initial] == [first.id]
        assert [item["id"] for item in tail] == [late.id]
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                run_routes.stream_run_events(
                    run_id,
                    SimpleNamespace(is_disconnected=lambda: None),
                    db,
                    SimpleNamespace(id=run_store.user_id),
                    last_event_id=foreign.id,
                )
            )
        assert error.value.status_code == 400
    finally:
        db.close()


def test_sse_closes_request_session_and_stops_after_disconnect(run_store, run_routes):
    run_id = _new_run(run_store, status="running")
    db = run_store.factory()
    original_close = db.close
    db.close = Mock(side_effect=original_close)

    async def disconnected():
        return True

    response = asyncio.run(
        run_routes.stream_run_events(
            run_id,
            SimpleNamespace(is_disconnected=disconnected),
            db,
            SimpleNamespace(id=run_store.user_id),
            last_event_id=None,
        )
    )

    async def consume():
        return [chunk async for chunk in response.body_iterator]

    try:
        assert response.ping_interval == 15
        assert db.close.call_count == 1
        assert asyncio.run(consume()) == []
    finally:
        db.close()


def test_cancel_endpoint_records_request_for_owned_run(run_store, run_routes):
    run_id = _new_run(run_store, status="running")
    db = run_store.factory()
    try:
        response = run_routes.cancel_run(
            run_id,
            db,
            SimpleNamespace(id=run_store.user_id),
        )
        assert response["status"] == "cancelling"
        assert response["cancellation_requested"] is True
    finally:
        db.close()


def test_worker_success_partial_report_failure_and_unexpected_failure(
    run_store, monkeypatch
):
    from app.tasks import agent_tasks

    monkeypatch.setattr(agent_tasks, "SessionLocal", run_store.factory)
    run_id = _new_run(run_store)

    async def completed_with_report_failure(*_args, **_kwargs):
        return {
            "status": "completed",
            "analysis_status": "success",
            "report_status": "failed",
            "content": "Analysis is available.",
            "tool_results": [],
        }

    monkeypatch.setattr(agent_tasks, "run_agent", completed_with_report_failure)
    result = agent_tasks.run_agent_task.run(
        run_id, run_store.session_id, "Analyze the dataset."
    )
    db = run_store.factory()
    try:
        assert result["analysis_status"] == "success"
        assert crud.get_run(db, run_id).status == "succeeded"
        assert [event.event_type for event in crud.list_events_for_run(db, run_id)][-1] == "run.succeeded"
    finally:
        db.close()

    partial_id = _new_run(run_store)

    async def partial(*_args, **_kwargs):
        return {"status": "partial_success", "analysis_status": "partial", "tool_results": []}

    monkeypatch.setattr(agent_tasks, "run_agent", partial)
    agent_tasks.run_agent_task.run(partial_id, run_store.session_id, "Analyze")
    db = run_store.factory()
    try:
        assert crud.get_run(db, partial_id).status == "partially_completed"
    finally:
        db.close()

    failed_id = _new_run(run_store)

    async def fail(*_args, **_kwargs):
        raise RuntimeError("private backend detail")

    monkeypatch.setattr(agent_tasks, "run_agent", fail)
    with pytest.raises(RuntimeError, match="private backend detail"):
        agent_tasks.run_agent_task.run(failed_id, run_store.session_id, "Analyze")
    db = run_store.factory()
    try:
        failed = crud.get_run(db, failed_id)
        failure_events = [
            event for event in crud.list_events_for_run(db, failed_id)
            if event.event_type == "run.failed"
        ]
        assert failed.status == "failed"
        assert failed.finished_at is not None
        assert len(failure_events) == 1
        assert "private backend detail" not in (failure_events[0].message or "")
        assert failure_events[0].payload == {"error_code": "agent_task_failed"}
    finally:
        db.close()

