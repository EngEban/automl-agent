"""Regression tests for ownership and file access in legacy analysis routes."""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from app.api.routes import analysis as analysis_routes
from app.api.routes import artifacts as artifact_routes
from app.config.settings import settings
from app.db.base import Base
from app.db.models import AnalysisRun, Artifact, Session as UserSession, User
from app.api.deps import get_current_user, security_scheme
from app.schemas.session import SessionRead


@pytest.fixture
def security_store():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    owner = User(email="analysis-owner@example.test", hashed_password="test-only")
    other = User(email="analysis-other@example.test", hashed_password="test-only")
    db.add_all([owner, other])
    db.flush()
    owner_session = UserSession(
        user_id=owner.id,
        name="owner data",
        source_type="upload",
        source_path="owner.csv",
        dataset_meta={},
    )
    other_session = UserSession(
        user_id=other.id,
        name="other data",
        source_type="upload",
        source_path="other.csv",
        dataset_meta={},
    )
    db.add_all([owner_session, other_session])
    db.commit()
    result = SimpleNamespace(
        engine=engine,
        factory=factory,
        owner_id=owner.id,
        other_id=other.id,
        owner_session_id=owner_session.id,
        other_session_id=other_session.id,
    )
    db.close()
    try:
        yield result
    finally:
        engine.dispose()


def _add_run(
    store,
    *,
    task_id: str | None = None,
    owner_id: str | None = None,
    session_id: str | None = None,
) -> str:
    task_id = task_id or str(uuid.uuid4())
    with store.factory() as db:
        db.add(
            AnalysisRun(
                task_id=task_id,
                requested_by=owner_id or store.owner_id,
                session_id=session_id or store.owner_session_id,
                status="queued",
                config_json={"legacy_analysis": True},
            )
        )
        db.commit()
    return task_id


def _user(user_id: str):
    return SimpleNamespace(id=user_id)


def _run_immediate(coroutine):
    """Drive test coroutines that only await immediate in-memory operations."""
    while True:
        try:
            yielded = coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        if yielded is not None:
            raise AssertionError("The test coroutine unexpectedly scheduled async I/O")


class _MemoryUpload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = io.BytesIO(content)

    async def read(self, size: int) -> bytes:
        return self._content.read(size)

    async def close(self) -> None:
        self._content.close()


@pytest.fixture
def storage_roots(monkeypatch, tmp_path):
    storage = tmp_path / "storage"
    uploads = storage / "uploads"
    reports = storage / "reports"
    models = storage / "models"
    for path in (uploads, reports, models):
        path.mkdir(parents=True)
    monkeypatch.setattr(settings, "STORAGE_DIR", storage)
    monkeypatch.setattr(settings, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(settings, "REPORT_DIR", reports)
    monkeypatch.setattr(settings, "MODEL_DIR", models)
    return SimpleNamespace(storage=storage, uploads=uploads, reports=reports, models=models)


def test_all_legacy_routes_declare_authenticated_user_dependency() -> None:
    protected_paths = {
        "/analyze/upload",
        "/analyze/url",
        "/status/{task_id}",
        "/download/report/{task_id}",
        "/download/model/{task_id}",
    }
    registered = {route.path: route for route in analysis_routes.router.routes}
    assert protected_paths <= registered.keys()

    def dependency_calls(dependant):
        for child in dependant.dependencies:
            yield child.call
            yield from dependency_calls(child)

    for path in protected_paths:
        route = registered[path]
        assert get_current_user in set(dependency_calls(route.dependant)), path


def test_missing_bearer_credentials_are_rejected() -> None:
    from starlette.requests import Request

    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    with pytest.raises(HTTPException) as error:
        _run_immediate(security_scheme(request))
    assert error.value.status_code in {401, 403}


def test_owned_task_requires_matching_run_and_parent_session(security_store) -> None:
    owned_task = _add_run(security_store)
    mismatched_parent_task = _add_run(
        security_store,
        owner_id=security_store.owner_id,
        session_id=security_store.other_session_id,
    )
    db = security_store.factory()
    try:
        assert analysis_routes._owned_task(
            owned_task, db, _user(security_store.owner_id)
        ).task_id == owned_task
        for task_id in (owned_task, mismatched_parent_task, str(uuid.uuid4())):
            with pytest.raises(HTTPException) as error:
                analysis_routes._owned_task(
                    task_id, db, _user(security_store.other_id)
                )
            assert error.value.status_code == 404
        with pytest.raises(HTTPException) as malformed:
            analysis_routes._owned_task("not-a-uuid", db, _user(security_store.owner_id))
        assert malformed.value.status_code == 404
    finally:
        db.close()


def test_duplicate_task_ids_are_rejected(security_store) -> None:
    task_id = str(uuid.uuid4())
    _add_run(security_store, task_id=task_id)
    _add_run(security_store, task_id=task_id)
    db = security_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            analysis_routes._owned_task(task_id, db, _user(security_store.owner_id))
        assert error.value.status_code == 404
    finally:
        db.close()


def test_status_is_only_queried_after_database_ownership(security_store, monkeypatch) -> None:
    task_id = _add_run(security_store)
    async_result = Mock(return_value=SimpleNamespace(state="PENDING", info=None))
    monkeypatch.setattr(analysis_routes, "AsyncResult", async_result)
    db = security_store.factory()
    try:
        own_status = _run_immediate(
            analysis_routes.get_status(task_id, db, _user(security_store.owner_id))
        )
        assert own_status.status == "queued"
        async_result.assert_called_once()

        async_result.reset_mock()
        with pytest.raises(HTTPException) as error:
            _run_immediate(
                analysis_routes.get_status(task_id, db, _user(security_store.other_id))
            )
        assert error.value.status_code == 404
        async_result.assert_not_called()
    finally:
        db.close()


def test_status_response_does_not_include_artifact_filesystem_paths(
    security_store, monkeypatch
) -> None:
    task_id = _add_run(security_store)
    monkeypatch.setattr(
        analysis_routes,
        "AsyncResult",
        Mock(
            return_value=SimpleNamespace(
                state="SUCCESS",
                info={
                    "best_model": "SyntheticClassifier",
                    "report_path": "C:/private/reports/report.pdf",
                    "model_path": "C:/private/models/model.pkl",
                },
            )
        ),
    )
    db = security_store.factory()
    try:
        response = _run_immediate(
            analysis_routes.get_status(
                task_id,
                db,
                _user(security_store.owner_id),
            )
        )
        assert response.result == {"best_model": "SyntheticClassifier"}
        assert "private" not in str(response.model_dump())
    finally:
        db.close()


@pytest.mark.parametrize(
    ("artifact_type", "route_name", "root_name", "expected_mime", "expected_suffix"),
    [
        ("report", "download_report", "reports", "application/pdf", ".pdf"),
        (
            "model",
            "download_model",
            "models",
            "application/octet-stream",
            ".pkl",
        ),
    ],
)
def test_legacy_report_and_model_downloads_are_owner_scoped(
    security_store, storage_roots, artifact_type, route_name, root_name, expected_mime, expected_suffix
) -> None:
    owned_task = _add_run(security_store)
    foreign_task = _add_run(
        security_store,
        owner_id=security_store.other_id,
        session_id=security_store.other_session_id,
    )
    root = getattr(storage_roots, root_name)
    artifact_path = root / f"owned{expected_suffix}"
    artifact_path.write_bytes(b"synthetic artifact")
    with security_store.factory() as db:
        db.add(
            Artifact(
                session_id=security_store.owner_session_id,
                artifact_type=artifact_type,
                file_path=str(artifact_path),
                meta={},
            )
        )
        db.commit()
        handler = getattr(analysis_routes, route_name)
        response = _run_immediate(
            handler(owned_task, db, _user(security_store.owner_id))
        )
        assert Path(response.path) == artifact_path.resolve()
        assert response.media_type == expected_mime
        assert response.filename == f"{artifact_type}_{owned_task}{expected_suffix}"
        with pytest.raises(HTTPException) as error:
            _run_immediate(handler(foreign_task, db, _user(security_store.owner_id)))
        assert error.value.status_code == 404


@pytest.mark.parametrize("api", ["legacy", "session"])
@pytest.mark.parametrize("bad_path_kind", ["outside", "traversal", "symlink", "missing"])
def test_artifact_download_rejects_unsafe_or_missing_paths(
    security_store, storage_roots, tmp_path, monkeypatch, api, bad_path_kind
) -> None:
    task_id = _add_run(security_store)
    session_id = security_store.owner_session_id
    if bad_path_kind == "outside":
        bad_path = tmp_path / "private.pdf"
        bad_path.write_bytes(b"outside")
    elif bad_path_kind == "traversal":
        outside = tmp_path / "private.pdf"
        outside.write_bytes(b"outside")
        bad_path = storage_roots.reports / ".." / ".." / "private.pdf"
    elif bad_path_kind == "symlink":
        outside = tmp_path / "private.pdf"
        outside.write_bytes(b"outside")
        bad_path = storage_roots.reports / "linked.pdf"
        try:
            bad_path.symlink_to(outside)
        except (OSError, NotImplementedError):
            # The sandbox may prohibit creating a real symlink. Simulate the
            # resolved target so the exact same route containment check runs.
            original_resolve = Path.resolve

            def resolve_with_link(path, *args, **kwargs):
                if Path(path) == bad_path:
                    return original_resolve(outside, *args, **kwargs)
                return original_resolve(path, *args, **kwargs)

            monkeypatch.setattr(Path, "resolve", resolve_with_link)
    else:
        bad_path = storage_roots.reports / "missing.pdf"

    with security_store.factory() as db:
        db.add(
            Artifact(
                session_id=session_id,
                artifact_type="report",
                file_path=str(bad_path),
                meta={},
            )
        )
        db.commit()
        if api == "legacy":
            call = lambda: analysis_routes.download_report(
                task_id, db, _user(security_store.owner_id)
            )
        else:
            call = lambda: artifact_routes.download_report(
                session_id, db, _user(security_store.owner_id)
            )
        with pytest.raises(HTTPException) as error:
            response = call()
            if hasattr(response, "send"):
                _run_immediate(response)
        assert error.value.status_code == 404
        assert str(bad_path) not in str(error.value.detail)


def test_modern_artifacts_are_owner_scoped_and_hide_storage_paths(
    security_store, storage_roots
) -> None:
    report = storage_roots.reports / "private-report.pdf"
    report.write_bytes(b"synthetic report")
    with security_store.factory() as db:
        db.add(
            Artifact(
                session_id=security_store.owner_session_id,
                artifact_type="report",
                file_path=str(report),
                meta={},
            )
        )
        db.commit()
        listing = artifact_routes.list_artifacts(
            security_store.owner_session_id,
            db,
            _user(security_store.owner_id),
        )
        assert len(listing) == 1
        assert "file_path" not in listing[0].model_dump()
        response = artifact_routes.download_report(
            security_store.owner_session_id,
            db,
            _user(security_store.owner_id),
        )
        assert Path(response.path) == report.resolve()
        with pytest.raises(HTTPException) as error:
            artifact_routes.download_report(
                security_store.owner_session_id,
                db,
                _user(security_store.other_id),
            )
        assert error.value.status_code == 404


def test_session_response_does_not_expose_source_path(security_store, tmp_path) -> None:
    private_path = tmp_path / "internal" / "private-dataset.csv"
    with security_store.factory() as db:
        session = db.get(UserSession, security_store.owner_session_id)
        session.source_path = str(private_path)
        db.commit()
        response = SessionRead.model_validate(session).model_dump()

    assert "source_path" not in response
    assert str(private_path) not in str(response)


def test_upload_streams_to_generated_path_and_registers_owner(
    security_store, storage_roots, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    enqueue = Mock()
    monkeypatch.setattr(analysis_routes.analyze_dataset, "apply_async", enqueue)
    db = security_store.factory()
    try:
        result = _run_immediate(
            analysis_routes.analyze_upload(
                db,
                _user(security_store.owner_id),
                _MemoryUpload(r"..\..\synthetic.csv", b"feature,target\n1,yes\n2,no\n"),
                "target",
            )
        )
        assert uuid.UUID(result.task_id)
        enqueue.assert_called_once()
        queued_args = enqueue.call_args.kwargs["args"]
        stored_path = Path(queued_args[1])
        assert stored_path.parent == storage_roots.uploads.resolve()
        assert stored_path.name == f"{result.task_id}.csv"
        assert stored_path.read_bytes().startswith(b"feature,target")
        assert queued_args == ["upload", str(stored_path), "target"]

        run = db.scalar(select(AnalysisRun).where(AnalysisRun.task_id == result.task_id))
        assert run is not None
        assert run.requested_by == security_store.owner_id
        session = db.get(UserSession, run.session_id)
        assert session.user_id == security_store.owner_id
        assert session.dataset_meta["original_filename"] == "synthetic.csv"
    finally:
        db.close()


def test_url_submission_creates_owner_record_before_queueing(
    security_store, monkeypatch
) -> None:
    enqueue = Mock()
    monkeypatch.setattr(analysis_routes.analyze_dataset, "apply_async", enqueue)
    db = security_store.factory()
    try:
        result = _run_immediate(
            analysis_routes.analyze_url(
                SimpleNamespace(
                    url="https://example.test/synthetic.csv",
                    target_column="target",
                ),
                db,
                _user(security_store.owner_id),
            )
        )
        enqueue.assert_called_once_with(
            args=["url", "https://example.test/synthetic.csv", "target"],
            task_id=result.task_id,
        )
        run = db.scalar(select(AnalysisRun).where(AnalysisRun.task_id == result.task_id))
        assert run is not None
        assert run.requested_by == security_store.owner_id
        session = db.get(UserSession, run.session_id)
        assert session.user_id == security_store.owner_id
        assert session.source_type == "url"
        assert session.name == "synthetic.csv"
    finally:
        db.close()


def test_dataset_download_uses_safe_filename_and_mime_type(
    security_store, storage_roots
) -> None:
    dataset = storage_roots.uploads / "generated.csv"
    dataset.write_bytes(b"x,y\n1,2\n")
    with security_store.factory() as db:
        session = db.get(UserSession, security_store.owner_session_id)
        session.source_type = "upload"
        session.source_path = str(dataset)
        session.dataset_meta = {"original_filename": r"..\..\private.csv"}
        db.commit()

        response = artifact_routes.download_dataset(
            session.id,
            db,
            _user(security_store.owner_id),
        )
        assert Path(response.path) == dataset.resolve()
        assert response.filename == "private.csv"
        assert response.media_type == "text/csv"


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("dataset.exe", b"anything", 400),
        ("dataset.csv", b"text\x00binary", 400),
        ("dataset.xlsx", b"not-a-zip", 400),
    ],
)
def test_upload_rejects_unsupported_or_mismatched_content(
    security_store, storage_roots, monkeypatch, filename, content, expected_status
) -> None:
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    enqueue = Mock()
    monkeypatch.setattr(analysis_routes.analyze_dataset, "apply_async", enqueue)
    db = security_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            _run_immediate(
                analysis_routes.analyze_upload(
                    db,
                    _user(security_store.owner_id),
                    _MemoryUpload(filename, content),
                )
            )
        assert error.value.status_code == expected_status
        enqueue.assert_not_called()
        assert list(storage_roots.uploads.iterdir()) == []
        assert db.scalar(select(func.count()).select_from(AnalysisRun)) == 0
    finally:
        db.close()


def test_upload_size_limit_removes_partial_file_and_never_queues(
    security_store, storage_roots, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    enqueue = Mock()
    monkeypatch.setattr(analysis_routes.analyze_dataset, "apply_async", enqueue)
    payload = b"a" * (1024 * 1024 + 1)
    db = security_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            _run_immediate(
                analysis_routes.analyze_upload(
                    db,
                    _user(security_store.owner_id),
                    _MemoryUpload("large.csv", payload),
                )
            )
        assert error.value.status_code == 413
        enqueue.assert_not_called()
        assert list(storage_roots.uploads.iterdir()) == []
        assert db.scalar(select(func.count()).select_from(AnalysisRun)) == 0
    finally:
        db.close()


def test_queue_failure_cleans_upload_and_ownership_records(
    security_store, storage_roots, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    enqueue = Mock(side_effect=RuntimeError("synthetic broker failure"))
    monkeypatch.setattr(analysis_routes.analyze_dataset, "apply_async", enqueue)
    db = security_store.factory()
    try:
        with pytest.raises(HTTPException) as error:
            _run_immediate(
                analysis_routes.analyze_upload(
                    db,
                    _user(security_store.owner_id),
                    _MemoryUpload("dataset.csv", b"x,y\n1,2\n"),
                )
            )
        assert error.value.status_code == 503
        assert list(storage_roots.uploads.iterdir()) == []
        assert db.scalar(select(func.count()).select_from(AnalysisRun)) == 0
    finally:
        db.close()


def test_worker_refuses_task_without_persisted_ownership(security_store, monkeypatch) -> None:
    from app.tasks import analyze as analyze_task

    monkeypatch.setattr(analyze_task, "SessionLocal", security_store.factory)
    with pytest.raises(RuntimeError, match="ownership record"):
        analyze_task._claim_legacy_analysis(str(uuid.uuid4()))
