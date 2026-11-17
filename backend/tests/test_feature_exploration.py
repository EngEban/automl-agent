from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import sessions as session_routes
from app.config.settings import settings
from app.core import feature_explorer
from app.db.base import Base
from app.db.models import Session as UserSession, User
from app.schemas.feature_analysis import FeatureAnalysisRequest


def _synthetic_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(441)
    signal = np.linspace(-2, 2, rows)
    target = np.where(signal >= 0, "positive", "negative")
    return pd.DataFrame({
        "row_id": np.arange(rows),
        "signal": signal,
        "signal_copy": signal * 0.99 + rng.normal(0, 0.01, rows),
        "independent": rng.normal(size=rows),
        "category": np.where(np.arange(rows) % 3 == 0, "a", "b"),
        "constant": 7,
        "sparse_value": np.where(np.arange(rows) % 11 == 0, "rare", None),
        "target": target,
    })


def test_feature_profile_reports_numeric_categorical_missing_constants_and_ids():
    frame = _synthetic_frame()
    frame.loc[1:8, "signal"] = np.nan
    result = feature_explorer.explore_features(frame)
    profile = result["profile"]
    columns = {item.name: item for item in profile.columns}

    assert result["n_rows"] == len(frame)
    assert columns["signal"].detected_type == "numeric"
    assert columns["signal"].missing_count == 8
    assert columns["category"].detected_type == "categorical"
    assert columns["category"].top_categories
    assert columns["constant"].is_constant is True
    assert columns["row_id"].likely_id is True
    assert result["selection_comparison"]["status"] == "requires_target"


def test_numeric_relationships_find_strong_pairs_and_leave_unrelated_pair_weak():
    result = feature_explorer.explore_features(_synthetic_frame())
    relationships = result["relationships"]
    pairs = {(pair["feature_a"], pair["feature_b"]): pair for pair in relationships["strong_pairs"]}

    assert relationships["status"] == "available"
    assert pairs[("signal", "signal_copy")]["pearson"] > 0.99
    assert pairs[("signal", "signal_copy")]["spearman"] > 0.99
    assert ("signal", "independent") not in pairs
    assert "Categorical association statistics are not implemented" in relationships["notes"][2]


def test_selection_comparison_runs_real_methods_and_is_deterministic():
    frame = _synthetic_frame()
    first = feature_explorer.explore_features(
        frame, target_column="target", task_type="classification", max_features=3,
    )["selection_comparison"]
    second = feature_explorer.explore_features(
        frame, target_column="target", task_type="classification", max_features=3,
    )["selection_comparison"]

    assert first["status"] == "success"
    assert first["holdout_rows_reserved"] > 0
    assert first["holdout_rows_used_for_selection"] == 0
    assert len(first["methods"]) == 5
    successful = {method["name"]: method for method in first["methods"] if method["status"] == "success"}
    assert len(successful) == 5
    assert successful["Constant / zero-variance filter"]["selected_count"] < len(frame.columns) - 1
    assert successful["Correlation filter"]["selected_count"] < 5
    assert all(len(method["selected_features"]) <= 3 for method in first["methods"] if method["name"] not in {"Constant / zero-variance filter", "Correlation filter"})
    assert any(item["feature"] == "signal_copy" for item in successful["Correlation filter"]["excluded_features"])
    assert [
        (method["name"], method["selected_features"], method["ranking"])
        for method in first["methods"]
    ] == [
        (method["name"], method["selected_features"], method["ranking"])
        for method in second["methods"]
    ]


def test_learned_selector_preprocessor_never_receives_holdout_rows(monkeypatch):
    frame = _synthetic_frame(90)
    from app.core.preprocessing import IntelligentPreprocessor, prepare_xy_for_training
    from app.core.trainer import _split_data

    X, y, _ = prepare_xy_for_training(frame, "target")
    expected_train, expected_holdout, _, _ = _split_data(X, y, "classification")
    expected_train_ids = set(expected_train["row_id"].tolist())
    expected_holdout_ids = set(expected_holdout["row_id"].tolist())
    seen_fit_ids: list[set[int]] = []
    seen_fit_targets: list[set[str]] = []
    original = IntelligentPreprocessor.fit_transform

    def recording_fit(self, values, labels=None, **kwargs):
        seen_fit_ids.append(set(values["row_id"].tolist()))
        seen_fit_targets.append(set(labels.tolist()))
        return original(self, values, labels, **kwargs)

    monkeypatch.setattr(IntelligentPreprocessor, "fit_transform", recording_fit)
    result = feature_explorer.explore_features(
        frame, target_column="target", task_type="classification",
    )["selection_comparison"]

    assert result["status"] == "success"
    assert seen_fit_ids == [expected_train_ids]
    assert seen_fit_ids[0].isdisjoint(expected_holdout_ids)
    assert seen_fit_targets == [set(y.loc[expected_train.index].tolist())]


def test_regression_selector_rejects_non_numeric_target_without_faking_scores():
    frame = _synthetic_frame()
    result = feature_explorer.explore_features(
        frame, target_column="category", task_type="regression",
    )["selection_comparison"]

    assert result["status"] in {"success", "unavailable"}
    if result["status"] == "success":
        failed = [method for method in result["methods"] if method["status"] in {"failed", "skipped"}]
        assert failed
        assert all(not method["ranking"] for method in failed)


def test_exploration_rejects_unknown_target_and_invalid_limits():
    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="target column"):
        feature_explorer.explore_features(frame, target_column="missing")
    with pytest.raises(ValueError, match="max_features"):
        feature_explorer.explore_features(frame, max_features=51)


def test_selection_comparison_respects_existing_dataset_memory_threshold(monkeypatch):
    monkeypatch.setattr(settings, "DATASET_MEMORY_WARNING_MB", 1)
    frame = _synthetic_frame()
    frame["large_text"] = ["x" * 10_000] * len(frame)
    result = feature_explorer.explore_features(
        frame, target_column="target", task_type="classification",
    )["selection_comparison"]

    assert result["status"] == "unavailable"
    assert "configured 1 MiB" in result["message"]


def test_selection_preserves_non_string_source_column_labels():
    frame = _synthetic_frame().rename(columns={"signal": 101})
    result = feature_explorer.explore_features(
        frame, target_column="target", task_type="classification",
    )["selection_comparison"]

    assert result["status"] == "success"
    assert any(
        "101" in method["selected_features"] or any(item["feature"] == "101" for item in method["ranking"])
        for method in result["methods"]
        if method["status"] == "success"
    )


@pytest.fixture
def feature_api_store(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    owner = User(email="feature-owner@example.test", hashed_password="test-only")
    other = User(email="feature-other@example.test", hashed_password="test-only")
    db.add_all([owner, other])
    db.flush()
    owned_session = UserSession(
        user_id=owner.id,
        name="synthetic features",
        source_type="upload",
        source_path="private-upload-name.csv",
        dataset_meta={},
        target_column="target",
        task_type="classification",
        status="created",
    )
    db.add(owned_session)
    db.commit()
    db.refresh(owned_session)

    storage = tmp_path / "storage"
    dataset_dir = storage / "sessions" / str(owned_session.id)
    dataset_dir.mkdir(parents=True)
    _synthetic_frame().to_pickle(dataset_dir / "raw.pkl")
    monkeypatch.setattr(settings, "STORAGE_DIR", storage)
    store = SimpleNamespace(
        engine=engine,
        factory=factory,
        db=db,
        owner=owner,
        other=other,
        session=owned_session,
    )
    try:
        yield store
    finally:
        db.close()
        engine.dispose()


def test_feature_analysis_api_returns_bounded_profile_for_owner(feature_api_store):
    result = session_routes.analyze_session_features(
        str(feature_api_store.session.id),
        FeatureAnalysisRequest(),
        feature_api_store.db,
        feature_api_store.owner,
    )

    assert result.n_rows == 120
    assert result.profiled_rows == 120
    assert result.selection_comparison["status"] == "success"


def test_feature_analysis_api_can_request_exploration_without_session_target(feature_api_store):
    result = session_routes.analyze_session_features(
        str(feature_api_store.session.id),
        FeatureAnalysisRequest(target_column=None, task_type=None),
        feature_api_store.db,
        feature_api_store.owner,
    )

    assert result.selection_comparison["status"] == "requires_target"


def test_feature_analysis_api_hides_other_users_sessions(feature_api_store):
    with pytest.raises(HTTPException) as error:
        session_routes.analyze_session_features(
            str(feature_api_store.session.id),
            FeatureAnalysisRequest(),
            feature_api_store.db,
            feature_api_store.other,
        )
    assert error.value.status_code == 404

