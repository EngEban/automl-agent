"""Focused tests for FastAPI application and API router registration."""
from __future__ import annotations

import os

# Keep application imports local to the test environment; importing LiteLLM
# must not trigger its optional remote model-cost-map lookup.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi.routing import APIRoute

from app.main import app


def _api_routes() -> list[APIRoute]:
    return [route for route in app.routes if isinstance(route, APIRoute)]


def test_main_application_imports_successfully() -> None:
    assert app.title == "AutoML Agent API"


def test_expected_api_routes_are_registered() -> None:
    paths = {route.path for route in _api_routes()}

    assert {
        "/api/sessions",
        "/api/chat/{session_id}",
        "/api/runs/{run_id}",
        "/api/artifacts/{session_id}",
        "/api/auth/login",
        "/api/predict/{model_version_id}",
        "/health",
    } <= paths


def test_prediction_route_is_registered_exactly_once() -> None:
    prediction_routes = [
        route
        for route in _api_routes()
        if route.path == "/api/predict/{model_version_id}"
    ]

    assert len(prediction_routes) == 1
    assert "POST" in prediction_routes[0].methods
