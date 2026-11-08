"""
API router aggregator. Mounts every route module under /api.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import analysis, artifacts, auth, chat, health, sessions

from app.api.routes import analysis, artifacts, auth, chat, health, sessions, runs 



router = APIRouter()

# Auth (must be accessible without a token)
router.include_router(auth.router, prefix="/auth", tags=["auth"])

# Ops / diagnostics
router.include_router(health.router, prefix="/health", tags=["health"])

# Protected routes (these require a valid token)
router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(artifacts.router, prefix="/artifacts", tags=["artifacts"])
router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
router.include_router(runs.router, prefix="/runs", tags=["runs"]) 