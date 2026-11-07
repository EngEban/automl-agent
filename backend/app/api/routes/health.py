"""
Health + configuration endpoints (useful for the frontend to bootstrap).
"""
from __future__ import annotations

from fastapi import APIRouter

from app.agent.llm import verify_connection
from app.agent.tools import get_tool_schemas
from app.config import settings
from app.schemas.agent import AgentConfig, ToolInfo

router = APIRouter()


@router.get("/")
def health() -> dict:
    return {"status": "healthy", "version": "0.2.0"}


@router.get("/agent/config", response_model=AgentConfig)
def agent_config() -> AgentConfig:
    return AgentConfig(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        has_api_key=bool(settings.LLM_API_KEY) or settings.LLM_PROVIDER == "ollama",
    )


@router.get("/agent/tools", response_model=list[ToolInfo])
def agent_tools() -> list[ToolInfo]:
    schemas = get_tool_schemas()
    return [
        ToolInfo(
            name=s["function"]["name"],
            description=s["function"]["description"],
            parameters=s["function"]["parameters"],
        )
        for s in schemas
    ]