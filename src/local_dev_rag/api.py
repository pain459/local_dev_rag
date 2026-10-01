"""HTTP application composition and operational endpoint contracts."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from local_dev_rag.config import Settings, get_settings

DependencyState = Literal["unknown", "healthy", "unhealthy"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DependencyStates(BaseModel):
    postgres: DependencyState
    chromadb: DependencyState
    ollama: DependencyState


class ReadinessResponse(BaseModel):
    status: Literal["starting", "healthy", "degraded"]
    dependencies: DependencyStates


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Local OpenCode RAG Memory")
    app.state.settings = settings if settings is not None else get_settings()

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/readyz", response_model=ReadinessResponse)
    async def readiness() -> ReadinessResponse:
        return ReadinessResponse(
            status="starting",
            dependencies=DependencyStates(postgres="unknown", chromadb="unknown", ollama="unknown"),
        )

    return app
