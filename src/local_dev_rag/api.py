"""HTTP application composition and operational endpoint contracts."""

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from local_dev_rag.config import Settings, get_settings
from local_dev_rag.domain import InvalidRequestError
from local_dev_rag.models import ModelRegistry

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
    registry = ModelRegistry(app.state.settings)
    app.state.model_registry = registry

    @app.exception_handler(InvalidRequestError)
    async def invalid_request(_request: Request, error: InvalidRequestError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(error),
                    "type": "invalid_request_error",
                    "param": error.param,
                    "code": None,
                },
            },
        )

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return registry.as_openai_models()

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
