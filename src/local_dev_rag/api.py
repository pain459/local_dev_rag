"""HTTP validation and composition for the foreground memory service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from local_dev_rag.config import Settings, get_settings
from local_dev_rag.db import Database
from local_dev_rag.domain import InvalidRequestError, RequestIdentity
from local_dev_rag.logging import RequestLoggingMiddleware, compact_id, configure_logging
from local_dev_rag.models import ModelRegistry, UnknownModelError
from local_dev_rag.observations import ObservationStore, valid_observation_id
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.proxy import MemorySearch, ProxyService
from local_dev_rag.readiness import HealthProbe, ReadinessService
from local_dev_rag.repository import PostgresCaptureStore, PostgresMemorySearch
from local_dev_rag.vector_store import VectorStore

DependencyState = Literal["healthy", "degraded", "unavailable"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DependencyStates(BaseModel):
    postgres: DependencyState
    chromadb: DependencyState
    ollama: DependencyState
    curator: DependencyState
    embedder: DependencyState
    memory_jobs: DependencyState


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    dependencies: DependencyStates


def create_app(
    settings: Settings | None = None,
    *,
    ollama_client: OllamaClient | None = None,
    database: Database | None = None,
    vector_store: MemorySearch | None = None,
    readiness_service: ReadinessService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            if database is None:
                await cast(Database, app.state.database).engine.dispose()

    app = FastAPI(title="Local OpenCode RAG Memory", lifespan=lifespan)
    app.state.settings = settings if settings is not None else get_settings()
    configure_logging(app.state.settings)
    app.add_middleware(RequestLoggingMiddleware)
    app.state.database = database or Database.create(app.state.settings)
    registry = ModelRegistry(app.state.settings)
    app.state.model_registry = registry
    app.state.ollama_client = ollama_client or OllamaClient(app.state.settings)
    observations = ObservationStore()
    app.state.observations = observations

    app.state.vector_store = vector_store or VectorStore(app.state.settings)
    app.state.readiness_service = readiness_service or ReadinessService(
        app.state.settings,
        database=app.state.database,
        vector_store=cast(HealthProbe, app.state.vector_store),
        ollama=app.state.ollama_client,
    )
    app.state.proxy_service = ProxyService(
        app.state.settings,
        registry=registry,
        ollama=app.state.ollama_client,
        capture_store=PostgresCaptureStore(app.state.database),
        vector_store=PostgresMemorySearch(app.state.database, app.state.vector_store),
    )

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

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        identity = RequestIdentity.from_headers(request.headers)
        observation_id = request.headers.get("x-opencode-rag-observation-id")
        if observation_id is not None and not valid_observation_id(observation_id):
            raise InvalidRequestError(
                "Observation ID must be a canonical UUID v4", "x-opencode-rag-observation-id"
            )
        request.state.log_identity = {
            "project_id": compact_id(identity.project_id),
            "session_id": compact_id(identity.session_id),
        }
        try:
            payload = await request.json()
        except (ValueError, UnicodeError):
            raise InvalidRequestError("Request body must be a JSON object", "body") from None
        if not isinstance(payload, dict):
            raise InvalidRequestError("Request body must be a JSON object", "body")
        payload = cast(dict[str, object], payload)
        model = payload.get("model")
        if not isinstance(model, str):
            raise InvalidRequestError("A generation model is required", "model")
        try:
            registry.get(model)
        except UnknownModelError as error:
            raise InvalidRequestError(str(error), "model") from error
        request.state.log_identity["model"] = model
        if not isinstance(payload.get("messages"), list):
            raise InvalidRequestError("messages must be an array", "messages")
        if "stream" in payload and not isinstance(payload["stream"], bool):
            raise InvalidRequestError("stream must be a boolean", "stream")

        result = await cast(ProxyService, app.state.proxy_service).complete(
            identity,
            payload,
            state=request.scope.setdefault("state", {}),
        )
        if observation_id is not None:
            observations.put(observation_id, identity, result.diagnostics.injected_memory_tokens)
        return result.response

    @app.get("/v1/rag/observations/{observation_id}")
    async def observation(observation_id: str, request: Request) -> JSONResponse:
        identity = RequestIdentity.from_headers(request.headers)
        tokens = observations.get(observation_id, identity) if valid_observation_id(
            observation_id
        ) else None
        if tokens is None:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return JSONResponse(
            content={"injected_memory_tokens": tokens}, headers={"Cache-Control": "no-store"}
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get(
        "/readyz", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}}
    )
    async def readiness() -> JSONResponse:
        report = await cast(ReadinessService, app.state.readiness_service).check()
        response = ReadinessResponse(
            status=report.status,
            dependencies=DependencyStates.model_validate(
                {name: result.state for name, result in report.dependencies.items()}
            ),
        )
        return JSONResponse(
            status_code=503 if report.status == "not_ready" else 200, content=response.model_dump()
        )

    return app
