"""HTTP application composition and operational endpoint contracts."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack
from typing import Literal, cast

import httpx
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.types import Message, Receive, Scope, Send

from local_dev_rag.config import Settings, get_settings
from local_dev_rag.domain import InvalidRequestError, RequestIdentity
from local_dev_rag.models import ModelRegistry, UnknownModelError
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.streaming import StreamAccumulator, relay_stream

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


class UpstreamStreamingResponse(StreamingResponse):
    """Keep the upstream context alive until delivery finishes or is cancelled."""

    def __init__(
        self,
        body: AsyncGenerator[bytes],
        *,
        stack: AsyncExitStack,
        accumulator: StreamAccumulator,
        status_code: int,
        headers: dict[str, str],
    ):
        super().__init__(body, status_code=status_code, headers=headers)
        self._stack = stack
        self._accumulator = accumulator
        self._body = body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        delivered = False

        async def observe_send(message: Message) -> None:
            nonlocal delivered
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                delivered = True

        try:
            await super().__call__(scope, receive, observe_send)
        finally:
            if not delivered:
                self._accumulator.abort()
            with CancelScope(shield=True):
                await self._body.aclose()
                await self._stack.aclose()


def upstream_error(
    message: str,
    status: int = 502,
    *,
    headers: dict[str, str] | None = None,
    fields: dict[str, object] | None = None,
) -> JSONResponse:
    error: dict[str, object] = {
        "message": message,
        "type": "upstream_error",
        "param": None,
        "code": None,
    }
    if fields:
        error.update(fields)
    json_headers = {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() not in {"content-type", "content-encoding", "content-length"}
    }
    return JSONResponse(
        status_code=status,
        content={"error": error},
        headers=json_headers,
    )


def safe_headers(headers: dict[str, str]) -> dict[str, str]:
    hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    hop.update(value.strip().lower() for value in headers.get("connection", "").split(","))
    return {key: value for key, value in headers.items() if key.lower() not in hop}


def create_app(
    settings: Settings | None = None, *, ollama_client: OllamaClient | None = None
) -> FastAPI:
    app = FastAPI(title="Local OpenCode RAG Memory")
    app.state.settings = settings if settings is not None else get_settings()
    registry = ModelRegistry(app.state.settings)
    app.state.model_registry = registry
    app.state.ollama_client = ollama_client or OllamaClient(app.state.settings)

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
        RequestIdentity.from_headers(request.headers)
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
        if not isinstance(payload.get("messages"), list):
            raise InvalidRequestError("messages must be an array", "messages")
        if "stream" in payload and not isinstance(payload["stream"], bool):
            raise InvalidRequestError("stream must be a boolean", "stream")

        stack = AsyncExitStack()
        try:
            client = cast(OllamaClient, app.state.ollama_client)
            upstream = await stack.enter_async_context(client.chat(payload))
            headers = safe_headers(dict(upstream.headers))
            if upstream.status_code >= 400:
                body = b"".join([chunk async for chunk in upstream.body])
                try:
                    error_payload = json.loads(body)
                    error = error_payload.get("error", "Ollama request failed")
                    if isinstance(error, dict):
                        return upstream_error(
                            "Ollama request failed",
                            upstream.status_code,
                            headers=headers,
                            fields=cast(dict[str, object], error),
                        )
                    message = str(error)
                except (ValueError, AttributeError):
                    message = "Ollama request failed"
                return upstream_error(message, upstream.status_code, headers=headers)
            if not payload.get("stream", False):
                body = b"".join([chunk async for chunk in upstream.body])
                return Response(body, status_code=upstream.status_code, headers=headers)
            accumulator = StreamAccumulator(model=model)
            request.state.stream_accumulator = accumulator
            first = await anext(upstream.body, None)

            async def fragments() -> AsyncIterator[bytes]:
                if first is not None:
                    yield first
                async for chunk in upstream.body:
                    yield chunk

            response = UpstreamStreamingResponse(
                relay_stream(fragments(), accumulator),
                stack=stack.pop_all(),
                accumulator=accumulator,
                status_code=upstream.status_code,
                headers=headers,
            )
            return response
        except httpx.HTTPError:
            return upstream_error("Unable to complete the Ollama request")
        finally:
            with CancelScope(shield=True):
                await stack.aclose()

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
