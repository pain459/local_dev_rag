"""HTTP application composition and operational endpoint contracts."""

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from typing import Literal, cast
from uuid import uuid4

import httpx
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import Message, Receive, Scope, Send

from local_dev_rag.config import Settings, get_settings
from local_dev_rag.db import Database
from local_dev_rag.domain import (
    AssistantCompletion,
    ConversationEventInput,
    InvalidRequestError,
    RequestIdentity,
)
from local_dev_rag.domain import (
    Scope as ConversationScope,
)
from local_dev_rag.events import content_hash, normalize_messages
from local_dev_rag.models import ModelRegistry, UnknownModelError
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.repository import ConversationRepository
from local_dev_rag.streaming import StreamAccumulator, relay_stream

DependencyState = Literal["unknown", "healthy", "unhealthy"]
logger = logging.getLogger(__name__)


@dataclass
class CaptureAttempt:
    database: Database
    scope: ConversationScope
    request_id: str
    parent_hash: str
    model: str
    request: Request
    finished: bool = False

    async def finish(self, completion: AssistantCompletion | None) -> None:
        if self.finished:
            return
        try:
            async with self.database.session() as session:
                repository = ConversationRepository(session)
                if completion is not None:
                    event = ConversationEventInput(
                        event_type="message",
                        role="assistant",
                        payload=completion.payload,
                        content_hash="",
                        request_id=self.request_id,
                        parent_hash=self.parent_hash,
                    )
                    await repository.finalize_assistant(
                        self.scope,
                        replace(
                            completion,
                            content_hash=content_hash(event),
                            request_id=self.request_id,
                        ),
                    )
                else:
                    attempt_id = str(uuid4())
                    await repository.append_events(
                        self.scope,
                        [
                            ConversationEventInput(
                                event_type="attempt",
                                role="proxy",
                                payload={"status": "incomplete", "attempt_id": attempt_id},
                                content_hash=f"attempt:{attempt_id}",
                                request_id=self.request_id,
                                model=self.model,
                                completed=False,
                            )
                        ],
                    )
            self.request.state.capture_status = "completed" if completion else "incomplete"
        except (SQLAlchemyError, OSError):
            self.request.state.capture_status = "unavailable"
            logger.warning("Conversation capture unavailable; completion was not persisted")
        self.finished = True


def nonstream_completion(body: bytes, model: str, request_id: str) -> AssistantCompletion | None:
    try:
        result = json.loads(body)
        if not isinstance(result, dict) or "error" in result:
            return None
        result = cast(dict[str, object], result)
        choices = result.get("choices")
        if not isinstance(choices, list):
            return None
        for value in cast(list[object], choices):
            if not isinstance(value, dict):
                continue
            choice = cast(dict[str, object], value)
            if choice.get("index", 0) != 0:
                continue
            message = choice.get("message")
            if choice.get("finish_reason") is None or not isinstance(message, dict):
                return None
            message = cast(dict[str, object], message)
            if message.get("role") != "assistant":
                return None
            source = result.get("id")
            selected_model = result.get("model")
            return AssistantCompletion(
                payload=message,
                content_hash="",
                request_id=request_id,
                model=selected_model if isinstance(selected_model, str) else model,
                source_message_id=source if isinstance(source, str) else None,
            )
    except (ValueError, UnicodeError):
        pass
    return None


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
        on_finished: Callable[[AssistantCompletion | None], Awaitable[None]] | None = None,
    ):
        super().__init__(body, status_code=status_code, headers=headers)
        self._stack = stack
        self._accumulator = accumulator
        self._body = body
        self._on_finished = on_finished

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
                try:
                    try:
                        await self._body.aclose()
                    finally:
                        await self._stack.aclose()
                except BaseException:
                    self._accumulator.abort()
                    raise
                finally:
                    if self._on_finished is not None:
                        await self._on_finished(self._accumulator.completion())


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
    settings: Settings | None = None,
    *,
    ollama_client: OllamaClient | None = None,
    database: Database | None = None,
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
    app.state.database = database or Database.create(app.state.settings)
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
        identity = RequestIdentity.from_headers(request.headers)
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

        events = normalize_messages(cast(list[Mapping[str, object]], payload["messages"]))
        parent_hash = events[-1].content_hash if events else ""
        request_id = parent_hash or "empty-conversation"
        capture: CaptureAttempt | None = None
        request.state.capture_status = "unavailable"
        try:
            db = cast(Database, app.state.database)
            async with db.session() as session:
                repository = ConversationRepository(session)
                scope = await repository.ensure_scope(identity)
                await repository.append_events(
                    scope, [replace(event, request_id=request_id, model=model) for event in events]
                )
            capture = CaptureAttempt(db, scope, request_id, parent_hash, model, request)
            request.state.capture_status = "inbound_persisted"
        except (SQLAlchemyError, OSError):
            logger.warning("PostgreSQL unavailable; forwarding the full request without capture")

        stack = AsyncExitStack()
        stream_owns_capture = False
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
                with CancelScope(shield=True):
                    await stack.aclose()
                if capture is not None:
                    await capture.finish(nonstream_completion(body, model, request_id))
                return Response(body, status_code=upstream.status_code, headers=headers)
            accumulator = StreamAccumulator(request_id=request_id, model=model)
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
                on_finished=capture.finish if capture is not None else None,
            )
            stream_owns_capture = True
            return response
        except httpx.HTTPError:
            return upstream_error("Unable to complete the Ollama request")
        finally:
            with CancelScope(shield=True):
                try:
                    await stack.aclose()
                finally:
                    if capture is not None and not stream_owns_capture:
                        await capture.finish(None)

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
