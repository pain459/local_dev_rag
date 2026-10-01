"""Foreground orchestration with durable capture and optional project memory."""

import asyncio
import json
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID

import httpx
from anyio import CancelScope
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from local_dev_rag.config import Settings
from local_dev_rag.context import ContextBuilder
from local_dev_rag.domain import (
    AssistantCompletion,
    ChatRequest,
    ConversationEventInput,
    MemoryCandidate,
    RequestIdentity,
    VectorHit,
)
from local_dev_rag.domain import Scope as ConversationScope
from local_dev_rag.events import json_value, normalize_messages
from local_dev_rag.logging import compact_id
from local_dev_rag.models import ModelRegistry, ModelSpec
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.ranking import rank_memories
from local_dev_rag.repository import CaptureUnavailable
from local_dev_rag.streaming import StreamAccumulator, relay_stream
from local_dev_rag.vector_store import VectorStoreUnavailable

logger = logging.getLogger(__name__)


class CaptureHandle(Protocol):
    scope: ConversationScope
    status: str

    async def finish(self, completion: AssistantCompletion | None) -> None: ...


class CaptureStore(Protocol):
    async def begin(
        self,
        identity: RequestIdentity,
        events: Sequence[ConversationEventInput],
        model: str,
        request_id: str,
        parent_hash: str,
    ) -> CaptureHandle: ...


class MemorySearch(Protocol):
    async def query(
        self, project_id: UUID, vector: Sequence[float], limit: int
    ) -> list[VectorHit]: ...


@dataclass(frozen=True)
class ProxyDiagnostics:
    retrieval_count: int
    injected_memory_tokens: int
    degraded_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class ProxyResult:
    response: Response
    diagnostics: ProxyDiagnostics


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
        disconnected = False
        state = scope.setdefault("state", {})

        async def observe_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect" and not delivered:
                disconnected = True
            return message

        async def observe_send(message: Message) -> None:
            nonlocal delivered
            try:
                await send(message)
            except OSError:
                # Normalize client transport failures here for every ASGI version,
                # without treating upstream read or cleanup errors as disconnects.
                raise ClientDisconnect() from None
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                delivered = True

        try:
            await super().__call__(scope, observe_receive, observe_send)
        except (asyncio.CancelledError, ClientDisconnect):
            state["stream_outcome"] = "cancelled"
            raise
        except BaseException:
            state["stream_outcome"] = "upstream_error"
            raise
        finally:
            if disconnected:
                state["stream_outcome"] = "cancelled"
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
                    if state.get("stream_outcome") != "cancelled":
                        state["stream_outcome"] = "upstream_error"
                    raise
                finally:
                    completion = self._accumulator.completion()
                    if "stream_outcome" not in state:
                        state["stream_outcome"] = (
                            "completed"
                            if delivered and completion is not None
                            else "upstream_error"
                        )
                    if self._on_finished is not None:
                        await self._on_finished(completion)


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


class ProxyService:
    def __init__(
        self,
        settings: Settings,
        *,
        registry: ModelRegistry,
        ollama: OllamaClient,
        capture_store: CaptureStore,
        vector_store: MemorySearch,
        context_builder_factory: Callable[[ModelSpec], ContextBuilder] | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.ollama = ollama
        self.capture_store = capture_store
        self.vector_store = vector_store
        self.context_builder_factory: Callable[[ModelSpec], ContextBuilder] = (
            context_builder_factory
            or (
                lambda spec: ContextBuilder(
                    spec,
                    safety_tokens=settings.model_budgets[spec.id].safety_tokens,
                    memory_token_budget=settings.memory_token_budget,
                )
            )
        )

    async def complete(
        self,
        identity: RequestIdentity,
        request: Mapping[str, object],
        *,
        state: MutableMapping[str, object] | None = None,
    ) -> ProxyResult:
        state = state if state is not None else {}
        model = cast(str, request["model"])
        spec = self.registry.get(model)
        messages = cast(Sequence[Mapping[str, object]], request["messages"])
        events = normalize_messages(messages)
        parent_hash = events[-1].content_hash if events else ""
        request_id = parent_hash or "empty-conversation"
        capture: CaptureHandle | None = None
        state["capture_status"] = "unavailable"
        degraded: list[str] = []
        try:
            capture = await self.capture_store.begin(
                identity, events, model, request_id, parent_hash
            )
            state["capture_status"] = capture.status
        except CaptureUnavailable:
            degraded.append("postgres")
            logger.warning(
                "dependency_unavailable",
                extra={
                    "degraded_dependencies": ["postgres"],
                    "error_category": "processing_error",
                },
            )
        payload = cast(dict[str, object], json_value(request))
        candidates: list[MemoryCandidate] = []
        injected_tokens = 0
        try:
            if capture is not None:
                query_text = self._query_text(messages)
                if query_text:
                    try:
                        vectors = await self.ollama.embed(
                            self.settings.embedding_model, [query_text]
                        )
                    except (httpx.HTTPError, ValueError):
                        degraded.append("embedder")
                    else:
                        try:
                            hits = await self.vector_store.query(
                                capture.scope.project_id,
                                vectors[0],
                                self.settings.retrieval_candidate_limit,
                            )
                            if any(
                                hit.memory.project_id != capture.scope.project_id for hit in hits
                            ):
                                raise VectorStoreUnavailable("Mismatched retrieval project")
                            candidates = [
                                candidate
                                for candidate in rank_memories(
                                    query_text,
                                    hits,
                                    datetime.now(UTC),
                                    weights=self.settings.ranking_weights,
                                )
                                if candidate.score >= self.settings.retrieval_min_score
                            ][: self.settings.retrieval_result_limit]
                        except VectorStoreUnavailable:
                            degraded.append("chromadb")
                        except CaptureUnavailable:
                            degraded.append("postgres")
                built = self.context_builder_factory(spec).build(
                    ChatRequest(
                        model=model,
                        messages=messages,
                        stream=bool(request.get("stream", False)),
                        extra={
                            key: value
                            for key, value in request.items()
                            if key not in {"model", "messages", "stream"}
                        },
                    ),
                    candidates,
                    history_durable=True,
                )
                payload = built.payload
                injected_tokens = built.injected_memory_tokens
            diagnostics = ProxyDiagnostics(len(candidates), injected_tokens, tuple(degraded))
            state["proxy_diagnostics"] = diagnostics
            logger.info(
                "foreground_context",
                extra={
                    "request_id": state.get("correlation_id", request_id),
                    "project_id": compact_id(identity.project_id),
                    "session_id": compact_id(identity.session_id),
                    "model": model,
                    "retrieval_count": diagnostics.retrieval_count,
                    "injected_memory_tokens": injected_tokens,
                    "degraded_dependencies": diagnostics.degraded_dependencies,
                    "ranking_components": [dict(c.score_components or {}) for c in candidates],
                },
            )
        except BaseException:
            if capture is not None:
                with CancelScope(shield=True):
                    await self._finish(capture, None, state)
            raise
        response = await self._forward(payload, capture, state, model, request_id)
        return ProxyResult(response, diagnostics)

    @staticmethod
    def _query_text(messages: Sequence[Mapping[str, object]]) -> str:
        start = next(
            (
                index
                for index in reversed(range(len(messages)))
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if start is None:
            return ""
        parts: list[str] = []
        for message in messages[start:]:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif content is not None:
                parts.append(json.dumps(json_value(content), ensure_ascii=False))
        return "\n".join(parts).strip()

    @staticmethod
    async def _finish(
        capture: CaptureHandle,
        completion: AssistantCompletion | None,
        state: MutableMapping[str, object],
    ) -> None:
        await capture.finish(completion)
        state["capture_status"] = capture.status

    async def _forward(
        self,
        payload: dict[str, object],
        capture: CaptureHandle | None,
        state: MutableMapping[str, object],
        model: str,
        request_id: str,
    ) -> Response:
        async def finish(completion: AssistantCompletion | None) -> None:
            if capture is not None:
                await self._finish(capture, completion, state)

        stack = AsyncExitStack()
        stream_owns_capture = False
        try:
            upstream = await stack.enter_async_context(self.ollama.chat(payload))
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
                    await finish(nonstream_completion(body, model, request_id))
                return Response(body, status_code=upstream.status_code, headers=headers)
            accumulator = StreamAccumulator(request_id=request_id, model=model)
            state["stream_accumulator"] = accumulator
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
                on_finished=finish if capture is not None else None,
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
                        await finish(None)
