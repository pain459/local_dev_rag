"""Allowlisted JSON telemetry: arbitrary messages, payloads and tracebacks never serialize."""

import json
import logging
import re
from contextvars import ContextVar
from hashlib import sha256
from time import perf_counter
from typing import cast
from uuid import UUID, uuid4

import httpx
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from local_dev_rag.config import Settings

_context: ContextVar[dict[str, object] | None] = ContextVar("log_context", default=None)
_identifier = re.compile(r"[A-Za-z0-9_-]{1,64}\Z", re.ASCII)
_digest = re.compile(r"[0-9a-f]{24}\Z", re.ASCII)
_events = {
    "request_completed",
    "foreground_context",
    "dependency_unavailable",
    "memory_job_completed",
    "memory_job_failed",
    "memory_job_lease_lost",
    "memory_worker_dependency_unavailable",
}
_categories = {"processing_error", "upstream_error", "timeout", "validation_error", "cancelled"}
_dependencies = {"postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs"}
_models = {
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
    "nomic-embed-text:latest",
}


def compact_id(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:24]


def correlation_id(value: str | None) -> str:
    return value if value is not None and _identifier.fullmatch(value) else str(uuid4())


def error_category(error: BaseException) -> str:
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, httpx.HTTPError):
        return "upstream_error"
    if isinstance(error, ValueError):
        return "validation_error"
    return "processing_error"


def _response_category(status: int) -> str | None:
    return "validation_error" if status == 400 else "upstream_error" if status >= 400 else None


class SafeJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        values: dict[str, object] = (_context.get() or {}) | record.__dict__
        event = record.msg if isinstance(record.msg, str) and record.msg in _events else "log"
        output: dict[str, object] = {"event": event, "level": record.levelname}
        for key in ("request_id", "project_id", "session_id", "job_id"):
            value = values.get(key)
            if isinstance(value, str):
                if key == "request_id" and _identifier.fullmatch(value) or _digest.fullmatch(value):
                    output[key] = value
                else:
                    try:
                        output[key] = str(UUID(value))
                    except ValueError:
                        output[key] = compact_id(value)
        model = values.get("model")
        if isinstance(model, str):
            output["model"] = model if model in _models else compact_id(model)
        for key in (
            "duration_ms",
            "retrieval_count",
            "retry_count",
            "attempt_count",
            "injected_memory_tokens",
            "estimated_input_tokens",
            "memory_count",
            "status_code",
        ):
            value = values.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and -(2**63) <= value <= 2**63 - 1
            ):
                output[key] = max(0, value)
        degraded = values.get("degraded_dependencies")
        if isinstance(degraded, (list, tuple)):
            output["degraded_dependencies"] = [
                item
                for item in cast(list[object], degraded)
                if isinstance(item, str) and item in _dependencies
            ]
        category = values.get("error_category")
        output["error_category"] = (
            category if isinstance(category, str) and category in _categories else None
        )
        if record.exc_info and record.exc_info[1] is not None:
            output["error_category"] = error_category(record.exc_info[1])
        # Deliberately do not call getMessage()/formatException(): both may contain secrets.
        return json.dumps(output, ensure_ascii=True, allow_nan=False)


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if not any(getattr(handler, "_local_rag_safe", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler._local_rag_safe = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    formatter = SafeJSONFormatter()
    # Existing server/library handlers must not bypass the serialization boundary.
    loggers = [root] + [
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    root.setLevel(settings.log_level)


class RequestLoggingMiddleware:
    """Observe final ASGI delivery so streaming timings include cleanup and capture."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id")
        request_id = correlation_id(supplied.decode("latin-1") if supplied else None)
        state = scope.setdefault("state", {})
        state["correlation_id"] = request_id
        token = _context.set({"request_id": request_id})
        status = 500
        category: str | None = None

        async def observe(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = list(message.get("headers", []))
                if not any(key.lower() == b"x-request-id" for key, _ in response_headers):
                    response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, observe)
        except BaseException as error:
            category = error_category(error)
            raise
        finally:
            if category is None:
                category = _response_category(status)
            diagnostics = state.get("proxy_diagnostics")
            degraded: list[str] = list(getattr(diagnostics, "degraded_dependencies", ()))
            if state.get("capture_status") == "unavailable" and "postgres" not in degraded:
                degraded.append("postgres")
                category = category or "processing_error"
            logging.getLogger(__name__).info(
                "request_completed",
                extra={
                    **cast(dict[str, object], state.get("log_identity", {})),
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                    "status_code": status,
                    "error_category": category,
                    "retry_count": 0,
                    "retrieval_count": getattr(diagnostics, "retrieval_count", 0),
                    "injected_memory_tokens": getattr(diagnostics, "injected_memory_tokens", 0),
                    "degraded_dependencies": degraded,
                },
            )
            _context.reset(token)
