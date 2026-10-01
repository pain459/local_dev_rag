"""Privacy at the serialized logging boundary, including untrusted failures."""

import importlib
import io
import json
import logging
from urllib.parse import quote

import httpx
import pytest

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.ollama import OllamaClient

PRIVATE = "prompt-tool-memory-secret"
ROOT = "/Users/private/项目/秘密"
MODEL = "qwen3-coder:30b"


def logging_module():
    assert importlib.util.find_spec("local_dev_rag.logging"), "Privacy-safe logging is missing"
    return importlib.import_module("local_dev_rag.logging")


def capture_logs():
    module = logging_module()
    module.configure_logging(Settings(_env_file=None))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(module.SafeJSONFormatter())
    logging.getLogger().addHandler(handler)
    return stream, handler


def test_formatter_drops_arbitrary_messages_exceptions_and_sensitive_extra():
    stream, handler = capture_logs()
    try:
        try:
            raise RuntimeError(PRIVATE + ROOT)
        except RuntimeError:
            logging.getLogger("httpx").exception(
                "Authorization: %s; %s",
                PRIVATE,
                ROOT,
                extra={
                    "prompt": PRIVATE,
                    "headers": {"Authorization": PRIVATE},
                    "memory_text": PRIVATE,
                    "project_root": ROOT,
                    "tool_output": PRIVATE,
                },
            )
    finally:
        logging.getLogger().removeHandler(handler)
    rendered = stream.getvalue()
    assert PRIVATE not in rendered
    assert ROOT not in rendered
    assert quote(ROOT) not in rendered
    value = json.loads(rendered)
    assert value["level"] == "ERROR"
    assert value["error_category"] == "processing_error"


def test_formatter_cannot_reflect_content_through_identifier_or_metric_fields():
    formatter = logging_module().SafeJSONFormatter()
    record = logging.LogRecord("untrusted", logging.ERROR, "", 1, PRIVATE, (), None)
    record.project_id = PRIVATE
    record.session_id = PRIVATE
    record.job_id = PRIVATE
    record.model = ROOT
    record.duration_ms = float("nan")
    record.retrieval_count = {"tool_output": PRIVATE}
    record.memory_count = 10**1000
    record.error_category = PRIVATE
    record.degraded_dependencies = [PRIVATE, "chromadb"]
    rendered = formatter.format(record)
    assert PRIVATE not in rendered and ROOT not in rendered
    value = json.loads(rendered)
    assert "duration_ms" not in value and "retrieval_count" not in value
    assert "memory_count" not in value
    assert value["degraded_dependencies"] == ["chromadb"]
    assert value["error_category"] is None


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_request_logs_safe_correlation_and_final_metrics(failure, streaming):
    stream, handler = capture_logs()
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://unused:unused@127.0.0.1:1/unused",
    )

    def upstream(request):
        if failure:
            raise httpx.ConnectError(PRIVATE + ROOT)
        if streaming:
            return httpx.Response(
                200, content=b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json={"choices": []})

    app = create_app(
        settings, ollama_client=OllamaClient(settings, transport=httpx.MockTransport(upstream))
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "x-request-id": "request-safe-1",
                    "x-opencode-project-id": quote(ROOT),
                    "x-opencode-session-id": PRIVATE,
                    "x-opencode-project-root": quote(ROOT),
                    "authorization": "Bearer " + PRIVATE,
                },
                json={
                    "model": MODEL,
                    "stream": streaming,
                    "messages": [{"role": "user", "content": PRIVATE}],
                },
            )
        assert response.status_code == (502 if failure else 200)
        assert response.headers["x-request-id"] == "request-safe-1"
    finally:
        await app.state.database.engine.dispose()
        logging.getLogger().removeHandler(handler)
    rendered = stream.getvalue()
    assert all(secret not in rendered for secret in (PRIVATE, ROOT, quote(ROOT)))
    records = [json.loads(line) for line in rendered.splitlines()]
    final = [record for record in records if record["event"] == "request_completed"]
    assert len(final) == 1
    record = final[0]
    assert record["request_id"] == "request-safe-1"
    assert record["project_id"] and record["session_id"]
    assert record["model"] == MODEL
    assert record["duration_ms"] >= 0
    assert record["retrieval_count"] == record["retry_count"] == 0
    assert record["status_code"] == (502 if failure else 200)
    assert record["error_category"] == ("upstream_error" if failure else None)
    assert record["degraded_dependencies"] == ["postgres"]


async def test_unsafe_request_id_and_invalid_request_are_logged_without_body_or_headers():
    stream, handler = capture_logs()
    app = create_app(Settings(_env_file=None))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            response = await client.post(
                "/v1/chat/completions", headers={"x-request-id": quote(ROOT)}, content=PRIVATE
            )
        assert response.status_code == 400
        assert response.headers["x-request-id"] != quote(ROOT)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        record = next(value for value in records if value["event"] == "request_completed")
        assert record["error_category"] == "validation_error"
        assert all(secret not in stream.getvalue() for secret in (PRIVATE, ROOT, quote(ROOT)))
    finally:
        await app.state.database.engine.dispose()
        logging.getLogger().removeHandler(handler)
