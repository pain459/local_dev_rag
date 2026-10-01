"""Final telemetry follows real ASGI delivery and durable capture outcomes."""

import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.requests import ClientDisconnect

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.logging import SafeJSONFormatter
from local_dev_rag.ollama import OllamaClient

from .capture_utils import Fragments, invoke_stream
from .test_capture import SSE, rows

PRIVATE = "private-stream-prompt-tool-authorization-memory"


@pytest.mark.parametrize(
    "case,category,outcome,capture_status",
    [
        ("success", None, "completed", "completed"),
        ("sse_error", "upstream_error", "upstream_error", "incomplete"),
        ("premature_eof", "upstream_error", "upstream_error", "incomplete"),
        ("read_error", "upstream_error", "upstream_error", "incomplete"),
        ("disconnect", "cancelled", "cancelled", "incomplete"),
        ("send_error", "cancelled", "cancelled", "incomplete"),
        ("cancel", "cancelled", "cancelled", "incomplete"),
        ("capture_unavailable", None, "completed", "unavailable"),
        ("finalize_unavailable", "processing_error", "completed", "unavailable"),
    ],
)
async def test_real_stream_logs_outcome_without_changing_status_capture_or_cleanup(
    database,
    monkeypatch,
    case,
    category,
    outcome,
    capture_status,
):
    disconnect, delivered = asyncio.Event(), asyncio.Event()

    class Waiting(Fragments):
        async def __aiter__(self):
            yield SSE
            await asyncio.Future()

    chunks = [SSE]
    if case == "sse_error":
        chunks.append(b'event: error\ndata: {"message":"' + PRIVATE.encode() + b'"}\n\n')
    if case == "premature_eof":
        chunks = [SSE.split(b"data: [DONE]")[0]]
    fragments = (
        Waiting([])
        if case in {"disconnect", "cancel"}
        else Fragments(
            chunks,
            httpx.ReadError(PRIVATE) if case == "read_error" else None,
        )
    )
    active_db = database
    if case == "capture_unavailable":
        active_db = Database.create(
            Settings(
                _env_file=None, database_url="postgresql+asyncpg://unused:unused@127.0.0.1:1/unused"
            )
        )

    @asynccontextmanager
    async def offline(self):
        raise OSError(PRIVATE)
        yield

    def upstream(request):
        if case == "finalize_unavailable":
            monkeypatch.setattr(Database, "session", offline)
        return httpx.Response(200, stream=fragments, headers={"content-type": "text/event-stream"})

    app = create_app(
        Settings(_env_file=None),
        database=active_db,
        ollama_client=OllamaClient(
            Settings(_env_file=None), transport=httpx.MockTransport(upstream)
        ),
    )
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SafeJSONFormatter())
    logging.getLogger().addHandler(handler)
    bodies, statuses = [], []

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        if message["type"] == "http.response.body" and message.get("body"):
            if case == "send_error":
                raise OSError(PRIVATE)
            bodies.append(message["body"])
            delivered.set()
            disconnect.set()

    try:
        if case == "disconnect":
            await asyncio.wait_for(
                invoke_stream(app, send, asgi_version="2.3", disconnect=disconnect), 2
            )
        elif case == "cancel":
            task = asyncio.create_task(invoke_stream(app, send))
            try:
                await asyncio.wait_for(delivered.wait(), 2)
            finally:
                task.cancel(PRIVATE)
                with pytest.raises(asyncio.CancelledError):
                    await task
        elif case in {"send_error", "read_error"}:
            with pytest.raises(ClientDisconnect if case == "send_error" else httpx.ReadError):
                await invoke_stream(app, send)
        else:
            await invoke_stream(app, send)
    finally:
        monkeypatch.undo()
        logging.getLogger().removeHandler(handler)
        if active_db is not database:
            await active_db.engine.dispose()
    assert statuses == [200]
    assert fragments.closed
    if case not in {"read_error", "disconnect", "send_error", "cancel"}:
        assert b"".join(bodies) == b"".join(chunks)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    finals = [record for record in records if record["event"] == "request_completed"]
    assert len(finals) == 1
    record = finals[0]
    assert record["status_code"] == 200
    assert record["error_category"] == category
    assert record.get("stream_outcome") == outcome
    assert record.get("capture_status") == capture_status
    assert record["degraded_dependencies"] == (["postgres"] if "unavailable" in case else [])
    assert PRIVATE not in output.getvalue()
    assert "answer" not in output.getvalue()
    events, jobs = await rows(database)
    if capture_status == "completed":
        assert [event["role"] for event in events] == ["assistant"]
        assert len(jobs) == 1
    elif capture_status == "incomplete":
        assert [event["role"] for event in events] == ["proxy"]
        assert events[0]["completed"] is False and jobs == []
    else:
        assert events == [] and jobs == []
