"""Operational states through PostgreSQL and the real foreground/worker adapters."""

import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.logging import SafeJSONFormatter, configure_logging
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.schema import conversation_events, memory_jobs
from local_dev_rag.vector_store import VectorStore

from .test_foreground_flow import RESULT, SSE, deliver, payload, settings
from .test_jobs import row
from .test_worker import seed, worker


@pytest.mark.parametrize("stall_query", [False, True])
async def test_readyz_bounds_real_sqlalchemy_stalled_rollback_without_orphan_tasks(
    database, monkeypatch, stall_query
):
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.util import await_only

    from local_dev_rag.domain import DependencyStatus
    from local_dev_rag.readiness import ReadinessService

    class Healthy:
        def __init__(self, name):
            self.name = name

        async def health(self):
            return DependencyStatus(self.name, "healthy")

        async def model_health(self, model, name):
            return DependencyStatus(name, "healthy")

    gate = asyncio.Event()
    entered = asyncio.Event()
    rollback = database.engine.sync_engine.dialect.do_rollback
    execute = AsyncSession.execute

    async def stalled_execute(self, *args, **kwargs):
        result = await execute(self, *args, **kwargs)
        if stall_query:
            await gate.wait()
        return result

    monkeypatch.setattr(AsyncSession, "execute", stalled_execute)

    def stalled_rollback(connection):
        entered.set()
        await_only(gate.wait())
        rollback(connection)

    monkeypatch.setattr(database.engine.sync_engine.dialect, "do_rollback", stalled_rollback)
    checker = ReadinessService(
        Settings(_env_file=None),
        database=database,
        vector_store=Healthy("chromadb"),
        ollama=Healthy("ollama"),
        timeout_seconds=0.02,
    )
    app = create_app(Settings(_env_file=None), database=database, readiness_service=checker)
    before = asyncio.all_tasks()
    factory = asyncio.get_running_loop().get_task_factory()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        task = asyncio.create_task(client.get("/readyz"))
        try:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            assert entered.is_set(), "The real SQLAlchemy rollback was not exercised"
            assert task in done, "readyz exceeded the independently bounded cleanup deadline"
            response = task.result()
            assert response.status_code == 200
            assert response.json()["status"] == "degraded"
            assert response.json()["dependencies"]["postgres"] == "unavailable"
            assert asyncio.get_running_loop().get_task_factory() is factory
            assert not (asyncio.all_tasks() - before - {asyncio.current_task()})
            assert database.engine.pool.checkedout() == 0
        finally:
            gate.set()
            await asyncio.wait_for(task, 2)
    monkeypatch.undo()
    # Pool remains usable for foreground ownership after the health failure.
    assert (await database.health()).state == "healthy"


async def test_database_and_real_chroma_health(database, chroma_url):
    assert callable(getattr(database, "health", None)), "Database health is missing"
    assert (await database.health()).state == "healthy"
    store = VectorStore(Settings(_env_file=None, chromadb_url=chroma_url))
    assert callable(getattr(store, "health", None)), "Chroma health is missing"
    assert (await store.health()).state == "healthy"


async def test_capture_failure_after_upstream_completion_is_visible_in_final_log(
    database, monkeypatch
):
    config = settings()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SafeJSONFormatter())
    logging.getLogger().addHandler(handler)

    @asynccontextmanager
    async def offline(self):
        raise OSError("private prompt tool memory /Users/private/root")
        yield

    def upstream(request):
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        monkeypatch.setattr(Database, "session", offline)
        return httpx.Response(200, json=RESULT)

    def chroma(request):
        if request.url.path == "/api/v1/collections":
            return httpx.Response(200, json={"id": "00000000-0000-0000-0000-000000000001"})
        return httpx.Response(
            200, json={"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        )

    app = create_app(
        config,
        database=database,
        vector_store=VectorStore(config, transport=httpx.MockTransport(chroma)),
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(upstream)),
    )
    try:
        body, state = await deliver(app, payload())
        assert json.loads(body) == RESULT and state["capture_status"] == "unavailable"
    finally:
        monkeypatch.undo()
        logging.getLogger().removeHandler(handler)
    final = next(
        json.loads(line)
        for line in output.getvalue().splitlines()
        if json.loads(line)["event"] == "request_completed"
    )
    assert final["degraded_dependencies"] == ["postgres"]
    assert final["error_category"] == "processing_error"
    assert "private" not in output.getvalue()
    async with database.session() as session:
        assert (await session.execute(select(memory_jobs))).all() == []


@pytest.mark.parametrize("failure", ["curator", "embedder", "chromadb"])
async def test_worker_failures_are_safe_logged_and_visible_in_durable_health(database, failure):
    scope, job = await seed(database)
    config = Settings(_env_file=None, retry_max_attempts=1)
    configure_logging(config)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SafeJSONFormatter())
    logging.getLogger().addHandler(handler)

    def upstream(request):
        if failure == "curator" or request.url.path == "/api/embed" and failure == "embedder":
            raise httpx.ReadTimeout("private prompt tool output memory /Users/private/root")
        if request.url.path == "/api/embed":
            return httpx.Response(
                200, json={"embeddings": [[1, 0]] * len(json.loads(request.content)["input"])}
            )
        from .test_worker import envelope

        return httpx.Response(200, json=envelope(["decision"]))

    def chroma(request):
        raise httpx.ConnectError("private memory authorization")

    store = VectorStore(
        config, collection_name=f"degraded-{uuid4()}", transport=httpx.MockTransport(chroma)
    )
    try:
        result = await worker(database, store, upstream, settings=config).run_once()
    finally:
        logging.getLogger().removeHandler(handler)
    assert result.state == "failed"
    assert (await row(database, job.id))["status"] == "failed"
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    record = next((record for record in records if record["event"] == "memory_job_failed"), None)
    assert record is not None, "Worker failure telemetry is missing"
    assert record["project_id"] == str(scope.project_id)
    assert record["session_id"] == str(scope.session_id)
    assert record["job_id"] == str(job.id)
    assert record["retry_count"] == 0 and record["attempt_count"] == 1
    assert record["duration_ms"] >= 0
    assert record["degraded_dependencies"] == [failure]
    assert record["error_category"] in {"timeout", "processing_error"}
    assert "private" not in output.getvalue()
    assert callable(getattr(database, "memory_health", None)), "Durable job health is missing"
    assert (await database.memory_health()).state == "degraded"


@pytest.mark.parametrize("failure", ["chromadb", "embedder", "curator", "postgres"])
@pytest.mark.parametrize("stream", [False, True])
async def test_degraded_readiness_still_completes_and_captures_foreground(
    database, failure, stream
):
    config = settings(chromadb_url="http://127.0.0.1:1")
    active_db = database
    if failure == "postgres":
        active_db = Database.create(
            Settings(
                _env_file=None, database_url="postgresql+asyncpg://unused:unused@127.0.0.1:1/unused"
            )
        )
    request_payload = payload(stream)

    def upstream(request):
        if request.url.path == "/api/tags":
            models = [config.default_model, config.curator_model, config.embedding_model]
            missing = (
                config.curator_model
                if failure == "curator"
                else config.embedding_model
                if failure == "embedder"
                else None
            )
            return httpx.Response(
                200, json={"models": [{"name": name} for name in models if name != missing]}
            )
        if request.url.path == "/api/embed":
            if failure == "embedder":
                return httpx.Response(503, json={"error": "private"})
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        body = json.loads(request.content)
        if failure == "postgres":
            assert body == request_payload
        return (
            httpx.Response(200, content=SSE, headers={"content-type": "text/event-stream"})
            if stream
            else httpx.Response(200, json=RESULT)
        )

    def chroma(request):
        if failure == "chromadb":
            raise httpx.ConnectError("private memory")
        if request.url.path == "/api/v1/heartbeat":
            return httpx.Response(200, json={"nanosecond heartbeat": 1})
        if request.url.path == "/api/v1/collections":
            return httpx.Response(200, json={"id": "00000000-0000-0000-0000-000000000001"})
        return httpx.Response(
            200, json={"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        )

    app = create_app(
        config,
        database=active_db,
        vector_store=VectorStore(config, transport=httpx.MockTransport(chroma)),
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(upstream)),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            response = await client.get("/readyz")
        assert response.status_code == 200 and response.json()["status"] == "degraded"
        assert response.json()["dependencies"][failure] != "healthy"
        body, state = await deliver(app, request_payload)
        assert body == SSE if stream else json.loads(body) == RESULT
        assert state["capture_status"] == ("unavailable" if failure == "postgres" else "completed")
        async with database.session() as session:
            events = (await session.execute(select(conversation_events))).all()
            jobs = (await session.execute(select(memory_jobs))).all()
        assert len(events) == (0 if failure == "postgres" else 5)
        assert len(jobs) == (0 if failure == "postgres" else 1)
    finally:
        if failure == "postgres":
            await active_db.engine.dispose()
