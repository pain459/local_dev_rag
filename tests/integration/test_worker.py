"""Durable worker effects through real PostgreSQL, Chroma and Ollama HTTP adapters."""

import asyncio
import importlib
import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from local_dev_rag.config import Settings
from local_dev_rag.curator import Curator
from local_dev_rag.domain import AssistantCompletion, RequestIdentity
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.repository import ConversationRepository
from local_dev_rag.schema import memory_items
from local_dev_rag.vector_store import VectorStore

from .test_jobs import ready, row

KINDS = ["requirement", "decision", "constraint", "preference", "outcome"]
FACTS = {
    "requirement": "The project requires transactional writes.",
    "decision": "Use PostgreSQL for durable memory.",
    "constraint": "Redis has a 256 MB limit.",
    "preference": "The team prefers Ruff for linting.",
    "outcome": "The PostgreSQL migration completed successfully.",
}


async def seed(database, *, project="project", session_id="session"):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity(session_id, project))
        event = await repository.finalize_assistant(
            scope,
            AssistantCompletion(
                payload={
                    "content": "We selected PostgreSQL for durable memory. "
                    + " ".join(text for kind, text in FACTS.items() if kind != "decision")
                },
                content_hash=str(uuid4()),
                request_id=str(uuid4()),
            ),
        )
        return scope, await repository.enqueue_memory_job(event.id)


def envelope(kinds=KINDS):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "memories": [
                                {
                                    "kind": kind,
                                    "text": FACTS[kind],
                                    "confidence": 0.9,
                                    "importance": 0.8,
                                }
                                for kind in kinds
                            ]
                        }
                    ),
                },
            }
        ]
    }


async def memories(database):
    async with database.session() as session:
        return (await session.execute(select(memory_items))).mappings().all()


def worker(database, store, handler, *, settings=None, **options):
    assert importlib.util.find_spec("local_dev_rag.worker") is not None, "Worker is missing"
    module = importlib.import_module("local_dev_rag.worker")
    settings = settings or Settings(_env_file=None, embedding_version=3)
    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    return module.Worker(
        database,
        settings,
        curator=Curator(client, settings),
        embedder=client,
        vector_store=store,
        batch_size=2,
        **options,
    )


def vector_store(chroma_url):
    return VectorStore(
        Settings(_env_file=None, chromadb_url=chroma_url), collection_name=f"worker-{uuid4()}"
    )


async def test_success_commits_memories_batches_vectors_and_only_then_completes(
    database, chroma_url
):
    scope, job = await seed(database)
    batches = []

    async def handler(request):
        payload = json.loads(request.content)
        assert (await row(database, job.id))["status"] == "running"
        if request.url.path == "/api/embed":
            assert len(await memories(database)) == 5  # committed before external embedding
            assert payload["model"] == "nomic-embed-text:latest"
            batches.append(len(payload["input"]))
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(payload["input"])})
        assert str(job.source_event_id) in payload["messages"][1]["content"]
        return httpx.Response(200, json=envelope())

    store = vector_store(chroma_url)
    instance = worker(database, store, handler)
    result = await instance.run_once()
    assert (result.state, result.job_id, result.memory_count) == ("completed", job.id, 5)
    assert batches == [2, 2, 1]
    stored = await memories(database)
    hits = await store.query(scope.project_id, (1, 0), 10)
    assert {hit.memory.id for hit in hits} == {item["id"] for item in stored}
    assert all(hit.memory.embedding_version == 3 for hit in hits)
    assert all(hit.memory.source_event_id == job.source_event_id for hit in hits)
    assert all(item["curator_model"] == "qwen2.5-coder:1.5b" for item in stored)
    assert (await row(database, job.id))["status"] == "completed"
    idle = await instance.run_once()
    assert (idle.state, idle.job_id, idle.memory_count) == ("idle", None, 0)


@pytest.mark.parametrize(
    ("failure", "category", "count"),
    [
        ("curator_timeout", "timeout", 0),
        ("malformed", "validation_error", 0),
        ("embedder", "upstream_error", 5),
        ("chroma", "processing_error", 5),
    ],
)
async def test_dependency_failure_keeps_retry_durable_and_source_intact(
    database, chroma_url, failure, category, count
):
    scope, job = await seed(database)

    def handler(request):
        if request.url.path == "/api/embed":
            if failure == "embedder":
                return httpx.Response(503, text="private source and credentials")
            return httpx.Response(200, json={"embeddings": [[1, 0], [1, 0]]})
        if failure == "curator_timeout":
            raise httpx.ReadTimeout("private source", request=request)
        return httpx.Response(200, json=envelope() if failure != "malformed" else {"private": 1})

    store = vector_store(chroma_url)
    if failure == "chroma":
        store = VectorStore(
            Settings(_env_file=None),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(503, text="private source")
            ),
        )
    result = await worker(database, store, handler).run_once()
    assert (result.state, result.memory_count) == ("retry", count)
    stored_job = await row(database, job.id)
    assert stored_job["status"] == "retry" and stored_job["lease_owner"] is None
    assert stored_job["next_attempt_at"] > stored_job["started_at"]
    assert stored_job["error_category"] == category
    assert "private" not in stored_job["error_message"]
    assert len(await memories(database)) == count
    # A subsequent foreground completion still persists/enqueues independently.
    other_scope, next_job = await seed(database, session_id="foreground")
    assert other_scope.project_id == scope.project_id
    assert (await row(database, next_job.id))["status"] == "pending"


async def test_retry_after_relational_commit_reuses_ids_and_skips_nondeterministic_curator(
    database, chroma_url
):
    scope, job = await seed(database)
    calls = 0
    fail = True

    def handler(request):
        nonlocal calls
        if request.url.path == "/api/embed":
            payload = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[1, 0]] * len(payload["input"])})
        calls += 1
        return httpx.Response(200, json=envelope())

    class FailingStore(VectorStore):
        async def upsert(self, items):
            if fail:
                from local_dev_rag.vector_store import VectorStoreUnavailable

                raise VectorStoreUnavailable("private")
            assert (await row(database, job.id))["status"] == "running"
            await super().upsert(items)

    store = FailingStore(
        Settings(_env_file=None, chromadb_url=chroma_url), collection_name=f"retry-{uuid4()}"
    )
    instance = worker(database, store, handler)
    assert (await instance.run_once()).state == "retry"
    ids = {item["id"] for item in await memories(database)}
    assert len(ids) == 5 and await store.query(scope.project_id, (1, 0), 10) == []
    await ready(database, job.id)
    fail = False
    # New process / repositories use the durable relational checkpoint.
    assert (await worker(database, store, handler).run_once()).state == "completed"
    assert calls == 1
    assert {item["id"] for item in await memories(database)} == ids
    assert {hit.memory.id for hit in await store.query(scope.project_id, (1, 0), 10)} == ids
    assert (await row(database, job.id))["attempt_count"] == 2


@pytest.mark.parametrize("stage", ["curator", "embedder"])
async def test_cancellation_releases_claim_and_preserves_checkpoint(database, chroma_url, stage):
    _, job = await seed(database)
    reached = asyncio.Event()

    async def handler(request):
        if (request.url.path == "/api/embed") == (stage == "embedder"):
            reached.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=envelope())

    instance = worker(database, vector_store(chroma_url), handler)
    task = asyncio.create_task(instance.run_once())
    await asyncio.wait_for(reached.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await row(database, job.id))["status"] == "retry"
    assert (await row(database, job.id))["lease_owner"] is None
    assert len(await memories(database)) == (5 if stage == "embedder" else 0)


async def test_retry_exhaustion_is_terminal_and_empty_curator_output_completes(
    database, chroma_url
):
    _, job = await seed(database)
    instance = worker(
        database,
        vector_store(chroma_url),
        lambda request: httpx.Response(200, json={}),
        settings=Settings(_env_file=None, retry_max_attempts=2),
    )
    assert (await instance.run_once()).state == "retry"
    await ready(database, job.id)
    assert (await instance.run_once()).state == "failed"
    assert (await row(database, job.id))["status"] == "failed"
    assert (await instance.run_once()).state == "idle"
    _, empty = await seed(database)
    result = await worker(
        database, vector_store(chroma_url), lambda request: httpx.Response(200, json=envelope([]))
    ).run_once()
    assert result.state == "completed" and result.memory_count == 0
    assert (await row(database, empty.id))["status"] == "completed"


async def test_forever_processes_jobs_and_stop_interrupts_idle_wait(database, chroma_url):
    _, job = await seed(database)
    stop = asyncio.Event()

    def handler(request):
        stop.set()
        return httpx.Response(200, json=envelope([]))

    instance = worker(database, vector_store(chroma_url), handler, poll_seconds=30)
    await asyncio.wait_for(instance.run_forever(stop), 3)
    assert (await row(database, job.id))["status"] == "completed"
    stop.clear()
    task = asyncio.create_task(instance.run_forever(stop))
    await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, 1)


async def test_real_curator_deadline_leaves_retryable_job(database, chroma_url):
    _, job = await seed(database)

    async def handler(request):
        await asyncio.Event().wait()

    instance = worker(database, vector_store(chroma_url), handler)
    instance.curator.timeout_seconds = 0.01
    assert (await instance.run_once()).state == "retry"
    assert (await row(database, job.id))["error_category"] == "timeout"
    assert await memories(database) == []


async def test_partial_vector_batch_failure_converges_without_duplicate_rows(database, chroma_url):
    scope, job = await seed(database)
    batches = 0
    fail = True
    store = vector_store(chroma_url)

    def handler(request):
        nonlocal batches
        if request.url.path == "/api/embed":
            batches += 1
            if fail and batches == 2:
                return httpx.Response(503)
            size = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"embeddings": [[1, 0]] * size})
        return httpx.Response(200, json=envelope())

    assert (await worker(database, store, handler).run_once()).state == "retry"
    assert len(await memories(database)) == 5
    assert len(await store.query(scope.project_id, (1, 0), 10)) == 2
    await ready(database, job.id)
    fail = False
    assert (await worker(database, store, handler).run_once()).state == "completed"
    assert len(await memories(database)) == 5
    assert len(await store.query(scope.project_id, (1, 0), 10)) == 5


async def test_process_loss_after_insert_recovers_expired_lease_from_snapshot(database, chroma_url):
    from local_dev_rag.domain import MemoryDraft
    from local_dev_rag.jobs import JobRepository
    from local_dev_rag.repository import MemoryRepository

    from .test_jobs import expire

    scope, job = await seed(database)
    settings = Settings(_env_file=None)
    await JobRepository(database, settings).claim("crashed", 60)
    async with database.session() as session:
        source = await ConversationRepository(session).curator_source(job.source_event_id)
        stored = await MemoryRepository(session).persist(
            source, [MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)], curator_model="curator"
        )
    await expire(database, job.id)

    def handler(request):
        assert request.url.path == "/api/embed", "Recovery reran the curator"
        return httpx.Response(200, json={"embeddings": [[1, 0]]})

    store = vector_store(chroma_url)
    assert (await worker(database, store, handler).run_once()).state == "completed"
    assert [hit.memory.id for hit in await store.query(scope.project_id, (1, 0), 10)] == [
        stored[0].id
    ]
    assert len(await memories(database)) == 1


async def test_background_failure_does_not_block_a_foreground_completion(database, chroma_url):
    from local_dev_rag.api import create_app

    from .test_foreground_flow import RESULT, deliver, payload

    _, job = await seed(database)
    store = vector_store(chroma_url)
    assert (
        await worker(database, store, lambda request: httpx.Response(503)).run_once()
    ).state == "retry"

    def handler(request):
        if request.url.path == "/api/embed":
            return httpx.Response(503)
        return httpx.Response(200, json=RESULT)

    settings = Settings(_env_file=None, chromadb_url=chroma_url)
    app = create_app(
        settings,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
    )
    body, state = await deliver(app, payload())
    assert json.loads(body) == RESULT and state["capture_status"] == "completed"
    assert (await row(database, job.id))["status"] == "retry"


async def test_malformed_embedding_is_retryable_validation_failure(database, chroma_url):
    _, job = await seed(database)

    def handler(request):
        return httpx.Response(
            200,
            json={"embeddings": [[True, 0]]} if request.url.path == "/api/embed" else envelope(),
        )

    assert (await worker(database, vector_store(chroma_url), handler).run_once()).state == "retry"
    assert (await row(database, job.id))["error_category"] == "validation_error"
    assert len(await memories(database)) == 5


async def test_worker_never_revives_retired_memories_on_retry(database, chroma_url):
    from sqlalchemy import update

    scope, job = await seed(database)
    store = vector_store(chroma_url)

    def handler(request):
        return (
            httpx.Response(503)
            if request.url.path == "/api/embed"
            else httpx.Response(200, json=envelope())
        )

    assert (await worker(database, store, handler).run_once()).state == "retry"
    async with database.session() as session:
        await session.execute(update(memory_items).values(state="deleted"))
    await ready(database, job.id)

    def forbidden(request):
        pytest.fail("Retired extraction was regenerated or embedded")

    result = await worker(database, store, forbidden).run_once()
    assert result.state == "completed" and result.memory_count == 0
    assert len(await memories(database)) == 5
    assert await store.query(scope.project_id, (1, 0), 10) == []


@pytest.mark.parametrize("state", ["deleted", "rejected", "superseded"])
async def test_retired_early_batch_memory_is_not_recalled_after_partial_success_and_retry(
    database, chroma_url, state
):
    from sqlalchemy import update

    from local_dev_rag.api import create_app
    from local_dev_rag.schema import conversation_events

    from .test_foreground_flow import RESULT, deliver

    scope, job = await seed(database)
    store = vector_store(chroma_url)
    fail = True
    batches = 0

    def handler(request):
        nonlocal batches
        if request.url.path == "/api/embed":
            batches += 1
            if fail and batches == 2:
                return httpx.Response(503)
            size = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"embeddings": [[1, 0]] * size})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "memories": [
                                        {
                                            "kind": "decision",
                                            "text": f"PostgreSQL {text} successfully.",
                                            "confidence": 0.9,
                                            "importance": 0.8,
                                        }
                                        for text in [
                                            "started",
                                            "restarted",
                                            "recovered",
                                            "connected",
                                            "migrated",
                                        ]
                                    ]
                                }
                            ),
                        },
                    }
                ]
            },
        )

    # Distinct fact bodies prove which exact vector is eligible in the actual prompt.
    async with database.session() as session:
        await session.execute(
            update(conversation_events)
            .where(conversation_events.c.id == job.source_event_id)
            .values(
                payload={
                    "content": " ".join(
                        f"PostgreSQL {text} successfully."
                        for text in ["started", "restarted", "recovered", "connected", "migrated"]
                    )
                }
            )
        )
    assert (await worker(database, store, handler).run_once()).state == "retry"
    early_hits = await store.query(scope.project_id, (1, 0), 10)
    assert len(early_hits) == 2
    retired = early_hits[0].memory
    replacement = early_hits[1].memory
    async with database.session() as session:
        await session.execute(
            update(memory_items)
            .where(memory_items.c.id == retired.id)
            .values(
                state=state,
                superseded_by_id=replacement.id if state == "superseded" else None,
            )
        )
    await ready(database, job.id)
    fail = False
    assert (await worker(database, store, handler).run_once()).state == "completed"

    observed = []

    def foreground(request):
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        observed.append(body)
        return httpx.Response(200, json=RESULT)

    settings = Settings(_env_file=None, retrieval_result_limit=10, retrieval_min_score=0)
    app = create_app(
        settings,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(settings, transport=httpx.MockTransport(foreground)),
    )
    query = {
        "model": "qwen3-coder:30b",
        "messages": [{"role": "user", "content": "Recall PostgreSQL successfully."}],
    }
    body, diagnostics = await deliver(app, query)
    assert json.loads(body) == RESULT
    prompt = json.dumps(observed[0]["messages"])
    assert retired.text not in prompt
    assert replacement.text in prompt
    assert diagnostics["proxy_diagnostics"].retrieval_count == 4
