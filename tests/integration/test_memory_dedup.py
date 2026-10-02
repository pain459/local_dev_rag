"""Project-level canonical memories retain per-source retry checkpoints."""

import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select, update

from local_dev_rag.domain import EmbeddedMemory, MemoryDraft
from local_dev_rag.repository import ConversationRepository, MemoryRepository
from local_dev_rag.schema import memory_items

from .test_worker import seed, vector_store


async def test_concurrent_workers_and_partial_vector_retry_keep_one_copy_per_fact(
    database, chroma_url
):
    from local_dev_rag.config import Settings
    from local_dev_rag.vector_store import VectorStore, VectorStoreUnavailable

    from .test_jobs import ready, row
    from .test_worker import envelope, worker

    scope, first = await seed(database, session_id="first")
    _, second = await seed(database, session_id="second")
    failed = False

    class PartialStore(VectorStore):
        async def upsert(self, items):
            nonlocal failed
            await super().upsert(items)
            if not failed:
                failed = True
                raise VectorStoreUnavailable("synthetic failure after a partial write")

    store = PartialStore(
        Settings(_env_file=None, chromadb_url=chroma_url), collection_name=f"dedup-{uuid4()}"
    )

    def handler(request):
        if request.url.path == "/api/embed":
            size = len(json.loads(request.content)["input"])
            return httpx.Response(200, json={"embeddings": [[1, 0]] * size})
        return httpx.Response(200, json=envelope())

    results = await asyncio.gather(*(worker(database, store, handler).run_once() for _ in range(2)))
    assert sorted(result.state for result in results) == ["completed", "retry"]
    retry = next(result for result in results if result.state == "retry")
    ids = {hit.memory.id for hit in await store.query(scope.project_id, (1, 0), 20)}
    assert len(ids) == 5
    await ready(database, retry.job_id)

    def retry_handler(request):
        assert request.url.path == "/api/embed", "A shared checkpoint was re-extracted"
        return handler(request)

    assert (await worker(database, store, retry_handler).run_once()).state == "completed"
    assert {hit.memory.id for hit in await store.query(scope.project_id, (1, 0), 20)} == ids
    assert (await row(database, first.id))["status"] == "completed"
    assert (await row(database, second.id))["status"] == "completed"
    async with database.session() as session:
        assert len((await session.execute(select(memory_items))).all()) == 5


async def test_shared_memory_allows_independent_job_completion_and_fencing(database):
    from local_dev_rag.jobs import JobRepository

    _, first = await seed(database, session_id="first")
    _, second = await seed(database, session_id="second")
    canonical = (await persist(database, first))[0]
    duplicate = (await persist(database, second, "  USE   Redis for caching. ", "preference"))[0]
    assert duplicate == canonical
    jobs = JobRepository(database)
    for _ in range(2):
        claimed = await jobs.claim("worker", 60)
        await jobs.complete(claimed.id, [canonical.id])
        with pytest.raises(ValueError):
            await jobs.complete(claimed.id, [canonical.id])
    assert await jobs.claim("worker", 60) is None


async def persist(database, job, text="Use Redis for caching.", kind="decision"):
    async with database.session() as session:
        source = await ConversationRepository(session).curator_source(job.source_event_id)
        return await MemoryRepository(session).persist(
            source, [MemoryDraft(kind, text, 0.9, 0.8)], curator_model="curator"
        )


async def test_concurrent_sources_share_one_active_memory_and_preserve_checkpoints(database):
    jobs = [(await seed(database, session_id=str(uuid4())))[1] for _ in range(8)]
    results = await asyncio.gather(*(persist(database, job) for job in jobs))
    assert len({items[0].id for items in results}) == 1
    canonical = results[0][0]
    assert canonical.source_event_id in {job.source_event_id for job in jobs}
    for job in jobs:
        # A retry must use the original checkpoint even if extraction would change.
        assert await persist(database, job, "A different fact.") == [canonical]


@pytest.mark.parametrize("state", ["deleted", "rejected", "superseded"])
async def test_new_source_can_reaffirm_retired_fact_but_retry_cannot_reactivate(database, state):
    _, first = await seed(database)
    old = (await persist(database, first))[0]
    replacement = None
    if state == "superseded":
        _, other = await seed(database)
        replacement = (await persist(database, other, "Use PostgreSQL."))[0].id
    async with database.session() as session:
        await session.execute(
            update(memory_items)
            .where(memory_items.c.id == old.id)
            .values(state=state, superseded_by_id=replacement)
        )
    assert (await persist(database, first))[0].state == state
    _, fresh = await seed(database, session_id="fresh")
    new = (await persist(database, fresh))[0]
    assert new.id != old.id and new.state == "active"
    assert (await persist(database, first))[0].id == old.id


async def test_duplicates_cannot_fill_chroma_candidate_limit(database, chroma_url):
    store = vector_store(chroma_url)
    for index in range(25):
        scope, job = await seed(database, session_id=f"source-{index}")
        items = await persist(database, job)
        await store.upsert([EmbeddedMemory(item, (1, 0)) for item in items])
    _, job = await seed(database, session_id="different")
    distinct = (await persist(database, job, "Redis requires port 6379."))[0]
    await store.upsert([EmbeddedMemory(distinct, (0.9, 0.1))])
    hits = await store.query(scope.project_id, (1, 0), 20)
    assert distinct.id in {hit.memory.id for hit in hits}
    assert len(hits) == 2
    async with database.session() as session:
        assert len((await session.execute(select(memory_items))).all()) == 2
