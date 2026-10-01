"""Real queue contracts: races, crashed workers, retry exhaustion and privacy."""

import asyncio
import importlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.domain import AssistantCompletion, ConversationEventInput, RequestIdentity
from local_dev_rag.repository import ConversationRepository


def queue(database, **overrides):
    assert importlib.util.find_spec("local_dev_rag.jobs") is not None, "JobRepository is missing"
    module = importlib.import_module("local_dev_rag.jobs")
    return module.JobRepository(database, Settings(_env_file=None, **overrides))


async def enqueue(database, count=1):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("session", "project"))
        jobs = []
        for index in range(count):
            event = await repository.finalize_assistant(
                scope,
                AssistantCompletion(
                    payload={"content": f"Decision {index}"},
                    content_hash=f"source-{index}",
                    request_id=f"request-{index}",
                ),
            )
            jobs.append(await repository.enqueue_memory_job(event.id))
        return jobs


async def row(database, job_id):
    async with database.session() as session:
        return (
            (await session.execute(text("SELECT * FROM memory_jobs WHERE id=:id"), {"id": job_id}))
            .mappings()
            .one()
        )


async def expire(database, job_id):
    async with database.session() as session:
        await session.execute(
            text(
                "UPDATE memory_jobs SET lease_expires_at = "
                "clock_timestamp() - interval '1 second' WHERE id=:id"
            ),
            {"id": job_id},
        )


async def ready(database, job_id):
    async with database.session() as session:
        await session.execute(
            text(
                "UPDATE memory_jobs SET next_attempt_at = "
                "clock_timestamp() - interval '1 second' WHERE id=:id"
            ),
            {"id": job_id},
        )


async def test_concurrent_claimants_take_one_job_once_and_commit_lease(database: Database):
    [original] = await enqueue(database)
    results = await asyncio.wait_for(
        asyncio.gather(*[queue(database).claim(f"worker-{index}", 60) for index in range(8)]), 10
    )
    claimed = [job for job in results if job is not None]
    assert len(claimed) == 1
    assert claimed[0].id == original.id
    assert claimed[0].attempt_count == 1
    stored = await row(database, original.id)
    assert stored["status"] == "running"
    assert stored["lease_owner"] == claimed[0].lease_owner
    assert (stored["lease_expires_at"] - stored["started_at"]).total_seconds() == 60


async def test_locked_job_is_skipped_without_blocking_the_next_job(database: Database):
    first, second = await enqueue(database, 2)
    async with database.session() as session:
        await session.execute(
            text("SELECT id FROM memory_jobs WHERE id=:id FOR UPDATE"), {"id": first.id}
        )
        claimed = await asyncio.wait_for(queue(database).claim("worker", 30), 2)
        assert claimed.id == second.id


async def test_expired_lease_is_recovered_with_attempt_fencing_even_for_same_worker(database):
    [original] = await enqueue(database)
    stale, replacement = queue(database), queue(database)
    first = await stale.claim("same-worker", 60)
    assert first.id == original.id
    assert await replacement.claim("same-worker", 60) is None
    await expire(database, original.id)
    recovered = await replacement.claim("same-worker", 60)
    assert recovered.id == original.id and recovered.attempt_count == 2
    with pytest.raises(ValueError, match="lease"):
        await stale.complete(original.id, [])
    with pytest.raises(ValueError, match="lease"):
        await stale.fail(original.id, ValueError("secret source"), None)
    await replacement.complete(original.id, [])
    stored = await row(database, original.id)
    assert stored["status"] == "completed" and stored["completed_at"] is not None
    assert stored["lease_owner"] is None and stored["lease_expires_at"] is None


async def test_retry_uses_database_time_bounded_exponential_backoff_and_safe_errors(database):
    [original] = await enqueue(database)
    instance = queue(database, retry_initial_seconds=2, retry_max_seconds=5)
    for delay in (2, 4, 5):
        claimed = await instance.claim("worker", 60)
        assert claimed is not None
        async with database.session() as session:
            before = await session.scalar(text("SELECT clock_timestamp()"))
        await instance.fail(original.id, RuntimeError("API_KEY=private; entire prompt here"), None)
        stored = await row(database, original.id)
        actual = (stored["next_attempt_at"] - before).total_seconds()
        assert delay <= actual < delay + 1
        assert stored["status"] == "retry" and stored["lease_owner"] is None
        assert stored["error_category"] == "processing_error"
        assert stored["error_message"] == "Memory processing failed"
        assert await instance.claim("worker", 60) is None
        await ready(database, original.id)


async def test_explicit_retry_time_is_clamped_to_database_backoff_window(database):
    [original] = await enqueue(database)
    instance = queue(database, retry_initial_seconds=2, retry_max_seconds=5)
    for requested, want in [
        (datetime(1970, 1, 1, tzinfo=UTC), 2),
        (datetime(2099, 1, 1, tzinfo=UTC), 5),
    ]:
        await instance.claim("worker", 60)
        async with database.session() as session:
            before = await session.scalar(text("SELECT clock_timestamp()"))
        await instance.fail(original.id, TimeoutError("private"), requested)
        stored = await row(database, original.id)
        assert want <= (stored["next_attempt_at"] - before).total_seconds() < want + 1
        assert stored["error_category"] == "timeout"
        await ready(database, original.id)


async def test_retry_exhaustion_and_expired_final_attempt_are_terminal(database):
    first, second = await enqueue(database, 2)
    async with database.session() as session:
        await session.execute(
            text(
                "UPDATE memory_jobs SET next_attempt_at = "
                "clock_timestamp() + interval '1 day' WHERE id=:id"
            ),
            {"id": second.id},
        )
    instance = queue(database, retry_max_attempts=2)
    for _ in range(2):
        claimed = await instance.claim("worker", 60)
        assert claimed.id == first.id
        await instance.fail(first.id, ValueError("invalid output with prompt"), None)
        await ready(database, first.id)
    stored = await row(database, first.id)
    assert stored["status"] == "failed" and stored["attempt_count"] == 2
    assert stored["completed_at"] is not None and stored["lease_owner"] is None
    await ready(database, second.id)
    for _ in range(2):
        assert (await instance.claim("worker", 60)).id == second.id
        await expire(database, second.id)
    assert await instance.claim("worker", 60) is None
    stored = await row(database, second.id)
    assert stored["status"] == "failed" and stored["attempt_count"] == 2
    assert stored["error_category"] == "lease_expired"


async def test_duplicate_enqueue_remains_one_job_after_retry_and_completion(database):
    [original] = await enqueue(database)
    instance = queue(database)
    await instance.claim("worker", 60)
    await instance.fail(original.id, RuntimeError("fail"), None)
    await ready(database, original.id)
    await instance.claim("worker", 60)
    await instance.complete(original.id, [])
    async with database.session() as session:
        repository = ConversationRepository(session)
        repeated = await repository.enqueue_memory_job(original.source_event_id)
        assert repeated.id == original.id and repeated.status == "completed"
        assert await session.scalar(text("SELECT count(*) FROM memory_jobs")) == 1


async def test_completion_rejects_unknown_or_unrelated_memory_ids(database):
    [original] = await enqueue(database)
    instance = queue(database)
    await instance.claim("worker", 60)
    with pytest.raises(ValueError, match="memory"):
        await instance.complete(original.id, [uuid4()])
    assert (await row(database, original.id))["status"] == "running"
    await instance.complete(original.id, [])


async def test_completion_accepts_only_active_memories_from_its_own_source(database):
    from sqlalchemy.dialects.postgresql import insert

    from local_dev_rag.schema import memory_items

    first, second = await enqueue(database, 2)
    instance = queue(database)
    claimed = await instance.claim("worker", 60)
    other = second if claimed.id == first.id else first
    valid, unrelated, rejected = uuid4(), uuid4(), uuid4()
    async with database.session() as session:
        for memory_id, job, state in [
            (valid, claimed, "active"),
            (unrelated, other, "active"),
            (rejected, claimed, "rejected"),
        ]:
            await session.execute(
                insert(memory_items).values(
                    id=memory_id,
                    project_id=job.project_id,
                    source_session_id=job.session_id,
                    source_event_id=job.source_event_id,
                    kind="decision",
                    text=str(memory_id),
                    confidence=0.9,
                    importance=0.8,
                    state=state,
                    curator_model="qwen2.5-coder:1.5b",
                )
            )
    for bad in (unrelated, rejected):
        with pytest.raises(ValueError, match="memory"):
            await instance.complete(claimed.id, [valid, bad])
    await instance.complete(claimed.id, [valid, valid])
    assert (await row(database, claimed.id))["status"] == "completed"


async def test_expired_lease_cannot_be_completed_or_failed_before_recovery(database):
    [original] = await enqueue(database)
    instance = queue(database)
    await instance.claim("worker", 60)
    await expire(database, original.id)
    with pytest.raises(ValueError, match="lease"):
        await instance.complete(original.id, [])
    with pytest.raises(ValueError, match="lease"):
        await instance.fail(original.id, RuntimeError("content"), None)


async def test_claim_validation_rejects_invalid_workers_and_leases(database):
    instance = queue(database)
    for worker, lease in [
        ("", 60),
        (" ", 60),
        ("x" * 129, 60),
        ("worker", 0),
        ("worker", -1),
        ("worker", True),
    ]:
        with pytest.raises(ValueError):
            await instance.claim(worker, lease)


async def test_curator_source_is_bounded_ordered_and_never_includes_future_or_other_scope(database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("s", "p"))
        for index in range(4):
            await repository.append_events(
                scope,
                [
                    ConversationEventInput(
                        event_type="message",
                        role="user",
                        payload={"content": str(index)},
                        content_hash=f"user-{index}",
                        request_id=str(index),
                    )
                ],
            )
        completed = await repository.finalize_assistant(
            scope,
            AssistantCompletion(
                payload={"content": "decision"},
                content_hash="answer",
                request_id="answer",
            ),
        )
        await repository.append_events(
            scope,
            [
                ConversationEventInput(
                    event_type="message",
                    role="user",
                    payload={"content": "future"},
                    content_hash="future",
                    request_id="future",
                )
            ],
        )
        assert hasattr(repository, "curator_source"), "Bounded curator source loader is missing"
        incoming = await repository.curator_source(completed.id, max_events=3)
        assert incoming.project_id == scope.project_id and incoming.session_id == scope.session_id
        assert incoming.source_event_id == completed.id
        assert [event.sequence for event in incoming.events] == [3, 4, 5]
        with pytest.raises(ValueError):
            await repository.curator_source(uuid4())
