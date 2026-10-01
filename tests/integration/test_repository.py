"""Storage contracts: lost scope isolation, event ordering, or dedupe must fail these tests."""

import asyncio
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from local_dev_rag.db import Database
from local_dev_rag.domain import (
    AssistantCompletion,
    ConversationEventInput,
    RequestIdentity,
)
from local_dev_rag.repository import ConversationRepository

from .conftest import migrate


def event(content_hash: str, role: str = "user", **extra: object) -> ConversationEventInput:
    return ConversationEventInput(
        event_type="message",
        role=role,
        payload={"content": content_hash, **extra},
        content_hash=content_hash,
        request_id="request-one",
    )


async def test_scope_upsert_uses_external_project_and_project_local_session(database: Database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        first = await repository.ensure_scope(RequestIdentity("session", "project", "/first"))
        retry = await repository.ensure_scope(RequestIdentity("session", "project", "/moved"))
        other = await repository.ensure_scope(RequestIdentity("session", "other-project", "/moved"))
        assert first == retry
        assert first.project_id != other.project_id
        assert first.session_id != other.session_id
        root = await session.scalar(
            text("SELECT root_label FROM projects WHERE id=:id"), {"id": first.project_id}
        )
        assert root == "/moved"
        assert await session.scalar(text("SELECT count(*) FROM projects")) == 2
        assert await session.scalar(text("SELECT count(*) FROM sessions")) == 2


async def test_jsonb_payload_preserves_nested_structure_and_event_order(database: Database):
    payload = {
        "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "local"}],
        "tool_calls": [{"id": "call-a", "function": {"arguments": '{"x":1}'}}],
    }
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("session", "project"))
        records = await repository.append_events(scope, [event("a", **payload), event("b")])
        assert [record.sequence for record in records] == [1, 2]
        rows = (
            (
                await session.execute(
                    text("SELECT payload FROM conversation_events ORDER BY sequence")
                )
            )
            .scalars()
            .all()
        )
        assert rows[0] == payload
        assert rows[1] == {"content": "b"}
        with pytest.raises(FrozenInstanceError):
            records[0].sequence = 100
        with pytest.raises(TypeError):
            records[0].payload["content"] = "changed"


async def test_duplicate_hash_returns_original_row_without_consuming_order(database: Database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("session", "project"))
        first = await repository.append_events(scope, [event("a"), event("b")])
        repeated = await repository.append_events(scope, [event("a"), event("b"), event("c")])
        assert repeated[:2] == first
        assert repeated[2].sequence == 3
        assert await session.scalar(text("SELECT count(*) FROM conversation_events")) == 3


async def test_same_content_can_belong_to_independent_sessions(database: Database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        a = await repository.ensure_scope(RequestIdentity("a", "project"))
        b = await repository.ensure_scope(RequestIdentity("b", "project"))
        stored_a = await repository.append_events(a, [event("same")])
        stored_b = await repository.append_events(b, [event("same")])
        assert stored_a[0].id != stored_b[0].id
        assert stored_a[0].sequence == stored_b[0].sequence == 1


async def test_completed_assistant_finalization_enqueues_one_job_atomically(database: Database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("session", "project"))
        completion = AssistantCompletion(
            payload={"content": "done"},
            content_hash="completed",
            request_id="request-one",
            model="qwen3-coder:30b",
        )
        original = await repository.finalize_assistant(scope, completion)
        retry = await repository.finalize_assistant(scope, completion)
        job = await repository.enqueue_memory_job(original.id)
        again = await repository.enqueue_memory_job(original.id)
        assert original == retry
        assert job == again
        assert job.source_event_id == original.id
        assert job.project_id == scope.project_id
        assert job.status == "pending"
        assert job.attempt_count == 0
        assert await session.scalar(text("SELECT count(*) FROM memory_jobs")) == 1
    async with database.session() as session:
        assert await session.scalar(text("SELECT count(*) FROM conversation_events")) == 1
        assert await session.scalar(text("SELECT count(*) FROM memory_jobs")) == 1


async def test_incomplete_or_non_assistant_event_cannot_be_curated(database: Database):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("session", "project"))
        incomplete = await repository.finalize_assistant(
            scope,
            AssistantCompletion(
                payload={"content": "partial"},
                content_hash="partial",
                request_id="request-one",
                completed=False,
            ),
        )
        user = (await repository.append_events(scope, [event("user")]))[0]
        for source in (incomplete, user):
            with pytest.raises(ValueError, match="completed assistant"):
                await repository.enqueue_memory_job(source.id)
        assert await session.scalar(text("SELECT count(*) FROM memory_jobs")) == 0


async def test_concurrent_duplicate_insert_returns_one_logical_event(database: Database):
    async with database.session() as session:
        scope = await ConversationRepository(session).ensure_scope(RequestIdentity("s", "p"))
    gate = asyncio.Event()

    async def insert_duplicate():
        async with database.session() as session:
            await gate.wait()
            return (await ConversationRepository(session).append_events(scope, [event("same")]))[0]

    tasks = [asyncio.create_task(insert_duplicate()) for _ in range(8)]
    gate.set()
    rows = await asyncio.wait_for(asyncio.gather(*tasks), timeout=15)
    assert len({row.id for row in rows}) == 1
    assert [row.sequence for row in rows] == [1] * 8
    async with database.session() as session:
        assert await session.scalar(text("SELECT count(*) FROM conversation_events")) == 1


async def test_concurrent_new_events_get_distinct_contiguous_order(database: Database):
    async with database.session() as session:
        scope = await ConversationRepository(session).ensure_scope(RequestIdentity("s", "p"))

    async def insert_unique(index: int):
        async with database.session() as session:
            return (
                await ConversationRepository(session).append_events(scope, [event(str(index))])
            )[0]

    rows = await asyncio.wait_for(asyncio.gather(*(insert_unique(i) for i in range(8))), timeout=15)
    assert sorted(row.sequence for row in rows) == list(range(1, 9))


async def test_concurrent_finalization_has_one_event_and_one_job(database: Database):
    async with database.session() as session:
        scope = await ConversationRepository(session).ensure_scope(RequestIdentity("s", "p"))
    completion = AssistantCompletion(
        payload={"content": "done"}, content_hash="done", request_id="request-one"
    )

    async def finalize():
        async with database.session() as session:
            return await ConversationRepository(session).finalize_assistant(scope, completion)

    rows = await asyncio.wait_for(asyncio.gather(*(finalize() for _ in range(8))), timeout=15)
    assert len({row.id for row in rows}) == 1
    async with database.session() as session:
        assert await session.scalar(text("SELECT count(*) FROM conversation_events")) == 1
        assert await session.scalar(text("SELECT count(*) FROM memory_jobs")) == 1


async def test_session_context_rolls_back_failed_operation(database: Database):
    with pytest.raises(RuntimeError):
        async with database.session() as session:
            await ConversationRepository(session).ensure_scope(RequestIdentity("s", "p"))
            raise RuntimeError("abort transaction")
    async with database.session() as session:
        assert await session.scalar(text("SELECT count(*) FROM projects")) == 0


async def test_cross_project_scope_cannot_insert_event(database: Database):
    from local_dev_rag.domain import Scope

    async with database.session() as session:
        repository = ConversationRepository(session)
        a = await repository.ensure_scope(RequestIdentity("s", "a"))
        b = await repository.ensure_scope(RequestIdentity("s", "b"))
    with pytest.raises(ValueError, match="scope"):
        async with database.session() as session:
            await ConversationRepository(session).append_events(
                Scope(a.project_id, b.session_id), [event("bad")]
            )


@pytest.mark.parametrize(
    "invalid",
    [
        {"kind": "unknown"},
        {"state": "unknown"},
        {"confidence": 1.01},
        {"importance": -0.1},
        {"state": "superseded", "superseded_by_id": None},
    ],
)
async def test_memory_lifecycle_and_scores_are_database_constraints(database: Database, invalid):
    async with database.session() as session:
        repository = ConversationRepository(session)
        scope = await repository.ensure_scope(RequestIdentity("s", "p"))
        source = (await repository.append_events(scope, [event("source")]))[0]
    values = dict(
        id=uuid4(),
        project_id=scope.project_id,
        source_session_id=scope.session_id,
        source_event_id=source.id,
        kind="decision",
        text="retain this",
        confidence=0.8,
        importance=0.5,
        state="active",
        superseded_by_id=None,
    )
    values.update(invalid)
    with pytest.raises(IntegrityError):
        async with database.session() as session:
            await session.execute(
                text("""
                INSERT INTO memory_items
                  (id, project_id, source_session_id, source_event_id, kind, text,
                   confidence, importance, state, superseded_by_id, curator_model)
                VALUES (:id, :project_id, :source_session_id, :source_event_id, :kind, :text,
                        :confidence, :importance, :state, :superseded_by_id, 'curator')
            """),
                values,
            )


def test_alembic_round_trip_uses_the_test_service(postgres_url: str):
    upgrade = migrate(postgres_url, "upgrade", "head")
    current = migrate(postgres_url, "current")
    assert "0001" in current.stdout
    downgrade = migrate(postgres_url, "downgrade", "base")
    assert "0001" in downgrade.stderr
    upgrade_again = migrate(postgres_url, "upgrade", "head")
    assert "0001" in upgrade_again.stderr
    assert upgrade.returncode == 0
