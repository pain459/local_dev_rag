"""Transaction-bound, retry-safe PostgreSQL conversation persistence."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import and_, func, literal_column, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from local_dev_rag.db import Database
from local_dev_rag.domain import (
    AssistantCompletion,
    ConversationEventInput,
    CuratorSource,
    MemoryDraft,
    MemoryItem,
    MemoryJob,
    RequestIdentity,
    Scope,
    StoredEvent,
    VectorHit,
)
from local_dev_rag.events import content_hash
from local_dev_rag.schema import (
    conversation_events,
    memory_items,
    memory_jobs,
    memory_sources,
    normalized_memory_text,
    projects,
    sessions,
)

logger = logging.getLogger(__name__)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in cast(Mapping[str, object], value).items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in cast(tuple[object, ...], value)]
    return value


def _stored_event(row: RowMapping) -> StoredEvent:
    return StoredEvent(**dict(row))


def _memory_job(row: RowMapping) -> MemoryJob:
    return MemoryJob(**dict(row))


class MemoryRepository:
    """Accepted extraction snapshots; source-event locking serializes competing retries."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def for_source(self, source: CuratorSource) -> list[MemoryItem]:
        rows = (
            (
                await self.session.execute(
                    select(memory_items)
                    .where(
                        memory_items.c.project_id == source.project_id,
                        or_(
                            and_(
                                memory_items.c.source_session_id == source.session_id,
                                memory_items.c.source_event_id == source.source_event_id,
                            ),
                            select(memory_sources.c.memory_id)
                            .where(
                                memory_sources.c.memory_id == memory_items.c.id,
                                memory_sources.c.project_id == source.project_id,
                                memory_sources.c.source_session_id == source.session_id,
                                memory_sources.c.source_event_id == source.source_event_id,
                            )
                            .exists(),
                        ),
                    )
                    .order_by(memory_items.c.created_at, memory_items.c.id)
                )
            )
            .mappings()
            .all()
        )
        return [MemoryItem(**dict(row)) for row in rows]

    async def persist(
        self, source: CuratorSource, drafts: Sequence[MemoryDraft], *, curator_model: str
    ) -> list[MemoryItem]:
        event = await self.session.scalar(
            select(conversation_events.c.id)
            .where(
                conversation_events.c.id == source.source_event_id,
                conversation_events.c.project_id == source.project_id,
                conversation_events.c.session_id == source.session_id,
                conversation_events.c.role == "assistant",
                conversation_events.c.completed.is_(True),
            )
            .with_for_update()
        )
        if event is None:
            raise ValueError("Memory source provenance mismatch")
        existing = await self.for_source(source)
        if existing:
            return existing
        # A consistent order avoids cross-job deadlocks on multiple canonical rows.
        for draft in sorted(drafts, key=lambda item: " ".join(item.text.lower().split())):
            memory_id = await self.session.scalar(
                insert(memory_items)
                .values(
                    id=uuid4(),
                    project_id=source.project_id,
                    source_session_id=source.session_id,
                    source_event_id=source.source_event_id,
                    kind=draft.kind,
                    text=draft.text,
                    confidence=draft.confidence,
                    importance=draft.importance,
                    curator_model=curator_model,
                )
                .on_conflict_do_update(
                    index_elements=[
                        memory_items.c.project_id,
                        normalized_memory_text(memory_items.c.text),
                    ],
                    index_where=memory_items.c.state == literal_column("'active'"),
                    # Lock/return the canonical row without changing its provenance,
                    # content, lifecycle, scores, or timestamps.
                    set_={"id": memory_items.c.id},
                )
                .returning(memory_items.c.id)
            )
            await self.session.execute(
                insert(memory_sources)
                .values(
                    project_id=source.project_id,
                    source_session_id=source.session_id,
                    source_event_id=source.source_event_id,
                    memory_id=memory_id,
                )
                .on_conflict_do_nothing()
            )
        return await self.for_source(source)

    async def project_id(self, external_id: str) -> UUID:
        if not external_id.strip():
            raise ValueError("An explicit project external ID is required")
        project_id = await self.session.scalar(
            select(projects.c.id).where(projects.c.external_project_id == external_id)
        )
        if project_id is None:
            raise ValueError("Unknown project external ID")
        return cast(UUID, project_id)

    async def active(self, project_id: UUID) -> list[MemoryItem]:
        rows = (
            (
                await self.session.execute(
                    select(memory_items)
                    .where(
                        memory_items.c.project_id == project_id,
                        memory_items.c.state == "active",
                    )
                    .order_by(memory_items.c.created_at, memory_items.c.id)
                )
            )
            .mappings()
            .all()
        )
        return [MemoryItem(**dict(row)) for row in rows]

    async def resolve_hits(self, project_id: UUID, hits: Sequence[VectorHit]) -> list[VectorHit]:
        rows = (
            (
                await self.session.execute(
                    select(memory_items).where(
                        memory_items.c.project_id == project_id,
                        memory_items.c.id.in_([hit.memory.id for hit in hits]),
                        memory_items.c.state == "active",
                    )
                )
            )
            .mappings()
            .all()
        )
        current = {row["id"]: MemoryItem(**dict(row)) for row in rows}
        return [
            VectorHit(current[hit.memory.id], hit.distance)
            for hit in hits
            if hit.memory.id in current
        ]

    async def record_embedding(
        self, project_id: UUID, ids: Sequence[UUID], model: str, version: int
    ) -> list[MemoryItem]:
        # UPDATE locks only still-active records until the vector upsert finishes.
        rows = (
            (
                await self.session.execute(
                    update(memory_items)
                    .where(
                        memory_items.c.project_id == project_id,
                        memory_items.c.id.in_(ids),
                        memory_items.c.state == "active",
                    )
                    .values(embedding_model=model, embedding_version=version, updated_at=func.now())
                    .returning(memory_items)
                )
            )
            .mappings()
            .all()
        )
        return [MemoryItem(**dict(row)) for row in rows]


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def ensure_scope(self, identity: RequestIdentity) -> Scope:
        project = insert(projects).values(
            id=uuid4(),
            external_project_id=identity.project_id,
            root_label=identity.project_root,
        )
        project_id = (
            await self.session.execute(
                project.on_conflict_do_update(
                    index_elements=[projects.c.external_project_id],
                    set_={
                        "root_label": func.coalesce(
                            project.excluded.root_label, projects.c.root_label
                        ),
                        "updated_at": func.now(),
                    },
                ).returning(projects.c.id)
            )
        ).scalar_one()
        session = insert(sessions).values(
            id=uuid4(),
            project_id=project_id,
            opencode_session_id=identity.session_id,
        )
        session_id = (
            await self.session.execute(
                session.on_conflict_do_update(
                    constraint="uq_session_external",
                    set_={"last_seen_at": func.now()},
                ).returning(sessions.c.id)
            )
        ).scalar_one()
        return Scope(project_id=project_id, session_id=session_id)

    async def append_events(
        self,
        scope: Scope,
        events: Sequence[ConversationEventInput],
    ) -> list[StoredEvent]:
        # Lock this session to order concurrent batches; other sessions stay independent.
        session_id = await self.session.scalar(
            select(sessions.c.id)
            .where(
                sessions.c.id == scope.session_id,
                sessions.c.project_id == scope.project_id,
            )
            .with_for_update()
        )
        if session_id is None:
            raise ValueError("Unknown or mismatched scope")
        sequence = (
            cast(
                int,
                (
                    await self.session.scalar(
                        select(
                            func.coalesce(
                                func.max(conversation_events.c.sequence),
                                0,
                            )
                        ).where(conversation_events.c.session_id == scope.session_id)
                    )
                ),
            )
            + 1
        )
        stored: list[StoredEvent] = []
        for event in events:
            statement = (
                insert(conversation_events)
                .values(
                    id=uuid4(),
                    project_id=scope.project_id,
                    session_id=scope.session_id,
                    sequence=sequence,
                    event_type=event.event_type,
                    role=event.role,
                    payload=_json_value(event.payload),
                    content_hash=event.content_hash,
                    request_id=event.request_id,
                    completion_request_id=event.completion_request_id,
                    source_message_id=event.source_message_id,
                    model=event.model,
                    completed=event.completed,
                )
                .on_conflict_do_nothing()
                .returning(conversation_events)
            )
            row = (await self.session.execute(statement)).mappings().one_or_none()
            if row is None:
                if event.completion_request_id is not None:
                    row = (
                        (
                            await self.session.execute(
                                select(conversation_events).where(
                                    conversation_events.c.session_id == scope.session_id,
                                    conversation_events.c.completion_request_id
                                    == event.completion_request_id,
                                )
                            )
                        )
                        .mappings()
                        .one_or_none()
                    )
                if row is None:
                    row = (
                        (
                            await self.session.execute(
                                select(conversation_events).where(
                                    conversation_events.c.session_id == scope.session_id,
                                    conversation_events.c.content_hash == event.content_hash,
                                )
                            )
                        )
                        .mappings()
                        .one()
                    )
                # An echo captured during recovery may precede its generation finalization.
                # Claim its request identity under the same session lock, retaining its content.
                if event.completion_request_id is not None and row["completion_request_id"] is None:
                    row = (
                        (
                            await self.session.execute(
                                update(conversation_events)
                                .where(conversation_events.c.id == row["id"])
                                .values(completion_request_id=event.completion_request_id)
                                .returning(conversation_events)
                            )
                        )
                        .mappings()
                        .one()
                    )
            else:
                sequence += 1
            stored.append(_stored_event(row))
        return stored

    async def finalize_assistant(
        self, scope: Scope, completion: AssistantCompletion
    ) -> StoredEvent:
        event = (
            await self.append_events(
                scope,
                [
                    ConversationEventInput(
                        event_type=(
                            "tool_call"
                            if completion.payload.get("tool_calls")
                            or completion.payload.get("function_call")
                            else "message"
                        ),
                        role="assistant",
                        payload=completion.payload,
                        content_hash=completion.content_hash,
                        request_id=completion.request_id,
                        source_message_id=completion.source_message_id,
                        model=completion.model,
                        completed=completion.completed,
                        completion_request_id=completion.request_id
                        if completion.completed
                        else None,
                    )
                ],
            )
        )[0]
        if completion.completed:
            await self.enqueue_memory_job(event.id)
        return event

    async def enqueue_memory_job(self, source_event_id: UUID) -> MemoryJob:
        source = (
            (
                await self.session.execute(
                    select(conversation_events).where(
                        conversation_events.c.id == source_event_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if source is None or source["role"] != "assistant" or not source["completed"]:
            raise ValueError("Memory jobs require a completed assistant source event")
        statement = (
            insert(memory_jobs)
            .values(
                id=uuid4(),
                project_id=source["project_id"],
                session_id=source["session_id"],
                source_event_id=source_event_id,
                job_kind="curate",
                deduplication_key=f"curate:{source_event_id}",
            )
            .on_conflict_do_nothing(constraint="uq_job_source_kind")
            .returning(memory_jobs)
        )
        row = (await self.session.execute(statement)).mappings().one_or_none()
        if row is None:
            row = (
                (
                    await self.session.execute(
                        select(memory_jobs).where(
                            memory_jobs.c.source_event_id == source_event_id,
                            memory_jobs.c.job_kind == "curate",
                        )
                    )
                )
                .mappings()
                .one()
            )
        return _memory_job(row)

    async def curator_source(self, source_event_id: UUID, *, max_events: int = 32) -> CuratorSource:
        """Read bounded same-session history ending at the completed source event."""
        if isinstance(max_events, bool) or not 1 <= max_events <= 256:
            raise ValueError("Invalid curator source event limit")
        source = (
            (
                await self.session.execute(
                    select(conversation_events).where(
                        conversation_events.c.id == source_event_id,
                        conversation_events.c.role == "assistant",
                        conversation_events.c.completed.is_(True),
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if source is None:
            raise ValueError("Curator source requires a completed assistant event")
        rows = (
            (
                await self.session.execute(
                    select(conversation_events)
                    .where(
                        conversation_events.c.project_id == source["project_id"],
                        conversation_events.c.session_id == source["session_id"],
                        conversation_events.c.sequence <= source["sequence"],
                        conversation_events.c.completed.is_(True),
                        conversation_events.c.role.in_(["user", "assistant", "tool", "function"]),
                    )
                    .order_by(conversation_events.c.sequence.desc())
                    .limit(max_events)
                )
            )
            .mappings()
            .all()
        )
        return CuratorSource(
            source["project_id"],
            source["session_id"],
            source_event_id,
            tuple(_stored_event(row) for row in reversed(rows)),
        )


@dataclass
class CaptureAttempt:
    database: Database
    scope: Scope
    request_id: str
    parent_hash: str
    model: str
    status: str = "inbound_persisted"
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
            self.status = "completed" if completion else "incomplete"
        except (SQLAlchemyError, OSError):
            self.status = "unavailable"
            logger.warning("Conversation capture unavailable; completion was not persisted")
        self.finished = True


class CaptureUnavailable(RuntimeError):
    """The inbound history could not be made durable."""


class VectorQuery(Protocol):
    async def query(
        self, project_id: UUID, vector: Sequence[float], limit: int
    ) -> list[VectorHit]: ...


class PostgresMemorySearch:
    """Chroma supplies distances; PostgreSQL supplies eligible memory records."""

    def __init__(self, database: Database, search: VectorQuery):
        self.database, self.search = database, search

    async def query(self, project_id: UUID, vector: Sequence[float], limit: int) -> list[VectorHit]:
        hits = await self.search.query(project_id, vector, limit)
        if not hits:
            return []
        try:
            async with self.database.session() as session:
                return await MemoryRepository(session).resolve_hits(project_id, hits)
        except (SQLAlchemyError, OSError) as error:
            raise CaptureUnavailable("Authoritative memory unavailable") from error


class PostgresCaptureStore:
    def __init__(self, database: Database):
        self.database = database

    async def begin(
        self,
        identity: RequestIdentity,
        events: Sequence[ConversationEventInput],
        model: str,
        request_id: str,
        parent_hash: str,
    ) -> CaptureAttempt:
        try:
            async with self.database.session() as session:
                repository = ConversationRepository(session)
                scope = await repository.ensure_scope(identity)
                await repository.append_events(
                    scope,
                    [replace(event, request_id=request_id, model=model) for event in events],
                )
            return CaptureAttempt(self.database, scope, request_id, parent_hash, model)
        except (SQLAlchemyError, OSError) as error:
            raise CaptureUnavailable("PostgreSQL capture unavailable") from error
