"""Transaction-bound, retry-safe PostgreSQL conversation persistence."""

from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from local_dev_rag.domain import (
    AssistantCompletion,
    ConversationEventInput,
    MemoryJob,
    RequestIdentity,
    Scope,
    StoredEvent,
)
from local_dev_rag.schema import conversation_events, memory_jobs, projects, sessions


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
