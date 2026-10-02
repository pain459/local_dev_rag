"""Short PostgreSQL queue transactions with database clocks and fenced leases."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from math import ceil, isfinite, log2
from typing import cast
from uuid import UUID

import httpx
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.domain import MemoryJob
from local_dev_rag.schema import memory_items, memory_jobs, memory_sources


def _compact_error(error: object) -> tuple[str, str]:
    # Arbitrary exception strings and even type names may contain conversation content.
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout", "Memory processing timed out"
    if isinstance(error, httpx.HTTPError):
        return "upstream_error", "Memory dependency request failed"
    if isinstance(error, ValueError):
        return "validation_error", "Memory validation failed"
    return "processing_error", "Memory processing failed"


class JobRepository:
    """One worker owns an instance across claim and transition calls.

    Claims commit before returning. In-memory owner/attempt tokens fence stale workers;
    only PostgreSQL holds durable job state. A process loss is recovered through expiry.
    """

    def __init__(self, database: Database, settings: Settings | None = None):
        self.database = database
        self.settings = settings if settings is not None else Settings()
        if not all(
            isfinite(value)
            for value in (self.settings.retry_initial_seconds, self.settings.retry_max_seconds)
        ):
            raise ValueError("Retry delays must be finite")
        self._claims: dict[UUID, MemoryJob] = {}

    async def claim(self, worker_id: str, lease_seconds: int) -> MemoryJob | None:
        if (
            not worker_id.strip()
            or len(worker_id) > 128
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise ValueError("Invalid worker or lease")
        async with self.database.session() as session:
            now = cast(datetime, await session.scalar(select(func.clock_timestamp())))
            eligible = or_(
                and_(
                    memory_jobs.c.status.in_(["pending", "retry"]),
                    memory_jobs.c.next_attempt_at <= now,
                ),
                and_(memory_jobs.c.status == "running", memory_jobs.c.lease_expires_at <= now),
            )
            exhausted = (
                select(memory_jobs.c.id)
                .where(eligible, memory_jobs.c.attempt_count >= self.settings.retry_max_attempts)
                .with_for_update(skip_locked=True)
            )
            await session.execute(
                update(memory_jobs)
                .where(memory_jobs.c.id.in_(exhausted))
                .values(
                    status="failed",
                    completed_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    error_category="lease_expired",
                    error_message="Memory attempt limit reached",
                )
            )
            candidate = await session.scalar(
                select(memory_jobs.c.id)
                .where(
                    eligible,
                    memory_jobs.c.attempt_count < self.settings.retry_max_attempts,
                )
                .order_by(memory_jobs.c.next_attempt_at, memory_jobs.c.created_at, memory_jobs.c.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if candidate is None:
                return None
            row = (
                (
                    await session.execute(
                        update(memory_jobs)
                        .where(memory_jobs.c.id == candidate)
                        .values(
                            status="running",
                            attempt_count=memory_jobs.c.attempt_count + 1,
                            lease_owner=worker_id,
                            lease_expires_at=now + timedelta(seconds=lease_seconds),
                            started_at=now,
                            error_category=None,
                            error_message=None,
                        )
                        .returning(memory_jobs)
                    )
                )
                .mappings()
                .one()
            )
            job = MemoryJob(**dict(row))
        self._claims[job.id] = job
        return job

    async def _lease(self, session: AsyncSession, job_id: UUID) -> tuple[RowMapping, datetime]:
        claimed = self._claims.get(job_id)
        if claimed is None:
            raise ValueError("No owned job lease")
        row = (
            (
                await session.execute(
                    select(memory_jobs)
                    .where(
                        memory_jobs.c.id == job_id,
                        memory_jobs.c.status == "running",
                        memory_jobs.c.lease_owner == claimed.lease_owner,
                        memory_jobs.c.attempt_count == claimed.attempt_count,
                    )
                    .with_for_update()
                )
            )
            .mappings()
            .one_or_none()
        )
        now = cast(datetime, await session.scalar(select(func.clock_timestamp())))
        if row is None or row["lease_expires_at"] <= now:
            raise ValueError("Job lease was lost or expired")
        return row, now

    async def complete(self, job_id: UUID, memory_ids: Sequence[UUID]) -> None:
        async with self.database.session() as session:
            row, now = await self._lease(session, job_id)
            unique_ids = set(memory_ids)
            if unique_ids:
                found = set(
                    (
                        await session.scalars(
                            select(memory_items.c.id).where(
                                memory_items.c.id.in_(unique_ids),
                                memory_items.c.project_id == row["project_id"],
                                or_(
                                    and_(
                                        memory_items.c.source_session_id == row["session_id"],
                                        memory_items.c.source_event_id == row["source_event_id"],
                                    ),
                                    select(memory_sources.c.memory_id)
                                    .where(
                                        memory_sources.c.memory_id == memory_items.c.id,
                                        memory_sources.c.project_id == row["project_id"],
                                        memory_sources.c.source_session_id == row["session_id"],
                                        memory_sources.c.source_event_id == row["source_event_id"],
                                    )
                                    .exists(),
                                ),
                                memory_items.c.state == "active",
                            )
                        )
                    ).all()
                )
                if found != unique_ids:
                    raise ValueError("Completion memory provenance mismatch")
            await session.execute(
                update(memory_jobs)
                .where(memory_jobs.c.id == job_id)
                .values(
                    status="completed",
                    completed_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    error_category=None,
                    error_message=None,
                )
            )
        self._claims.pop(job_id, None)

    async def fail(self, job_id: UUID, error: object, retry_at: datetime | None = None) -> None:
        if retry_at is not None and (retry_at.tzinfo is None or retry_at.utcoffset() is None):
            raise ValueError("Retry timestamp must include a timezone")
        category, message = _compact_error(error)
        async with self.database.session() as session:
            row, now = await self._lease(session, job_id)
            attempts = cast(int, row["attempt_count"])
            terminal = attempts >= self.settings.retry_max_attempts
            initial, maximum = (
                self.settings.retry_initial_seconds,
                self.settings.retry_max_seconds,
            )
            exponent = min(attempts - 1, max(0, ceil(log2(maximum / initial))))
            delay = min(maximum, initial * 2**exponent)
            earliest, latest = now + timedelta(seconds=delay), now + timedelta(seconds=maximum)
            next_attempt = earliest if retry_at is None else max(earliest, min(latest, retry_at))
            await session.execute(
                update(memory_jobs)
                .where(memory_jobs.c.id == job_id)
                .values(
                    status="failed" if terminal else "retry",
                    next_attempt_at=next_attempt,
                    completed_at=now if terminal else None,
                    lease_owner=None,
                    lease_expires_at=None,
                    error_category=category,
                    error_message=message,
                )
            )
        self._claims.pop(job_id, None)
