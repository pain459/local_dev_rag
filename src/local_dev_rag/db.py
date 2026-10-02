"""Async database lifetime and transaction boundaries."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql.elements import TextClause

from local_dev_rag.config import Settings
from local_dev_rag.domain import DependencyStatus
from local_dev_rag.logging import error_category
from local_dev_rag.schema import memory_jobs


@dataclass(frozen=True)
class Database:
    engine: AsyncEngine

    @classmethod
    def create(cls, settings: Settings) -> "Database":
        return cls(create_async_engine(settings.database_url, pool_pre_ping=True))

    async def _probe(
        self, name: str, statement: Select[int] | TextClause, timeout_seconds: float
    ) -> DependencyStatus:
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Invalid database probe timeout")
        # Explicit ownership avoids SQLAlchemy context exits, which create shielded
        # child tasks. Read-only health transactions never commit. Both execution
        # and rollback/close are awaited here under separate deadlines, so no
        # cleanup is detached and the application pool is never disposed.
        session = async_sessionmaker(self.engine, expire_on_commit=False)()
        cleanup_seconds = min(timeout_seconds, 0.25)
        status = DependencyStatus(name, "unavailable")
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await session.execute(statement)
                failures = result.scalar() if name == "memory_jobs" else 0
                status = DependencyStatus(name, "degraded" if failures else "healthy")
        except Exception as error:
            status = DependencyStatus(name, "unavailable", error_category(error))
        finally:
            try:
                async with asyncio.timeout(cleanup_seconds):
                    await session.close()
            except (Exception, asyncio.CancelledError) as error:
                status = DependencyStatus(name, "unavailable", error_category(error))
                # Cancellation during driver rollback invalidates its connection.
                # Explicitly invalidate remaining session state as a final bounded
                # cleanup; this operates only on this probe's checked-out resources.
                try:
                    async with asyncio.timeout(cleanup_seconds):
                        await session.invalidate()
                except Exception as cleanup_error:
                    status = DependencyStatus(name, "unavailable", error_category(cleanup_error))
                if isinstance(error, asyncio.CancelledError):
                    raise
        return status

    async def health(self, *, timeout_seconds: float = 2) -> DependencyStatus:
        return await self._probe("postgres", text("SELECT 1"), timeout_seconds)

    async def memory_health(self, *, timeout_seconds: float = 2) -> DependencyStatus:
        return await self._probe(
            "memory_jobs",
            select(func.count())
            .select_from(memory_jobs)
            .where(
                memory_jobs.c.status.in_(["failed", "retry"]),
            ),
            timeout_seconds,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with factory.begin() as session:
            yield session
