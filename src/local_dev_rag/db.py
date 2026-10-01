"""Async database lifetime and transaction boundaries."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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

    async def health(self) -> DependencyStatus:
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return DependencyStatus("postgres", "healthy")
        except Exception as error:
            return DependencyStatus("postgres", "unavailable", error_category(error))

    async def memory_health(self) -> DependencyStatus:
        try:
            async with self.session() as session:
                failures = await session.scalar(
                    select(func.count())
                    .select_from(memory_jobs)
                    .where(
                        memory_jobs.c.status.in_(["failed", "retry"]),
                    )
                )
            return DependencyStatus("memory_jobs", "degraded" if failures else "healthy")
        except Exception as error:
            return DependencyStatus("memory_jobs", "unavailable", error_category(error))

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with factory.begin() as session:
            yield session
