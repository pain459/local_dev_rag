"""Async database lifetime and transaction boundaries."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from local_dev_rag.config import Settings


@dataclass(frozen=True)
class Database:
    engine: AsyncEngine

    @classmethod
    def create(cls, settings: Settings) -> "Database":
        return cls(create_async_engine(settings.database_url, pool_pre_ping=True))

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with factory.begin() as session:
            yield session
