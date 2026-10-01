"""Asynchronous semantic-memory processing; PostgreSQL is the durable checkpoint."""

import asyncio
import logging
import signal
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal
from uuid import UUID, uuid4

from local_dev_rag.config import Settings
from local_dev_rag.curator import Curator
from local_dev_rag.db import Database
from local_dev_rag.domain import EmbeddedMemory, MemoryItem, MemoryJob
from local_dev_rag.jobs import JobRepository
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.repository import ConversationRepository, MemoryRepository
from local_dev_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkResult:
    state: Literal["idle", "completed", "retry", "failed"]
    job_id: UUID | None
    memory_count: int


async def index_memories(
    database: Database,
    settings: Settings,
    embedder: OllamaClient,
    vector_store: VectorStore,
    items: Sequence[MemoryItem],
    *,
    batch_size: int,
) -> int:
    """Embed bounded batches and attach the authoritative PostgreSQL IDs to vectors."""
    count = 0
    for offset in range(0, len(items), batch_size):
        batch = [
            replace(
                item,
                embedding_model=settings.embedding_model,
                embedding_version=settings.embedding_version,
            )
            for item in items[offset : offset + batch_size]
            if item.state == "active"
        ]
        if not batch:
            continue
        vectors = await embedder.embed(settings.embedding_model, [item.text for item in batch])
        by_id = {item.id: vector for item, vector in zip(batch, vectors, strict=True)}
        async with database.session() as session:
            current = await MemoryRepository(session).record_embedding(
                batch[0].project_id,
                [item.id for item in batch],
                settings.embedding_model,
                settings.embedding_version,
            )
            await vector_store.upsert(
                [EmbeddedMemory(item, tuple(by_id[item.id])) for item in current]
            )
        count += len(current)
    return count


class Worker:
    def __init__(
        self,
        database: Database,
        settings: Settings,
        *,
        curator: Curator | None = None,
        embedder: OllamaClient | None = None,
        vector_store: VectorStore | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 600,
        batch_size: int = 16,
        poll_seconds: float = 1,
    ):
        if (
            isinstance(batch_size, bool)
            or batch_size <= 0
            or isinstance(lease_seconds, bool)
            or lease_seconds < 2
            or not isfinite(poll_seconds)
            or poll_seconds <= 0
        ):
            raise ValueError("Invalid worker limits")
        self.database, self.settings = database, settings
        client = OllamaClient(settings)
        self.curator = curator if curator is not None else Curator(client, settings)
        self.embedder = embedder if embedder is not None else client
        self.vector_store = vector_store if vector_store is not None else VectorStore(settings)
        self.jobs = JobRepository(database, settings)
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self.lease_seconds, self.batch_size, self.poll_seconds = (
            lease_seconds,
            batch_size,
            poll_seconds,
        )

    async def _fail(self, job: MemoryJob, error: BaseException) -> None:
        try:
            await self.jobs.fail(job.id, error)
        except ValueError:
            # A replacement worker owns an expired lease; never transition its claim.
            logger.warning("memory_job_lease_lost", extra={"job_id": str(job.id)})

    async def run_once(self) -> WorkResult:
        job = await self.jobs.claim(self.worker_id, self.lease_seconds)
        if job is None:
            return WorkResult("idle", None, 0)
        count = 0
        try:
            # Leave time to transition failure before the durable lease expires.
            async with asyncio.timeout(self.lease_seconds - 1):
                async with self.database.session() as session:
                    source = await ConversationRepository(session).curator_source(
                        job.source_event_id
                    )
                    stored = await MemoryRepository(session).for_source(source)
                if not stored:
                    drafts = await self.curator.extract(source)
                    async with self.database.session() as session:
                        stored = await MemoryRepository(session).persist(
                            source,
                            drafts,
                            curator_model=self.settings.curator_model,
                        )
                active = [item for item in stored if item.state == "active"]
                count = len(active)
                await index_memories(
                    self.database,
                    self.settings,
                    self.embedder,
                    self.vector_store,
                    active,
                    batch_size=self.batch_size,
                )
                await self.jobs.complete(job.id, [item.id for item in active])
            return WorkResult("completed", job.id, count)
        except asyncio.CancelledError as error:
            cleanup = asyncio.create_task(self._fail(job, error))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            raise
        except Exception as error:
            await self._fail(job, error)
            state = "failed" if job.attempt_count >= self.settings.retry_max_attempts else "retry"
            return WorkResult(state, job.id, count)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                result = await self.run_once()
                if result.state != "idle":
                    continue
            except Exception:
                # Database recovery is independent of the foreground process.
                logger.warning("memory_worker_dependency_unavailable")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass


async def serve() -> None:
    settings = Settings()
    database = Database.create(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await Worker(database, settings).run_forever(stop)
    finally:
        await database.engine.dispose()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
