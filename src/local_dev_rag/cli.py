"""Explicitly scoped maintenance commands; output contains counts, never memory bodies."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.repository import MemoryRepository
from local_dev_rag.vector_store import VectorStore
from local_dev_rag.worker import index_memories


async def reindex(
    database: Database,
    settings: Settings,
    project: str,
    *,
    embedder: OllamaClient | None = None,
    vector_store: VectorStore | None = None,
    batch_size: int = 16,
) -> int:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Invalid embedding batch size")
    async with database.session() as session:
        repository = MemoryRepository(session)
        project_id = await repository.project_id(project)
    store = vector_store if vector_store is not None else VectorStore(settings)
    client = embedder if embedder is not None else OllamaClient(settings)
    await store.delete_project(project_id)
    # Accepted rows commit before workers write vectors. Enumerating after deletion
    # includes every completed write that deletion could have removed; later writes
    # stay indexed independently of this rebuild's snapshot.
    async with database.session() as session:
        items = await MemoryRepository(session).active(project_id)
    return await index_memories(database, settings, client, store, items, batch_size=batch_size)


async def _command(project: str) -> int:
    settings = Settings()
    database = Database.create(settings)
    try:
        return await reindex(database, settings, project)
    finally:
        await database.engine.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="local-dev-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser("reindex")
    rebuild.add_argument("--project", required=True, help="Exact external project ID")
    args = parser.parse_args(argv)
    try:
        count = asyncio.run(_command(args.project))
    except Exception:
        print("reindex failed", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"reindexed={count}")


if __name__ == "__main__":
    main()
