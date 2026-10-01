"""Chroma HTTP adapter; all vector searches require an exact project scope."""

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from uuid import UUID

import httpx
from anyio import CancelScope
from pydantic import BaseModel, TypeAdapter, ValidationError

from local_dev_rag.config import Settings
from local_dev_rag.domain import EmbeddedMemory, MemoryItem, VectorHit


class VectorStoreUnavailable(RuntimeError):
    """Semantic recall could not be completed safely."""


class _Collection(BaseModel):
    id: UUID


class _QueryResult(BaseModel):
    ids: list[list[str]]
    metadatas: list[list[dict[str, object]]]
    documents: list[list[str]]
    distances: list[list[float]]


_memory_adapter = TypeAdapter(MemoryItem)


def _metadata(memory: MemoryItem) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in asdict(memory).items():
        if key == "text":
            continue
        if isinstance(value, UUID):
            value = str(value)
        elif isinstance(value, datetime):
            value = value.isoformat()
        elif value is None:
            # Chroma merges metadata on update; empty sentinels clear previous values.
            value = ""
        result["memory_id" if key == "id" else key] = value
    return result


class VectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        collection_name: str = "local_dev_rag_memory",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._url = settings.chromadb_url.rstrip("/")
        self._collection_name = collection_name
        self._transport = transport

    @asynccontextmanager
    async def _client(self) -> AsyncGenerator[httpx.AsyncClient]:
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(
                httpx.AsyncClient(base_url=self._url, timeout=5, transport=self._transport)
            )
            yield client
        finally:
            with CancelScope(shield=True):
                await stack.aclose()

    @staticmethod
    async def _post(
        client: httpx.AsyncClient,
        path: str,
        *,
        json: Mapping[str, object],
    ) -> httpx.Response:
        stack = AsyncExitStack()
        try:
            response = await stack.enter_async_context(client.stream("POST", path, json=json))
            await response.aread()
            return response
        finally:
            with CancelScope(shield=True):
                await stack.aclose()

    async def _collection(self, client: httpx.AsyncClient) -> str:
        response = await self._post(
            client,
            "/api/v1/collections",
            json={
                "name": self._collection_name,
                "get_or_create": True,
                "metadata": {"hnsw:space": "cosine"},
            },
        )
        response.raise_for_status()
        return str(_Collection.model_validate(response.json()).id)

    async def upsert(self, items: Sequence[EmbeddedMemory]) -> None:
        if not items:
            return
        try:
            async with self._client() as client:
                collection = await self._collection(client)
                response = await self._post(
                    client,
                    f"/api/v1/collections/{collection}/upsert",
                    json={
                        "ids": [str(item.memory.id) for item in items],
                        "embeddings": [list(item.vector) for item in items],
                        "documents": [item.memory.text for item in items],
                        "metadatas": [_metadata(item.memory) for item in items],
                    },
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as error:
            raise VectorStoreUnavailable("Chroma upsert unavailable") from error

    async def query(
        self,
        project_id: UUID,
        vector: Sequence[float],
        limit: int,
    ) -> list[VectorHit]:
        if limit <= 0:
            raise ValueError("Vector query limit must be positive")
        try:
            async with self._client() as client:
                collection = await self._collection(client)
                response = await self._post(
                    client,
                    f"/api/v1/collections/{collection}/query",
                    json={
                        "query_embeddings": [list(vector)],
                        "n_results": limit,
                        "where": {"project_id": {"$eq": str(project_id)}},
                        "include": ["metadatas", "documents", "distances"],
                    },
                )
                response.raise_for_status()
                result = _QueryResult.model_validate(response.json())
            hits: list[VectorHit] = []
            for id, metadata, text, distance in zip(
                result.ids[0],
                result.metadatas[0],
                result.documents[0],
                result.distances[0],
                strict=True,
            ):
                data = {**metadata, "id": metadata["memory_id"], "text": text}
                for field in ("superseded_by_id", "embedding_model"):
                    if data.get(field) == "":
                        data[field] = None
                memory = _memory_adapter.validate_python(data)
                if memory.project_id != project_id or str(memory.id) != id:
                    raise ValueError("Chroma returned mismatched memory provenance")
                hits.append(VectorHit(memory, distance))
            return hits
        except (httpx.HTTPError, ValidationError, ValueError, KeyError, IndexError) as error:
            raise VectorStoreUnavailable("Chroma query unavailable") from error

    async def delete_project(self, project_id: UUID) -> None:
        try:
            async with self._client() as client:
                collection = await self._collection(client)
                response = await self._post(
                    client,
                    f"/api/v1/collections/{collection}/delete",
                    json={
                        "where": {"project_id": {"$eq": str(project_id)}},
                    },
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as error:
            raise VectorStoreUnavailable("Chroma deletion unavailable") from error
