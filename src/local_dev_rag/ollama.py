"""Ollama transport adapter; chat uses its OpenAI compatibility endpoint."""

from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from math import isfinite
from typing import cast

import httpx

from local_dev_rag.config import Settings
from local_dev_rag.domain import DependencyStatus, UpstreamResponse


class OllamaClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self._settings = settings
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._settings.ollama_url.rstrip("/"),
            timeout=self._settings.upstream_timeout_seconds,
            transport=self._transport,
            headers={"accept-encoding": "identity"},
        )

    @asynccontextmanager
    async def chat(self, payload: Mapping[str, object]) -> AsyncGenerator[UpstreamResponse]:
        async with self._client() as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json=dict(payload)
            ) as response:

                async def body() -> AsyncIterator[bytes]:
                    # Mock transports can return already-consumed response content.
                    if response.is_stream_consumed:
                        yield response.content
                    else:
                        async for chunk in response.aiter_raw():
                            yield chunk

                yield UpstreamResponse(response.status_code, dict(response.headers), body())

    async def embed(self, model: str, inputs: Sequence[str]) -> list[list[float]]:
        async with self._client() as client:
            response = await client.post("/api/embed", json={"model": model, "input": list(inputs)})
            response.raise_for_status()
            payload = cast(dict[str, object], response.json())
            vectors = payload.get("embeddings")
            if not isinstance(vectors, list) or len(cast(list[object], vectors)) != len(inputs):
                raise ValueError("Ollama returned an invalid embedding batch")
            result: list[list[float]] = []
            for vector in cast(list[object], vectors):
                if not isinstance(vector, list) or not vector:
                    raise ValueError("Ollama returned an invalid embedding vector")
                values = cast(list[object], vector)
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    for value in values
                ):
                    raise ValueError("Ollama returned an invalid embedding value")
                result.append([float(cast(float, value)) for value in values])
            return result

    async def health(self) -> DependencyStatus:
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
            return DependencyStatus("ollama", "healthy")
        except httpx.HTTPError as error:
            return DependencyStatus("ollama", "unavailable", str(error))
