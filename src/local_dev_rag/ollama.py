"""Ollama transport adapter; chat uses its OpenAI compatibility endpoint."""

from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from math import isfinite
from typing import cast

import httpx
from anyio import CancelScope

from local_dev_rag.config import Settings
from local_dev_rag.domain import DependencyStatus, UpstreamResponse
from local_dev_rag.logging import error_category


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
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(self._client())
            response = await stack.enter_async_context(
                client.stream(
                    "POST",
                    "/api/embed",
                    json={"model": model, "input": list(inputs)},
                )
            )
            await response.aread()
            response.raise_for_status()
            payload: object = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Ollama returned an invalid embedding response")
            payload = cast(dict[str, object], payload)
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
        finally:
            with CancelScope(shield=True):
                await stack.aclose()

    async def _available_models(self) -> set[str]:
        async with self._client() as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid health response")
        models = cast(dict[str, object], payload).get("models")
        if not isinstance(models, list):
            raise ValueError("Invalid health response")
        names: set[str] = set()
        for model in cast(list[object], models):
            if not isinstance(model, dict):
                raise ValueError("Invalid health response")
            name = cast(dict[str, object], model).get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid health response")
            names.add(name)
        return names

    async def health(self) -> DependencyStatus:
        try:
            await self._available_models()
            return DependencyStatus("ollama", "healthy")
        except httpx.HTTPError as error:
            return DependencyStatus("ollama", "unavailable", error_category(error))
        except (ValueError, UnicodeError):
            return DependencyStatus("ollama", "unavailable", "invalid_health_response")

    async def model_health(self, model: str, name: str) -> DependencyStatus:
        try:
            models = await self._available_models()
            return DependencyStatus(name, "healthy" if model in models else "degraded")
        except httpx.HTTPError as error:
            return DependencyStatus(name, "unavailable", error_category(error))
        except (ValueError, UnicodeError):
            return DependencyStatus(name, "unavailable", "invalid_health_response")
