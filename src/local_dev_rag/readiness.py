"""Bounded dependency checks; semantic memory degradation never gates chat."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal, Protocol

from local_dev_rag.config import Settings
from local_dev_rag.domain import DependencyStatus
from local_dev_rag.logging import error_category


class HealthProbe(Protocol):
    async def health(self) -> DependencyStatus: ...


class DatabaseProbe(Protocol):
    async def health(self, *, timeout_seconds: float) -> DependencyStatus: ...

    async def memory_health(self, *, timeout_seconds: float) -> DependencyStatus: ...


class RuntimeProbe(HealthProbe, Protocol):
    async def model_health(self, model: str, name: str) -> DependencyStatus: ...


@dataclass(frozen=True)
class ReadinessReport:
    status: Literal["ready", "degraded", "not_ready"]
    dependencies: Mapping[str, DependencyStatus]


class ReadinessService:
    def __init__(
        self,
        settings: Settings,
        *,
        database: DatabaseProbe,
        vector_store: HealthProbe,
        ollama: RuntimeProbe,
        timeout_seconds: float = 2,
    ):
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Invalid readiness timeout")
        self.settings, self.database, self.vector_store, self.ollama = (
            settings,
            database,
            vector_store,
            ollama,
        )
        self.timeout_seconds = timeout_seconds

    async def _check(self, name: str, probe: Callable[[], Awaitable[object]]) -> DependencyStatus:
        try:
            # Database probes own distinct operation/cleanup budgets themselves;
            # cancelling around their cleanup recreates the session-exit failure.
            async with asyncio.timeout(
                None if name in {"postgres", "memory_jobs"} else self.timeout_seconds
            ):
                result = await probe()
            if (
                not isinstance(result, DependencyStatus)
                or result.name != name
                or result.state
                not in {
                    "healthy",
                    "degraded",
                    "unavailable",
                }
            ):
                return DependencyStatus(name, "unavailable", "invalid_health_response")
            # Adapter detail is not an API/logging payload; expose only controlled categories.
            return DependencyStatus(name, result.state)
        except Exception as error:
            return DependencyStatus(name, "unavailable", error_category(error))

    async def check(self) -> ReadinessReport:
        probes: dict[str, Callable[[], Awaitable[object]]] = {
            "postgres": lambda: self.database.health(timeout_seconds=self.timeout_seconds),
            "chromadb": lambda: self.vector_store.health(),
            "ollama": lambda: self.ollama.health(),
            "curator": lambda: self.ollama.model_health(self.settings.curator_model, "curator"),
            "embedder": lambda: self.ollama.model_health(self.settings.embedding_model, "embedder"),
            "memory_jobs": lambda: self.database.memory_health(
                timeout_seconds=self.timeout_seconds
            ),
        }
        results = await asyncio.gather(
            *(self._check(name, probe) for name, probe in probes.items())
        )
        dependencies = {result.name: result for result in results}
        status = (
            "not_ready"
            if dependencies["ollama"].state != "healthy"
            else "degraded"
            if any(result.state != "healthy" for result in results)
            else "ready"
        )
        return ReadinessReport(status, MappingProxyType(dependencies))
