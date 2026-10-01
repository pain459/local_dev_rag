"""Dependency failures may degrade memory; only Ollama gates foreground readiness."""

import asyncio
import importlib

import httpx
import pytest

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.domain import DependencyStatus
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.vector_store import VectorStore

NAMES = ("postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs")


class Probe:
    def __init__(self, name, failure=None):
        self.name, self.failure = name, failure

    async def health(self):
        if self.failure == "timeout":
            await asyncio.sleep(30)
        if self.failure == "exception":
            raise RuntimeError("private prompt /Users/secret")
        if self.failure == "malformed":
            return {"state": "healthy"}
        return DependencyStatus(self.name, "unavailable" if self.failure else "healthy")

    async def memory_health(self):
        return DependencyStatus("memory_jobs", "healthy")


class RuntimeProbe(Probe):
    def __init__(self, failure_name=None, failure=None):
        super().__init__("ollama", failure if failure_name == "ollama" else None)
        self.failure_name, self.model_failure = failure_name, failure

    async def model_health(self, model, name):
        return await Probe(name, self.model_failure if name == self.failure_name else None).health()


def service(failure_name=None, failure=None):
    assert importlib.util.find_spec("local_dev_rag.readiness"), "ReadinessService is missing"
    module = importlib.import_module("local_dev_rag.readiness")
    return module.ReadinessService(
        Settings(_env_file=None),
        database=Probe("postgres", failure if failure_name == "postgres" else None),
        vector_store=Probe("chromadb", failure if failure_name == "chromadb" else None),
        ollama=RuntimeProbe(failure_name, failure),
        timeout_seconds=0.02,
    )


@pytest.mark.parametrize("name", [None, "postgres", "chromadb", "ollama", "curator", "embedder"])
@pytest.mark.parametrize("failure", ["unavailable", "exception", "timeout", "malformed"])
async def test_readiness_status_and_http_code_preserve_foreground_availability(name, failure):
    checker = service(name, failure)
    app = create_app(Settings(_env_file=None), readiness_service=checker)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            response = await client.get("/readyz")
            health = await client.get("/healthz")
        assert health.status_code == 200 and health.json() == {"status": "ok"}
        assert response.status_code == (503 if name == "ollama" else 200)
        body = response.json()
        assert body["status"] == (
            "ready" if name is None else "not_ready" if name == "ollama" else "degraded"
        )
        assert set(body["dependencies"]) == set(NAMES)
        assert all(state == "healthy" for key, state in body["dependencies"].items() if key != name)
        if name:
            assert body["dependencies"][name] == "unavailable"
        assert "private" not in response.text
    finally:
        await app.state.database.engine.dispose()


@pytest.mark.parametrize(
    "payload", [[], {}, {"models": None}, {"models": [{}]}, {"models": [{"name": 3}]}]
)
async def test_malformed_ollama_health_is_unavailable_without_raising(payload):
    client = OllamaClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    status = await client.health()
    assert status.state == "unavailable"
    assert status.detail == "invalid_health_response"


async def test_model_presence_is_checked_without_running_inference():
    settings = Settings(_env_file=None)

    def handler(request):
        assert request.method == "GET" and request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": settings.default_model}, {"name": settings.embedding_model}]},
        )

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    assert (await client.health()).state == "healthy"
    assert (await client.model_health(settings.curator_model, "curator")).state == "degraded"
    assert (await client.model_health(settings.embedding_model, "embedder")).state == "healthy"


@pytest.mark.parametrize(
    "payload", [None, [], {}, {"nanosecond heartbeat": "private"}, {"nanosecond heartbeat": True}]
)
async def test_chroma_health_rejects_malformed_heartbeat(payload):
    store = VectorStore(
        Settings(_env_file=None),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    assert callable(getattr(store, "health", None)), "Chroma health is missing"
    status = await store.health()
    assert status.state == "unavailable"
    assert status.detail == "invalid_health_response"


async def test_http_health_failures_never_return_exception_urls_or_messages():
    def handler(request):
        raise httpx.ReadTimeout("Authorization: secret /Users/private/root", request=request)

    client = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    status = await client.health()
    assert status.state == "unavailable"
    assert status.detail == "timeout"
