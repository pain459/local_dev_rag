from httpx import ASGITransport, AsyncClient

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings


async def test_health_is_exact_process_liveness_contract():
    transport = ASGITransport(app=create_app(Settings(_env_file=None)))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_exposes_initial_dependency_states():
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "starting",
        "dependencies": {"postgres": "unknown", "chromadb": "unknown", "ollama": "unknown"},
    }


def test_openapi_describes_typed_readiness_dependencies():
    schema = create_app(Settings(_env_file=None)).openapi()
    response_schema = schema["paths"]["/readyz"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert response_schema == {"$ref": "#/components/schemas/ReadinessResponse"}
    dependencies = schema["components"]["schemas"]["DependencyStates"]
    assert set(dependencies["required"]) == {"postgres", "chromadb", "ollama"}
