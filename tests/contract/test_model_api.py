import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings


async def test_model_discovery_lists_only_generation_models_without_identity_headers():
    async with AsyncClient(
        transport=ASGITransport(app=create_app(Settings(_env_file=None))), base_url="http://test"
    ) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert {model["id"] for model in payload["data"]} == {
        "qwen3-coder:30b",
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:7b",
        "llama3.1:8b",
        "qwen2.5:7b",
    }


@pytest.mark.parametrize("missing", ["x-opencode-session-id", "x-opencode-project-id"])
async def test_identity_errors_have_openai_compatible_400_envelope(missing):
    from local_dev_rag.domain import RequestIdentity

    app = create_app(Settings(_env_file=None))

    # Exercise the reusable error handler through a test-only route; chat is Task 4.
    @app.get("/identity-check")
    async def identity_check(request: Request):
        return RequestIdentity.from_headers(request.headers)

    headers = {"x-opencode-session-id": "session", "x-opencode-project-id": "project"}
    del headers[missing]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/identity-check", headers=headers)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == missing
    assert error["message"]
