from uuid import uuid4

import httpx
import pytest
from starlette.responses import Response, StreamingResponse

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.proxy import ProxyDiagnostics, ProxyResult

HEADERS = {"x-opencode-session-id": "session", "x-opencode-project-id": "project"}
MODEL = "qwen3-coder:30b"


class DiagnosticProxy:
    async def complete(self, identity, payload, *, state):
        response = (
            StreamingResponse(iter([b"data: [DONE]\n\n"]))
            if payload.get("stream")
            else Response(b'{"choices":[]}', media_type="application/json")
        )
        return ProxyResult(response, ProxyDiagnostics(2, 137, ()))


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_registers_only_scoped_content_free_observation(stream):
    app = create_app(Settings(_env_file=None))
    app.state.proxy_service = DiagnosticProxy()
    observation_id = str(uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={**HEADERS, "x-opencode-rag-observation-id": observation_id},
            json={"model": MODEL, "messages": [], "stream": stream},
        )
        assert response.status_code == 200
        route = f"/v1/rag/observations/{observation_id}"
        for headers in [
            {**HEADERS, "x-opencode-session-id": "other"},
            {**HEADERS, "x-opencode-project-id": "other"},
        ]:
            denied = await client.get(route, headers=headers)
            absent = await client.get(f"/v1/rag/observations/{uuid4()}", headers=headers)
            assert denied.status_code == absent.status_code == 404
            assert denied.json() == absent.json()
        found = await client.get(route, headers=HEADERS)
        assert found.status_code == 200
        assert found.json() == {"injected_memory_tokens": 137}
        assert (await client.get(route)).status_code == 400


async def test_invalid_observation_ids_are_rejected_before_generation():
    app = create_app(Settings(_env_file=None))
    app.state.proxy_service = DiagnosticProxy()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={**HEADERS, "x-opencode-rag-observation-id": "arbitrary-content"},
            json={"model": MODEL, "messages": []},
        )
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "x-opencode-rag-observation-id"
        response = await client.get("/v1/rag/observations/arbitrary-content", headers=HEADERS)
        assert response.status_code == 404
