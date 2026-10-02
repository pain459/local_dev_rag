"""The optional operator smoke exits with actionable prerequisite remedies."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smoke.sh"

# Run the actual shell/heredoc with only Docker, database and model I/O replaced.
# This catches a query/connection await placed outside the worker deadline, or
# cleanup that hangs after cancellation. Real HTTPX/SQLAlchemy/settings stay loaded.
DOCKER_RUNNER = r"""
import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

args = sys.argv[1:]
if args == ["compose", "ps", "--status", "running", "--services"]:
    print("proxy\nworker\npostgres\nchromadb")
elif args == ["compose", "port", "proxy", "8080"]:
    print("127.0.0.1:8080")
elif args[:3] == ["compose", "exec", "-T"]:
    import httpx
    import local_dev_rag.db
    import local_dev_rag.ollama
    import local_dev_rag.vector_store

    mode = os.environ["SMOKE_TEST_MODE"]
    events = []
    def record(event):
        events.append(event)
        Path(os.environ["SMOKE_TEST_EVENTS"]).write_text(json.dumps(events))

    async def stall(stage):
        record(stage + "_started")
        try:
            await asyncio.Event().wait()
        finally:
            record(stage + "_cancelled")

    state = {"marker": "", "scalar_count": 0}
    model = "qwen2.5-coder:1.5b"
    models = ["qwen3-coder:30b", model, "qwen2.5-coder:7b", "llama3.1:8b", "qwen2.5:7b"]
    own, foreign, memory = UUID(int=1), UUID(int=2), UUID(int=3)

    def transport(request):
        path = request.url.path
        if path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/api/tags":
            tags = [{"name": m} for m in [*models, "nomic-embed-text:latest"]]
            return httpx.Response(200, json={"models": tags})
        if path == "/readyz":
            names = ["postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs"]
            states = {n: "healthy" for n in names}
            return httpx.Response(200, json={"status": "ready", "dependencies": states})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        body = json.loads(request.content)
        if body["stream"]:
            choice = {"delta": {"content": state["marker"]}, "finish_reason": "stop"}
            chunk = {"model": model, "choices": [choice]}
            sse = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        prompt = body["messages"][-1]["content"]
        match = re.search(r"quartz[0-9a-f]+", prompt)
        if match:
            state["marker"] = match[0]
        reply = state["marker"] if match else "UNKNOWN"
        choice = {"finish_reason": "stop", "message": {"content": reply}}
        return httpx.Response(200, json={"model": model, "choices": [choice]})

    class Client(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(transport), **kwargs)

        async def aclose(self):
            record("client_close_started")
            try:
                if mode == "cleanup_stalls":
                    await stall("client_close")
            finally:
                await super().aclose()
                record("client_closed")

        async def __aexit__(self, *args):
            await self.aclose()

    httpx.AsyncClient = Client

    class Session:
        def __init__(self):
            self.queries = 0

        async def scalar(self, statement):
            if mode in {"project_query", "cleanup_stalls"}:
                await stall("query")
            if mode == "cancelled_query":
                raise asyncio.CancelledError("private exception payload")
            state["scalar_count"] += 1
            return own if state["scalar_count"] == 1 else foreign

        async def execute(self, statement):
            self.queries += 1
            if mode == f"query_{self.queries}":
                await stall("query")
            rows = [("completed", None)] if self.queries == 1 else [(memory,)]
            return SimpleNamespace(all=lambda: rows)

    class Database:
        def __init__(self):
            self.engine = SimpleNamespace(dispose=self.dispose)

        @asynccontextmanager
        async def session(self):
            record("session_opened")
            try:
                if mode == "connection":
                    await stall("connection")
                yield Session()
            finally:
                record("session_closed")

        async def dispose(self):
            record("database_close_started")
            try:
                if mode == "cleanup_stalls":
                    await stall("database_close")
            finally:
                record("database_closed")

    local_dev_rag.db.Database.create = lambda settings: Database()

    async def embed(self, model, inputs):
        return [[1.0, 0.0]]

    async def query(self, project_id, vector, limit):
        return [SimpleNamespace(memory=SimpleNamespace(id=memory))] if project_id == own else []

    local_dev_rag.ollama.OllamaClient.embed = embed
    local_dev_rag.vector_store.VectorStore.query = query
    exec(compile(sys.stdin.read(), "smoke-heredoc", "exec"), {"__name__": "__main__"})
"""


def run_smoke(tmp_path, *, mode="normal", timeout="0.05"):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n{DOCKER_RUNNER}")
    docker.chmod(0o755)
    events = tmp_path / "events.json"
    started = time.monotonic()
    with subprocess.Popen(
        ["/bin/sh", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "SMOKE_TEST_MODE": mode,
            "SMOKE_TEST_EVENTS": str(events),
            "SMOKE_MODEL": "qwen2.5-coder:1.5b",
            "SMOKE_TIMEOUT_SECONDS": timeout,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            pytest.fail("smoke exceeded its worker/cleanup deadline (2-second safety bound)")
        result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    return (
        result,
        json.loads(events.read_text()) if events.exists() else [],
        time.monotonic() - started,
    )


@pytest.mark.parametrize(
    "mode",
    ["connection", "project_query", "query_1", "query_2", "cleanup_stalls", "cancelled_query"],
)
def test_worker_deadline_bounds_all_database_awaits_and_cleanup(tmp_path, mode):
    result, events, elapsed = run_smoke(tmp_path, mode=mode)
    assert result.returncode == 1
    assert "Worker progress timed out. Check docker compose logs worker" in result.stderr
    assert "increase SMOKE_TIMEOUT_SECONDS for a slow host." in result.stderr
    assert elapsed < 2
    assert "session_closed" in events
    assert "database_closed" in events
    assert "client_closed" in events
    assert "Traceback" not in result.stderr
    assert "private exception payload" not in result.stderr
    assert "quartz" not in result.stdout + result.stderr


@pytest.mark.parametrize("timeout", ["0.05", "1", "3600"])
def test_normal_worker_completion_and_valid_timeout_boundaries(tmp_path, timeout):
    result, events, _ = run_smoke(tmp_path, timeout=timeout)
    assert result.returncode == 0, result.stderr
    assert "worker durable completion and accepted memory (jobs=1, memories=1)" in result.stdout
    assert "live smoke complete" in result.stdout
    assert "database_closed" in events and "client_closed" in events
    assert "quartz" not in result.stdout


@pytest.mark.parametrize("timeout", ["0", "-1", "3600.01", "nan", "inf", "not-seconds"])
def test_invalid_worker_timeout_fails_before_resource_acquisition(tmp_path, timeout):
    result, events, _ = run_smoke(tmp_path, timeout=timeout)
    assert result.returncode == 1
    assert "Set SMOKE_TIMEOUT_SECONDS" in result.stderr
    assert not events
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("docker_body", "remedy"),
    [
        (None, "Install Docker"),
        ("exit 1", "Start Docker"),
        (
            'case "$*" in "info") exit 0;; "compose version") exit 1;; esac',
            "Install Docker Compose",
        ),
        (
            'case "$*" in "compose ps --status running --services") echo postgres;; esac',
            "docker compose up --build -d --wait",
        ),
        (
            'case "$*" in "compose ps --status running --services") '
            'printf "proxy\\nworker\\npostgres\\nchromadb\\n";; '
            '"compose port proxy 8080") echo 0.0.0.0:8080;; esac',
            "127.0.0.1",
        ),
    ],
)
def test_prerequisite_failure_has_precise_remedy(tmp_path, docker_body, remedy):
    if docker_body is not None:
        docker = tmp_path / "docker"
        docker.write_text(f"#!/bin/sh\n{docker_body}\n")
        docker.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert remedy in result.stderr
    assert "Traceback" not in result.stderr
