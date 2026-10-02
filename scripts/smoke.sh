#!/bin/sh
# Optional live verification. Run against the active Compose project, never print bodies.
set -eu
script_dir=${0%/*}
CDPATH= cd "$script_dir/.."

fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || fail "Install Docker with the Compose plugin, then rerun ./scripts/smoke.sh."
docker info >/dev/null 2>&1 || fail "Start Docker Desktop or the Docker daemon; verify docker info."
docker compose version >/dev/null 2>&1 || fail "Install Docker Compose v2+; verify docker compose version."
docker compose config --quiet >/dev/null 2>&1 || fail "Fix .env/compose.yaml; run docker compose config --quiet."
running=$(docker compose ps --status running --services 2>/dev/null) || fail "Inspect docker compose ps; run docker compose up --build -d --wait."
for service in proxy worker postgres chromadb; do
    case "
$running
" in *"
$service
"*) ;; *) fail "$service is not running. Run docker compose up --build -d --wait; inspect docker compose logs $service.";; esac
done
published=$(docker compose port proxy 8080 2>/dev/null) || fail "Publish proxy port 8080 on host loopback in compose.yaml."
case "$published" in 127.0.0.1:*) ;; *) fail "Bind the published proxy only to 127.0.0.1 in compose.yaml.";; esac
printf 'PASS: Docker, four services, loopback proxy publication\n'

docker compose exec -T \
    -e SMOKE_MODEL="${SMOKE_MODEL:-qwen2.5-coder:1.5b}" \
    -e SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-300}" \
    proxy python - <<'PY'
import asyncio
import json
import logging
import os
import sys
from time import monotonic
from uuid import uuid4

import httpx
from sqlalchemy import select

from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.schema import memory_items, memory_jobs, projects
from local_dev_rag.vector_store import VectorStore

# Never expose HTTP bodies, exceptions, credentials, prompts, or memory text.
logging.disable(logging.CRITICAL)
MODELS = {
    "qwen3-coder:30b", "qwen2.5-coder:1.5b", "qwen2.5-coder:7b",
    "llama3.1:8b", "qwen2.5:7b",
}


class SmokeFailure(Exception):
    pass


def require(condition, remedy):
    if not condition:
        raise SmokeFailure(remedy)


def passed(message):
    print(f"PASS: {message}", flush=True)


async def run():
    settings = Settings()
    model = os.environ["SMOKE_MODEL"]
    require(model in MODELS, "Set SMOKE_MODEL to one of the five /v1/models generation IDs.")
    try:
        deadline_seconds = float(os.environ["SMOKE_TIMEOUT_SECONDS"])
        require(0 < deadline_seconds <= 3600, "Set SMOKE_TIMEOUT_SECONDS between 1 and 3600.")
    except ValueError:
        raise SmokeFailure("Set SMOKE_TIMEOUT_SECONDS to seconds, e.g. 300.") from None
    async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds + 10) as client:
        proxy = "http://127.0.0.1:8080"
        try:
            health = await client.get(f"{proxy}/healthz")
            require(health.status_code == 200 and health.json() == {"status": "ok"},
                    "Proxy liveness failed. Inspect docker compose logs proxy; rebuild/restart proxy.")
        except httpx.HTTPError:
            raise SmokeFailure("Proxy unreachable. Inspect docker compose logs proxy; run docker compose up --build -d --wait.") from None
        try:
            tags = await client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            require(tags.status_code == 200, "Ollama /api/tags failed. Start ollama serve; correct OLLAMA_URL and restart proxy/worker.")
            installed = {m["name"] for m in tags.json()["models"]}
        except httpx.HTTPError:
            raise SmokeFailure("Ollama unreachable from containers. Start ollama serve; check host.docker.internal:11434 and the host listener/firewall; set OLLAMA_URL and recreate proxy/worker.") from None
        required = MODELS | {settings.curator_model, settings.embedding_model}
        missing = required - installed
        # Model IDs are a fixed catalog or operator configuration, never response bodies.
        require(not missing, "Missing required models. On the host run: " +
                "; ".join(f"ollama pull {name}" for name in sorted(missing)) +
                ". Then rerun smoke.")
        passed("container-to-Ollama reachability and required model presence")
        ready = await client.get(f"{proxy}/readyz")
        report = ready.json()
        expected = {"postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs"}
        require(set(report.get("dependencies", {})) == expected,
                "Readiness schema mismatch. Rebuild proxy/worker from this checkout.")
        unhealthy = [name for name, state in report["dependencies"].items() if state != "healthy"]
        require(ready.status_code == 200 and report.get("status") == "ready" and not unhealthy,
                "Full memory smoke needs ready dependencies (unhealthy: " + ",".join(unhealthy) +
                "). Inspect /readyz and docker compose logs proxy worker. For memory_jobs, inspect durable failed/retry counts; restore the dependency and reindex affected projects. Reindex alone does not clear failed job state.")
        passed("proxy liveness and readiness (six healthy dependencies)")
        catalog = await client.get(f"{proxy}/v1/models")
        require(catalog.status_code == 200 and {m["id"] for m in catalog.json()["data"]} == MODELS,
                "Proxy model catalog mismatch. Rebuild proxy and restore the five configured generation models; keep the embedder hidden.")
        passed("exactly five selectable generation models, embedder hidden")

        suffix = uuid4().hex
        project_a, project_b = f"smoke-a-{suffix}", f"smoke-b-{suffix}"
        # Unpredictable identifier: an unrelated session cannot guess it from the query.
        marker = "quartz" + uuid4().hex[:12]
        decision = f"We selected {marker} for durable memory."

        async def chat(project, session, prompt, *, stream=False):
            payload = {"model": model, "stream": stream, "temperature": 0,
                       "max_tokens": 128, "messages": [
                           {"role": "system", "content": "Follow the user request. Historical project evidence may be supplied separately."},
                           {"role": "user", "content": prompt},
                       ]}
            headers = {"x-opencode-project-id": project, "x-opencode-session-id": session}
            if not stream:
                response = await client.post(f"{proxy}/v1/chat/completions", json=payload, headers=headers)
                require(response.status_code == 200,
                        "Non-streaming inference failed. Check selected model with ollama run on the host, GPU/RAM, timeout and proxy error categories.")
                body = response.json()
                require(body.get("model") == model, "Upstream changed the selected model; inspect Ollama compatibility.")
                choice = body["choices"][0]
                require(choice.get("finish_reason") == "stop", "Non-streaming inference did not finish. Increase max_tokens/timeout or select a smaller installed model.")
                return choice["message"].get("content") or ""
            text, done, finished = [], False, False
            async with client.stream("POST", f"{proxy}/v1/chat/completions", json=payload, headers=headers) as response:
                require(response.status_code == 200 and response.headers.get("content-type", "").startswith("text/event-stream"),
                        "Streaming inference failed. Inspect proxy/upstream error categories and selected model capability.")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        continue
                    chunk = json.loads(data)
                    require("error" not in chunk, "Ollama emitted a streaming error. Check model inference, memory pressure and proxy categories.")
                    if "model" in chunk:
                        require(chunk["model"] == model, "Streaming response changed the selected model.")
                    for choice in chunk.get("choices", []):
                        text.append(choice.get("delta", {}).get("content") or "")
                        finished |= choice.get("finish_reason") == "stop"
            require(done and finished and bool("".join(text).strip()),
                    "Stream incomplete/empty. Check Ollama timeout, model inference and proxy stream_outcome.")
            return "".join(text)

        reply = await chat(project_a, "establish", f"Reply exactly with this confirmed project decision, without commentary: {decision}")
        require(marker in reply, "Selected model did not confirm the synthetic decision. Choose SMOKE_MODEL=qwen2.5-coder:7b or inspect local model inference.")
        passed(f"non-streaming completion ({model})")
        database = Database.create(settings)
        try:
            deadline = monotonic() + deadline_seconds
            while True:
                async with database.session() as session:
                    project_id = await session.scalar(select(projects.c.id).where(projects.c.external_project_id == project_a))
                    jobs = (await session.execute(select(memory_jobs.c.status, memory_jobs.c.error_category).where(memory_jobs.c.project_id == project_id))).all()
                    active = (await session.execute(select(memory_items.c.id).where(
                        memory_items.c.project_id == project_id, memory_items.c.state == "active",
                        memory_items.c.text.contains(marker)))).all()
                require(not any(status == "failed" for status, _ in jobs),
                        "Synthetic memory job failed terminally. Inspect worker categories, curator schema/evidence and model inference; fix dependencies before a new smoke run.")
                if jobs and all(status == "completed" for status, _ in jobs):
                    require(bool(active), "Worker finished with no supported synthetic memory. Curator may have omitted/paraphrased the decision; verify CURATOR_MODEL structured output/evidence, or use a stronger curator, restart worker and rerun.")
                    break
                require(monotonic() < deadline, "Worker progress timed out. Check docker compose logs worker, durable job status/lease/retry categories, curator/embedder inference; increase SMOKE_TIMEOUT_SECONDS for a slow host.")
                await asyncio.sleep(1)
            passed(f"worker durable completion and accepted memory (jobs={len(jobs)}, memories={len(active)})")
            query = "Which exact backend name did we select for durable memory? Reply only with its name; reply UNKNOWN if no project evidence identifies it."
            same = await chat(project_a, "fresh-recall", query, stream=True)
            require(marker in same, "Same-project streaming recall failed. Check Chroma index, embedding configuration/version, relevance budget and reindex this synthetic project using its external ID from PostgreSQL.")
            passed("streaming completion and same-project recall in a fresh session")
            other = await chat(project_b, "fresh-recall", query)
            require(marker not in other, "Cross-project isolation failed. Stop using the proxy; inspect project headers and project filters before resuming.")
            vector = (await OllamaClient(settings).embed(settings.embedding_model, [query]))[0]
            store = VectorStore(settings)
            own_hits = await store.query(project_id, vector, settings.retrieval_candidate_limit)
            async with database.session() as session:
                foreign_id = await session.scalar(select(projects.c.id).where(projects.c.external_project_id == project_b))
            foreign_hits = await store.query(foreign_id, vector, settings.retrieval_candidate_limit)
            require(any(hit.memory.id in {row[0] for row in active} for hit in own_hits) and not foreign_hits,
                    "Vector recall/isolation failed. Inspect exact project filtering and reindex the affected project; PostgreSQL is authoritative.")
            passed(f"cross-project isolation (foreign vector hits={len(foreign_hits)})")
        finally:
            await database.engine.dispose()
    passed("live smoke complete; synthetic project records retained locally")


try:
    asyncio.run(run())
except SmokeFailure as error:
    print(f"FAIL: {error}", file=sys.stderr)
    sys.exit(1)
except Exception:
    print("FAIL: unexpected transport/schema/database failure; inspect /readyz, docker compose logs proxy worker and installed versions. Response bodies and exception text suppressed.", file=sys.stderr)
    sys.exit(1)
PY
