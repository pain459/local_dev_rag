# Local OpenCode RAG Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local OpenAI-compatible proxy that gives OpenCode durable, project-scoped conversation memory while serving selectable Ollama models.

**Architecture:** OpenCode and Ollama run on the host. A FastAPI proxy, asynchronous worker, PostgreSQL, and ChromaDB run under Docker Compose; PostgreSQL is authoritative and Chroma is a rebuildable semantic index. The proxy preserves OpenCode's streaming tool loop while replacing old raw turns with a bounded block of relevant project memories.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic Settings, HTTPX, SQLAlchemy 2 async, asyncpg, Alembic, PostgreSQL, ChromaDB, Docker Compose, pytest, pytest-asyncio, Ruff, Pyright, OpenCode JavaScript plugin.

**Spec:** `docs/superpowers/specs/2026-10-01-local-opencode-rag-memory-design.md`

## Global Constraints

- Target OpenCode 1.18.30 and its `chat.headers` plugin hook.
- Target Ollama 0.35.0 through its OpenAI-compatible chat endpoint and native/OpenAI-compatible embedding endpoint.
- Expose exactly these chat models: `qwen3-coder:30b`, `qwen2.5-coder:1.5b`, `qwen2.5-coder:7b`, `llama3.1:8b`, and `qwen2.5:7b`.
- Default the foreground model to `qwen3-coder:30b`, curator to `qwen2.5-coder:1.5b`, and embedder to `nomic-embed-text:latest`.
- Never expose `nomic-embed-text:latest` as a selectable chat model.
- Retrieval is project-scoped; no request may retrieve another project's memory.
- PostgreSQL is authoritative; every Chroma record must reference an active PostgreSQL memory ID.
- Preserve system instructions, tool definitions, current user input, and complete tool-call/result chains.
- Bind the proxy only to `127.0.0.1`; do not publish PostgreSQL or ChromaDB ports by default.
- Do not log prompt, source, tool-result, or memory bodies at ordinary log levels.

## Review Focus

- Interleaved or incomplete tool-call chains must never be split into invalid outbound history; Task 6 adds exact chain-preservation tests.
- A retried OpenCode request or replayed stream must not create duplicate events or memory jobs; Tasks 3 and 7 add concurrency and idempotency tests.
- A cancelled or broken stream must never be marked complete or curated; Tasks 4 and 7 test partial-stream behavior.
- Missing or mismatched project/session headers must not permit cross-project retrieval; Tasks 2 and 7 test rejection and isolation.
- PostgreSQL, ChromaDB, curator, embedder, and Ollama outages need distinct safe behaviors; Tasks 4, 7, 9, and 10 pin each degraded mode.

---

## Planned file structure

```text
.
├── .env.example
├── .opencode/plugins/rag-memory.js
├── alembic.ini
├── compose.yaml
├── Dockerfile
├── opencode.json
├── pyproject.toml
├── uv.lock
├── migrations/
│   ├── env.py
│   └── versions/0001_initial_memory_schema.py
├── src/local_dev_rag/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── context.py
│   ├── curator.py
│   ├── db.py
│   ├── domain.py
│   ├── events.py
│   ├── jobs.py
│   ├── logging.py
│   ├── models.py
│   ├── ollama.py
│   ├── proxy.py
│   ├── ranking.py
│   ├── repository.py
│   ├── streaming.py
│   ├── vector_store.py
│   ├── worker.py
│   └── cli.py
└── tests/
    ├── contract/
    ├── integration/
    ├── unit/
    └── e2e/
```

Each module owns one responsibility: API translation, model/runtime I/O, relational persistence, vector persistence, context selection, memory extraction, or job execution. The proxy composes those interfaces rather than embedding their implementation details.

### Task 1: Service skeleton, configuration, and Compose topology

**Files:**
- Create: `pyproject.toml`
- Create: `src/local_dev_rag/__init__.py`
- Create: `src/local_dev_rag/config.py`
- Create: `src/local_dev_rag/api.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/contract/test_health.py`
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `.env.example`

**Interfaces:**
- Produces: `Settings(BaseSettings)`, `get_settings() -> Settings`, and `create_app(settings: Settings | None = None) -> FastAPI`.
- Produces: `GET /healthz -> {"status": "ok"}` and the dependency-aware `GET /readyz` response contract.

- [ ] **Step 1: Write failing configuration tests**

Assert that `Settings()` defaults to proxy port `8080`, Ollama URL `http://host.docker.internal:11434`, curator `qwen2.5-coder:1.5b`, embedder `nomic-embed-text:latest`, and rejects a non-loopback bind unless `ALLOW_REMOTE_BIND=true`.

- [ ] **Step 2: Run the configuration tests and confirm failure**

Run: `uv run pytest tests/unit/test_config.py -v`  
Expected: FAIL because `local_dev_rag.config` does not exist.

- [ ] **Step 3: Create the package metadata and `Settings`**

Declare Python `>=3.12,<3.13`, application/runtime dependencies, test groups, Ruff, Pyright, and pytest settings. Implement typed settings for service URLs, database, model IDs, per-model budgets, retrieval limits, retry values, bind policy, and log level.

- [ ] **Step 4: Write failing health endpoint tests**

Assert `/healthz` is `200` with the exact body and `/readyz` returns a typed JSON object with `status` plus dependency entries.

- [ ] **Step 5: Implement `create_app()` and the two endpoint shells**

Until adapters exist, return `{"status":"starting","dependencies":{"postgres":"unknown","chromadb":"unknown","ollama":"unknown"}}`; later tasks replace each value with a live check without changing the shape.

- [ ] **Step 6: Add the container skeleton**

Create a non-root production image and Compose services named `proxy`, `worker`, `postgres`, and `chromadb`. Publish only `127.0.0.1:${PROXY_PORT:-8080}:8080`; add `host.docker.internal:host-gateway`; define named database volumes. Use `pg_isready` for PostgreSQL, Chroma's heartbeat endpoint for ChromaDB, `/healthz` for the proxy, and a worker process check that Task 10 hardens with dependency state.

- [ ] **Step 7: Verify configuration and topology**

Run: `uv sync --all-groups && uv run pytest tests/unit/test_config.py tests/contract/test_health.py -v && docker compose config`  
Expected: all tests PASS and Compose configuration renders without errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src tests Dockerfile compose.yaml .env.example
git commit -m "feat: scaffold local RAG services"
```

### Task 2: OpenCode plugin, provider configuration, and model catalog

**Files:**
- Create: `.opencode/plugins/rag-memory.js`
- Create: `opencode.json`
- Create: `src/local_dev_rag/models.py`
- Create: `tests/unit/test_models.py`
- Create: `tests/contract/test_opencode_config.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: `ModelSpec(id: str, display_name: str, context_tokens: int, output_tokens: int)` and `ModelRegistry.get(model_id: str) -> ModelSpec`.
- Produces: plugin headers `x-opencode-session-id`, `x-opencode-project-id`, and `x-opencode-project-root` for provider `local-rag`.

- [ ] **Step 1: Write failing model-catalog tests**

Assert the registry returns exactly the five generation model IDs, defaults to `qwen3-coder:30b`, rejects `nomic-embed-text:latest`, and raises `UnknownModelError` for any unconfigured ID.

- [ ] **Step 2: Run the model tests and confirm failure**

Run: `uv run pytest tests/unit/test_models.py -v`  
Expected: FAIL because `ModelRegistry` is missing.

- [ ] **Step 3: Implement the registry**

Load explicit context/output budgets from `Settings`; do not infer them from marketing limits. Provide `as_openai_models() -> dict[str, object]` for `/v1/models`.

- [ ] **Step 4: Write failing configuration-contract tests**

Parse `opencode.json` and assert provider ID `local-rag`, npm adapter `@ai-sdk/openai-compatible`, base URL `http://localhost:8080/v1`, exact model membership, and no embedder. Statically inspect the plugin export and required headers.

- [ ] **Step 5: Create the OpenCode artifacts**

Use the installed 1.18.30 `chat.headers` hook. Precompute a stable project hash from OpenCode's project ID plus normalized Git remote when available, falling back to the normalized project directory. Add headers only when `input.provider.info.id === "local-rag"`.

- [ ] **Step 6: Wire `GET /v1/models` and header validation**

Add `RequestIdentity.from_headers(headers: Mapping[str, str]) -> RequestIdentity` in `domain.py`. Missing session or project headers return a structured `400 invalid_request_error`; a mismatched raw root is diagnostic only and never changes the project ID.

- [ ] **Step 7: Verify with the installed OpenCode configuration loader**

Run: `uv run pytest tests/unit/test_models.py tests/contract/test_opencode_config.py -v && opencode debug config`  
Expected: tests PASS and OpenCode loads provider `local-rag` plus the plugin without a configuration error.

- [ ] **Step 8: Commit**

```bash
git add .opencode opencode.json src/local_dev_rag/models.py src/local_dev_rag/domain.py src/local_dev_rag/api.py tests
git commit -m "feat: configure OpenCode local models"
```

### Task 3: PostgreSQL schema and idempotent repositories

**Files:**
- Create: `src/local_dev_rag/db.py`
- Create: `src/local_dev_rag/repository.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/versions/0001_initial_memory_schema.py`
- Create: `tests/integration/test_repository.py`

**Interfaces:**
- Consumes: `RequestIdentity` and settings from Tasks 1-2.
- Produces: `Database.create(settings) -> Database` and async `Database.session()` context manager.
- Produces: `ConversationRepository.ensure_scope(identity) -> Scope`, `append_events(scope, events) -> list[StoredEvent]`, `finalize_assistant(scope, completion) -> StoredEvent`, and `enqueue_memory_job(source_event_id) -> MemoryJob`.
- Produces domain records used later: `MemoryCandidate`, `EmbeddedMemory`, `VectorHit`, `CuratorSource`, and `MemoryDraft`.

- [ ] **Step 1: Write failing repository integration tests**

Against a temporary PostgreSQL instance, assert project/session upsert, ordered JSONB event insertion, duplicate content-hash insertion returning the original rows, one memory job per completed source event, and concurrent duplicate inserts producing one logical event.

- [ ] **Step 2: Run the repository tests and confirm failure**

Run: `uv run pytest tests/integration/test_repository.py -v`  
Expected: FAIL because database modules and migrations do not exist.

- [ ] **Step 3: Define domain records and the initial migration**

Create typed immutable records for `Scope`, `ConversationEventInput`, `StoredEvent`, `AssistantCompletion`, `MemoryItem`, `MemoryCandidate`, `EmbeddedMemory`, `VectorHit`, `CuratorSource`, `MemoryDraft`, and `MemoryJob`. Implement the five spec tables, foreign keys, lifecycle constraints, useful indexes, and retry-safe unique constraints.

- [ ] **Step 4: Implement the async database and repository**

Use SQLAlchemy Core/ORM with asyncpg. Upserts must use PostgreSQL conflict handling rather than read-then-write logic.

- [ ] **Step 5: Verify migration round-trip and repository behavior**

Run: `uv run alembic upgrade head && uv run pytest tests/integration/test_repository.py -v`  
Expected: migration succeeds and all repository tests PASS.

- [ ] **Step 6: Commit**

```bash
git add alembic.ini migrations src/local_dev_rag/db.py src/local_dev_rag/domain.py src/local_dev_rag/repository.py tests/integration/test_repository.py
git commit -m "feat: add durable conversation storage"
```

### Task 4: Ollama client and transparent streaming proxy

**Files:**
- Create: `src/local_dev_rag/ollama.py`
- Create: `src/local_dev_rag/streaming.py`
- Modify: `src/local_dev_rag/api.py`
- Create: `tests/contract/test_chat_proxy.py`
- Create: `tests/unit/test_streaming.py`

**Interfaces:**
- Consumes: `ModelRegistry` from Task 2.
- Produces: `UpstreamResponse(status_code: int, headers: Mapping[str, str], body: AsyncIterator[bytes])` and `DependencyStatus(name: str, state: Literal["healthy", "degraded", "unavailable"], detail: str | None)` domain records.
- Produces: `OllamaClient.chat(payload: Mapping[str, object]) -> AsyncContextManager[UpstreamResponse]`, `OllamaClient.embed(model: str, inputs: Sequence[str]) -> list[list[float]]`, and `OllamaClient.health() -> DependencyStatus`.
- Produces: `StreamAccumulator.feed(chunk: bytes) -> bytes` and `StreamAccumulator.completion() -> AssistantCompletion | None`.

- [ ] **Step 1: Write failing streaming parser tests**

Cover fragmented SSE lines, multibyte UTF-8 boundaries, content deltas, tool-call argument fragments, `[DONE]`, upstream error events, and a stream ending before completion returning `None`.

- [ ] **Step 2: Run the parser tests and confirm failure**

Run: `uv run pytest tests/unit/test_streaming.py -v`  
Expected: FAIL because `StreamAccumulator` is missing.

- [ ] **Step 3: Implement the Ollama adapter and stream accumulator**

Keep transport injectable for tests. Relay bytes promptly while accumulating only the structured assistant completion required for persistence.

- [ ] **Step 4: Write failing chat contract tests**

Using an HTTPX mock transport, assert non-streaming pass-through, streaming byte order, tools/tool-choice preservation, unknown-model rejection, Ollama connection failure mapped to `502`, and cancelled/broken streams never yielding a completed assistant record.

- [ ] **Step 5: Implement the baseline `/v1/chat/completions` route**

Validate identity and model, forward the request, preserve status and safe headers, and emit OpenAI-compatible errors. Do not add memory behavior in this task.

- [ ] **Step 6: Verify the proxy contract**

Run: `uv run pytest tests/unit/test_streaming.py tests/contract/test_chat_proxy.py -v`  
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/local_dev_rag/api.py src/local_dev_rag/ollama.py src/local_dev_rag/streaming.py tests
git commit -m "feat: proxy Ollama chat completions"
```

### Task 5: Conversation normalization and durable capture

**Files:**
- Create: `src/local_dev_rag/events.py`
- Modify: `src/local_dev_rag/api.py`
- Modify: `src/local_dev_rag/repository.py`
- Create: `tests/unit/test_events.py`
- Create: `tests/integration/test_capture.py`

**Interfaces:**
- Consumes: repository and streaming completion interfaces from Tasks 3-4.
- Produces: `normalize_messages(messages: Sequence[Mapping[str, object]]) -> list[ConversationEventInput]` and `content_hash(event) -> str`.

- [ ] **Step 1: Write failing normalization tests**

Assert stable hashes across JSON key order, distinct hashes for ordered content changes, retention of multimodal/text parts, exact tool-call IDs and arguments, and correct role/event classification.

- [ ] **Step 2: Run normalization tests and confirm failure**

Run: `uv run pytest tests/unit/test_events.py -v`  
Expected: FAIL because normalization functions are missing.

- [ ] **Step 3: Implement canonical normalization**

Canonicalize only representation details; never reorder message arrays, content parts, tool calls, or tool results.

- [ ] **Step 4: Write failing capture integration tests**

Assert a full request and completion persist once, a retried identical request persists no duplicate, a replayed successful stream creates no duplicate job, and an interrupted stream stores diagnostic attempt state without a completed assistant event or memory job.

- [ ] **Step 5: Integrate capture around the baseline proxy**

Persist unseen inbound events before forwarding. Finalize and enqueue only after successful stream/non-stream completion. If PostgreSQL is unavailable, call the baseline passthrough path without trimming or claiming persistence.

- [ ] **Step 6: Verify capture behavior**

Run: `uv run pytest tests/unit/test_events.py tests/integration/test_capture.py -v`  
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/local_dev_rag/events.py src/local_dev_rag/api.py src/local_dev_rag/repository.py tests
git commit -m "feat: capture OpenCode conversation events"
```

### Task 6: Token-bounded context construction

**Files:**
- Create: `src/local_dev_rag/context.py`
- Create: `tests/unit/test_context.py`

**Interfaces:**
- Consumes: `ModelSpec` and `MemoryCandidate` domain values.
- Produces: `ChatRequest(model: str, messages: list[dict[str, object]], stream: bool, extra: dict[str, object])` and `ContextBuildResult(payload: dict[str, object], included_event_ids: tuple[str, ...], dropped_event_ids: tuple[str, ...], estimated_input_tokens: int, injected_memory_tokens: int)`.
- Produces: `TokenEstimator.estimate(value: object) -> int` and `ContextBuilder.build(request: ChatRequest, memories: Sequence[MemoryCandidate], history_durable: bool) -> ContextBuildResult`.

- [ ] **Step 1: Write failing budget and ordering tests**

Assert system messages and current user input survive; recent turns are preferred; output reserve and safety margin are honored; memories fit only the memory partition; and selected memories appear in one delimited historical-evidence block with kind/date provenance.

- [ ] **Step 2: Add failing tool-chain and durability tests**

Cover one tool call/result, several calls in one assistant message, interleaved results, an incomplete chain, and a PostgreSQL-degraded request. Assert chains are kept whole or removed whole, and `history_durable=False` disables destructive trimming.

- [ ] **Step 3: Run context tests and confirm failure**

Run: `uv run pytest tests/unit/test_context.py -v`  
Expected: FAIL because `ContextBuilder` is missing.

- [ ] **Step 4: Implement the conservative estimator and builder**

Use `ceil(serialized_utf8_bytes / 3)` initially. Build atomic conversation units, select newest units backward, then inject deduplicated memories in chronological order. Return included/dropped event IDs and token estimates for diagnostics.

- [ ] **Step 5: Verify context behavior**

Run: `uv run pytest tests/unit/test_context.py -v`  
Expected: all tests PASS, including the Review Focus tool-chain cases.

- [ ] **Step 6: Commit**

```bash
git add src/local_dev_rag/context.py src/local_dev_rag/domain.py tests/unit/test_context.py
git commit -m "feat: build bounded model context"
```

### Task 7: Chroma adapter, retrieval, ranking, and foreground integration

**Files:**
- Create: `src/local_dev_rag/vector_store.py`
- Create: `src/local_dev_rag/ranking.py`
- Create: `src/local_dev_rag/proxy.py`
- Modify: `src/local_dev_rag/api.py`
- Create: `tests/unit/test_ranking.py`
- Create: `tests/integration/test_retrieval.py`
- Create: `tests/integration/test_foreground_flow.py`

**Interfaces:**
- Consumes: `OllamaClient.embed`, repositories, `ContextBuilder`, and `ModelRegistry`.
- Produces: `VectorStore.upsert(items: Sequence[EmbeddedMemory])`, `VectorStore.query(project_id: UUID, vector: Sequence[float], limit: int) -> list[VectorHit]`, and `VectorStore.delete_project(project_id: UUID)`.
- Produces: `rank_memories(query_text: str, hits: Sequence[VectorHit], now: datetime) -> list[MemoryCandidate]`.
- Produces: `ProxyDiagnostics(retrieval_count: int, injected_memory_tokens: int, degraded_dependencies: tuple[str, ...])`, `ProxyResult(response: Response, diagnostics: ProxyDiagnostics)`, and `ProxyService.complete(identity, request) -> ProxyResult` used by the API route.

- [ ] **Step 1: Write failing ranking tests**

Assert semantic score, importance, recency, identifier/error-string overlap, inactive-memory exclusion, diversity, stable ties, and duplicate suppression produce deterministic ordering.

- [ ] **Step 2: Implement the pure ranking function**

Keep weights in typed settings. Normalize every component before applying weights; return score components for structured diagnostics.

- [ ] **Step 3: Write failing vector integration tests**

Against ChromaDB, upsert two projects with similar text and assert exact project filtering prevents cross-project hits. Assert repeated upsert is idempotent and an unavailable Chroma service raises `VectorStoreUnavailable`.

- [ ] **Step 4: Implement the Chroma adapter**

Store PostgreSQL memory ID, project ID, kind, importance, created timestamp, lifecycle/version fields, and text. Use deterministic Chroma IDs derived from PostgreSQL IDs.

- [ ] **Step 5: Write failing foreground-flow tests**

Assert retrieval query embedding, top-candidate reranking, bounded memory injection, no cross-project memory, one upstream call, streaming preservation, and these fallbacks: Chroma down means recent-context-only; PostgreSQL down means untrimmed plain passthrough; embedder down means no retrieval.

- [ ] **Step 6: Implement `ProxyService` and make the API thin**

The service coordinates persistence, retrieval, context building, upstream streaming, finalization, and job enqueueing through injected interfaces. It must not contain database- or Chroma-specific queries.

- [ ] **Step 7: Verify retrieval and foreground behavior**

Run: `uv run pytest tests/unit/test_ranking.py tests/integration/test_retrieval.py tests/integration/test_foreground_flow.py -v`  
Expected: all tests PASS, including project isolation and degraded modes.

- [ ] **Step 8: Commit**

```bash
git add src/local_dev_rag/vector_store.py src/local_dev_rag/ranking.py src/local_dev_rag/proxy.py src/local_dev_rag/api.py tests
git commit -m "feat: inject project-scoped memory"
```

### Task 8: Curator schema and durable memory-job queue

**Files:**
- Create: `src/local_dev_rag/curator.py`
- Create: `src/local_dev_rag/jobs.py`
- Modify: `src/local_dev_rag/repository.py`
- Create: `tests/unit/test_curator.py`
- Create: `tests/integration/test_jobs.py`

**Interfaces:**
- Consumes: `OllamaClient`, `ConversationRepository`, and domain values.
- Produces: `Curator.extract(source: CuratorSource) -> list[MemoryDraft]`.
- Produces: `JobRepository.claim(worker_id: str, lease_seconds: int) -> MemoryJob | None`, `complete(job_id, memory_ids)`, and `fail(job_id, error, retry_at)`.

- [ ] **Step 1: Write failing curator validation tests**

Assert the prompt requests only the seven allowed memory kinds; valid JSON becomes typed drafts; prose-wrapped, malformed, unknown-kind, oversized, low-confidence, empty, and secret-like candidates are rejected; speculative unselected ideas are omitted.

- [ ] **Step 2: Implement the curator**

Call `qwen2.5-coder:1.5b` with non-streaming structured output, low temperature, configurable timeout, and a hard input/output budget. Treat validation as application logic, never as model trust.

- [ ] **Step 3: Write failing job-queue integration tests**

Assert `FOR UPDATE SKIP LOCKED` allows one claimant, leases can be recovered after expiry, retries use bounded exponential backoff, a duplicate source event has one job, and the terminal-attempt limit is enforced.

- [ ] **Step 4: Implement durable claiming and transitions**

Use database time for lease comparisons. Persist compact error category/message without source content.

- [ ] **Step 5: Verify curator and queue behavior**

Run: `uv run pytest tests/unit/test_curator.py tests/integration/test_jobs.py -v`  
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/local_dev_rag/curator.py src/local_dev_rag/jobs.py src/local_dev_rag/repository.py tests
git commit -m "feat: add durable memory extraction jobs"
```

### Task 9: Worker processing and Chroma rebuild

**Files:**
- Create: `src/local_dev_rag/worker.py`
- Create: `src/local_dev_rag/cli.py`
- Modify: `src/local_dev_rag/repository.py`
- Modify: `src/local_dev_rag/vector_store.py`
- Create: `tests/integration/test_worker.py`
- Create: `tests/integration/test_reindex.py`

**Interfaces:**
- Consumes: curator, job queue, Ollama embedder, memory repository, and vector store.
- Produces: `WorkResult(state: Literal["idle", "completed", "retry", "failed"], job_id: UUID | None, memory_count: int)`.
- Produces: `Worker.run_once() -> WorkResult`, `Worker.run_forever(stop: asyncio.Event) -> None`, and CLI command `local-dev-rag reindex --project <external-id>`.

- [ ] **Step 1: Write failing worker success-path tests**

Assert a claimed job extracts drafts, stores accepted memories, embeds them in batches, upserts Chroma records linked to PostgreSQL IDs, and marks the job complete only after both stores succeed.

- [ ] **Step 2: Write failing worker failure-path tests**

Cover curator timeout, malformed curator output, embedder outage, Chroma outage, worker cancellation, and retry exhaustion. Assert the foreground path is unaffected and retryable state remains durable.

- [ ] **Step 3: Implement worker orchestration**

Make every stage idempotent. A repeat after PostgreSQL insert but before Chroma upsert must converge without duplicate memory items.

- [ ] **Step 4: Write failing reindex tests**

Delete a project's Chroma collection entries, run reindex, and assert all active PostgreSQL memories reappear with the current embedding version while superseded/rejected/deleted memories remain absent.

- [ ] **Step 5: Implement the reindex CLI**

Require an explicit project external ID. Batch embeddings and upserts; print counts only, never memory bodies.

- [ ] **Step 6: Verify worker and recovery behavior**

Run: `uv run pytest tests/integration/test_worker.py tests/integration/test_reindex.py -v`  
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/local_dev_rag/worker.py src/local_dev_rag/cli.py src/local_dev_rag/repository.py src/local_dev_rag/vector_store.py tests
git commit -m "feat: process and rebuild semantic memory"
```

### Task 10: Readiness, structured logging, and operational hardening

**Files:**
- Create: `src/local_dev_rag/logging.py`
- Modify: `src/local_dev_rag/api.py`
- Modify: `src/local_dev_rag/proxy.py`
- Modify: `src/local_dev_rag/worker.py`
- Modify: `compose.yaml`
- Create: `tests/unit/test_logging.py`
- Create: `tests/contract/test_readiness.py`
- Create: `tests/integration/test_degraded_modes.py`

**Interfaces:**
- Consumes: dependency `health()` methods from database, vector store, and Ollama adapters.
- Produces: `ReadinessReport(status: Literal["ready", "degraded", "not_ready"], dependencies: Mapping[str, DependencyStatus])`, `ReadinessService.check() -> ReadinessReport`, and `configure_logging(settings) -> None`.

- [ ] **Step 1: Write failing privacy-safe logging tests**

Capture logs for successful and failing requests. Assert request/project/session IDs, model, timing, retrieval count, retry count, and error category are present while prompt text, tool output, memory text, authorization values, and project-root contents are absent.

- [ ] **Step 2: Implement structured logging and request correlation**

Generate or accept a safe request ID, bind compact identifiers, and apply explicit redaction before serialization.

- [ ] **Step 3: Write failing readiness and degraded-mode tests**

Assert healthy dependencies return `200 ready`; Chroma, curator, or embedder degradation returns `200 degraded` because foreground passthrough works; PostgreSQL degradation returns `200 degraded` with memory disabled; Ollama unavailability returns `503 not_ready`.

- [ ] **Step 4: Implement readiness and final Compose health checks**

Keep `/healthz` process-only. Make Compose wait on PostgreSQL/Chroma startup health, run migrations before proxy/worker startup, and restart failed long-running services without exposing internal ports.

- [ ] **Step 5: Verify operational behavior**

Run: `uv run pytest tests/unit/test_logging.py tests/contract/test_readiness.py tests/integration/test_degraded_modes.py -v && docker compose config`  
Expected: all tests PASS and Compose renders four hardened services.

- [ ] **Step 6: Commit**

```bash
git add src/local_dev_rag compose.yaml tests
git commit -m "feat: harden local RAG operations"
```

### Task 11: End-to-end verification and user documentation

**Files:**
- Create: `tests/e2e/test_project_memory.py`
- Create: `scripts/smoke.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: every prior task's public interfaces and Compose services.
- Produces: one documented setup path and a repeatable smoke test.

- [ ] **Step 1: Write the end-to-end test against deterministic fakes**

Exercise: first session establishes a decision; worker curates and embeds it; second session in the same project retrieves it; a second project with the same query cannot retrieve it; changing the foreground model preserves memory behavior; a tool-call round trip remains valid.

- [ ] **Step 2: Run the end-to-end test and fix only integration defects**

Run: `uv run pytest tests/e2e/test_project_memory.py -v`  
Expected: PASS through the real proxy, PostgreSQL, and ChromaDB with fake Ollama responses.

- [ ] **Step 3: Add the optional live Ollama smoke script**

Check Docker, proxy readiness, Ollama reachability, required model presence, `/v1/models`, one non-streaming completion, one streaming completion, worker progress, same-project recall, and cross-project isolation. Exit with a non-zero status and a precise remedy for any failed prerequisite.

- [ ] **Step 4: Rewrite the README as an operator guide**

Document prerequisites, environment configuration, `docker compose up --build`, migrations, OpenCode launch/model selection, health interpretation, model budgets, backups, restore, Chroma reindex, logs, degraded modes, test commands, and troubleshooting. State clearly that unlimited history is durable storage plus bounded retrieval, not an infinite model context.

- [ ] **Step 5: Run the full verification gate**

Run: `uv run ruff check .`  
Expected: PASS.

Run: `uv run pyright`  
Expected: PASS with zero errors.

Run: `uv run pytest -v`  
Expected: all non-live tests PASS.

Run: `docker compose config`  
Expected: PASS with only the proxy port published to loopback.

Run: `opencode debug config`  
Expected: provider and plugin load without errors.

- [ ] **Step 6: Run the live smoke test when Ollama is available**

Run: `./scripts/smoke.sh`  
Expected: all five generation models are listed, the selected installed model streams a response, and project memory isolation passes. If Ollama is intentionally stopped, record this one verification as skipped rather than claiming it passed.

- [ ] **Step 7: Commit**

```bash
git add README.md scripts/smoke.sh tests/e2e/test_project_memory.py
git commit -m "docs: complete local RAG setup and verification"
```

## Final branch verification

- [ ] Confirm `git status --short` contains no unintended files.
- [ ] Run the full Task 11 verification commands again from a clean process state.
- [ ] Inspect `docker compose config` for accidental public database/vector ports or embedded secrets.
- [ ] Review the branch diff against the design spec, especially project isolation, tool-chain preservation, fallback behavior, and log redaction.
- [ ] Request a whole-branch code review before integration.
