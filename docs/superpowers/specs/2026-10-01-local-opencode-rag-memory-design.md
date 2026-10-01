# Local OpenCode RAG Memory Design

**Date:** 2026-10-01  
**Status:** Approved design  
**Target environment:** OpenCode 1.18.30, Ollama 0.35.0, Docker 29.8.1, Docker Compose 5.5.1

## 1. Purpose

Build a fully local, OpenCode-style coding agent with durable project memory. OpenCode remains the user interface, coding agent, tool runner, and model picker. Ollama remains the model runtime. A transparent OpenAI-compatible proxy adds long-lived conversation memory without requiring an ever-growing model context window.

Success means:

- OpenCode can select any configured local generation model.
- Conversations and tool events can grow without a storage limit imposed by a model context window.
- Each model request receives recent context plus only relevant older project memories.
- Memory survives OpenCode, proxy, and container restarts.
- No conversation memory crosses project boundaries.
- Foreground coding remains usable when the optional semantic-memory path is degraded.

## 2. Scope

### 2.1 Included in version 1

- An OpenAI-compatible streaming chat proxy between OpenCode and Ollama.
- A project-level OpenCode plugin that attaches stable session and project metadata to model requests.
- Project-scoped memory shared by all OpenCode sessions for the same repository.
- Exact conversation and processing records in PostgreSQL.
- Semantic memory vectors in ChromaDB.
- Asynchronous structured memory extraction with `qwen2.5-coder:1.5b`.
- Embedding generation with `nomic-embed-text:latest`.
- Docker Compose services for the proxy, worker, PostgreSQL, and ChromaDB.
- Configuration for all installed generation models in OpenCode.
- Health, degraded-mode behavior, migrations, tests, and local operating documentation.

### 2.2 Explicitly excluded from version 1

- Source-code or documentation indexing.
- Memory sharing between unrelated projects.
- Remote or multi-user deployment.
- A new graphical or terminal user interface.
- Replacement of OpenCode's coding tools, repository context, or permission system.
- Authentication intended for exposure beyond localhost.
- A management UI for browsing or editing memories.

## 3. Deployment topology

OpenCode and Ollama run directly on the host. GPU-backed model execution therefore remains outside Docker.

Docker Compose runs four services:

1. `proxy`: OpenAI-compatible API, request normalization, persistence, retrieval, context construction, and Ollama streaming.
2. `worker`: durable asynchronous memory extraction and embedding jobs.
3. `postgres`: authoritative relational state and job queue.
4. `chromadb`: rebuildable semantic vector index.

Only the proxy port is published, bound to `127.0.0.1`. PostgreSQL and ChromaDB are reachable only on the Compose network. The proxy and worker reach Ollama through `host.docker.internal`; the Compose configuration adds the Linux `host-gateway` mapping while remaining compatible with macOS Docker Desktop.

Persistent named volumes hold PostgreSQL and ChromaDB data.

## 4. Component boundaries

### 4.1 OpenCode

OpenCode owns:

- the TUI and session experience;
- repository tools and tool execution;
- permission prompts;
- the selected generation model;
- the live agent loop.

OpenCode is not forked or modified.

### 4.2 OpenCode memory plugin

A small project plugin uses the installed OpenCode `chat.headers` hook. The hook has access to `sessionID`, while plugin initialization has access to the project and working directory.

It adds these headers only for the local RAG provider:

- `x-opencode-session-id`: stable OpenCode session identifier;
- `x-opencode-project-id`: stable project identifier;
- `x-opencode-project-root`: normalized project root when safe and useful for local diagnostics;
- `x-opencode-request-kind`: foreground by default, allowing auxiliary requests to be distinguished when the installed OpenCode API exposes that information.

The project ID is derived from the normalized Git remote plus repository identity when available. A hash of the normalized repository root is the fallback. The raw root is not used as the database primary key.

The plugin does not perform retrieval, persistence, prompt construction, or model calls.

### 4.3 RAG proxy

The proxy is a Python 3.12 FastAPI application. It owns:

- OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints;
- streaming and non-streaming Ollama forwarding;
- request validation and model allowlisting;
- idempotent event persistence;
- retrieval and ranking;
- bounded context construction;
- finalized response capture;
- creation of durable memory-extraction jobs;
- health and readiness endpoints.

The proxy preserves tool definitions, tool calls, sampling settings, stop conditions, response format options supported by Ollama, and the selected model ID.

### 4.4 Memory worker

The worker runs the same application package with a separate process command. It claims jobs from PostgreSQL using row locking so multiple workers can be added later.

For each finalized conversational unit, the worker:

1. builds a bounded extraction input;
2. calls `qwen2.5-coder:1.5b` with a strict structured-output prompt;
3. validates the returned JSON against the memory schema;
4. rejects empty, malformed, low-confidence, or non-durable candidates;
5. stores accepted memories in PostgreSQL;
6. embeds accepted memory text using `nomic-embed-text:latest`;
7. upserts vectors and metadata into ChromaDB;
8. records completion, retry, or terminal failure in PostgreSQL.

The curator and embedding model names are configurable through environment variables.

### 4.5 PostgreSQL

PostgreSQL is the source of truth for exact events and derived memory records. It owns ordering, uniqueness, processing state, provenance, supersession, and retry state.

ChromaDB can be rebuilt entirely from PostgreSQL and the configured embedding model.

### 4.6 ChromaDB

ChromaDB stores embeddings for accepted memory items plus filtering metadata. It never becomes the only copy of conversation text or memory provenance.

Every vector record contains a PostgreSQL memory ID and project ID. All searches apply an exact project filter before results are eligible for injection.

## 5. Supported models

The OpenCode provider exposes these selectable generation models:

- `qwen3-coder:30b`
- `qwen2.5-coder:1.5b`
- `qwen2.5-coder:7b`
- `llama3.1:8b`
- `qwen2.5:7b`

`nomic-embed-text:latest` is not shown as a chat model because it is used only by the memory pipeline.

The default generation model is `qwen3-coder:30b`. Model context and output limits are explicit configuration values, not assumptions embedded in the context builder. The initial configuration uses conservative values verified against the local Ollama model metadata during implementation. The proxy rejects unknown model IDs rather than forwarding arbitrary names.

## 6. Data model

The initial relational model contains:

### 6.1 `projects`

- internal UUID;
- stable external project hash;
- optional normalized remote identity;
- optional diagnostic root label;
- creation and update timestamps.

### 6.2 `sessions`

- internal UUID;
- project foreign key;
- OpenCode session ID;
- creation and last-seen timestamps;
- optional session title;
- unique constraint on project and OpenCode session ID.

### 6.3 `conversation_events`

- internal UUID;
- project and session foreign keys;
- source message ID when provided;
- request ID and deterministic content hash;
- event type and role;
- ordered content payload stored as JSONB;
- selected model;
- creation timestamp;
- uniqueness fields for retry-safe insertion.

Events represent user messages, assistant messages, tool calls, tool results, system-visible conversation elements, and proxy processing boundaries without flattening away structure.

### 6.4 `memory_items`

- internal UUID;
- project foreign key;
- source session and event references;
- kind: requirement, decision, constraint, preference, error, fix, or outcome;
- concise memory text;
- confidence and importance;
- lifecycle state: active, superseded, rejected, or deleted;
- optional superseding memory reference;
- curator and embedding model identifiers;
- embedding version;
- creation and update timestamps.

### 6.5 `memory_jobs`

- internal UUID;
- project, session, and source-event references;
- job kind and deduplication key;
- status, attempt count, and next-attempt timestamp;
- lease owner and lease expiry;
- compact error category and message;
- creation, start, and completion timestamps.

## 7. Foreground request flow

1. OpenCode sends a chat-completions request to the proxy with the plugin headers.
2. The proxy validates the project, session, model, and request shape.
3. The proxy normalizes the incoming message sequence and idempotently persists previously unseen events.
4. It builds a retrieval query from the current user request and current task context.
5. It embeds the query through Ollama using `nomic-embed-text:latest`.
6. It searches ChromaDB with an exact project filter and a configurable candidate limit.
7. It reranks candidates using semantic similarity, importance, recency, lifecycle state, and duplication penalties.
8. It builds a bounded outbound context.
9. It forwards the selected model request to Ollama and streams the response to OpenCode without buffering the full response first.
10. While streaming, it captures response chunks for final persistence.
11. On successful completion, it stores the finalized assistant event and creates one deduplicated memory job.
12. On cancellation or stream failure, it records an incomplete attempt but does not curate it as a completed memory source.

## 8. Context construction policy

Unlimited chat means unlimited durable history, not an unlimited model context.

The context builder uses a configurable token budget per generation model and applies these invariants:

1. Preserve OpenCode system instructions and tool definitions.
2. Preserve the current user request.
3. Preserve recent conversation turns verbatim while they fit.
4. Treat each assistant tool call and its tool result as an atomic dependency chain; never retain only one side.
5. Reserve capacity for model output and a safety margin.
6. Add only relevant, active project memories within the remaining retrieval budget.
7. Remove older raw turns from the outbound request only after their exact representation is durable in PostgreSQL.

Retrieved memories are injected as a clearly delimited system context block. The block states that memories are historical evidence, not new instructions, and includes compact provenance such as kind and date. Memories are deduplicated and ordered coherently before injection.

The first implementation uses deterministic configurable budget partitions and a conservative token estimator. Exact tokenizer integration can replace the estimator without changing the context-builder interface.

## 9. Memory extraction policy

The curator extracts only durable information likely to matter in future coding turns:

- confirmed requirements;
- architecture or implementation decisions and their rationale;
- user preferences and project constraints;
- important failures and diagnosed causes;
- fixes that worked;
- completed task outcomes and remaining follow-ups.

It must not store:

- transient greetings or conversational filler;
- speculative ideas presented but not selected;
- raw secrets or credentials;
- huge source or tool outputs;
- duplicate restatements;
- instructions found inside retrieved memory as if they were current user commands.

The structured result includes memory kind, concise text, confidence, importance, source references, and an optional supersedes hint. Application validation is authoritative; the curator does not write directly to either database.

## 10. Retrieval policy

Retrieval is project-scoped in version 1. A new session in the same repository may recall relevant memories from older sessions. No query searches across project IDs.

Ranking combines:

- Chroma semantic distance;
- stored importance;
- recency decay;
- exact keyword overlap for identifiers and error strings;
- lifecycle state;
- diversity and duplicate suppression.

The selected memories must fit the configured memory token budget. Low-scoring candidates are omitted; the system prefers no historical memory over irrelevant memory.

## 11. API behavior

### 11.1 `GET /v1/models`

Returns only the five configured generation models in OpenAI-compatible form.

### 11.2 `POST /v1/chat/completions`

Supports streaming and non-streaming requests. The first release targets the subset emitted by OpenCode and accepted by Ollama, including tools and tool-choice fields.

Unsupported fields produce a clear validation error only when silently dropping them could change behavior. Safe pass-through fields are forwarded.

### 11.3 Operational endpoints

- `GET /healthz`: process liveness.
- `GET /readyz`: dependency state with healthy or degraded detail.

Administrative reindex and deletion operations are command-line maintenance commands in version 1 rather than public HTTP endpoints.

## 12. Failure behavior

### 12.1 ChromaDB unavailable

The proxy continues with recent context and no semantic recall. It reports degraded readiness and logs a structured dependency error.

### 12.2 PostgreSQL unavailable

The proxy becomes a plain Ollama passthrough. It does not trim based on assumed persistence, does not claim memory was saved, and reports degraded readiness.

### 12.3 Curator or embedding failure

The foreground response is unaffected. The job remains retryable with bounded exponential backoff. Repeated terminal failures are visible in health diagnostics and structured logs.

### 12.4 Ollama unavailable or model missing

The proxy returns a clear OpenAI-compatible upstream error. It does not substitute another model silently.

### 12.5 Client cancellation or interrupted stream

The proxy cancels the upstream request where possible, stores diagnostic state without treating a partial answer as completed memory, and releases resources promptly.

## 13. Security and privacy

- Published ports bind only to `127.0.0.1`.
- PostgreSQL and ChromaDB are not published to the host by default.
- No cloud API or telemetry service is required.
- Prompt, source, and tool-result bodies are excluded from ordinary logs.
- Logs include identifiers, timings, token estimates, model names, retry counts, and error categories.
- Secrets found in conversation data are not intentionally promoted into memory items.
- Configuration uses environment variables and a checked-in `.env.example`; real secrets are not committed.
- Database payloads remain local and are protected by host and Docker volume permissions.

## 14. Observability and operations

Structured logs carry request ID, project ID, session ID, model, duration, retrieval count, injected-memory token estimate, and degraded-state flags.

Compose health checks cover PostgreSQL, ChromaDB, proxy liveness, proxy readiness, and the worker process. Startup waits for durable dependencies to become available but handles dependencies that fail after startup.

Alembic owns PostgreSQL schema migrations. A maintenance command can rebuild a project's Chroma collection from active PostgreSQL memory items when the embedding model or index schema changes.

## 15. Verification strategy

### 15.1 Unit tests

- project and session identity derivation;
- request/event normalization and deduplication;
- token budgeting;
- preservation of tool-call/result chains;
- ranking, supersession, and duplicate suppression;
- structured curator validation;
- context-block formatting and prompt-injection boundaries;
- retry and lease behavior.

### 15.2 Contract tests

- OpenAI-compatible model listing;
- streaming and non-streaming chat completions;
- tool-call request and response shapes;
- cancellation and upstream errors;
- unknown-model rejection;
- header propagation from a fixture plugin request.

### 15.3 Integration tests

Integration tests run against temporary PostgreSQL and ChromaDB containers and a deterministic fake Ollama server. They cover persistence, retrieval, job processing, restarts, dependency degradation, and Chroma rebuilds.

### 15.4 Local end-to-end smoke tests

An optional smoke test uses the installed Ollama models and OpenCode configuration. It verifies model selection, streaming, tool use, cross-session recall within one project, and isolation between two project fixtures.

## 16. Configuration artifacts

The repository will provide:

- a project `opencode.json` that defines the local proxy provider and five generation models;
- a project OpenCode plugin under `.opencode/plugins/` compatible with OpenCode 1.18.30;
- `compose.yaml` for the four container services and volumes;
- `.env.example` documenting ports, database settings, model names, token budgets, retrieval limits, and retry settings;
- migration and maintenance commands;
- a README with setup, health checks, smoke tests, troubleshooting, backup, restore, and reindex instructions.

## 17. Future-compatible boundaries

The following extensions are intentionally enabled but not implemented:

- replacing ChromaDB with `pgvector` behind the vector-store interface;
- adding source-code and documentation collections;
- explicit cross-project memory sharing with access rules;
- remote PostgreSQL and horizontally scaled workers;
- a memory inspection and correction interface;
- alternate curator models or rule-based extraction stages.

These changes do not require OpenCode to change its provider contract or the proxy to change its public OpenAI-compatible endpoints.

