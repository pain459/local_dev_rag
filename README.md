# Local OpenCode project memory

OpenCode keeps its interface, coding tools, permissions, sessions, and model picker. This local OpenAI-compatible proxy adds durable project memory to Ollama conversations. A new session can recall relevant decisions from earlier sessions in the same project, including after a foreground model change.

Durable unlimited history means storage that is independent of the model context window. It does **not** mean infinite model context, unlimited disk space, or perfect recall. Each inference request uses bounded recent history and a small selection of relevant memories. Source-code/documentation indexing and cross-project memory sharing are outside this version.

## Topology and prerequisites

OpenCode and Ollama run on the **host**; GPU inference stays outside Docker. Compose runs exactly four services:

| Service | Responsibility | Persistence / access |
| --- | --- | --- |
| `proxy` | Capture, retrieval, bounded context, OpenAI-compatible HTTP/SSE | Only published port: `127.0.0.1:8080` (configurable host port) |
| `worker` | Curate completed turns, embed accepted memories, durable retries | No host port |
| `postgres` | Authoritative events, memories, provenance, sessions, job queue | Named `postgres_data` volume; Compose network only |
| `chromadb` | Rebuildable semantic index | Named `chroma_data` volume; Compose network only |

OpenCode calls `http://localhost:8080/v1`. The proxy and worker reach host Ollama at `http://host.docker.internal:11434`; both have the Linux `host-gateway` mapping. They use `postgres:5432` and `chromadb:8000` inside Compose. Container proxy binding is `0.0.0.0:8080`, but host publication is loopback only. There is no authentication suitable for remote exposure: keep this a single-user local deployment.

The design targets OpenCode **1.18.30**, Ollama **0.35.0**, Docker **29.8.1**, and Compose **5.5.1**. The image uses Python 3.12; local verification requires Python 3.12, `uv`, and Node.js for the OpenCode plugin contract tests. Node.js **24.21.0** is the verified local test version; earlier versions have not been validated and no lower minimum is claimed. Integration/E2E tests require a running Docker daemon and the PostgreSQL/Chroma images. The optional smoke requires only Docker/Compose on the host; its Python checks run in the proxy image.

Ollama must listen on an address reachable from containers. On macOS, Docker Desktop supplies `host.docker.internal`. On Linux, a host Ollama listener bound only to `127.0.0.1` generally cannot accept traffic from the Docker gateway. Configure `OLLAMA_HOST` for a reachable host interface and restrict port 11434 with your host firewall; check connectivity from the proxy container. Setting `OLLAMA_URL=http://127.0.0.1:11434` inside Compose points at that container, not the host.

## First launch

On the host, start Ollama (`ollama serve` if your installed service is not already running), then install the five generation models and hidden embedder:

```sh
ollama pull qwen3-coder:30b
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
ollama list
```

The 30B default needs considerably more memory than the smaller models. Model presence is not proof that inference fits your machine. Select a smaller installed generation model if necessary.

From this checkout:

```sh
cp .env.example .env
# Edit .env for your machine before starting.
docker compose config --quiet
docker compose up --build -d --wait
docker compose ps
curl --fail http://127.0.0.1:8080/healthz
curl --silent --show-error http://127.0.0.1:8080/readyz
curl --fail http://127.0.0.1:8080/v1/models
```

`.env` is ignored by Git. The checked-in `local_rag` PostgreSQL password is a local example, not a production secret. Choose local credentials before initializing the volume. Leave `DATABASE_URL` blank: Compose passes the same raw `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` to PostgreSQL, proxy and worker, and application settings construct a correctly percent-encoded URL at runtime. Passwords containing `@`, `:`, or `/` work without manual URL construction. An explicit `DATABASE_URL` in `.env` overrides this construction in both apps; percent-encode its username/password and keep them consistent with PostgreSQL. For example, the synthetic password `demo@pass:word` encodes as `demo%40pass%3Aword`. Changing a password in `.env` does not rotate an already initialized PostgreSQL user's password. Keep credentials consistent with the existing database or rotate them explicitly.

`PROXY_PORT` changes the published host port only; update `opencode.json`'s provider `baseURL` to match. Compose deliberately overrides host `PROXY_HOST`/`ALLOW_REMOTE_BIND` for the internal container bind, defaults the database host to `postgres`, and fixes the vector URL to its service name. `OLLAMA_URL`, model roles, budgets, retrieval/ranking settings, retry settings, and `LOG_LEVEL` come from `.env`. Add `EMBEDDING_VERSION=1` to `.env` when you want to manage the index version explicitly.

Alembic migrations run **before** each proxy/worker process starts. PostgreSQL and Chroma must first be healthy; the worker then waits for the migrated proxy process health check, serializing the first migration without a fifth service. Verify the revision with:

```sh
docker compose exec -T proxy alembic current
```

For explicit migration maintenance, back up PostgreSQL, stop application writers, then run a one-off migrated image:

```sh
docker compose stop proxy worker
docker compose run --rm --no-deps proxy alembic upgrade head
docker compose up -d --wait proxy worker
```

The unreleased initial revision is `0001`. If you manually initialized a database from an earlier intermediate commit of this branch, that revision's schema changed during development; a matching revision string alone does not establish schema compatibility. The final schema includes a partial unique index for active project memory text and a `memory_sources` checkpoint table. Preserve a backup and explicitly migrate existing records/checkpoints and deduplicate/reindex vectors before using an intermediate database, or recreate only a disposable development database. Running `upgrade head` on an older `0001` alone does not install these changes.

## OpenCode configuration and launch

Launch OpenCode from this repository to load `opencode.json` and the automatically discovered `.opencode/plugins/rag-memory.js`:

```sh
opencode debug config
opencode models local-rag
opencode --model local-rag/qwen3-coder:30b
```

Use OpenCode's `/models` picker to switch among:

| Selectable generation ID | Default context / output limits |
| --- | --- |
| `qwen3-coder:30b` (default foreground) | 8192 / 2048 |
| `qwen2.5-coder:1.5b` | 8192 / 2048 |
| `qwen2.5-coder:7b` | 8192 / 2048 |
| `llama3.1:8b` | 8192 / 2048 |
| `qwen2.5:7b` | 8192 / 2048 |

For another coding repository, merge the `local-rag` provider and desired default model into that project's `opencode.json`, and copy `rag-memory.js` into that project's `.opencode/plugins/`. Preserve existing provider/plugin configuration. Run `opencode debug config` in that project and check both provider and plugin entries before launching. The databases stay in this Compose project; there is no need for a separate stack per coding repository.

The plugin uses the OpenCode 1.18.30 `chat.headers` hook only for provider `local-rag`. It supplies `x-opencode-project-id` and `x-opencode-session-id`, plus a percent-encoded diagnostic root. Project identity hashes OpenCode's project ID with a normalized `origin` remote, falling back to the canonical worktree/directory root. New sessions in the same identity share memory; another identity does not. Changing remote/project identity may create a separate memory namespace. The raw root never overrides the project ID. Direct API clients must supply both identity headers; missing/invalid identity returns HTTP 400. Local callers can choose their own headers, so project filtering is a data-isolation rule, not authentication against a malicious host user.

`CURATOR_MODEL=qwen2.5-coder:1.5b` performs background extraction; it is independently configurable. Although the same model is selectable for foreground chat, curator requests never appear as a separate picker entry. `EMBEDDING_MODEL=nomic-embed-text:latest` embeds queries and memories and is **never** a selectable generation model. Changing the foreground model leaves project identity and stored memory intact. The proxy rejects unknown generation IDs and never silently substitutes a model.

## Context, extraction, and retrieval limits

`MODEL_BUDGETS` in `.env.example` sets explicit per-model context, output reserve, and a 512-token safety reserve. The initial input target is therefore 5632 estimated tokens. The conservative estimator uses serialized UTF-8 bytes divided by three, rounded up; it is not an exact model tokenizer. Keep OpenCode's `limit.context`/`limit.output` in sync with proxy limits when changing budgets. Inspect your local Ollama metadata (`ollama show MODEL`) and runtime behavior before increasing them; model metadata and actual runtime context allocation can differ.

`MEMORY_TOKEN_BUDGET=1024`, `RETRIEVAL_CANDIDATE_LIMIT=20`, `RETRIEVAL_RESULT_LIMIT=6`, and `RETRIEVAL_MIN_SCORE=0.35` bound semantic work and injection. Ranking combines semantic distance, importance, recency, token overlap, and diversity. The default raw semantic floor is 0.2, or exact query-token overlap must support eligibility before ranking bonuses. Only active PostgreSQL-backed memories from the exact project qualify, even if Chroma contains stale records.

System instructions, tool definitions, the current user request, and output/safety reserves are protected. Recent turns stay verbatim while they fit. Assistant tool calls and their matching results remain complete atomic chains; protected incomplete/orphaned chains produce a validation error. Oversized mandatory input is preserved and honestly included in the estimate even when it exceeds the configured target; it may exceed the actual model window and fail upstream. Reduce a huge current request/tool payload or increase verified limits in that case. PostgreSQL-degraded passthrough can likewise exceed the target because nothing is trimmed on assumed durability. Older raw turns are trimmed only after durable PostgreSQL capture. Retrieved memories use a delimited historical-evidence system block with kind/date provenance, not new instructions.

The worker processes only successfully finalized assistant completions. Interrupted/cancelled/error streams keep diagnostic attempt state but create no completed-source curation job. Extraction accepts only supported requirements, decisions, constraints, preferences, errors, fixes, and outcomes; application validation rejects malformed, oversized, low-confidence, secret-like, or unsupported candidates. Conditional statements fail closed **as a whole** and contribute no evidence; separate unconditional statements may still support memory. A legitimate decision expressed inside a conditional statement can be omitted until restated unconditionally. A completed job may legitimately yield zero memories, so worker completion alone does not prove recall.

Evidence must be an assertion. Questions cannot confirm a fact. Accepted text must match an ordered source clause (ignoring case/spacing and normalizing supported negation contractions), or use a bounded selection paraphrase that preserves the selected subject and its local purpose/rationale. Arbitrary English paraphrases may be omitted; plain extractive diagnosed errors, successful fixes, and outcomes remain useful. Facts are never assembled from words in different statements.

Active memories with the same case/whitespace-normalized text share one canonical row per project, even if curator labels differ. Its original provenance is retained, and source checkpoints retain subsequent supporting events and independent job retries. Retrying an old source never revives a retired memory; a new source can reaffirm it as a new active row. This is exact normalized-text deduplication, not semantic equivalence of different wording. The canonical PostgreSQL ID is also its stable Chroma vector ID.

## Health and degraded operation

`/healthz` returns HTTP 200 `{"status":"ok"}` when the proxy process serves requests. `/readyz` checks six dependencies with bounded deadlines:

| Entry | What the check establishes |
| --- | --- |
| `postgres` | A database `SELECT 1` succeeds |
| `chromadb` | A valid Chroma heartbeat succeeds |
| `ollama` | A valid model catalog is reachable |
| `curator` | The configured curator model is present |
| `embedder` | The configured embedding model is present |
| `memory_jobs` | Durable failed/retrying job state is visible |

All healthy means HTTP 200 `ready`. Optional memory dependencies unavailable/degraded mean HTTP 200 `degraded`, while foreground chat can continue. Ollama unreachable means HTTP 503 `not_ready`. Probes validate model **presence**, not generation/embedding inference or available GPU/RAM. A missing selected foreground model still fails that request, even if `/readyz` is ready; `/v1/models` is the configured allowlist, not an installation/inference test.

Compose proxy health accepts a valid `ready`, `degraded`, or `not_ready` report after checking liveness. Worker health checks the actual worker process, database, and dependency report, without claiming a job; optional inference/index failure does not kill an otherwise recoverable worker. Thus `docker compose up --wait` can succeed while HTTP readiness is `not_ready`. Always inspect `/readyz` separately.

| Failure | Foreground behavior / recovery |
| --- | --- |
| PostgreSQL down | Full untrimmed Ollama passthrough; no claim that history was saved; restore database |
| Chroma down | Durable capture and recent bounded context continue; no semantic recall; restore/reindex |
| Query embedder down | No semantic retrieval; foreground chat/capture continue |
| Curator or worker embedder failure | Foreground unaffected; jobs retry with bounded exponential backoff |
| Repeated job failure | Terminal after `RETRY_MAX_ATTEMPTS=5`; durable failure stays visible |
| Ollama or selected model unavailable | OpenAI-compatible upstream error; no model substitution |
| Client disconnect/interrupted SSE | Diagnostic capture only; partial answers never curated |

Failed jobs keep `memory_jobs` degraded until the durable failure state is resolved, even after the dependency recovers. Retry settings default to 2 seconds initially and at most 300 seconds between attempts. Worker leases allow recovery after a process restart. Reindex can rebuild accepted memories but does not rerun failed extraction or clear failed jobs. There is no public job-retry/deletion API in this version; investigate failures before an explicitly scoped administrative database transition.

## Daily commands, logs, and privacy

```sh
docker compose ps
docker compose logs --tail=100 proxy worker
docker compose restart worker
docker compose stop
docker compose up -d --wait
docker compose down                 # preserves named volumes
```

Do **not** use `docker compose down --volumes` on your memory stack unless you intend to delete its durable data. Use a stable Compose project name/directory so you reconnect to the same volumes.

Structured logs report opaque request correlation, hashed external project/session identifiers, internal UUIDs, models, timings, retrieval/token estimates, attempts/retries, capture state, stream outcome, dependency flags, and controlled error categories. Prompt/source/tool/memory bodies, authorization, roots, arbitrary exception text, and raw access logs are suppressed, including at DEBUG. Database JSONB payloads retain exact conversation/tool content and may contain sensitive source material; local storage is not automatic redaction or encryption. Protect volumes and backups with host permissions and disk encryption. Candidate secret screening is conservative validation, not a guarantee that all sensitive data can be recognized.

Two telemetry limits matter during troubleshooting: indexing-stage worker logs currently label the combined persistence/embed/upsert span `embedder`, so a PostgreSQL failure in that span may be misattributed; rare raw iterator/cleanup `OSError` outside the scoped client-send boundary can have ambiguous cancellation/upstream categories. Standard HTTPX read/close, SSE errors, and client send disconnects are covered. Cross-check durable job state and readiness instead of inferring a root cause from a single category. These limits do not change capture safety or permit bodies in logs.

Inspect count-only durable job state without dumping content:

```sh
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT status, error_category, count(*) FROM memory_jobs GROUP BY status, error_category ORDER BY status;"'
```

## PostgreSQL backup and restore

PostgreSQL is the authority for projects, sessions, exact events, provenance, lifecycle, and processing state. Chroma is disposable derived state, not a substitute for a database backup. Schedule backups according to your own retention policy; no automatic backup is configured. Use a private backup directory outside this checkout, and verify restoration on a separate disposable deployment before relying on it:

```sh
umask 077
docker compose exec -T postgres sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > /path/to/private-backups/local-rag.dump
```

For restore, select the intended Compose project and database, take a fresh backup, and stop proxy/worker writers. The following command **replaces matching database objects/data** from the dump:

```sh
docker compose stop proxy worker
docker compose exec -T postgres sh -c 'exec pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --exit-on-error' < /path/to/private-backups/local-rag.dump
docker compose run --rm --no-deps proxy alembic upgrade head
# Reindex every restored project before relying on semantic recall (see below).
docker compose up -d --wait proxy worker
```

Verify database revision, event/memory/job counts, readiness, and scoped recall after restoration. Existing Chroma data may be stale relative to the restored snapshot; rebuild each restored project from active PostgreSQL memories.

## Reindex and embedding changes

Find external project IDs without reading memory text:

```sh
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT external_project_id FROM projects ORDER BY created_at;"'
docker compose run --rm --no-deps proxy local-dev-rag reindex --project '<exact-external-project-id>'
```

Reindex deletes that project's entries in the **current** collection, reads only active PostgreSQL memories, re-embeds them in bounded batches, revalidates/locks rows before upsert, and prints counts only (`reindexed=N`). Keep PostgreSQL/Chroma/Ollama available. Stopping foreground/worker writers during maintenance reduces concurrent changes and recall gaps. Unknown projects or dependency failure exit nonzero with content-free output.

After changing `EMBEDDING_MODEL`, updating a model under the same name, or changing embedding/index semantics, increment `EMBEDDING_VERSION`, pull the model, recreate proxy/worker to load the settings, and reindex **every** project. Default collections include embedding version and a digest of the model name (`local_dev_rag_memory_v<VERSION>_<digest>`). This avoids incompatible vector dimensions sharing one Chroma collection. A new configuration starts with an empty index; existing memories do not become searchable there until rebuilt. Old collections remain dormant and consume disk until explicitly retired after backup/verification; reindex does not delete them. Collection-name overrides are available to programmatic tests/operators, not a documented environment setting.

## Verification and optional live smoke

```sh
uv sync --all-groups
uv run ruff check .
uv run pyright
uv run pytest -v
uv run pytest tests/e2e/test_project_memory.py -v
docker compose config
opencode debug config
opencode models local-rag
./scripts/smoke.sh
```

The default suite performs no live model inference: disposable PostgreSQL/Chroma containers and deterministic fake Ollama responses exercise actual proxy HTTP/SSE, capture, curation, indexing, fresh-session recall, all five foreground model switches, strict project isolation, and tool-call/result round trips. Docker fixtures stop their own temporary containers; they never use your Compose volumes. Test output/cache is not application data.

The optional smoke uses the currently selected Compose project (`COMPOSE_PROJECT_NAME` is honored) and its proxy image. It checks Docker, four running services, loopback publication, liveness, six healthy dependencies, container-to-host Ollama reachability, all five generation models plus curator/embedder installation, and the exact exposed generation catalog. It performs one non-streaming completion, waits for durable worker completion and an accepted synthetic memory, then checks streaming recall in a fresh same-project session and unrelated-project isolation in both response and vector search. It prints only statuses/counts, never generated replies or memory text. It writes two unique synthetic project namespaces into the local database and retains those records; it does not delete operator data.

```sh
SMOKE_MODEL=qwen2.5-coder:7b SMOKE_TIMEOUT_SECONDS=600 ./scripts/smoke.sh
```

The smoke defaults to the small installed `qwen2.5-coder:1.5b` foreground model and a 300-second worker deadline. All five models must be installed/listed, but only the selected model runs foreground inference; the curator and embedder also run actual inference. The deterministic E2E covers tool transport; live smoke does not claim every model can execute OpenCode's tools. Any prerequisite or behavioral failure exits nonzero with a remedy. If Ollama is unavailable, record live verification as skipped rather than interpreting process health or deterministic tests as a live pass.

## Troubleshooting

- **HTTP 400 identity/model/history errors:** verify plugin loading, provider ID `local-rag`, both identity headers, a configured generation ID, and complete tool chains. Keep budgets aligned with OpenCode.
- **Proxy process healthy but `/readyz` 503:** check host Ollama, container reachability, listener/firewall, and `OLLAMA_URL`; restart/recreate proxy and worker after environment changes.
- **Models listed but HTTP 502 or inference timeout:** `/v1/models` is a static allowlist. Pull the requested model, test `ollama run MODEL` on the host, select a smaller model, and check memory/timeout pressure. Readiness model presence is not inference validation.
- **Chat works but no older recall:** check `postgres`, `chromadb`, `embedder`, and `memory_jobs`; verify stable project identity, completed jobs and active-memory counts, extraction evidence, relevance thresholds and memory budget. Rebuild after embedding changes. Empty curation is possible by policy.
- **Worker retry/failed counts rise:** inspect safe error categories and real inference capability. Restore the affected dependency. Completed extraction with zero candidates differs from failed processing; terminal jobs require explicit administrative recovery.
- **Startup migration/database failure:** check credentials against the initialized volume, PostgreSQL health, installed revision/schema, and app image build. Do not erase volumes to fix a configuration mismatch.
- **OpenCode provider/plugin missing:** run `opencode debug config` in the coding project; install both project artifacts, check OpenCode 1.18.30 compatibility, and update the base URL after changing the host proxy port.
