# Local OpenCode project memory: user guide

This guide is for a developer setting up this project for the first time. It explains how to run OpenCode with downloaded Ollama models, keep project decisions across conversations, and distinguish saved sessions from retrieved memory. For database backups, migrations, reindexing, failure recovery, and destructive maintenance, use the [operations guide](operations-guide.md). The [README](../README.md) is the shorter entry point.

Commands below run from the `local_dev_rag` checkout unless a different directory is stated. Replace `/path/to/...` and `ses_REPLACE_WITH_REAL_ID` with your own values. HTTP examples assume the default host port, `8080`; change those URLs if you change `PROXY_PORT`.

## Contents

- [What this project adds](#what-this-project-adds)
- [Five-minute path](#five-minute-path)
- [Host assumptions and prerequisites](#host-assumptions-and-prerequisites)
- [Clone, configure, and start](#clone-configure-and-start)
- [Models, capacity, and runtime context](#models-capacity-and-runtime-context)
- [Setup test ladder](#setup-test-ladder)
- [Everyday use and model switching](#everyday-use-and-model-switching)
- [Saved sessions, exports, and RAG memory](#saved-sessions-exports-and-rag-memory)
- [Choosing this project or Claude Code](#choosing-this-project-or-claude-code)
- [Troubleshooting decision tree](#troubleshooting-decision-tree)
- [Limitations and safe next steps](#limitations-and-safe-next-steps)

## What this project adds

OpenCode is the coding interface: it owns the terminal UI, coding tools, permissions, model picker, and saved conversations. This project's `local-rag` provider sends requests through a local proxy to host Ollama. A plugin identifies the project and session. The proxy saves conversation events to PostgreSQL, and a background worker extracts supported facts and embeds accepted memories into Chroma. A later request can retrieve a small amount of relevant evidence from earlier conversations in the same project.

| Component | Runs where | What you should expect |
| --- | --- | --- |
| OpenCode | Host | Interactive coding and its own session history |
| Ollama | Host | Generation, background curation, and embedding; host GPU access |
| `proxy` | Docker | OpenAI-compatible endpoint and bounded context assembly |
| `worker` | Docker | Background extraction, embedding, and durable retry jobs |
| `postgres` | Docker | Authoritative events, memories, provenance, and processing state |
| `chromadb` | Docker | Rebuildable semantic index of accepted memories |

The stack publishes only `127.0.0.1:8080`. PostgreSQL and Chroma have no published host ports. Keep this a single-user deployment: the proxy has no authentication suitable for an untrusted network. Do not publish these services through a tunnel or change the proxy publication to a public interface.

“Durable history” means history can outlive a model's context window and an OpenCode session. It does not mean infinite context, unlimited disk space, or guaranteed recall. By default each request can inject at most six relevant memories within a 1024-token memory budget, alongside bounded recent conversation. A completed turn can produce no accepted memories. Store requirements that must never be missed in a reviewed repository document too.

## Five-minute path

This is a short path once host tools, images, and models are installed. First-time downloads and a full smoke test can take substantially longer than five minutes.

```sh
git clone https://github.com/pain459/local_dev_rag.git
cd local_dev_rag
make help
make precheck
# Start the installed Ollama app/service first, or use ollama serve in another terminal.
make essentials
# Review the newly created .env before initializing the database.
docker compose config --quiet
make up
make doctor
make ready
opencode --model local-rag/qwen2.5-coder:7b
```

Healthy outcome: four running services, `doctor` reports a loaded provider/plugin and installed models, and `ready` passes both HTTP probes with six healthy dependencies. In OpenCode, ask it to reply with a short greeting without using tools. After the reply, run `ollama ps` in another terminal and verify the context allocation for the selected model. Follow the [test ladder](#setup-test-ladder) to verify memory, tests, and the default Qwen3 model before relying on the setup.

If the tools are already configured and the stack is running, start at `make doctor`. `make essentials` preserves an existing `.env`, but it does download/update assets and build images.

## Host assumptions and prerequisites

The operator scripts target macOS and Linux. Windows/WSL is not a verified operator environment for this checkout. The reference host is an Apple M5 Max with 36 GB unified memory; it establishes a working example, not a minimum requirement or a guarantee that a 30B model with 64K context fits every 36 GB machine.

Install these host tools yourself before `make precheck`; Make never installs them:

| Requirement | Purpose and verified baseline | Installation reference |
| --- | --- | --- |
| Git, `make`, POSIX shell, `curl` | Clone, operate, and inspect HTTP; scripts use `/bin/sh` | Use your OS packages/developer tools |
| Docker with running daemon and Compose plugin | Four-service stack and disposable test containers; Docker 29.8.1 / Compose 5.5.1 | [Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [Linux](https://docs.docker.com/engine/install/) |
| Ollama | Host inference server; 0.35.0 | [Ollama download](https://ollama.com/download) |
| OpenCode | Coding client; 1.18.30, including this version's plugin hook | [OpenCode installation](https://opencode.ai/docs/) |
| Python 3.12 | Required local interpreter (`>=3.12,<3.13`); image also uses 3.12 | [Python downloads](https://www.python.org/downloads/) |
| `uv` | Locked Python dependencies, Ruff, Pyright, pytest | [uv installation](https://docs.astral.sh/uv/getting-started/installation/) |
| Node.js | JavaScript plugin contract tests; verified 24.21.0 | [Node.js download](https://nodejs.org/en/download) |

Other versions are not certified by this project; no lower Node.js minimum is claimed. Recheck provider/plugin resolution and tests after an OpenCode upgrade. You also need network access for initial model/package/image downloads, sufficient disk for all downloaded models and growing database volumes, and enough free RAM/GPU memory for foreground, curator, embedder, context caches, Docker, and your development tools. Download size alone is not the loaded inference footprint. There is no universal RAM or disk minimum established by this checkout.

Ollama must be reachable both from the host CLI and from containers. Docker Desktop supplies `host.docker.internal` on macOS; Compose adds a Linux `host-gateway` mapping too. A Linux Ollama server listening only on host loopback generally cannot accept Docker-gateway traffic. Select a host interface reachable from Docker and restrict port `11434` with a firewall before changing the listener. See [Ollama's server environment instructions](https://docs.ollama.com/faq#how-do-i-configure-ollama-server). Do not expose an unauthenticated Ollama listener to an untrusted network.

## Clone, configure, and start

### 1. Check the host

After cloning as above:

```sh
make precheck
docker info
docker compose version
python3.12 --version
node --version
opencode --version
```

`precheck` is read-only: it checks tools and versions, the Docker daemon, rendered configuration, provider/plugin files, and port availability. An already running local proxy on the selected port is acceptable. It does not certify installed models or live inference. Fix its `FAIL` remedies before continuing; Python must be 3.12.

### 2. Prepare assets and review configuration

Start the installed Ollama app/service. If there is no existing server, run this in another terminal and leave it running:

```sh
ollama serve
```

Then:

```sh
make essentials
```

This creates `.env` with private permissions only when absent, runs `uv sync --all-groups`, pulls PostgreSQL/Chroma images, builds the app image, and pulls the five generation models plus configured curator/embedder. It never overwrites `.env` or credentials. It never starts the Compose stack. `make doctor-fix` is the repair variant: it pulls missing images/models, syncs/builds, and runs a final read-only diagnosis. That diagnosis can fail when the stack is stopped; finish host configuration and run `make up`.

Read [.env.example](../.env.example), then edit your private `.env` using your editor. Do this before the first `make up` initializes PostgreSQL.

| Setting | What to configure |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Replace the local example credentials before first initialization. Keep `.env` out of Git. Changing these later does not rotate credentials in an existing database. |
| `DATABASE_URL` | Leave blank so both apps construct an encoded URL from raw `POSTGRES_*`. Explicit overrides must percent-encode reserved credential characters and match PostgreSQL. |
| `OLLAMA_URL` | Container address, default `http://host.docker.internal:11434`. Container `localhost` points to the container itself. |
| `PROXY_PORT` | Host publication, default `8080`. If changed, also edit `provider.local-rag.options.baseURL` in [opencode.json](../opencode.json). Keep host publication on loopback. |
| `DEFAULT_MODEL` | Model registry setting; the HTTP chat endpoint still requires an explicit `model`. OpenCode's initial choice is separately set by `opencode.json`'s `model`. |
| `CURATOR_MODEL`, `EMBEDDING_MODEL` | Background roles; require explicit tags such as `nomic-embed-text:latest`. The embedder is not a chat model. |
| `MODEL_BUDGETS` | Per-model context/output/safety limits; synchronize OpenCode limits and verified runtime allocation before changing them. |
| `UPSTREAM_TIMEOUT_SECONDS` | Each upstream request timeout, default `120`; slow inference can require more time. |

Keep host CLI `OLLAMA_HOST` pointed at the same server that containers reach through `OLLAMA_URL`. These are different settings. Compose overrides the container's bind address to `0.0.0.0:8080` so the service can accept container traffic; the host port stays loopback. Render safely without displaying credentials:

```sh
docker compose config --quiet
```

Use a stable checkout directory and Compose project name to reconnect to the same named volumes. If you choose a `COMPOSE_PROJECT_NAME`, export that same value for every operator command. An accidental project-name change can look like lost memory while selecting different volumes.

### 3. Start and diagnose

```sh
make up
make status
make doctor
make ready
```

`make up` starts services and waits for Compose health; use `make recreate` to rebuild and force recreation after app or `.env` changes. Startup automatically runs Alembic before the proxy and worker. The worker waits for the migrated proxy's process health. `doctor` diagnoses installed models, OpenCode provider/plugin resolution, four running services, liveness/readiness, database revision, and worker health without running inference or repairing anything.

Useful direct equivalents with default Docker/Compose settings:

```sh
docker compose up -d --wait --wait-timeout 120
docker compose ps
docker compose exec -T proxy alembic current
curl --fail --silent --show-error http://127.0.0.1:8080/healthz
curl --silent --show-error http://127.0.0.1:8080/readyz
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

Expected revision: `0001 (head)` for this unreleased initial schema. An older intermediate checkout also used `0001`; the revision string alone does not prove compatibility with that intermediate schema. Consult the operations guide before reusing old data.

`/healthz` should return `{"status":"ok"}`. `/readyz` should say `ready` with `postgres`, `chromadb`, `ollama`, `curator`, `embedder`, and `memory_jobs` all `healthy`. HTTP 200 `degraded` means chat may work with reduced memory functionality. HTTP 503 `not_ready` means Ollama is unreachable. Compose health accepts valid degraded/not-ready reports, so `up --wait` can succeed before usable inference. `make ready` requires full readiness. The model endpoint is an allowlist, not proof that a model is installed or fits in memory.

## Models, capacity, and runtime context

The default installation pulls these exact IDs. To install or inspect them directly:

```sh
ollama pull qwen3-coder:30b
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
ollama list
```

| OpenCode selector | Configured context / output / safety tokens | Practical role |
| --- | --- | --- |
| `local-rag/qwen3-coder:30b` | 65536 / 8192 / 4096 | Default foreground coding model; largest of this catalog |
| `local-rag/qwen2.5-coder:1.5b` | 32768 / 4096 / 2048 | Small foreground option; also the default background curator |
| `local-rag/qwen2.5-coder:7b` | 32768 / 4096 / 2048 | Smaller coding option for initial interactive checks |
| `local-rag/llama3.1:8b` | 65536 / 8192 / 4096 | Alternative generation model |
| `local-rag/qwen2.5:7b` | 32768 / 4096 / 2048 | Alternative generation model |
| No selector: `nomic-embed-text:latest` | Embedding role | Hidden from the generation picker; query/memory embeddings only |

The table describes configured budgets, not demonstrated model quality or hardware fit. `tool_call: true` in OpenCode enables the transport; it is not a benchmark showing all five models reliably complete coding tasks. Try a small, reviewable task before trusting a model with a large change. The default curator is independently configurable and can omit valid facts; stronger curation may cost more memory and latency.

### Verify the actual allocation

There are three different limits: advertised model capacity (`ollama show`), OpenCode/proxy policy (`opencode.json` and `MODEL_BUDGETS`), and the loaded Ollama runtime (`ollama ps`). All must support the selected policy. This proxy forwards OpenAI-compatible chat requests and does not set native Ollama `num_ctx`. Declaring 65536 in project configuration does not allocate it in Ollama. See [Ollama context length](https://docs.ollama.com/context-length) and [OpenAI compatibility context settings](https://docs.ollama.com/api/openai-compatibility#setting-the-local-context-size).

Reference-host observation: a real Qwen3 OpenCode request succeeded on the Apple M5 Max / 36 GB host while `ollama ps` showed `qwen3-coder:30b` at **32768**, although the project declared **65536**. That short prompt proves basic inference, not long-request safety at 64K. Check your own allocation; do not infer it from host RAM or this observation.

Ollama documents memory-dependent defaults and warns that increasing context increases memory use. For a manually served Ollama instance, stop its existing process cleanly first, then start the same reachable listener with a matching context setting:

```sh
OLLAMA_CONTEXT_LENGTH=65536 ollama serve
```

For an app or system service, configure the setting in that service and restart it; setting a variable in a client terminal does not reconfigure an already running daemon. The app also offers a context slider. Follow the [Ollama environment setup](https://docs.ollama.com/faq#how-do-i-configure-ollama-server) appropriate to your host. Preserve the listener/firewall setup that allows container access. Do not run two servers on the same port.

After restarting Ollama, make a real request through this checkout's provider, then check immediately while the model remains loaded:

```sh
ollama show qwen3-coder:30b
opencode run --model local-rag/qwen3-coder:30b \
  'Without using tools, reply exactly CONTEXT_CHECK_OK.'
ollama ps
```

Acceptance check: the `qwen3-coder:30b` row has `CONTEXT` at least `65536`; also inspect `PROCESSOR` for GPU/CPU offloading. A missing row means the model is not currently loaded, so run a fresh prompt and check again. Repeat with `llama3.1:8b` before relying on its 64K policy, and verify at least `32768` for the three 32K models. Do not count `ollama show` metadata as the runtime acceptance check.

If a 64K allocation does not fit, select a smaller model, reduce the task payload, or lower that model's project policy to a verified capacity. For a 32K Qwen3/Llama policy, edit **both** `opencode.json` (`context=32768`, `output=4096`) and the corresponding `.env` `MODEL_BUDGETS` entry (`context_tokens=32768`, `output_tokens=4096`, `safety_tokens=2048`), keeping the other entries intact. Recreate proxy/worker with `make recreate`, restart OpenCode, and rerun readiness, prompt, and allocation checks. Never leave the client advertising 64K while the runtime supplies 32K.

Configured 64K leaves 53248 estimated input tokens after output and safety reserves; 32K leaves 26624. The estimator uses serialized UTF-8 bytes divided by three, rounded up, rather than the exact tokenizer. Recent history and memories share finite space. Very large current requests, system instructions, or tool payloads are protected rather than silently discarded, and can still exceed the model window. Split large work into focused requests.

## Setup test ladder

Run each layer for what it establishes; a pass at one layer does not imply the next. The first-time asset preparation belongs between steps 1 and 2.

| Step | Command | Healthy outcome and scope |
| --- | --- | --- |
| 1. Host | `make precheck` | Tools/daemon/config/loopback port checks pass; no inference |
| 2. Render | `docker compose config --quiet` | Exit 0; configuration parses without printing secrets |
| 3. Start | `make up` | Four services running with Compose health; full readiness still separate |
| 4. Diagnose | `make doctor` | Installed models, loaded provider/plugin, migrations, worker, HTTP checks pass |
| 5. Readiness | `make ready` | Liveness and exactly six healthy dependencies |
| 6. Live memory | `make smoke` | Completion, accepted memory, fresh-session streaming recall, unrelated-project isolation |
| 7. Static checks | `uv run ruff check .` and `uv run pyright` | Exit 0 for code style/type checks |
| 8. Tests | `make test` or `make check` | pytest passes; `check` also runs Ruff, Pyright, and Compose render |
| 9. Client | `opencode debug config` and `opencode models local-rag` | Correct loopback baseURL, discovered memory plugin, exactly five generation selectors |
| 10. Prompt/runtime | Qwen3 prompt above, then `ollama ps` | A real reply and a matching loaded context allocation |

`make smoke` writes unique synthetic project records into your current Compose database and retains them. Its foreground default is `qwen2.5-coder:1.5b`; it uses real curator and embedder inference. All five generation IDs plus curator/embedder must be installed, even though only the selected foreground model is exercised. If the smallest model cannot follow the synthetic prompt or the host needs more time:

```sh
SMOKE_MODEL=qwen2.5-coder:7b SMOKE_TIMEOUT_SECONDS=600 make smoke
```

Success ends with `PASS: live smoke complete; synthetic project records retained locally`. A worker job that completes with zero accepted memories fails this smoke's recall requirement. A failure is evidence to investigate, not a reason to claim memory is verified.

For a lighter local code check without Docker-backed integration/E2E execution:

```sh
uv run ruff check .
uv run pyright
uv run pytest tests/unit tests/contract -q
```

Some contract tests exercise the installed OpenCode resolver; missing executable/plugin dependencies can produce skips. Review skipped tests before claiming coverage. The full pytest suite starts disposable PostgreSQL/Chroma containers and uses deterministic fake Ollama responses; it does not run live model inference or use the operator stack's volumes. It covers actual proxy HTTP/SSE, tool transport, curation, retrieval, five model switches, and project isolation. The live smoke separately checks real inference, but does not prove every model's OpenCode tool quality.

Inspect `opencode debug config` locally: the resolved output can include settings from global configuration and other providers, so do not paste it publicly without reviewing it. Expect the memory plugin path ending in `.opencode/plugins/rag-memory.js`, provider ID `local-rag`, and `http://localhost:8080/v1` (or your chosen port). Model discovery should contain exactly the five `local-rag/...` generation selectors from the table, never the embedder.

For a final interactive check:

```sh
opencode --model local-rag/qwen3-coder:30b
```

Prompt: “Without using tools, reply exactly LOCAL_RAG_OK.” Verify the reply, then test a small code explanation before authorizing edits. If Ollama is unavailable, record live smoke/prompt checks as unavailable or skipped; deterministic tests and process health are not substitutes.

## Everyday use and model switching

From this checkout:

```sh
make ready
opencode
# Or choose a configured model explicitly:
opencode --model local-rag/qwen2.5-coder:7b
```

From this checkout, launch a different existing coding repository:

```sh
make launch REPO=/path/to/repo
make launch REPO="/path/to/coding repo" MODEL=local-rag/qwen2.5-coder:7b
```

`REPO` must exist and be readable; relative paths resolve from the local RAG checkout. The launcher opens that project's absolute path and loads this checkout's provider and memory plugin without copying files into the target. It defaults to this checkout's `opencode.json` model. `MODEL` accepts only `local-rag/<configured-model>`; other values fail. Keep this checkout available while using the launcher.

The launcher supplies `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, inline provider/plugin options, and `OPENCODE_PURE=0`. In verified OpenCode 1.18.30, the plugin replaces the entire `local-rag` provider after configuration merging, preventing target settings from redirecting this provider or changing its limits. Other target settings/providers still merge; check their permissions and instructions. See [OpenCode configuration precedence](https://opencode.ai/docs/config/#precedence-order). Normal coding activity can change target files according to OpenCode permissions; the launcher itself does not edit the target repository.

In the TUI, use `/models` to switch among the local selectors; `/new` starts a fresh session, `/sessions` selects a saved session, and `/compact` summarizes the current session. File references with `@` provide task-specific file context. These are OpenCode features described in its [TUI documentation](https://opencode.ai/docs/tui/). Select the `local-rag` provider to use this memory system; another provider bypasses the proxy.

Changing the foreground model does not change project identity, stored events, or accepted memories. A fresh session in the same project can retrieve older relevant evidence without resuming the original transcript. The curator and embedder roles stay as configured independently of your foreground choice. Unknown generation IDs are rejected rather than silently substituted.

The plugin hashes OpenCode's project ID with normalized `origin`, falling back to a canonical root when no usable remote exists. Moving a no-remote repository, changing its remote, or changing OpenCode project identity can select a different memory namespace. There is no cross-project memory sharing or namespace migration command. Use a stable project and verify scoped recall after moving it.

For a manual memory experiment, state an unconditional, non-sensitive decision such as “We selected PostgreSQL for durable project memory.” Complete the response, allow the background worker to process it, then use `/new` in the same project and ask which backend was selected. This is a useful observation, not a deterministic promise: extraction and relevance filtering can omit it. `make smoke` provides the stronger automated recall/isolation check.

Common daily commands:

```sh
make status
docker compose logs --tail=100 proxy worker
make logs LOG_TAIL=200
# Ctrl-C stops following logs, not the stack.
make down
make up
```

`down` preserves named volumes. `restart` restarts existing containers; `recreate` rebuilds/reloads configuration and waits for Compose health. Do not use volume deletion or `make reset` as routine troubleshooting. Detailed backup and recovery procedures belong in the [operations guide](operations-guide.md).

## Saved sessions, exports, and RAG memory

Three stores have different purposes:

| Store | Preserves | How to use/back up it |
| --- | --- | --- |
| OpenCode's host session storage | Client transcript and session state | List/resume sessions; JSON export/import for a selected session |
| PostgreSQL named volume | Proxy-captured events, accepted memories, provenance, jobs | PostgreSQL backup/restore in the operations guide |
| Chroma named volume | Search vectors for active memories | Rebuild from PostgreSQL after restoring/changing embeddings |

OpenCode saves its session data independently on the host (standard macOS/Linux data directory: `~/.local/share/opencode/`; actual paths can vary with environment). `opencode debug paths` shows paths for your installation. See [OpenCode storage documentation](https://opencode.ai/docs/troubleshooting/#storage). Do not clear this directory as a casual cache remedy: it contains application/session data.

The following forms were checked against OpenCode **1.18.30** help and its [CLI reference](https://opencode.ai/docs/cli/). Later releases can reorganize commands; use your installed `--help` before adapting them.

```sh
opencode session list --max-count 20
opencode session list --max-count 20 --format json
opencode --continue
opencode --session ses_REPLACE_WITH_REAL_ID
opencode --session ses_REPLACE_WITH_REAL_ID --fork
```

Use the appropriate repository directory when continuing. In another repository launched by `make launch`, use `/sessions` inside the TUI to select the transcript while preserving the launcher configuration. The launcher has no dedicated continue/session variable. `/sessions` also has `/resume` and `/continue` aliases; consult [TUI sessions](https://opencode.ai/docs/tui/#sessions).

### Export privately, inspect, then import

Choose a real session ID from the list. The example uses a new private temporary directory to avoid overwriting an existing archive. Keep it outside the coding repository and move a needed backup to protected durable storage before temporary-directory cleanup.

```sh
umask 077
session_export_dir=$(mktemp -d)
opencode export ses_REPLACE_WITH_REAL_ID --sanitize > "$session_export_dir/session.json"
python3.12 -m json.tool "$session_export_dir/session.json" > /dev/null
printf 'Export location: %s\n' "$session_export_dir/session.json"
```

`--sanitize` redacts sensitive transcript/file data according to OpenCode's export behavior; inspect the actual file before sharing, since redaction is not a guarantee of removing every secret. A sanitized export can omit information needed for a complete private transcript archive. If you need full fidelity, create a separate private raw export and never treat it as safe to share:

```sh
opencode export ses_REPLACE_WITH_REAL_ID > "$session_export_dir/session-private-raw.json"
python3.12 -m json.tool "$session_export_dir/session-private-raw.json" > /dev/null
```

Only proceed after the export command and JSON check both succeed. Avoid `/share` for sensitive sessions: a local export does not require publishing a share link. TUI `/export` is a Markdown conversation export opened in your editor; it differs from CLI JSON import/export. See [OpenCode TUI export](https://opencode.ai/docs/tui/#export).

To restore a trusted JSON export, run in the intended project directory:

```sh
opencode import /path/to/private-exports/session.json
opencode session list --max-count 20
opencode --session ses_REPLACE_WITH_IMPORTED_ID
```

Check the imported session ID/result before resuming. Import restores OpenCode session data; it does not restore PostgreSQL memories/jobs or Chroma vectors. Future messages sent through the proxy follow normal capture/curation; importing by itself is not an ingestion or backfill command for RAG.

### Delete only the intended session

After checking the exact ID and exporting anything needed:

```sh
opencode session delete ses_REPLACE_WITH_REAL_ID
```

This deletes OpenCode session data. It does not send a memory-deletion request to this proxy: accepted facts and captured events can remain in PostgreSQL. This version has no public RAG-memory deletion or job-retry API. If your intent is privacy erasure, deleting/exporting the client session is insufficient: erasure requires a separately reviewed administrative procedure covering the intended PostgreSQL records, derived Chroma data, and retained backups, including the risk of restoring erased data. The operations guide describes storage and backup boundaries but does not provide an erasure runbook. Deleting Compose volumes, conversely, does not delete OpenCode's host history or host Ollama models.

## Choosing this project or Claude Code

This is a workflow choice, not a claim that one agent always codes better. The checkout has transport, isolation, and recall tests plus a live smoke; it has no comparative coding-quality, throughput, or cost benchmark against Claude Code. Compare the same small task on your own hardware and account before committing a large workflow.

| Dimension | This project's OpenCode + local RAG stack | Claude Code |
| --- | --- | --- |
| Inference/backend | Five allowlisted downloaded models through host Ollama; you manage capacity and context | Claude models through supported providers with `/model` and provider-specific choices ([model configuration](https://code.claude.com/docs/en/model-config)). Ollama also documents a Claude Code integration for local or cloud models ([official integration](https://docs.ollama.com/integrations/claude-code)); local inference is not exclusive to this project. |
| Cross-session memory | Background extractive curation, PostgreSQL provenance/jobs, project-filtered semantic retrieval with bounded injection | Written `CLAUDE.md` instructions and automatically maintained repository memory notes, inspectable through `/memory` ([memory](https://code.claude.com/docs/en/memory)). Both systems have finite context and can miss useful information. |
| Session continuity | OpenCode list/continue/resume/fork/export/import, separate from this stack's RAG database | Local conversation persistence and continuation/resume ([how it works](https://code.claude.com/docs/en/how-claude-code-works), [resume workflow](https://code.claude.com/docs/en/common-workflows#resume-previous-conversations)); persistence is distinct from auto memory. |
| Coding tools and permissions | OpenCode's tools/permissions remain in charge; small-model tool reliability needs task-level evaluation | Agent workflow with file/shell tools and permission controls ([how it works](https://code.claude.com/docs/en/how-claude-code-works)); capabilities still depend on model, provider, and allowed tools. |
| Model capability decisions | Evaluate the allowlisted local models on your tasks for reasoning, tool-call reliability, latency, and actual allocated context within host memory limits | Hosted Claude choices include model-dependent reasoning effort and extended-context options ([model configuration](https://code.claude.com/docs/en/model-config)). Consider these when complex tasks or larger working context matter; verify account/provider availability and task results rather than assuming a quality advantage. |
| Integration ecosystem | OpenCode supports local and remote MCP servers ([MCP servers](https://opencode.ai/docs/mcp-servers/)); this checkout adds its own proxy/plugin memory path | Documented MCP integrations connect external tools, databases, and APIs ([MCP integrations](https://code.claude.com/docs/en/mcp)). Compare the integrations your team needs and their permissions; this project's RAG adapter is not shared with Claude Code. |
| Vendor support | You own diagnosis and maintenance of the custom RAG services, using these guides and upstream component channels | Anthropic documents product issue reporting and account/billing support ([getting help](https://code.claude.com/docs/en/troubleshooting#get-more-help)); human support access depends on the plan ([support options](https://support.claude.com/en/articles/9015913-how-to-get-support)). Check the applicable support terms before relying on a response commitment. |
| Data handling | Configured generation/curation/embedding traffic stays on the host/Compose network. Database captures can contain sensitive code and prompts; protect disk and backups. Optional tools/providers can make their own external requests. | Data handling depends on account/provider and settings. Consumer training is controlled by user choice; commercial terms differ ([data usage](https://code.claude.com/docs/en/data-usage)). A local execution environment alone does not imply local inference. |
| Cost and upkeep | Hardware, electricity, storage, downloads, and maintenance; no remote per-token inference bill for these local model calls | Subscription allowances or API/provider usage billing, depending on authentication; do not infer actual billing from a session estimate ([costs](https://code.claude.com/docs/en/costs)). Local Ollama inference has a different cost profile. |
| Operational responsibility | You operate four services, models, backups, schema, index rebuilds, retries, and identity stability | Standard hosted inference avoids this project's database/index stack; a local Ollama integration still requires host model operations. Consult the [official setup](https://code.claude.com/docs/en/overview) for the chosen environment. |

Choose this stack when its particular project-scoped retrieval/provenance model is useful and you are prepared to own local operations. Consider Claude Code when its supported tools, account/provider setup, editable memory files, and session workflow fit your team more directly. If your main requirement is simply local model inference, evaluate both OpenCode and Ollama's documented Claude Code integration; this project adds its own RAG behavior rather than making local inference possible for the first time.

Do not point Claude Code at this project's proxy as a drop-in client: the implemented endpoint is OpenAI-compatible chat with OpenCode identity headers, not an Anthropic Messages endpoint. No Claude Code adapter or shared-memory integration is verified here. Neither choice promises unlimited context, accurate recall, or safe unattended edits; review changes and test them.

## Troubleshooting decision tree

Start with `make doctor`; follow the first failing layer, then repeat `make ready` and the smallest live prompt. Use bounded logs rather than dumping conversations or secrets.

```text
make precheck fails?
  -> Fix missing tools, Docker daemon, invalid configuration, or port conflict.
precheck passes but make up fails?
  -> Inspect Compose status and the failing service's startup logs.
up passes but make ready fails?
  -> Read /readyz and repair the named dependency; process health is insufficient.
ready passes but OpenCode fails?
  -> Check provider/plugin, model presence, a short host inference, runtime context.
chat succeeds but old facts are absent?
  -> Check project identity, capture/worker/jobs, supported extraction, and index.
all checks pass but coding loops or becomes slow?
  -> Reduce task/tool payload; verify context and memory pressure; try another model.
```

| Symptom | Checks and safe next action |
| --- | --- |
| Missing executable or wrong Python | Follow `precheck` installation remedies. Install Python 3.12 yourself. Select existing tools with `make precheck PYTHON=/path/to/python3.12 OPENCODE=/path/to/opencode`; paths containing spaces must be quoted. `UV_PYTHON` can select uv's existing interpreter. Operator targets force `UV_PYTHON_DOWNLOADS=never`. |
| Docker unavailable or startup failure | Start Docker Desktop/daemon; check `docker info`, `make status`, and `docker compose logs --tail=100 postgres chromadb proxy worker`. Fix the failing service rather than resetting volumes. |
| Port 8080 occupied | An existing proxy is accepted only if liveness matches. Stop the identified conflicting service or change `.env` `PROXY_PORT` and OpenCode `baseURL` together; render, recreate, and rerun `precheck`. |
| Ready is `not_ready` / Ollama unreachable | Check `ollama list`, host listener/firewall, `OLLAMA_HOST`, and container `OLLAMA_URL`. Container loopback is not host loopback. Recreate after `.env` changes. |
| Ready is `degraded` | Inspect which of six dependency names is unhealthy. PostgreSQL loss means untrimmed passthrough without durable-capture assurance; Chroma/embedder loss means no semantic recall. Restore dependencies and follow operator recovery procedures. |
| Model listed but request fails | `/v1/models` is static. Run `ollama list`, pull the exact tagged ID, and test `ollama run qwen2.5-coder:7b 'Reply with OK.'`. Check model-specific memory/timeout errors. A ready report does not test foreground inference. |
| Provider/plugin error or HTTP 400 | Run `opencode debug config` and `opencode models local-rag` in this checkout. Confirm local baseURL and the memory plugin; avoid `--pure` for normal memory-enabled use. Use `make launch` for another repository. Raw proxy requests need project/session headers and complete tool-call/result chains. |
| Repeated model loops or compaction | First verify `ollama ps` matches the configured context/output policy. Reduce large file/tool payloads, make one focused request, use `/new` or `/compact`, and try a different local model. More context cannot guarantee tool quality. Export a needed transcript first; compaction is not a database backup. |
| 64K configured but `ollama ps` says 32K | Apply the runtime-context procedure above, or lower both client and proxy limits to verified capacity. Restart the responsible host daemon and recreate/restart the affected clients/services; check again after a real request. |
| Slow response, OOM, or upstream timeout | Inspect `ollama ps` offloading and host memory pressure; free capacity, use a smaller model/context, and split the task. Curator/embedder inference can compete with foreground work. Increase timeout only after confirming that inference makes progress; it does not fix insufficient memory. |
| Chat works but facts are not recalled | Verify the same project/provider, a successfully completed response, worker activity, six dependencies, and accepted-memory existence. Questions/conditionals are not assertions; restrictive extraction can accept zero memories. Retrieval is relevance-bounded. Do not assume every saved event becomes a memory. |
| `memory_jobs` remains degraded after recovery | Durable failed/retrying jobs stay visible. Default retries stop after five attempts. Reindex restores vectors for accepted memories; it does not rerun failed extraction or clear job state. Consult operations before an administrative job change. |
| Recall disappeared after embedding change | Increment `EMBEDDING_VERSION`, recreate the apps, and reindex every affected project as described in operations. The new version/model collection starts empty; old collections can still consume disk. |

Useful logs and diagnostic equivalents:

```sh
docker compose logs --tail=100 proxy worker
docker compose logs --tail=100 postgres chromadb
make ready
ollama ps
opencode debug paths
```

Proxy/worker logs intentionally omit prompt, tool, memory, credential, and raw exception bodies, even at DEBUG. They expose controlled categories, timing, capture/stream state, and opaque IDs. Check readiness and durable job state alongside a category: indexing failures can currently be labeled `embedder` even when the persistence step failed. OpenCode's own logs have separate handling and can contain sensitive context; see [OpenCode troubleshooting](https://opencode.ai/docs/troubleshooting/#logs). Review them before sharing.

Operator deadlines default to 15 seconds per diagnostic, 300 for Compose, 120 for its startup health wait, and 3600 for dependency downloads/model pulls/the complete smoke. For a demonstrably slow operation, use whole seconds from 1 to 86400, for example `make up STARTUP_TIMEOUT_SECONDS=240 COMPOSE_TIMEOUT_SECONDS=360`. `make logs` follows until interrupted. `make test` and the test/static-check portion of `make check` have no outer deadline. A timed-out local command may already have submitted work to Docker/Ollama; inspect status before retrying.

## Limitations and safe next steps

- This version retrieves curated conversation facts, not a prebuilt source-code/documentation index. OpenCode file tools remain how you inspect the current repository.
- Memory is scoped to an exact project and selection is bounded. It does not guarantee perfect recall, semantic deduplication, or cross-project knowledge sharing. Similar facts with different wording may remain separate; obsolete memories need operator attention.
- Only successfully finalized assistant completions produce curation jobs. Cancelled/error/partial streams keep diagnostic capture, not accepted completion memories. Conditional evidence and unsupported paraphrases can be omitted deliberately.
- Saved source events can contain sensitive code, credentials, and personal information. Candidate secret screening is not full redaction; this stack does not provide automatic encryption, authentication for remote use, or an end-user erasure API. Protect host storage and private backups.
- PostgreSQL outages permit untrimmed passthrough rather than assuming persistence; this can exceed configured context. Never rely on durable capture while degraded.
- No automatic backup or retention schedule is configured. Durable storage grows; a Chroma index or OpenCode export is not a PostgreSQL backup.
- There is no live proof that every listed model fits your host, handles every tool call, or works at 64K. Use actual runtime allocation and task checks, and revalidate after tool/model upgrades.

Before a long project, pass the setup ladder, choose a model/context that fits, make and verify a PostgreSQL backup using the [operations guide](operations-guide.md), and preserve essential decisions in reviewable project files. Stop normally with `make down`. Volume deletion is a deliberate data-loss operation, not a performance fix.
