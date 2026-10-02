# Local Dev RAG

OpenCode keeps its interface, coding tools, permissions, sessions, and model picker. This local OpenAI-compatible proxy adds durable project memory to host Ollama conversations: new sessions can retrieve relevant decisions from earlier sessions in the same project, including after a foreground model change.

- [Product overview](docs/product-overview.md): why Local Dev RAG, who it serves, how project memory works, and how it compares with Claude Code.
- [User guide](docs/user-guide.md): first-time setup, models, everyday use, saved sessions, and verification.
- [Operations guide](docs/operations-guide.md): topology, configuration, health, backups, migrations, reindexing, and incident recovery.

OpenCode and Ollama run on the host. Docker Compose runs exactly four services: `proxy`, `worker`, `postgres`, and `chromadb`. PostgreSQL is authoritative durable state; Chroma is a rebuildable semantic index. OpenCode sessions and Ollama models are separate host state.

## Fast path

Install Git, Make, Docker/Compose, Ollama, OpenCode, Python 3.12, uv, and Node.js first; see the [tested versions and host requirements](docs/user-guide.md#host-assumptions-and-prerequisites). macOS/Linux are the operator targets. Initial downloads, especially the 30B model, require substantial time, disk, and available memory.

```sh
git clone https://github.com/pain459/local_dev_rag.git
cd local_dev_rag
make precheck
# Start the installed Ollama app/service, or run ollama serve in another terminal.
make essentials
# Review the new .env and choose PostgreSQL credentials before first initialization.
docker compose config --quiet
make up
make doctor
make ready
opencode --model local-rag/qwen2.5-coder:7b
```

`essentials` preserves an existing `.env`, downloads/builds assets, and does not start the stack. Ollama must be reachable from containers at `OLLAMA_URL`; see the [listener guidance](docs/user-guide.md#host-assumptions-and-prerequisites). `make launch REPO=/path/to/repo` opens another coding project with this checkout's provider and memory plugin. Use `make help` for commands and the [setup test ladder](docs/user-guide.md#setup-test-ladder) before relying on recall. `make smoke` runs real inference and retains synthetic records in your database.

## Safety and limits

Keep this a single-user local deployment. Only the proxy is published, on `127.0.0.1:8080` by default; PostgreSQL and Chroma have no host ports. The proxy and Ollama do not provide authentication suitable for an untrusted network. Do not expose them through a public bind or tunnel. Conversation/tool content is stored in PostgreSQL and can be sensitive; protect volumes and backups. Changing `.env` credentials does not rotate an initialized database's credentials.

Compose health is weaker than inference readiness: `make up` can succeed with unavailable Ollama. Require `make ready`, a real prompt, and matching runtime context. The default Qwen3 and Llama policies declare 65536 tokens, but **that does not allocate 64K in Ollama**. Verify `ollama ps` after a request and follow [both supported remedies](docs/operations-guide.md#runtime-context-acceptance) if allocation is smaller. A short successful reply does not prove safe long-context operation.

Durable history does not mean infinite context, unlimited disk, or perfect recall. Memory injection is bounded; extraction can accept zero facts. Source-code/document indexing and cross-project sharing are outside this version. The project does not benchmark every model's coding/tool quality.

`make down` preserves volumes. **`make reset` deletes the selected Compose project's PostgreSQL data and Chroma index** after exact confirmation; read the [reset and recovery boundaries](docs/operations-guide.md#reset-and-disaster-recovery) and verify backups first. Keep the same Compose project name to reconnect to the same data. An older intermediate database may also report revision `0001` while having an incompatible schema; read the [migration runbook](docs/operations-guide.md#migrations) before reusing it.
