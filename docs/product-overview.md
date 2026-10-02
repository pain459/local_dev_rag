# Local Dev RAG

Local-first AI coding with durable project memory, a choice of local models, and infrastructure you own.

Local Dev RAG connects OpenCode to host Ollama through a project-memory proxy. Keep OpenCode's coding tools and interactive workflow, choose a configured local model, and let later conversations retrieve relevant decisions from earlier work in the same project. PostgreSQL holds the durable records; a semantic index helps select what belongs in the next request.

The value is continuity under your control: less repeated explanation, local generation and memory processing, and project knowledge that can outlive a session or foreground model change. These are benefits to evaluate on your own work, not measured productivity guarantees.

Local Dev RAG is a credible alternative to Claude Code for developers who prioritize this combination of privacy, model choice, semantic recall, and infrastructure ownership. **Claude Code also supports Ollama through Ollama's documented integration.** Local inference alone is not the differentiator, and this project does not claim feature parity or universally better coding results. See the [official Ollama integration](https://docs.ollama.com/integrations/claude-code).

For installation and everyday commands, start with the [user guide](user-guide.md). For backups, maintenance, and recovery, use the [operations guide](operations-guide.md).

## The problem and the vision

A repository evolves through decisions as well as commits. Why a backend was selected, which constraint shaped an implementation, or what an investigation established may be scattered across earlier coding conversations. Starting a fresh session or trying another model can mean explaining that background again. Keeping an entire transcript in every prompt eventually runs into context and resource limits.

Local Dev RAG's vision is to make useful project history available when it matters while keeping the memory system understandable and locally operated. It separates durable storage from the model's working context: save conversation evidence, curate supported memories, and retrieve a bounded selection for the current task. Reviewed repository documents remain the place for requirements that must never be missed.

## Who it serves

| User | High-value use case | What to evaluate |
| --- | --- | --- |
| An independent developer maintaining a long-lived project | Return to a project in a new session and recover relevant earlier decisions | Whether extraction and recall capture the facts that matter to the work |
| A developer handling sensitive code who can operate a local stack | Keep configured model calls and memory processing on the development machine | Host security, tool/network permissions, and backup handling |
| A local-model practitioner | Compare configured models while retaining the same project's memory | Task quality, tool-call reliability, latency, and hardware fit |
| An engineer studying or adapting a memory system | Inspect durable events, provenance, jobs, and retrieval behavior in owned infrastructure | Willingness to work with the database and code; there is no memory management UI |

The current deployment is for one user on a trusted machine. It is not a shared team memory service or a managed enterprise platform.

## What makes it useful

The core distinction is a separate, project-scoped semantic memory layer with durable evidence and processing state. It sits between an existing coding client and a local model runtime, so model selection and memory storage have separate lifecycles.

| Product pillar | Shipped behavior | Practical benefit |
| --- | --- | --- |
| Local-first execution | Configured generation, curation, and embedding calls go to host Ollama | Control where these model calls run and which hardware serves them |
| Local-model choice | Five explicit generation models are available through `local-rag`; curator and embedder roles are configured separately | Select a foreground model that fits the task and host without replacing the memory store |
| Project-scoped RAG memory | Accepted conversation memories are retrieved within the current project | Bring relevant earlier evidence into a new task without injecting every past conversation |
| OpenCode workflow | OpenCode retains its interface, file and shell tools, permissions, sessions, and model picker | Add memory to a coding workflow without learning a new client |
| Cross-session and model continuity | A project's accepted memories remain available across sessions and foreground model changes | Revisit earlier decisions even when the original session is not resumed |
| Owned infrastructure and data | PostgreSQL stores events, memories, provenance, and jobs; Chroma holds a rebuildable index | Back up authoritative state and rebuild retrieval data under your own operating policies |

Continuity depends on the same project identity, retained data, healthy dependencies, and relevant accepted memories. It is not a promise that every fact will be recalled.

## How it works

OpenCode and Ollama run on the host. Docker Compose runs exactly four services: `proxy`, `worker`, `postgres`, and `chromadb`.

```text
OpenCode + project/session plugin
    -> proxy -> host Ollama -> reply to OpenCode
         |  ^
         |  +-- bounded project-memory retrieval from Chroma
         v
    PostgreSQL: events, accepted memories, provenance, jobs
         |
         v
    worker -> host Ollama: curate and embed -> Chroma index
```

1. The plugin attaches project and session identity to `local-rag` requests. The proxy captures conversation and tool events in PostgreSQL.
2. For the current request, the proxy embeds the query through Ollama, searches the project's indexed memories, ranks eligible results, and assembles bounded recent context plus selected historical evidence.
3. The selected Ollama model generates the response. OpenCode continues to own tool execution and its permission flow.
4. A successfully finalized assistant response creates durable background work. The worker curates supported facts, validates candidates, saves accepted memories with provenance, and indexes their embeddings for later retrieval.

By default, a request can include at most six memories within an estimated 1,024-token memory budget. Curation is asynchronous and can accept zero facts; a new session started immediately may precede indexing. Conditional statements, unsupported paraphrases, and secret-like candidates can be rejected. Retrieved memory is marked as historical evidence, not current instructions.

This is **bounded semantic recall**. It is neither infinite context nor a verbatim archive delivered to the model. PostgreSQL's captured events and OpenCode's saved transcripts are separate stores; neither means every event becomes a retrievable memory. The current RAG pipeline does not index repository source files or documentation.

## Available today

| Implemented capability | What it provides |
| --- | --- |
| OpenAI-compatible chat proxy with streaming and non-streaming paths | Transport between OpenCode and Ollama, including tool-call message handling |
| Explicit generation catalog | `qwen3-coder:30b` by default, plus `qwen2.5-coder:1.5b`, `qwen2.5-coder:7b`, `llama3.1:8b`, and `qwen2.5:7b`; unknown generation IDs are rejected |
| Durable capture and background jobs | Persistent evidence and processing state, with bounded retries and worker leases |
| Project-filtered retrieval and context budgets | Selected historical context within the configured request policy |
| Another-repository launcher | Open an existing repository using this checkout's provider and plugin without copying them into that repository |
| Operator commands and diagnostics | Prerequisite checks, readiness reports, migrations, project reindexing, and a live memory smoke test |
| Documented recovery procedures | PostgreSQL backup/restore and index rebuild guidance, with separate treatment of host sessions and model files |

The implementation is grounded in the checked-in [provider configuration](../opencode.json), [memory settings](../src/local_dev_rag/config.py), [foreground proxy](../src/local_dev_rag/proxy.py), [curator](../src/local_dev_rag/curator.py), and [deployment configuration](../compose.yaml). These are current capabilities; the [future direction](#future-direction) below is not a list of delivered features.

## An end-to-end example

Imagine working on an existing service repository over several days. You want to use a smaller local model initially, retain a backend decision, and revisit it in a fresh session.

1. Complete the user guide's [setup and verification ladder](user-guide.md#setup-test-ladder), including a real prompt and runtime context check. Keep the Local Dev RAG checkout available.
2. From that checkout, launch the other repository, replacing the example path with an existing directory:

   ```sh
   make ready
   make launch REPO=/path/to/service-repo MODEL=local-rag/qwen2.5-coder:7b
   ```

3. In OpenCode, give a small coding task and an explicit, non-sensitive decision: “We selected PostgreSQL for durable project memory.” Review the proposed edits and run the repository's tests. Allow the response to finish and the background worker to process it.
4. Use `/new` in that same project, then ask, “Which backend did we select for durable project memory?” A relevant accepted memory can supply the earlier decision. Treat a missing or incorrect answer as a reason to inspect capture, curation, and retrieval; repeating the sentence is not a guarantee of storage or recall.
5. Use `/models` to select another configured `local-rag` model that fits the machine, then continue the project. The foreground model changes; the project's memory store does not. Use `/sessions` if you instead want to reopen the original OpenCode transcript. These client commands are described in the [OpenCode TUI reference](https://opencode.ai/docs/tui/).
6. End normal stack use with `make down`, which preserves named volumes. Follow the [operations guide](operations-guide.md#backup-and-restore) to protect the durable database before relying on it for future work.

The launcher itself does not edit the target repository; coding tools can edit it under OpenCode permissions. Target instructions, tool settings, and other providers still matter. Changing the remote or moving a repository without a remote can change its memory namespace; see [project identity and everyday use](user-guide.md#everyday-use-and-model-switching).

## Local Dev RAG and Claude Code

Choose on workflow fit, data handling, and operating responsibility. Neither a local runtime nor durable memory establishes coding quality by itself. The upstream descriptions below were checked against official documentation on October 2, 2026; provider capabilities and terms can change.

| Dimension | Local Dev RAG | Claude Code |
| --- | --- | --- |
| Data path and privacy | Configured generation, curation, and embedding stay on the host/Compose network. External tools or other providers can transmit data separately. | Depends on provider and account settings; hosted processing and retention follow the applicable [data-usage terms](https://code.claude.com/docs/en/data-usage). Ollama supports a local inference path too. |
| Memory model | Curated conversation facts, PostgreSQL provenance, and project-filtered semantic retrieval with bounded injection; no memory editing UI. | Persistent `CLAUDE.md` instructions and auto memory notes, with files users can inspect and edit through `/memory`. See [Claude Code memory](https://code.claude.com/docs/en/memory). |
| Model and provider choices | This memory path uses five allowlisted Ollama generation models. OpenCode supports other [providers](https://opencode.ai/docs/providers/), but selecting another provider bypasses this proxy's memory. | Claude model/provider choices are documented in [model configuration](https://code.claude.com/docs/en/model-config). [Ollama's integration](https://docs.ollama.com/integrations/claude-code) also connects local and cloud models; verify feature compatibility for the chosen model. |
| Integrations and support | OpenCode supports [local and remote MCP servers](https://opencode.ai/docs/mcp-servers/). You maintain the custom memory stack using this repository and upstream resources. | Documented [MCP integrations](https://code.claude.com/docs/en/mcp) and [product/account support routes](https://code.claude.com/docs/en/troubleshooting). Support access depends on the applicable plan; no response commitment is assumed here. |
| Operations | You run four services and manage model capacity, backups, migrations, indexing, and failures. | A standard hosted-model setup does not require this project's PostgreSQL/Chroma stack. A local Ollama setup still requires local model operations; consult the [setup overview](https://code.claude.com/docs/en/overview). |
| Costs | Local model calls avoid remote per-token inference charges, but hardware, power, storage, downloads, and maintenance have costs. External services can add charges. | Subscription or API/provider costs depend on the chosen setup; [cost guidance](https://code.claude.com/docs/en/costs) distinguishes usage estimates from billing. Local Ollama changes that cost profile. |
| Capability tradeoffs | Coding, reasoning, tool use, and latency depend on the selected local model and hardware. The project adds memory infrastructure, not a new coding model. | Evaluate the chosen model's reasoning, context, tools, and provider support on the same tasks. This repository has no comparative coding-quality or speed benchmark. |

Local Dev RAG's proxy is not a verified Claude Code adapter: it exposes OpenAI-compatible chat with OpenCode identity headers, not an Anthropic Messages endpoint. The two products do not share this memory database through a supported integration.

### Choose Local Dev RAG

Choose it when you want project-scoped semantic recall with inspectable database provenance, local model processing, and direct control of the storage and runtime—and are prepared to operate that stack. A good first adoption is one repository with a few repeatable, reviewable tasks and decisions whose recall you can check.

### Choose Claude Code

Choose it when its client workflow, supported integrations, editable memory files, and account/provider arrangements fit your work more directly. A hosted setup can remove this custom memory stack from your operational responsibilities. Evaluate its Ollama integration if local inference is also a requirement.

### Evaluate both

If your main requirement is local models, both deserve consideration. Run the same small coding task, inspect the diff and test results, try a fresh-session memory question, and compare elapsed time, resource use, setup effort, and total cost in your environment. Avoid assuming that either workflow transfers every capability unchanged between providers.

## Privacy, boundaries, and prerequisites

The default Compose setup publishes only the proxy on `127.0.0.1:8080`; PostgreSQL and Chroma have no published host ports. Ollama must be reachable by the containers through a suitably restricted host listener. Keep the deployment single-user and local: project filtering is not access control, and the proxy/Ollama listeners do not provide authentication suitable for untrusted networks.

Raw conversation and tool events can contain sensitive code, credentials, and personal information. Candidate screening does not redact PostgreSQL events or guarantee secret removal. Protect volumes, host sessions, exports, and backups. The stack provides no automatic encryption, retention policy, deletion UI, or public memory-erasure API. Deleting an OpenCode session does not erase the RAG database.

External MCP servers, shell commands, web tools, sharing features, or alternative/cloud providers can move data outside the local stack when enabled. Initial model, package, and image downloads also need network access. Local Dev RAG is not a network sandbox or a compliance certification; inspect the complete workflow when assessing privacy.

Operators need macOS or Linux, Git/Make, Docker/Compose, host Ollama and OpenCode, Python 3.12, uv, and Node.js, plus capacity for models, context caches, containers, and growing storage. The guides record tested versions, not universal minimum hardware requirements. Foreground and background inference share resources. Read [host prerequisites](user-guide.md#host-assumptions-and-prerequisites) and [model capacity guidance](user-guide.md#models-capacity-and-runtime-context) before adopting it.

## Limits and non-goals

- Recall is selective and fallible. Accepted facts can become obsolete, and curation can omit valid information. Keep essential requirements in reviewed project documents.
- A configured context budget does not allocate that context in Ollama. Verify the loaded runtime after a real request; a successful short reply does not establish long-context safety.
- Database failure can allow untrimmed passthrough without durable capture; semantic-index or embedding failure can remove recall. Process health alone does not prove inference or memory readiness.
- Source-code/document indexing, cross-project memory sharing, remote multi-user operation, and a memory inspection/correction UI are outside this version. OpenCode's repository tools remain the way to inspect current source.
- Storage is finite and operator-managed. There is no automatic backup or retention schedule, and an index or client export is not an authoritative database backup.
- Local Dev RAG does not replace OpenCode's tools or permissions, guarantee model/tool correctness, or promise safe unattended changes.

Use the [limitations and troubleshooting guidance](user-guide.md#limitations-and-safe-next-steps) and [operations guide](operations-guide.md) when assessing these boundaries.

## Evidence and quality

The repository includes unit and contract coverage for identity, context budgeting, curation, streaming, configuration, and operator commands. Integration and end-to-end tests use disposable PostgreSQL/Chroma services and deterministic model fixtures to exercise durable capture, recall, isolation, and recovery. The separate live smoke uses installed Ollama models and retains synthetic records in the operator database; it is an explicit validation step, not a read-only probe.

`make check` runs Ruff, Pyright, pytest, and a Compose configuration check. The [setup test ladder](user-guide.md#setup-test-ladder) adds readiness, real inference, and runtime context checks. This evidence supports specific implementation behaviors; it does not establish comparative model quality, throughput, perfect recall, an SLA, or suitability for every machine. Revalidate after upgrading the client, models, or runtime.

## Future direction

The [design's future-compatible boundaries](superpowers/specs/2026-10-01-local-opencode-rag-memory-design.md#17-future-compatible-boundaries) identify possible extension areas: alternative vector storage such as `pgvector`, source/document collections, explicitly controlled cross-project sharing, remote database and worker scaling, memory inspection/correction, and richer curation or rule-based extraction. Choosing a different curator model is already configurable; additional extraction approaches are future scope.

These are exploration themes, not scheduled releases or commitments. Shared or remote operation would need additional security and operational work, and new retrieval sources would need their own quality evaluation.

## FAQ

### Does it remember everything?

No. Durable event capture, accepted memories, and the evidence selected for a request are different things. Semantic recall is bounded and can miss facts; it does not replay the full transcript.

### Can I change models without losing memory?

Yes, within the configured `local-rag` catalog and the same project identity. Stored memories are independent of the foreground model. Different models may use the same retrieved evidence differently; changing the embedder requires versioning and reindexing as documented in operations.

### Can I use it with another repository?

Yes. Run `make launch REPO=/path/to/repo` from this checkout. Each project has its own memory namespace; this does not enable sharing between unrelated projects.

### Does Claude Code lack local models or persistent memory?

No. Claude Code has persistent instructions and auto memory, and Ollama documents a local-model integration. Local Dev RAG offers a different memory architecture and operating model; see the [comparison](#local-dev-rag-and-claude-code).

### Is it free to run or fully offline?

Local inference avoids remote token charges for the configured downloaded models, but operating costs remain. With assets installed, the core inference/memory path uses local services. Downloads, updates, and enabled external tools or providers require their own network access; this is not a verified air-gapped distribution.

### Where do I start?

Follow the [user guide](user-guide.md) for installation, model selection, another-repository use, sessions, and verification. Before trusting durable memory with important project history, use the [operations guide](operations-guide.md) to establish backups and understand recovery. The [README](../README.md) remains the compact command entry point.
