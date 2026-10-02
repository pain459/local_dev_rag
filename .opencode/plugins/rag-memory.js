import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { realpathSync } from "node:fs";
import { resolve } from "node:path";

// Canonicalize common SSH/HTTPS transports without including credentials.
function normalizeRemote(remote, root) {
  const value = remote.trim();
  const scp = value.match(/^(?:[^@/]+@)?([^/:]+):(.+)$/);
  if (scp && !value.includes("://")) {
    return `${scp[1].toLowerCase()}/${scp[2]}`.replace(/\/+$/, "").replace(/\.git$/, "");
  }
  try {
    const url = new URL(value);
    if (url.protocol === "file:") return realpathSync(decodeURIComponent(url.pathname));
    return `${url.host.toLowerCase()}${url.pathname}`.replace(/\/+$/, "").replace(/\.git$/, "");
  } catch {
    // A local filesystem remote is relative to the repository root.
    return realpathSync(resolve(root, value));
  }
}

/** @type {import("@opencode-ai/plugin").Plugin} */
export const RagMemoryPlugin = async ({ project, directory, worktree, client }, options = {}) => {
  const root = realpathSync(worktree || directory);
  let source = root;
  try {
    const remote = execFileSync("git", ["-C", root, "remote", "get-url", "origin"], {
      encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 2000,
    });
    source = normalizeRemote(remote, root);
  } catch {
    // Non-Git projects and repositories without origin use the canonical root.
  }
  const projectID = createHash("sha256").update(JSON.stringify([project.id, source])).digest("hex");
  // Both maps expire after five minutes and hold at most 512 entries per plugin.
  const ttl = 300_000;
  const limit = 512;
  const pending = new Map();
  const completed = new Map();
  const prune = (entries) => {
    for (const [id, entry] of entries) {
      if (Date.now() - entry.created >= ttl) entries.delete(id);
    }
    while (entries.size > limit) entries.delete(entries.keys().next().value);
  };
  return {
    config: async (config) => {
      if (!options.launchConfig) return;
      // OpenCode 1.18.30 runs this after config merging and before constructing
      // providers. Assignment removes target-only aliases, SDKs, options and limits.
      const launch = structuredClone(options.launchConfig);
      config.provider ??= {};
      config.provider["local-rag"] = launch.provider["local-rag"];
      config.model = launch.model;
      if (config.disabled_providers) {
        config.disabled_providers = config.disabled_providers.filter((id) => id !== "local-rag");
      }
      if (config.enabled_providers && !config.enabled_providers.includes("local-rag")) {
        config.enabled_providers.push("local-rag");
      }
    },
    "chat.headers": async (input, output) => {
      const providerID = input.provider?.id ?? input.provider?.info?.id ?? input.model?.providerID;
      if (providerID !== "local-rag") return;
      output.headers["x-opencode-session-id"] = input.sessionID;
      output.headers["x-opencode-project-id"] = projectID;
      // Fetch headers require HTTP-safe bytes, including for Unicode project paths.
      output.headers["x-opencode-project-root"] = encodeURIComponent(root);
      const id = randomUUID();
      output.headers["x-opencode-rag-observation-id"] = id;
      const baseURL = input.provider?.options?.baseURL
        ?? input.provider?.info?.options?.baseURL
        ?? input.model?.api?.url;
      pending.set(id, {
        sessionID: input.sessionID, parentID: input.message?.id,
        baseURL, created: Date.now(),
      });
      prune(pending);
    },
    event: async ({ event }) => {
      if (event.type !== "message.updated") return;
      const info = event.properties.info;
      if (info.role !== "assistant" || info.providerID !== "local-rag"
        || info.time?.completed === undefined || info.error || info.summary) return;
      prune(pending);
      prune(completed);
      const key = `${info.sessionID}:${info.id}`;
      if (completed.has(key)) return;
      completed.set(key, { created: Date.now() });
      prune(completed);
      // Drain before any await so repeated/concurrent events cannot duplicate notices.
      const observations = [];
      for (const [id, entry] of pending) {
        if (entry.sessionID === info.sessionID && entry.parentID === info.parentID) {
          pending.delete(id);
          observations.push([id, entry]);
        }
      }
      try {
        const counts = await Promise.all(observations.map(async ([id, entry]) => {
          try {
            if (!entry.baseURL) return 0;
            const url = new URL(`${entry.baseURL.replace(/\/+$/, "")}/rag/observations/${id}`);
            const response = await fetch(url, {
              headers: {
                "x-opencode-session-id": info.sessionID,
                "x-opencode-project-id": projectID,
                "x-opencode-project-root": encodeURIComponent(root),
              },
              signal: AbortSignal.timeout(2000),
            });
            if (!response.ok) return 0;
            const count = (await response.json()).injected_memory_tokens;
            return Number.isSafeInteger(count) && count > 0 ? count : 0;
          } catch { return 0; }
        }));
        const tokens = counts.reduce((total, count) => total + count, 0);
        if (!Number.isSafeInteger(tokens) || tokens <= 0) return;
        await client.session.prompt({
          path: { id: info.sessionID }, query: { directory: root },
          body: {
            noReply: true,
            parts: [{
              type: "text", text: `🧠 RAG memory applied · ${tokens} context tokens`,
              // OpenCode 1.18.30 hides synthetic text in its user-message renderer.
              // ignored excludes this programmatic status from model conversion;
              // noReply above keeps its append from starting a provider request.
              ignored: true,
            }],
          },
        });
      } catch {
        // Status diagnostics and notice failures never interrupt coding.
      }
    },
  };
};
