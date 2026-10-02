import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
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
export const RagMemoryPlugin = async ({ project, directory, worktree }, options = {}) => {
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
    },
  };
};
