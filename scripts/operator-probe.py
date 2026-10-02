#!/usr/bin/env python3
"""Read-only, stdlib-only config/HTTP probes. Never print config or HTTP bodies."""

import json
import re
import socket
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

GENERATION_MODELS = (
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
)
DEPENDENCIES = {"postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs"}


def failure(remedy):
    print(f"FAIL: {remedy}", file=sys.stderr)
    return 1


def proxy_port(config):
    ports = config["services"]["proxy"]["ports"]
    if len(ports) != 1:
        raise ValueError
    port = ports[0]
    if port.get("host_ip") != "127.0.0.1" or int(port["target"]) != 8080:
        raise ValueError
    value = int(port["published"])
    if not 1 <= value <= 65535:
        raise ValueError
    return value


def request(port, path):
    # Ignore host proxy env; these checks are exclusively local loopback HTTP.
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
        return json.load(response)


def ready(config):
    port = proxy_port(config)
    failures = 0
    for path in ("/healthz", "/readyz"):
        try:
            payload = request(port, path)
            if path == "/healthz":
                valid = payload == {"status": "ok"}
            else:
                valid = (
                    isinstance(payload, dict)
                    and payload.get("status") == "ready"
                    and isinstance(payload.get("dependencies"), dict)
                    and set(payload["dependencies"]) == DEPENDENCIES
                    and all(value == "healthy" for value in payload["dependencies"].values())
                )
            if not valid:
                raise ValueError
            print(f"PASS: {path}")
        except (HTTPError, URLError, OSError, ValueError, TypeError):
            failures = failure(
                f"{path} unavailable or unhealthy on port {port}. Run make up; "
                "inspect docker compose logs proxy worker and dependency health. "
                "For memory_jobs investigate durable failures; reindex does not clear failed jobs."
            )
    return failures


def opencode(config, port):
    provider = config.get("provider", {}).get("local-rag", {})
    url = urlparse(provider.get("options", {}).get("baseURL", ""))
    valid = (
        provider.get("npm") == "@ai-sdk/openai-compatible"
        and set(provider.get("models", {})) == set(GENERATION_MODELS)
        and config.get("model") in {f"local-rag/{model}" for model in GENERATION_MODELS}
        and url.scheme == "http"
        and url.hostname in {"localhost", "127.0.0.1"}
        and url.port == port
        and url.path.rstrip("/") == "/v1"
    )
    result = 0
    if not valid:
        result = failure(
            f"OpenCode local-rag provider/models/baseURL mismatch. Set opencode.json baseURL "
            f"to http://localhost:{port}/v1; restore the five generation models; "
            "run opencode debug config and opencode models local-rag."
        )
    plugins = config.get("plugin", [])
    if not isinstance(plugins, list) or not any(
        isinstance(plugin, str) and plugin.endswith("/.opencode/plugins/rag-memory.js")
        for plugin in plugins
    ):
        result = failure(
            "OpenCode rag-memory.js plugin missing from resolved config. Restore "
            ".opencode/plugins/rag-memory.js; launch from this checkout; run opencode debug config."
        )
    return result


def main():
    mode = sys.argv[1]
    try:
        config = json.load(sys.stdin)
        if not isinstance(config, dict):
            raise ValueError
        if mode == "opencode":
            return opencode(config, int(sys.argv[2]))
        if mode == "reset":
            name = config["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
                raise ValueError
            volumes = []
            for volume in config.get("volumes", {}).values():
                if volume.get("external"):
                    continue
                value = volume["name"]
                if not isinstance(value, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
                ):
                    raise ValueError
                volumes.append(value)
            print(f"RESET target Compose project: {name}")
            print("Volumes to delete: " + (", ".join(volumes) or "none configured"))
            return 0
        port = proxy_port(config)
        if mode == "port":
            print(port)
        elif mode == "ready":
            return ready(config)
        elif mode == "ports":
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    if request(port, "/healthz") != {"status": "ok"}:
                        raise ValueError
                print(f"PASS: port {port} already serves proxy liveness")
            except ConnectionRefusedError:
                print(f"PASS: proxy port {port} available")
            except (HTTPError, URLError, OSError, ValueError):
                return failure(
                    f"Port {port} occupied/unavailable. Stop the conflicting listener or "
                    "set PROXY_PORT in .env and update opencode.json baseURL; rerun make precheck."
                )
        elif mode in {"config", "models"}:
            environment = config["services"]["proxy"].get("environment", {})
            url = urlparse(environment.get("OLLAMA_URL", "http://host.docker.internal:11434"))
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.hostname
                in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            ):
                return failure(
                    "Container OLLAMA_URL must reach the host, e.g. "
                    "http://host.docker.internal:11434; container loopback points at itself."
                )
            models = list(GENERATION_MODELS)
            for key, default in (
                ("CURATOR_MODEL", "qwen2.5-coder:1.5b"),
                ("EMBEDDING_MODEL", "nomic-embed-text:latest"),
            ):
                value = environment.get(key, default)
                if not isinstance(value, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", value
                ):
                    return failure(f"Invalid {key}. Set a valid Ollama model ID in .env.")
                if not re.fullmatch(r"[^:]+:[A-Za-z0-9_.-]+", value):
                    return failure(
                        f"Invalid {key}. Require an explicit Ollama tag, e.g. model:latest."
                    )
                if value not in models:
                    models.append(value)
            if mode == "models":
                print("\n".join(models))
            else:
                local = json.loads(Path("opencode.json").read_text())
                plugin = Path(".opencode/plugins/rag-memory.js")
                local["plugin"] = [str(plugin.resolve())] if plugin.is_file() else []
                return opencode(local, port)
        else:
            raise ValueError
        return 0
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError):
        return failure(
            "Invalid operator config/port. Keep proxy host publication at 127.0.0.1, "
            "target 8080, and PROXY_PORT between 1 and 65535; "
            "run docker compose config --quiet and repair opencode.json."
        )


if __name__ == "__main__":
    sys.exit(main())
