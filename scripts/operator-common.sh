#!/bin/sh
# Sourced by the focused operator scripts. Never source .env or eval config.
DOCKER=${DOCKER:-docker}
COMPOSE=${COMPOSE:-}
UV=${UV:-uv}
PYTHON=${PYTHON:-python3.12}
NODE=${NODE:-node}
OLLAMA=${OLLAMA:-ollama}
OPENCODE=${OPENCODE:-opencode}
operator_dir=${0%/*}
probe=$operator_dir/operator-probe.py
runner=$operator_dir/operator-run.py
DIAGNOSTIC_TIMEOUT_SECONDS=${DIAGNOSTIC_TIMEOUT_SECONDS:-15}
COMPOSE_TIMEOUT_SECONDS=${COMPOSE_TIMEOUT_SECONDS:-300}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-120}
DOWNLOAD_TIMEOUT_SECONDS=${DOWNLOAD_TIMEOUT_SECONDS:-3600}
# A functioning Python runtime supervises even a broken/hanging PYTHON override.
runner_python=$(command -v python3 || command -v python3.12 || command -v "$PYTHON") || {
    printf '%s\n' 'FAIL: Python runtime missing. Install Python 3.12 (including python3) before using operator commands.' >&2
    exit 1
}
problems=0

fail() { printf 'FAIL: %s\n' "$1" >&2; }
issue() { fail "$1"; problems=1; }
pass() { printf 'PASS: %s\n' "$1"; }
run_deadline() { "$runner_python" "$runner" "$@"; }
run_probe() { run_deadline "$DIAGNOSTIC_TIMEOUT_SECONDS" "$@"; }
# The default preserves DOCKER as one executable even when its path has spaces.
# Explicit COMPOSE is a command plus words, never evaluated shell syntax.
compose_with_timeout() {
    deadline=$1
    shift
    if [ -n "$COMPOSE" ]; then
        (set -f; run_deadline "$deadline" $COMPOSE "$@")
    else
        run_deadline "$deadline" "$DOCKER" compose "$@"
    fi
}
compose() { compose_with_timeout "$COMPOSE_TIMEOUT_SECONDS" "$@"; }
compose_probe() { compose_with_timeout "$DIAGNOSTIC_TIMEOUT_SECONDS" "$@"; }
compose_follow() {
    if [ -n "$COMPOSE" ]; then (set -f; $COMPOSE "$@"); else "$DOCKER" compose "$@"; fi
}
render() { compose_probe config --format json 2>/dev/null; }
inspect_config() { printf '%s\n' "$config" | run_probe "$PYTHON" "$probe" "$1"; }

host_checks() {
    platform=$(run_probe uname -s) || { issue 'Platform detection failed or timed out. Check uname and DIAGNOSTIC_TIMEOUT_SECONDS.'; return 1; }
    case "$platform" in
        Darwin) platform_name=macOS; docker_remedy='Install Docker Desktop for Mac (https://docs.docker.com/desktop/setup/install/mac-install/).';;
        Linux) platform_name=Linux; docker_remedy='Install Docker Engine and the Compose plugin for your distribution (https://docs.docker.com/engine/install/).';;
        *) platform_name=$platform; docker_remedy='Install Docker with the Compose plugin (https://docs.docker.com/get-started/get-docker/).';;
    esac
    for tool in "$DOCKER" "$OLLAMA" "$OPENCODE" "$UV" "$PYTHON" "$NODE"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            case "$tool" in
                "$DOCKER") remedy=$docker_remedy;;
                "$OLLAMA") remedy='Install Ollama from https://ollama.com/download; start ollama serve.';;
                "$OPENCODE") remedy='Install OpenCode from https://opencode.ai/docs/; target 1.18.30.';;
                "$UV") remedy='Install uv from https://docs.astral.sh/uv/getting-started/installation/.';;
                "$PYTHON") remedy='Install Python 3.12 from https://www.python.org/downloads/ or select an existing interpreter with PYTHON=.';;
                "$NODE") remedy='Install Node.js from https://nodejs.org/en/download; verified version 24.21.0.';;
            esac
            issue "$platform_name: missing $tool. $remedy"
            continue
        fi
        version=$(run_probe "$tool" --version 2>/dev/null) || { issue "$tool --version failed or timed out. Repair/select your $platform_name installation; check DIAGNOSTIC_TIMEOUT_SECONDS."; continue; }
        # Host tools print their own public version; never print config/debug output.
        pass "$tool: $version"
        if [ "$tool" = "$PYTHON" ]; then
            case "$version" in 'Python 3.12.'*) ;; *) issue 'Python 3.12 is required. Select PYTHON=python3.12; this command never installs Python.';; esac
        fi
    done
    if command -v "$DOCKER" >/dev/null 2>&1; then
        run_probe "$DOCKER" info >/dev/null 2>&1 || issue "Docker daemon unavailable or timed out. On $platform_name start Docker Desktop/the Docker daemon; verify docker info."
        if compose_version=$(compose_probe version 2>/dev/null); then
            pass "$compose_version"
        else
            issue "$platform_name: Docker Compose unavailable. $docker_remedy Verify docker compose version."
        fi
    fi
    if [ "$platform" = Linux ]; then
        printf '%s\n' 'NOTE: Linux containers need a reachable Ollama interface. Configure OLLAMA_HOST and restrict port 11434 with your firewall; verify container connectivity.'
    fi
    [ "$problems" -eq 0 ]
}

config_checks() {
    config=$(render) || { issue 'Compose render failed. Fix .env/COMPOSE_FILE; run docker compose config --quiet (do not print credentials).'; return 1; }
    inspect_config config || { problems=1; return 1; }
    pass 'Compose configuration and loopback publication'
}

models() { inspect_config models; }
model_present() {
    printf '%s\n' "$installed" | awk -v model="$1" '$1 == model { found=1 } END { exit !found }'
}

create_env() {
    if [ -e .env ] || [ -L .env ]; then
        pass 'Existing .env preserved'
    else
        [ -f .env.example ] && [ -r .env.example ] || {
            fail 'Missing/unreadable .env.example; restore the template before creating .env.'; return 1;
        }
        # noclobber + private mode also protects against concurrent creation.
        (umask 077; set -C; /bin/cat .env.example > .env) || {
            fail 'Could not create .env safely. Check .env.example and permissions.'; return 1;
        }
        pass 'Created .env from .env.example; review credentials before make up'
    fi
}
