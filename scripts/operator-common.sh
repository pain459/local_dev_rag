#!/bin/sh
# Sourced by the focused operator scripts. Never source .env or eval config.
DOCKER=${DOCKER:-docker}
COMPOSE=${COMPOSE:-"$DOCKER compose"}
UV=${UV:-uv}
PYTHON=${PYTHON:-python3.12}
NODE=${NODE:-node}
OLLAMA=${OLLAMA:-ollama}
OPENCODE=${OPENCODE:-opencode}
operator_dir=${0%/*}
probe=$operator_dir/operator-probe.py
problems=0

fail() { printf 'FAIL: %s\n' "$1" >&2; }
issue() { fail "$1"; problems=1; }
pass() { printf 'PASS: %s\n' "$1"; }
# COMPOSE is an operator command plus arguments; split words, never eval syntax.
compose() { (set -f; $COMPOSE "$@"); }
render() { compose config --format json 2>/dev/null; }
inspect_config() { printf '%s\n' "$config" | "$PYTHON" "$probe" "$1"; }

host_checks() {
    platform=$(uname -s)
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
        version=$("$tool" --version 2>/dev/null) || { issue "$tool --version failed. Repair/select your $platform_name installation."; continue; }
        # Host tools print their own public version; never print config/debug output.
        pass "$tool: $version"
        if [ "$tool" = "$PYTHON" ]; then
            case "$version" in 'Python 3.12.'*) ;; *) issue 'Python 3.12 is required. Select PYTHON=python3.12; this command never installs Python.';; esac
        fi
    done
    if command -v "$DOCKER" >/dev/null 2>&1; then
        "$DOCKER" info >/dev/null 2>&1 || issue "Docker daemon unavailable. On $platform_name start Docker Desktop/the Docker daemon; verify docker info."
        if compose_version=$(compose version 2>/dev/null); then
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
