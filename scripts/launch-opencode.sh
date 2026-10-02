#!/bin/sh
# Keep the coding project separate from this checkout's provider/plugin resources.
set -eu

fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }

[ -n "${REPO:-}" ] || fail 'REPO is required: make launch REPO=/path/to/repo'
[ -d "$REPO" ] || fail 'REPO must name an existing directory.'
[ -r "$REPO" ] || fail 'REPO must name a readable directory.'
case "$REPO" in
    /*) launch_repo=$REPO ;;
    *) launch_repo=$PWD/$REPO ;;
esac
launch_repo=$(CDPATH= cd "$launch_repo" && pwd -P) || fail 'Cannot access REPO directory.'
launch_root=$(CDPATH= cd "${0%/*}/.." && pwd -P) || fail 'Cannot locate local RAG checkout.'

OPENCODE=${OPENCODE:-opencode}
command -v "$OPENCODE" >/dev/null 2>&1 || fail 'OpenCode executable missing; install OpenCode or select OPENCODE=/path/to/opencode.'
launch_python=$(command -v python3 || command -v python3.12 || command -v "${PYTHON:-python3.12}") || fail 'Python runtime missing; select an existing Python installation.'

OPENCODE_CONFIG=$launch_root/opencode.json
OPENCODE_CONFIG_DIR=$launch_root/.opencode
# Project config loads after OPENCODE_CONFIG. Inline config restores our provider
# and default/selected model after that merge, without editing the coding project.
OPENCODE_CONFIG_CONTENT=$("$launch_python" - "$OPENCODE_CONFIG" "${MODEL:-}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as source:
        config = json.load(source)
    provider = config["provider"]["local-rag"]
    model = sys.argv[2] or config["model"]
    if not model.startswith("local-rag/") or model.removeprefix("local-rag/") not in provider["models"]:
        raise ValueError("MODEL must be local-rag/<configured-model> from this checkout's opencode.json")
    print(json.dumps({"model": model, "provider": {"local-rag": provider}}))
except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
    print(f"FAIL: Invalid local OpenCode configuration or MODEL: {error}", file=sys.stderr)
    sys.exit(1)
PY
) || exit 1
export OPENCODE_CONFIG OPENCODE_CONFIG_DIR OPENCODE_CONFIG_CONTENT

if [ -n "${MODEL:-}" ]; then
    exec "$OPENCODE" "$launch_repo" --model "$MODEL"
fi
exec "$OPENCODE" "$launch_repo"
