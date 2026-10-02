#!/bin/sh
# Every invocation below is a read-only probe, never a repair or inference.
set -eu
. "${0%/*}/operator-common.sh"
if [ "${1:-doctor}" = ready ]; then
    config_checks || exit 1
    inspect_config ready
    exit
fi
host_checks || exit 1
config_checks || exit 1
inspect_config ports || problems=1
[ -f .env ] || issue '.env missing. Run make doctor-fix or make essentials; review local credentials before startup.'
[ -d .venv ] || issue 'Python project dependencies missing. Run uv sync --all-groups or make doctor-fix.'
installed=$(run_probe "$OLLAMA" list 2>/dev/null) || { installed=''; issue 'Ollama unreachable or timed out. Start ollama serve; check OLLAMA_HOST; rerun make doctor.'; }
required=$(models) || exit 1
for model in $required; do
    model_present "$model" || issue "Missing Ollama model. Run: ollama pull $model (or make doctor-fix)."
done
resolved=$(run_probe "$OPENCODE" debug config 2>/dev/null) || { resolved=''; issue 'OpenCode resolved config unavailable or timed out. Run opencode debug config in this checkout; repair provider/plugin configuration.'; }
if [ -n "$resolved" ]; then
    port=$(inspect_config port) || exit 1
    printf '%s\n' "$resolved" | run_probe "$PYTHON" "$probe" opencode "$port" || problems=1
fi
catalog=$(run_probe "$OPENCODE" models local-rag 2>/dev/null) || { catalog=''; issue 'OpenCode model discovery failed or timed out. Run opencode models local-rag; restore local-rag provider.'; }
for model in qwen3-coder:30b qwen2.5-coder:1.5b qwen2.5-coder:7b llama3.1:8b qwen2.5:7b; do
    printf '%s\n' "$catalog" | awk -v model="local-rag/$model" '$0 == model { found=1 } END { exit !found }' || issue "OpenCode missing local-rag/$model. Restore opencode.json; run opencode models local-rag."
done
running=$(compose_probe ps --status running --services 2>/dev/null) || { running=''; issue 'Cannot inspect stack or probe timed out. Run docker compose ps; check daemon and project overrides.'; }
for service in proxy worker postgres chromadb; do
    if ! printf '%s\n' "$running" | awk -v service="$service" '$0 == service { found=1 } END { exit !found }'; then
        issue "$service is stopped. Run make up; inspect docker compose logs $service."
        continue
    fi
    pass "$service running"
    case "$service" in
        proxy)
            inspect_config ready || problems=1
            revision=$(compose_probe exec -T proxy alembic current 2>/dev/null) || revision=''
            case "$revision" in *'0001 (head)'*) pass 'Alembic current 0001 (head)';;
                *) issue 'Migration revision unavailable/not head. Back up PostgreSQL, stop writers, then make migrate; verify docker compose exec -T proxy alembic current.';; esac
            ;;
        worker)
            compose_probe exec -T worker python -m local_dev_rag.worker --healthcheck >/dev/null 2>&1 || issue 'Worker health failed or timed out. Inspect docker compose logs worker and make ready; restore dependencies and make restart.'
            ;;
    esac
done
[ "$problems" -eq 0 ] || exit 1
pass 'doctor (read-only); model presence does not certify inference or schema compatibility'
