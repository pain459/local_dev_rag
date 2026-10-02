#!/bin/sh
# Destruction is exclusively current Compose-project-scoped and confirmed.
set -eu
. "${0%/*}/operator-common.sh"
config=$(render) || { fail 'Cannot resolve Compose config; reset refused. Run docker compose config --quiet.'; exit 1; }
inspect_config reset || exit 1
printf '%s\n' 'WARNING: deletes durable PostgreSQL memory and the derived Chroma index in the volumes above.' \
    'Preserves .env, host Ollama models, and Docker images. Back up durable memory first.'
if [ "${CONFIRM:-}" != RESET ]; then
    if [ -n "${CONFIRM:-}" ] || [ ! -t 0 ]; then
        fail 'Reset refused. Supply exact CONFIRM=RESET or type RESET at an interactive terminal.'
        exit 1
    fi
    printf 'Type RESET to delete these project volumes: '
    IFS= read -r confirmation || { fail 'Reset refused; confirmation unreadable.'; exit 1; }
    [ "$confirmation" = RESET ] || { fail 'Reset refused; confirmation must be exact RESET.'; exit 1; }
fi
compose down --volumes --remove-orphans
