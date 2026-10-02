#!/bin/sh
set -eu
. "${0%/*}/operator-common.sh"
case "$1" in
    up) compose up -d --wait;;
    down) compose down;;
    restart) compose restart;;
    recreate) compose up --build --force-recreate -d --wait;;
    status) compose ps;;
    logs) compose logs --tail "${LOG_TAIL:-100}" --follow;;
    config) compose config --quiet;;
    migrate)
        printf '%s\n' 'Migration maintenance: back up PostgreSQL and stop application writers first.'
        compose exec -T proxy alembic upgrade head;;
    reindex)
        project=${PROJECT:-}
        # Exact opaque IDs: no whitespace, shell syntax, options, or truncation.
        case "$project" in ''|-*|*[!A-Za-z0-9_.:/@+-]*) fail 'Set PROJECT=<exact-id> (1-256 safe ID characters, no whitespace or leading dash).'; exit 1;; esac
        [ "${#project}" -le 256 ] || { fail 'PROJECT exceeds 256 characters; supply the exact project ID.'; exit 1; }
        compose exec -T proxy python -m local_dev_rag.cli reindex --project "$project";;
    *) fail 'Unknown stack command'; exit 1;;
esac
