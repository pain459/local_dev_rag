#!/bin/sh
set -eu
. "${0%/*}/operator-common.sh"
startup_timeout() {
    case "$STARTUP_TIMEOUT_SECONDS" in
        ''|*[!0-9]*) fail 'STARTUP_TIMEOUT_SECONDS must be whole seconds between 1 and 86400.'; return 1;;
    esac
    if ! [ "$STARTUP_TIMEOUT_SECONDS" -ge 1 ] 2>/dev/null || ! [ "$STARTUP_TIMEOUT_SECONDS" -le 86400 ] 2>/dev/null; then
        fail 'STARTUP_TIMEOUT_SECONDS must be whole seconds between 1 and 86400.'
        return 1
    fi
}
case "$1" in
    up) startup_timeout; compose up -d --wait --wait-timeout "$STARTUP_TIMEOUT_SECONDS";;
    down) compose down;;
    restart) compose restart;;
    recreate) startup_timeout; compose up --build --force-recreate -d --wait --wait-timeout "$STARTUP_TIMEOUT_SECONDS";;
    status) compose ps;;
    logs) compose_follow logs --tail "${LOG_TAIL:-100}" --follow;;
    smoke) run_deadline "$DOWNLOAD_TIMEOUT_SECONDS" /bin/sh "$operator_dir/smoke.sh";;
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
