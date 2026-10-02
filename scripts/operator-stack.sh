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
database_exposure() {
    case "${EXPOSE_DB:-}" in
        ''|0|1) ;;
        *) fail 'EXPOSE_DB must be empty, 0 (private), or 1 (loopback PostgreSQL access).'; return 1;;
    esac
    POSTGRES_INSPECT_PORT=${POSTGRES_INSPECT_PORT-5433}
    case "$POSTGRES_INSPECT_PORT" in
        ''|*[!0-9]*) fail 'POSTGRES_INSPECT_PORT must be a decimal port between 1 and 65535.'; return 1;;
    esac
    # Normalize decimal leading zeroes before comparison and Compose interpolation.
    # Bound the length before numeric tests so arbitrarily large input cannot overflow.
    while [ "${POSTGRES_INSPECT_PORT#0}" != "$POSTGRES_INSPECT_PORT" ]; do
        POSTGRES_INSPECT_PORT=${POSTGRES_INSPECT_PORT#0}
    done
    if [ -z "$POSTGRES_INSPECT_PORT" ] || [ "${#POSTGRES_INSPECT_PORT}" -gt 5 ] ||
        ! [ "$POSTGRES_INSPECT_PORT" -le 65535 ]; then
        fail 'POSTGRES_INSPECT_PORT must be a decimal port between 1 and 65535.'
        return 1
    fi
    export POSTGRES_INSPECT_PORT
    inspect_cli_files=0
    [ "${EXPOSE_DB:-}" = 1 ] || return 0
    inspect_overlay=$PWD/compose.inspect.yaml
    # Explicit CLI file flags take precedence over COMPOSE_FILE. Inspect the same
    # literal words as compose_with_timeout, with glob expansion disabled.
    if compose_has_file_flags; then
        inspect_cli_files=1
    else
        COMPOSE_FILE=${COMPOSE_FILE:-compose.yaml}${COMPOSE_PATH_SEPARATOR:-:}$inspect_overlay
        export COMPOSE_FILE
    fi
    printf '%s\n' 'To remove PostgreSQL host access: make recreate EXPOSE_DB=0 (preserves named volumes).'
}
compose_has_file_flags() (
    set -f
    for compose_word in $COMPOSE; do
        case "$compose_word" in -f|--file|--file=*|-f?*) return 0;; esac
    done
    return 1
)
compose_startup() {
    if [ "$inspect_cli_files" = 1 ]; then
        compose -f "$inspect_overlay" "$@"
    else
        compose "$@"
    fi
}
preflight_database_exposure() {
    [ "${EXPOSE_DB:-}" = 1 ] || return 0
    # Capture the complete merged publication set: Compose port reports only the
    # first binding. Keep rendered credentials and render diagnostics private.
    if ! database_config=$(compose_startup config --format json 2>/dev/null); then
        fail 'Database publication preflight could not render the selected Compose files; inspect them privately before retrying startup.'
        return 1
    fi
    printf '%s\n' "$database_config" |
        run_probe "$PYTHON" "$probe" database-exposure "$POSTGRES_INSPECT_PORT"
}
verify_database_exposure() {
    [ "${EXPOSE_DB:-}" = 1 ] || return 0
    expected_mapping=127.0.0.1:$POSTGRES_INSPECT_PORT
    if ! mapping=$(compose_startup port postgres 5432); then
        fail "Could not verify PostgreSQL mapping; expected $expected_mapping."
        return 1
    fi
    if [ "$mapping" != "$expected_mapping" ]; then
        fail "Unexpected PostgreSQL mapping; expected only $expected_mapping."
        return 1
    fi
    pass "PostgreSQL host access verified: $expected_mapping"
}
case "$1" in
    up)
        startup_timeout; database_exposure
        preflight_database_exposure
        compose_startup up -d --wait --wait-timeout "$STARTUP_TIMEOUT_SECONDS"
        verify_database_exposure;;
    down) compose down;;
    restart) compose restart;;
    recreate)
        startup_timeout; database_exposure
        preflight_database_exposure
        compose_startup up --build --force-recreate -d --wait --wait-timeout "$STARTUP_TIMEOUT_SECONDS"
        verify_database_exposure;;
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
