#!/bin/sh
# Both services run migrations; worker startup waits for the migrated proxy's liveness.
set -eu
alembic upgrade head
exec "$@"
