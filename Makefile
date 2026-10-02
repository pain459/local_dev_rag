# Thin command surface; reusable checks and safeguards live under scripts/.
override SHELL := /bin/sh
.DEFAULT_GOAL := help
DOCKER ?= docker
# Empty COMPOSE selects the distinct, quoted DOCKER executable + compose argument.
COMPOSE ?=
COMPOSE_FILE ?= compose.yaml
EXPOSE_DB ?= 0
POSTGRES_INSPECT_PORT ?= 5433
UV ?= uv
PYTHON ?= python3.12
NODE ?= node
OLLAMA ?= ollama
OPENCODE ?= opencode
LOG_TAIL ?= 100
DIAGNOSTIC_TIMEOUT_SECONDS ?= 15
COMPOSE_TIMEOUT_SECONDS ?= 300
STARTUP_TIMEOUT_SECONDS ?= 120
DOWNLOAD_TIMEOUT_SECONDS ?= 3600
# Capture user values once, literally, before export can expand Make expressions.
# Defer $(value ...) through eval's first pass; := performs its sole expansion.
# Internal defaults remain deliberate Make expressions. Keep Make's own control
# metadata intact so its command-line precedence and recursive invocations work.
# Protect the capture machinery itself from user overrides/iterator collisions.
override operator_literal_iterator := $(value operator_literal_iterator)
export operator_literal_iterator
override operator_user_variables := $(filter-out MAKEFLAGS MAKEOVERRIDES MFLAGS MAKELEVEL GNUMAKEFLAGS,$(.VARIABLES))
$(foreach operator_literal_iterator,$(operator_user_variables),$(if $(filter command line environment,$(origin $(operator_literal_iterator))),$(eval override $(operator_literal_iterator) := $$(value $(operator_literal_iterator)))$(eval export $(operator_literal_iterator))))
# Operator setup may download project dependencies, never a host Python runtime.
override UV_PYTHON_DOWNLOADS := never
export UV_PYTHON_DOWNLOADS
export DOCKER COMPOSE COMPOSE_FILE UV PYTHON NODE OLLAMA OPENCODE LOG_TAIL
export EXPOSE_DB POSTGRES_INSPECT_PORT
export PROJECT CONFIRM
export REPO MODEL
export DIAGNOSTIC_TIMEOUT_SECONDS COMPOSE_TIMEOUT_SECONDS STARTUP_TIMEOUT_SECONDS DOWNLOAD_TIMEOUT_SECONDS

.PHONY: help precheck essentials doctor doctor-fix up down restart recreate status logs ready migrate reindex smoke test check reset launch
help:
	@printf '%s\n' \
	  'Local RAG operator commands (macOS/Linux)' \
	  '  precheck     Read-only host tools, versions, daemon, ports/config checks' \
	  '  essentials   Create missing .env, sync dependencies, download/build assets' \
	  '  doctor       Read-only end-to-end diagnosis with exact remedies' \
	  '  doctor-fix   Project-local/download remediation; then doctor (no startup)' \
	  '  up           Start stack and wait for Compose health' \
	  '  down         Stop stack; preserve named volumes' \
	  '  restart      Restart services' \
	  '  recreate     Build, force recreate, and wait' \
	  '  status       Compose service state' \
	  '  logs         Follow logs (LOG_TAIL=100)' \
	  '  ready        Check liveness and full six-dependency readiness' \
	  '  migrate      Upgrade database in running proxy; back up and stop writers first' \
	  '  reindex      Rebuild one exact project: PROJECT=<exact-id>' \
	  '  smoke        Live memory smoke (creates isolated smoke records)' \
	  '  launch       OpenCode for REPO=/path/to/repo (optional MODEL=local-rag/<configured-model>)' \
	  '  test         Run pytest' \
	  '  check        Ruff, Pyright, pytest, and Compose render' \
	  '  reset        Delete current Compose project volumes: exact RESET required' \
	  'Safety: never installs host tools; never overwrites .env or credentials.' \
	  'doctor is read-only; doctor-fix downloads/builds without starting services.' \
	  'reset preserves .env, host Ollama models, and Docker images. Back up first.' \
	  'Database GUI: make up EXPOSE_DB=1 or make recreate EXPOSE_DB=1;' \
	  '  POSTGRES_INSPECT_PORT=5433 (optional port 1-65535), PostgreSQL on 127.0.0.1 only.' \
	  '  Preflight rejects extra PostgreSQL mappings or any Chroma host publication.' \
	  '  Default EXPOSE_DB=0 (or empty) keeps databases private; Chroma stays private.' \
	  '  Remove host access: make recreate EXPOSE_DB=0 (preserves named volumes).' \
	  'Overrides: COMPOSE_PROJECT_NAME, COMPOSE_FILE, COMPOSE, DOCKER, UV, PYTHON, NODE, OLLAMA, OPENCODE.' \
	  'Deadlines (seconds): DIAGNOSTIC_TIMEOUT_SECONDS=15, COMPOSE_TIMEOUT_SECONDS=300,' \
	  '  STARTUP_TIMEOUT_SECONDS=120, DOWNLOAD_TIMEOUT_SECONDS=3600; logs --follow is unbounded.'

precheck:
	@/bin/sh scripts/precheck.sh
essentials:
	@/bin/sh scripts/operator-assets.sh essentials
doctor:
	@/bin/sh scripts/doctor.sh
doctor-fix:
	@/bin/sh scripts/operator-assets.sh doctor-fix

up down restart recreate status logs migrate reindex:
	@/bin/sh scripts/operator-stack.sh $@
ready:
	@/bin/sh scripts/doctor.sh ready
reset:
	@/bin/sh scripts/reset.sh
smoke:
	@/bin/sh scripts/operator-stack.sh smoke
launch:
	@/bin/sh scripts/launch-opencode.sh
test:
	@"$$UV" run pytest
check:
	@"$$UV" run ruff check .
	@"$$UV" run pyright
	@"$$UV" run pytest
	@/bin/sh scripts/operator-stack.sh config
