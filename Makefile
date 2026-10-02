# Thin command surface; reusable checks and safeguards live under scripts/.
SHELL := /bin/sh
.DEFAULT_GOAL := help
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose
COMPOSE_FILE ?= compose.yaml
UV ?= uv
PYTHON ?= python3.12
NODE ?= node
OLLAMA ?= ollama
OPENCODE ?= opencode
LOG_TAIL ?= 100
export DOCKER COMPOSE COMPOSE_FILE UV PYTHON NODE OLLAMA OPENCODE LOG_TAIL
export PROJECT CONFIRM

.PHONY: help precheck essentials doctor doctor-fix up down restart recreate status logs ready migrate reindex smoke test check reset
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
	  '  test         Run pytest' \
	  '  check        Ruff, Pyright, pytest, and Compose render' \
	  '  reset        Delete current Compose project volumes: exact RESET required' \
	  'Safety: never installs host tools; never overwrites .env or credentials.' \
	  'doctor is read-only; doctor-fix downloads/builds without starting services.' \
	  'reset preserves .env, host Ollama models, and Docker images. Back up first.' \
	  'Overrides: COMPOSE_PROJECT_NAME, COMPOSE_FILE, COMPOSE, DOCKER, UV, PYTHON, NODE, OLLAMA, OPENCODE.'

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
	@/bin/sh scripts/smoke.sh
test:
	@"$(UV)" run pytest
check:
	@"$(UV)" run ruff check .
	@"$(UV)" run pyright
	@"$(UV)" run pytest
	@/bin/sh scripts/operator-stack.sh config
