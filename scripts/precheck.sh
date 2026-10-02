#!/bin/sh
set -eu
. "${0%/*}/operator-common.sh"
host_checks || exit 1
config_checks || exit 1
inspect_config ports
printf '%s\n' 'PASS: precheck (read-only). Verified targets: Docker 29.8.1, Compose 5.5.1, Ollama 0.35.0, OpenCode 1.18.30, Node.js 24.21.0; other versions are not certified.'
[ -e .env ] || printf '%s\n' 'NOTE: .env absent; make essentials or make doctor-fix creates it without overwriting.'
