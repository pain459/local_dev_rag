#!/bin/sh
# Only explicitly requested project-local dependency/download remediation.
set -eu
. "${0%/*}/operator-common.sh"
host_checks || exit 1
config_checks || exit 1
inspect_config ports
installed=$(run_probe "$OLLAMA" list 2>/dev/null) || { fail 'Ollama unreachable or timed out. Start ollama serve; check OLLAMA_HOST before downloads.'; exit 1; }
required=$(models) || exit 1
create_env
run_deadline "$DOWNLOAD_TIMEOUT_SECONDS" "$UV" sync --all-groups
if [ "${1:-essentials}" = doctor-fix ]; then
    compose pull --ignore-buildable --policy missing
else
    compose pull --ignore-buildable
fi
compose build
printf '%s\n' 'NOTE: qwen3-coder:30b is large; downloads need disk space and inference needs substantial RAM/GPU memory.'
for model in $required; do
    if [ "${1:-essentials}" = essentials ] || ! model_present "$model"; then
        run_deadline "$DOWNLOAD_TIMEOUT_SECONDS" "$OLLAMA" pull "$model"
    fi
done
if [ "${1:-essentials}" = doctor-fix ]; then
    /bin/sh "$operator_dir/doctor.sh"
else
    printf '%s\n' 'Assets prepared. Review .env, start host Ollama, then make up and make doctor.'
fi
