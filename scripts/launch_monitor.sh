#!/usr/bin/env bash

set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"

PORT=8501
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --headless) EXTRA_ARGS+=(--server.headless true); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

exec dl_venv/bin/streamlit run app/monitor.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    "${EXTRA_ARGS[@]}"
