#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJ_ROOT}"

if [[ ! -x dl_venv/bin/streamlit ]]; then
    echo "ERROR: dl_venv/bin/streamlit not found." >&2
    echo "  The dashboard runs from dl_venv (streamlit + plotly installed there)." >&2
    echo "  Restore it with: python3 -m venv dl_venv && dl_venv/bin/pip install streamlit plotly pandas requests" >&2
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "WARN: .env missing — Alpaca account/positions sections will show errors" >&2
fi

if pgrep -f "streamlit run app/monitor.py" > /dev/null; then
    echo "Stopping running dashboard instance..."
    pkill -f "streamlit run app/monitor.py"
    sleep 1
fi

mkdir -p logs
nohup bash scripts/launch_monitor.sh --headless >> logs/monitor.log 2>&1 &
PID=$!

for _ in $(seq 1 20); do
    if curl -sf -o /dev/null http://localhost:8501; then
        echo "OK: dashboard running at http://localhost:8501 (PID ${PID}, logs: logs/monitor.log)"
        exit 0
    fi
    sleep 1
done

echo "ERROR: dashboard did not answer within 20s — check logs/monitor.log" >&2
exit 1
