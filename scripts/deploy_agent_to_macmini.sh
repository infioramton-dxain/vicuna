#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: deploy_agent_to_macmini.sh [--dry-run]

Pushes the agent Python package to the Mac mini, sets up a venv,
installs pip deps. Reads MACMINI_HOST / MACMINI_USER from .env.
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
for var in MACMINI_HOST MACMINI_USER; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} not set in .env" >&2; exit 3
    fi
done

source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
macmini_resolve || true
REMOTE="${MACMINI_USER}@${MACMINI_ADDR}"
AGENT_ROOT="/home/${MACMINI_USER}/agent"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o BatchMode=yes"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/deploy_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }
run() {
    log "+ $*"
    [[ "${DRY}" -eq 1 ]] && return 0
    "$@"
}
remote() {
    log "+ ssh ${REMOTE} -- $1"
    [[ "${DRY}" -eq 1 ]] && return 0
    ssh ${SSH_OPTS} "${REMOTE}" "$1"
}

log "Phase 9b deploy starting → ${REMOTE}:${AGENT_ROOT}"

remote "mkdir -p '${AGENT_ROOT}/src/agent' '${AGENT_ROOT}/logs' '${AGENT_ROOT}/db'"

run rsync -avz --delete \
    -e "ssh ${SSH_OPTS}" \
    --exclude '__pycache__' --exclude '*.pyc' --exclude 'recs/' \
    "${PROJ_ROOT}/src/agent/" \
    "${REMOTE}:${AGENT_ROOT}/src/agent/"

run rsync -avz \
    -e "ssh ${SSH_OPTS}" \
    "${PROJ_ROOT}/src/agent/requirements.txt" \
    "${REMOTE}:${AGENT_ROOT}/requirements.txt"

remote "if [ ! -x '${AGENT_ROOT}/venv/bin/pip' ]; then
    rm -rf '${AGENT_ROOT}/venv'
    python3 -m venv '${AGENT_ROOT}/venv'
fi"

remote "'${AGENT_ROOT}/venv/bin/pip' install --upgrade pip --quiet"
remote "'${AGENT_ROOT}/venv/bin/pip' install -r '${AGENT_ROOT}/requirements.txt' --quiet"

remote "cd '${AGENT_ROOT}' && PYTHONPATH=src '${AGENT_ROOT}/venv/bin/python' -c \"
from agent.config import SHIPPED_DB
from agent.db import open_shipped_ro
from agent.alpaca_client import StubAlpacaClient
from agent.llm_router import rank_picks
from agent.risk_overlay import filter_and_size
import agent.main as m
print('imports ok; shipped_db =', SHIPPED_DB)
print('shipped db exists:', SHIPPED_DB.exists())
if SHIPPED_DB.exists():
    with open_shipped_ro() as c:
        n = c.execute('SELECT COUNT(*) FROM mm_bars').fetchone()[0]
        print('mm_bars rows readable:', n)
\""

remote "if [ ! -f '${AGENT_ROOT}/.env' ]; then
    cat > '${AGENT_ROOT}/.env.example' <<'EOF'
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

AGENT_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash-exp


AGENT_DRY_RUN=1
AGENT_MAX_PICKS=5
EOF
    echo 'wrote .env.example — copy to .env and fill in credentials'
fi"

log "OK: agent deployed → ${REMOTE}:${AGENT_ROOT}"
log ""
log "Next steps on the Mac mini:"
log "  1. cp ${AGENT_ROOT}/.env.example ${AGENT_ROOT}/.env"
log "  2. Edit ${AGENT_ROOT}/.env with real ALPACA + DEEPSEEK keys"
log "  3. Test: PYTHONPATH=${AGENT_ROOT}/src ${AGENT_ROOT}/venv/bin/python -m agent.main --dry-run --as-of 2026-06-26"
log "  4. Once happy: flip AGENT_DRY_RUN=0 in .env"
