#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${PROJ_ROOT}/data/.vicuna_halt" ]]; then
    echo "$(date -u +%H:%M:%S) HALTED (data/.vicuna_halt present) — skipping run"
    exit 0
fi
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/notify.sh"

if [[ -f "${PROJ_ROOT}/venv/bin/activate" ]]; then
    source "${PROJ_ROOT}/venv/bin/activate"
fi

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/sweep_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

log "capturing pre-job hardware snapshot"
bash scripts/pre_job_snapshot.sh > "${LOG}.pre_snapshot.txt" 2>&1 || true

log "nightly sweep starting (full universe × 55 strategies, --parallel)"
python -m macmini.backtest.runner --all --parallel >> "${LOG}" 2>&1 \
    || { log "ERROR: backtest runner failed"; notify high "ALPACA sweep FAILED" "nightly_sweep backtest runner exit 20 — tomorrow's ranking re-uses the previous run_id"; bash scripts/pre_job_snapshot.sh > "${LOG}.post_snapshot.txt" 2>&1 || true; exit 20; }
log "OK: nightly sweep complete"
hc_ping "${HC_UUID_SWEEP:-}"   # dead-man's-switch success ping
touch "${PROJ_ROOT}/logs/.hb_nightly_sweep"   # success-only heartbeat for cron_catchup (log mtime lies when the job hangs mid-run)

log "capturing post-job hardware snapshot"
bash scripts/pre_job_snapshot.sh > "${LOG}.post_snapshot.txt" 2>&1 || true
