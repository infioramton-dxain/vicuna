#!/usr/bin/env bash
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/notify.sh"

if [[ -f "${PROJ_ROOT}/venv/bin/activate" ]]; then
    source "${PROJ_ROOT}/venv/bin/activate"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/weekly_alt_data_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

FAILED=()

run_step() {
    local name="$1"; shift
    log "Step: ${name}"
    if "$@" >> "${LOG}" 2>&1; then
        log "OK: ${name}"
    else
        log "WARN: ${name} failed (continuing — sources are independent)"
        FAILED+=("${name}")
    fi
}

run_step "finnhub_fund (fundamentals/analyst/earnings)" \
    python -m macmini.data.finnhub_fund
run_step "sec_form4 (insider)" \
    python -m macmini.data.sec_form4
run_step "sec_xbrl (company facts → mm_sec_facts, Titan DCF inputs)" \
    python -m macmini.data.sec_xbrl
run_step "cftc_cot (COT)" \
    python -m macmini.data.cftc_cot
run_step "etf_flows" \
    python -m macmini.data.etf_flows
run_step "alpaca_dividends (cash-dividend history, D43)" \
    python -m macmini.data.alpaca_dividends --start 2018-01-01

if [[ ${#FAILED[@]} -eq 0 ]]; then
    log "OK: weekly alt-data refresh complete"
    hc_ping "${HC_UUID_ALT_DATA:-}"
    touch "${PROJ_ROOT}/logs/.hb_weekly_alt_data"   # success-only heartbeat
    exit 0
fi

log "FAILED sources: ${FAILED[*]}"
notify high "ALPACA weekly alt-data INCOMPLETE" \
    "failed: ${FAILED[*]} — other sources refreshed; see $(basename "${LOG}")"
exit 1
