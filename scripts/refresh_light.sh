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

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/refresh_light_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

log "Step 1: sentiment refresh"
python -m macmini.data.sentiment_sync >> "${LOG}" 2>&1 \
    || { log "ERROR: sentiment_sync failed"; notify high "ALPACA light-refresh FAILED" "midday sentiment_sync exit 12 — no push this cycle"; exit 12; }

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
if [[ -n "${MACMINI_USER:-}" && -n "${MACMINI_HOST:-}" && -n "${MACMINI_DEST_DIR:-}" ]]; then
    source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
    macmini_resolve 2>>"${LOG}" || true   # MagicDNS drops intermittently (2026-07-16); falls back to cached/static IP
    SSH_OPTS="-o StrictHostKeyChecking=accept-new -o BatchMode=yes"

    log "Step 1a: pull mini's trades.db → mm_recs_ledger"
    TRADES_LOCAL="${PROJ_ROOT}/exports/mini_trades_snapshot.db"
    mkdir -p "${PROJ_ROOT}/exports"
    rsync -av --timeout=30 -e "ssh ${SSH_OPTS}" \
        "${MACMINI_USER}@${MACMINI_ADDR}:${MACMINI_DEST_DIR%/}/../trades.db" \
        "${TRADES_LOCAL}" \
        >> "${LOG}" 2>&1 \
        || { log "WARN: trades.db pull failed; mm_recs_ledger stays at the last sync"; notify default "ALPACA light-refresh WARN" "step 1a trades.db pull failed (non-fatal)"; }
    python -m macmini.recs.ledger_sync --snapshot "${TRADES_LOCAL}" >> "${LOG}" 2>&1 \
        || { log "WARN: ledger_sync failed (non-fatal)"; notify default "ALPACA light-refresh WARN" "step 1a ledger_sync failed (non-fatal)"; }
    python -m macmini.data.alpaca_broker_mirror >> "${LOG}" 2>&1 \
        || { log "WARN: alpaca_broker_mirror failed (non-fatal)"; notify default "ALPACA light-refresh WARN" "step 1a broker mirror failed (non-fatal)"; }
    DIVERGENCE_SUMMARY="${PROJ_ROOT}/exports/divergence_summary.txt"
    rm -f "${DIVERGENCE_SUMMARY}"
    div_rc=0
    python -m macmini.recs.divergence --summary-file "${DIVERGENCE_SUMMARY}" >> "${LOG}" 2>&1 || div_rc=$?
    if [ "${div_rc}" -eq 3 ]; then
        log "WARN: broker-vs-ledger divergence detected (see mm_broker_divergences + dashboard)"
        notify high "ALPACA broker-ledger DIVERGENCE" "$(cat "${DIVERGENCE_SUMMARY}" 2>/dev/null || echo 'mismatch found — see dashboard')"
        "${PROJ_ROOT}/scripts/notify_operator.sh" "broker-ledger DIVERGENCE" \
            "$(cat "${DIVERGENCE_SUMMARY}" 2>/dev/null || echo 'mismatch found — see dashboard')" \
            >> "${LOG}" 2>&1 || log "WARN: alert-drop for divergence failed (ntfy already fired above)"
    elif [ "${div_rc}" -ne 0 ]; then
        log "WARN: divergence check crashed (rc=${div_rc}, non-fatal)"
    fi

    log "Step 1b: pull investment cases from mini"
    mkdir -p "${PROJ_ROOT}/data/investment_cases"
    rsync -av --timeout=30 -e "ssh ${SSH_OPTS}" \
        "${MACMINI_USER}@${MACMINI_ADDR}:${MACMINI_DEST_DIR%/}/../investment-cases/" \
        "${PROJ_ROOT}/data/investment_cases/" \
        >> "${LOG}" 2>&1 \
        || { log "WARN: investment-cases pull failed; dashboard keeps whatever is already local"; notify default "ALPACA light-refresh WARN" "investment-cases pull failed (mini unreachable?) — push may fail next"; }
fi

log "Step 1c: vs-SPY scoreboard (non-fatal)"
python -m macmini.recs.vs_spy >> "${LOG}" 2>&1 \
    || { log "WARN: vs_spy scoreboard failed (non-fatal)"; notify default "ALPACA light-refresh WARN" "vs_spy scoreboard failed (non-fatal)"; }

log "Step 2: push to Mac mini"
bash scripts/push_to_macmini.sh >> "${LOG}" 2>&1 \
    || { log "ERROR: push failed"; notify high "ALPACA light-refresh FAILED" "midday push exit 15 — mini keeps previous (stale) DB"; exit 15; }

log "OK: light refresh + push complete"
hc_ping "${HC_UUID_LIGHT:-}"
touch "${PROJ_ROOT}/logs/.hb_refresh_light"   # success-only heartbeat for cron_catchup (log mtime lies when the job hangs mid-run)
