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
LOG="${PROJ_ROOT}/logs/refresh_and_push_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

log "capturing pre-job hardware snapshot"
bash scripts/pre_job_snapshot.sh > "${LOG}.pre_snapshot.txt" 2>&1 || true

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o BatchMode=yes"
source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
macmini_resolve 2>>"${LOG}" || true   # MagicDNS drops intermittently (2026-07-16); helper falls back to cached/static IP
DECISIONS_REMOTE="${MACMINI_USER}@${MACMINI_ADDR}:${MACMINI_DEST_DIR%/}/../decisions.jsonl"
DECISIONS_LOCAL="${PROJ_ROOT}/exports/decisions.jsonl"
TRADES_REMOTE="${MACMINI_USER}@${MACMINI_ADDR}:${MACMINI_DEST_DIR%/}/../trades.db"
TRADES_LOCAL="${PROJ_ROOT}/exports/mini_trades_snapshot.db"
mkdir -p "${PROJ_ROOT}/exports"

log "Step 0a: pull decisions.jsonl from mini"
rsync -av --timeout=30 -e "ssh ${SSH_OPTS}" \
    "${DECISIONS_REMOTE}" \
    "${DECISIONS_LOCAL}" \
    >> "${LOG}" 2>&1 \
    || { log "WARN: decisions.jsonl pull failed; attribution will use whatever is already at exports/decisions.jsonl"; notify default "ALPACA refresh WARN" "step 0a decisions.jsonl pull failed (non-fatal)"; }

log "Step 0b: pull mini's trades.db → mm_recs_ledger"
rsync -av --timeout=30 -e "ssh ${SSH_OPTS}" \
    "${TRADES_REMOTE}" \
    "${TRADES_LOCAL}" \
    >> "${LOG}" 2>&1 \
    || { log "WARN: trades.db pull failed; ledger_sync will skip and mm_recs_ledger will be stale"; notify default "ALPACA refresh WARN" "step 0b trades.db pull failed (non-fatal)"; }

python -m macmini.recs.ledger_sync --snapshot "${TRADES_LOCAL}" >> "${LOG}" 2>&1 \
    || { log "WARN: ledger_sync failed; mm_recs_ledger not updated this cycle"; notify default "ALPACA refresh WARN" "step 0b ledger_sync failed (non-fatal)"; }

CASES_REMOTE="${MACMINI_USER}@${MACMINI_ADDR}:${MACMINI_DEST_DIR%/}/../investment-cases/"
CASES_LOCAL="${PROJ_ROOT}/data/investment_cases/"
mkdir -p "${CASES_LOCAL}"

log "Step 0c: pull investment cases from mini"
rsync -av --timeout=30 -e "ssh ${SSH_OPTS}" \
    "${CASES_REMOTE}" \
    "${CASES_LOCAL}" \
    >> "${LOG}" 2>&1 \
    || { log "WARN: investment-cases pull failed; dashboard keeps whatever is already local"; notify default "ALPACA refresh WARN" "step 0c investment-cases pull failed (non-fatal)"; }

log "Step 0d: Alpaca broker-mirror snapshot"
python -m macmini.data.alpaca_broker_mirror >> "${LOG}" 2>&1 \
    || { log "WARN: alpaca_broker_mirror failed; mm_alpaca_* tables not refreshed this cycle"; notify default "ALPACA refresh WARN" "step 0d alpaca_broker_mirror failed (non-fatal)"; }

log "Step 0e: broker-vs-ledger divergence check"
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
    notify default "ALPACA refresh WARN" "step 0e divergence check crashed rc=${div_rc} (non-fatal)"
fi

log "Step 0f: vs-SPY scoreboard"
python -m macmini.recs.vs_spy >> "${LOG}" 2>&1 \
    || { log "WARN: vs_spy scoreboard failed (non-fatal)"; notify default "ALPACA refresh WARN" "step 0f vs_spy scoreboard failed (non-fatal)"; }

BARS_START="$(date -u -d '3 days ago' +%Y-%m-%d)"
BARS_END="$(date -u -d 'tomorrow' +%Y-%m-%d)"

log "Step 1: bars ingest (${BARS_START} → ${BARS_END} exclusive)"
python -m macmini.data.alpaca_bars --start "${BARS_START}" --end "${BARS_END}" >> "${LOG}" 2>&1 \
    || { log "ERROR: alpaca_bars failed"; notify high "ALPACA refresh FAILED" "step 1 alpaca_bars exit 10 — mini will trade STALE DB"; exit 10; }

log "Step 2: VIX refresh (yahoo)"
python -m macmini.data.yahoo_macro >> "${LOG}" 2>&1 \
    || { log "ERROR: yahoo_macro failed"; notify high "ALPACA refresh FAILED" "step 2 yahoo_macro exit 11 — mini will trade STALE DB"; exit 11; }

log "Step 3: sentiment refresh"
python -m macmini.data.sentiment_sync >> "${LOG}" 2>&1 \
    || { log "ERROR: sentiment_sync failed"; notify high "ALPACA refresh FAILED" "step 3 sentiment_sync exit 12 — mini will trade STALE DB"; exit 12; }

log "Step 4: indicators refresh"
python -m macmini.data.indicators >> "${LOG}" 2>&1 \
    || { log "ERROR: indicators failed"; notify high "ALPACA refresh FAILED" "step 4 indicators exit 13 — mini will trade STALE DB"; exit 13; }

log "Step 4.5: recommendation attribution"
python -m macmini.recs.attribution >> "${LOG}" 2>&1 \
    || { log "WARN: attribution failed; scoreboard will be stale until next cycle"; notify default "ALPACA refresh WARN" "step 4.5 attribution failed (non-fatal)"; }

log "Step 4.6: exit-signal intelligence (mm_exit_signals)"
python -m macmini.recs.exit_signals >> "${LOG}" 2>&1 \
    || { log "WARN: exit_signals failed (non-fatal)"; notify default "ALPACA refresh WARN" "step 4.6 exit_signals failed (non-fatal)"; }

log "Step 4.7: forward earnings calendar (audit C6 — blackout input)"
python -m macmini.data.earnings_calendar >> "${LOG}" 2>&1 \
    || { log "WARN: earnings_calendar failed; blackout runs on the previous pull's window"; notify default "ALPACA refresh WARN" "step 4.7 earnings_calendar failed (non-fatal)"; }

log "Step 4.8: fundamentals growth derivation"
python -m macmini.data.fundamentals_growth >> "${LOG}" 2>&1 \
    || { log "WARN: fundamentals_growth failed; dashboard shows the previous day's growth rows"; notify default "ALPACA refresh WARN" "step 4.8 fundamentals_growth failed (non-fatal)"; }

now_et_hm=$(TZ='America/New_York' date +%H%M)
now_et_wd=$(TZ='America/New_York' date +%u)   # 1=Mon..7=Sun
if [[ ${now_et_wd} -le 5 && 10#${now_et_hm} -ge 858 && 10#${now_et_hm} -lt 1945 ]]; then
    log "FREEZE: mm_active_picks 08:58–19:45 ET freeze active (weekday, now ${now_et_hm} ET) — skipping Step 5 (exporter) + Step 6 (push)"
    notify default "ALPACA refresh FREEZE-SKIP" "steps 5+6 skipped: 08:58 freeze active at ${now_et_hm} ET; steps 1-4.5 completed"
    hc_ping "${HC_UUID_REFRESH:-}"   # data refresh succeeded; freeze skip is not a failure
    touch "${PROJ_ROOT}/logs/.hb_refresh_and_push"   # success-only heartbeat for cron_catchup; freeze-skip counts (replaying inside the freeze would be wrong anyway)
    log "capturing post-job hardware snapshot"
    bash scripts/pre_job_snapshot.sh > "${LOG}.post_snapshot.txt" 2>&1 || true
    exit 0
fi

log "Step 5: ranking exporter"
python -m macmini.ranking.exporter >> "${LOG}" 2>&1 \
    || { log "ERROR: exporter failed"; notify high "ALPACA refresh FAILED" "step 5 exporter exit 14 — mini will trade STALE DB"; exit 14; }

log "Step 6: push to Mac mini"
bash scripts/push_to_macmini.sh >> "${LOG}" 2>&1 \
    || { log "ERROR: push failed"; notify high "ALPACA refresh FAILED" "step 6 push exit 15 — mini will trade STALE DB"; exit 15; }

log "OK: refresh + push complete"
hc_ping "${HC_UUID_REFRESH:-}"   # dead-man's-switch success ping
touch "${PROJ_ROOT}/logs/.hb_refresh_and_push"   # success-only heartbeat for cron_catchup (log mtime lies when the job hangs mid-run)

log "capturing post-job hardware snapshot"
bash scripts/pre_job_snapshot.sh > "${LOG}.post_snapshot.txt" 2>&1 || true
