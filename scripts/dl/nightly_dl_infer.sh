#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${PROJ_ROOT}/data/.vicuna_halt" ]]; then
    echo "$(date -u +%H:%M:%S) HALTED (data/.vicuna_halt present) — skipping run"
    exit 0
fi
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/notify.sh"
if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/dl_infer_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

DEPLOYED="${PROJ_ROOT}/data/checkpoints/patchtst/deployed.json"
if [[ ! -f "${DEPLOYED}" ]]; then
    log "ERROR: ${DEPLOYED} missing — no deployed DL model"
    notify high "DL nightly infer" "deployed.json missing; nightly DL inference skipped"
    exit 21
fi

if crontab -l 2>/dev/null | grep -q 'ONESHOT-dl-retrain'; then
    log "skip: DL retrain one-shot is scheduled tonight; it covers cache + inference"
    exit 0
fi

exec 9> "${PROJ_ROOT}/data/.dl_pipeline.lock"
if ! flock -n 9; then
    log "skip: DL pipeline lock held (retrain in progress)"
    exit 0
fi

for _ in $(seq 1 12); do
    pgrep -f "scripts/nightly_sweep.sh" >/dev/null || break
    log "sweep still running; waiting 5 min"
    sleep 300
done

log "Step 1: rebuild DL feature cache from macmini.db"
rm -rf "${PROJ_ROOT}/data/dl_cache_build"
PYTHONPATH=src venv/bin/python -m macmini.dl.data --output "${PROJ_ROOT}/data/dl_cache_build" >> "${LOG}" 2>&1 \
    || { log "ERROR: cache rebuild failed"; notify high "DL nightly infer" "cache rebuild failed; see ${LOG##*/}"; exit 22; }

N_SHARDS="$(find "${PROJ_ROOT}/data/dl_cache_build" -name '*.parquet' | wc -l)"
if (( N_SHARDS < 380 )); then
    log "ERROR: only ${N_SHARDS} parquet shards built (expected ~395); keeping old cache"
    notify high "DL nightly infer" "cache rebuild produced only ${N_SHARDS} shards; old cache kept"
    exit 23
fi
rm -rf "${PROJ_ROOT}/data/dl_cache_prev"
[[ -d "${PROJ_ROOT}/data/dl_cache" ]] && mv "${PROJ_ROOT}/data/dl_cache" "${PROJ_ROOT}/data/dl_cache_prev"
mv "${PROJ_ROOT}/data/dl_cache_build" "${PROJ_ROOT}/data/dl_cache"
log "cache swapped in: ${N_SHARDS} shards"

log "Step 2: inference from deployed.json (backfill 3 days, idempotent)"
dl_venv/bin/python scripts/dl/populate_dl_signals.py \
    --deployed-config "${DEPLOYED}" --backfill-days 3 >> "${LOG}" 2>&1 \
    || { log "ERROR: inference failed"; notify high "DL nightly infer" "populate_dl_signals failed; see ${LOG##*/}"; exit 24; }

log "OK: nightly DL cache rebuild + inference complete"
hc_ping "${HC_UUID_DL_INFER:-}"
touch "${PROJ_ROOT}/logs/.hb_dl_infer"
