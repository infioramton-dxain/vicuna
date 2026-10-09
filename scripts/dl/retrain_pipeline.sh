#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/notify.sh"
if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${PROJ_ROOT}/logs/dl_retrain_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }
fail() { log "ERROR: $2"; notify high "DL retrain (D33)" "$2 — see ${LOG##*/}"; exit "$1"; }

RUN_DATE="$(date +%Y%m%d)"
MODEL_ID="patchtst_seq64_dm128_${RUN_DATE}"
TRAIN_END="2025-12-31"
VAL_END="2026-06-30"
CKPT_DIR="${PROJ_ROOT}/data/checkpoints/patchtst/retrain_${RUN_DATE}"
DEPLOYED="${PROJ_ROOT}/data/checkpoints/patchtst/deployed.json"

exec 9> "${PROJ_ROOT}/data/.dl_pipeline.lock"
flock -w 7200 9 || fail 31 "could not acquire DL pipeline lock within 2h"

for _ in $(seq 1 24); do
    pgrep -f "scripts/nightly_sweep.sh" >/dev/null || break
    log "sweep still running; waiting 5 min"
    sleep 300
done
pgrep -f "scripts/nightly_sweep.sh" >/dev/null && fail 32 "sweep still running after 2h wait"

log "Step 1: refresh mm_training_labels"
PYTHONPATH=src venv/bin/python -m macmini.dl.labels >> "${LOG}" 2>&1 || fail 33 "label refresh failed"

log "Step 2: rebuild DL feature cache"
rm -rf "${PROJ_ROOT}/data/dl_cache_build"
PYTHONPATH=src venv/bin/python -m macmini.dl.data --output "${PROJ_ROOT}/data/dl_cache_build" >> "${LOG}" 2>&1 \
    || fail 34 "cache rebuild failed"
N_SHARDS="$(find "${PROJ_ROOT}/data/dl_cache_build" -name '*.parquet' | wc -l)"
(( N_SHARDS >= 380 )) || fail 35 "only ${N_SHARDS} shards built (expected ~395); old cache kept"
rm -rf "${PROJ_ROOT}/data/dl_cache_prev"
[[ -d "${PROJ_ROOT}/data/dl_cache" ]] && mv "${PROJ_ROOT}/data/dl_cache" "${PROJ_ROOT}/data/dl_cache_prev"
mv "${PROJ_ROOT}/data/dl_cache_build" "${PROJ_ROOT}/data/dl_cache"
log "cache swapped in: ${N_SHARDS} shards"

log "Step 3: train PatchTST (train<=${TRAIN_END}, val<=${VAL_END})"
mkdir -p "${CKPT_DIR}"
dl_venv/bin/python scripts/dl/train_patchtst.py \
    --cache "${PROJ_ROOT}/data/dl_cache" \
    --train-end "${TRAIN_END}" --val-end "${VAL_END}" \
    --epochs 30 --batch-size 512 \
    --out "${CKPT_DIR}" >> "${LOG}" 2>&1 || fail 36 "training failed"

BEST="$(grep 'best ckpt: ' "${LOG}" | tail -1 | sed 's/.*best ckpt: //')"
[[ -f "${BEST}" ]] || fail 37 "best checkpoint not found (parsed: '${BEST}')"
log "best checkpoint: ${BEST}"

log "Step 4: smoke inference (backfill 15 days — covers the post-${VAL_END} OOS window)"
dl_venv/bin/python scripts/dl/populate_dl_signals.py \
    --checkpoint "${BEST}" --model-id "${MODEL_ID}" \
    --train-end "${TRAIN_END}" --backfill-days 15 >> "${LOG}" 2>&1 \
    || fail 38 "smoke inference failed"

COVERAGE="$(venv/bin/python - "$MODEL_ID" <<'EOF'
import sqlite3, sys
db = sqlite3.connect("file:macmini.db?mode=ro", uri=True)
row = db.execute(
    "SELECT COUNT(*) FROM mm_dl_signals WHERE model_id = ? "
    "AND date = (SELECT MAX(date) FROM mm_dl_signals WHERE model_id = ?)",
    (sys.argv[1], sys.argv[1])).fetchone()
print(row[0])
EOF
)"
log "coverage on latest date: ${COVERAGE} symbols"
(( COVERAGE >= 300 )) || fail 39 "coverage ${COVERAGE} < 300 symbols; deployed.json NOT updated"

log "Step 5: deploy ${MODEL_ID}"
VAL_METRIC="$(basename "${BEST}" | sed -n 's/.*val_dir_acc=\([0-9.]*\)\.ckpt/\1/p')"
venv/bin/python - "$BEST" "$MODEL_ID" "$TRAIN_END" "$VAL_METRIC" "$DEPLOYED" <<'EOF'
import json, sys
from datetime import datetime, timezone
ckpt, model_id, train_end, val_metric, out = sys.argv[1:6]
json.dump({
    "checkpoint": ckpt,
    "model_id": model_id,
    "train_end": train_end,
    "seq_len": 64,
    "best_val_metric": float(val_metric) if val_metric else None,
    "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, open(out, "w"), indent=2)
EOF
log "wrote ${DEPLOYED}"

log "Step 6: informational gate reading (expected FAIL until ~60 OOS days accumulate)"
OOS_START="$(date -d "${VAL_END} + 1 day" +%Y-%m-%d)"
PYTHONPATH=src venv/bin/python scripts/dl/eval_dl_gate.py \
    --model-id "${MODEL_ID}" --min-date "${OOS_START}" >> "${LOG}" 2>&1 \
    || log "WARN: eval_dl_gate reading failed (non-fatal)"

CRON_TMP="$(mktemp)"
if crontab -l > "${CRON_TMP}" 2>/dev/null && [[ -s "${CRON_TMP}" ]]; then
    grep -v 'ONESHOT-dl-retrain' "${CRON_TMP}" > "${CRON_TMP}.new"
    if [[ -s "${CRON_TMP}.new" ]] && crontab "${CRON_TMP}.new"; then
        log "removed ONESHOT crontab line"
    else
        log "WARN: filtered crontab empty or install failed — ONESHOT line NOT removed; remove manually"
    fi
else
    log "WARN: could not read current crontab — ONESHOT line NOT removed; remove manually"
fi
rm -f "${CRON_TMP}" "${CRON_TMP}.new"

log "OK: retrain pipeline complete — ${MODEL_ID} deployed for accumulation"
notify default "DL retrain (D33)" "retrain complete: ${MODEL_ID} deployed, coverage ${COVERAGE} symbols; gate clock started ${OOS_START}"
