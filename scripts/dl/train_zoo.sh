#!/usr/bin/env bash
set -uo pipefail

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
LOG="${PROJ_ROOT}/logs/train_zoo_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

exec 9> "${PROJ_ROOT}/data/.dl_pipeline.lock"
if ! flock -n 9; then
    log "skip: DL pipeline lock held"
    exit 0
fi

if [[ "$(date +%-d)" -le 7 ]]; then
    log "Step 0: monthly Optuna search (first Sunday) — 100 trials/model"
    dl_venv/bin/python scripts/dl/tune_zoo.py --all --trials 100 >> "${LOG}" 2>&1 \
        || { log "ERROR: tuning failed"; notify high "Zoo weekly train" "tune_zoo failed; see ${LOG##*/}"; exit 40; }
fi

log "Step 1: train all zoo models"
dl_venv/bin/python scripts/dl/train_zoo.py --all >> "${LOG}" 2>&1 \
    || { log "ERROR: zoo training failed"; notify high "Zoo weekly train" "train_zoo --all failed; see ${LOG##*/}"; exit 41; }

log "Step 2: immediate OOS inference from the fresh checkpoints"
dl_venv/bin/python scripts/dl/zoo_infer.py >> "${LOG}" 2>&1 \
    || { log "ERROR: post-train inference failed"; notify high "Zoo weekly train" "zoo_infer failed; see ${LOG##*/}"; exit 42; }

log "Step 3: exit-policy trajectory cache refresh (Phase C1, CPU, ~40 min at 8 workers)"
PYTHONPATH=src venv/bin/python -m macmini.rl.exit_policy.trajectories --workers 8 >> "${LOG}" 2>&1 \
    || log "WARN: trajectory extraction failed (non-fatal) — see ${LOG##*/}"

log "Step 4: train exit policies (Phase C2 — DiscreteCQL + DiscreteBCQ)"
PYTHONPATH=src dl_venv/bin/python -m macmini.rl.exit_policy.train --algo all >> "${LOG}" 2>&1 \
    || log "WARN: exit-policy training failed (non-fatal) — see ${LOG##*/}"

log "Step 5: exit-uplift eval reports (Phase C3)"
for _algo in cql bcq; do
    PYTHONPATH=src dl_venv/bin/python -m macmini.rl.exit_policy.eval \
        --algo "${_algo}" --out "${PROJ_ROOT}/exports/rl_exit_uplift_${_algo}.json" >> "${LOG}" 2>&1 \
        || log "WARN: exit-uplift eval (${_algo}) failed (non-fatal) — see ${LOG##*/}"
done

log "Step 3: scoreboard refresh"
PYTHONPATH=src venv/bin/python scripts/dl/eval_model_scoreboard.py >> "${LOG}" 2>&1 \
    || { log "ERROR: scoreboard failed"; notify high "Zoo weekly train" "scoreboard failed; see ${LOG##*/}"; exit 43; }

log "OK: weekly zoo retrain complete"
hc_ping "${HC_UUID_ZOO:-}"
touch "${PROJ_ROOT}/logs/.hb_train_zoo"
