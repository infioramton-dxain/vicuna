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
LOG="${PROJ_ROOT}/logs/kronos_${STAMP}.log"
mkdir -p "${PROJ_ROOT}/logs"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }

for _ in $(seq 1 12); do
    if ! pgrep -f "scripts/nightly_sweep.sh" >/dev/null \
        && ! pgrep -f "scripts/dl/nightly_dl_infer.sh" >/dev/null; then
        break
    fi
    log "sweep/DL infer still running; waiting 5 min"
    sleep 300
done

log "Step 1: Kronos full-universe forecast"
dl_venv/bin/python scripts/dl/kronos_nightly.py >> "${LOG}" 2>&1 \
    || { log "ERROR: kronos forecast failed"; notify high "Kronos nightly" "kronos_nightly.py failed; see ${LOG##*/}"; exit 31; }

log "Step 1b: zoo OOS inference (skips models not yet trained)"
dl_venv/bin/python scripts/dl/zoo_infer.py >> "${LOG}" 2>&1 \
    || { log "ERROR: zoo inference failed"; notify high "Kronos nightly" "zoo_infer failed; see ${LOG##*/}"; exit 33; }

log "Step 1b4: RL exit-policy annotation (non-fatal)"
PYTHONPATH=src dl_venv/bin/python -m macmini.rl.exit_policy.infer >> "${LOG}" 2>&1 \
    || log "WARN: exit-policy annotation failed or no deployable checkpoint (non-fatal) — see ${LOG##*/}"

log "Step 1b2: BMA blend (shadow model)"
PYTHONPATH=src venv/bin/python scripts/dl/blend_bma.py >> "${LOG}" 2>&1 \
    || { log "ERROR: blend_bma failed"; notify high "Kronos nightly" "blend_bma failed; see ${LOG##*/}"; exit 35; }

log "Step 1b3: LLM shadow arbiter (non-fatal)"
PYTHONPATH=src dl_venv/bin/python -m macmini.llm.arbiter >> "${LOG}" 2>&1 \
    || log "WARN: arbiter failed or refused (non-fatal) — see ${LOG##*/}"

log "Step 1b5: TITAN-1 quality-candidate scan (non-fatal, annotation only)"
PYTHONPATH=src venv/bin/python -m macmini.recs.titan_ql >> "${LOG}" 2>&1 \
    || log "WARN: titan_ql scan failed (non-fatal) — see ${LOG##*/}"

log "Step 1c: model scoreboard refresh"
PYTHONPATH=src venv/bin/python scripts/dl/eval_model_scoreboard.py >> "${LOG}" 2>&1 \
    || { log "ERROR: scoreboard eval failed"; notify high "Kronos nightly" "eval_model_scoreboard failed; see ${LOG##*/}"; exit 34; }

log "Step 1c2: selector scores (shadow — non-fatal)"
PYTHONPATH=src dl_venv/bin/python -m macmini.selector.score >> "${LOG}" 2>&1 \
    || log "WARN: selector scoring failed (non-fatal) — see ${LOG##*/}"

log "Step 1d: Titan analyst notes (context only — non-fatal)"
PYTHONPATH=src venv/bin/python -m macmini.recs.titan_notes >> "${LOG}" 2>&1 \
    || log "WARN: titan_notes failed (non-fatal) — see ${LOG##*/}"

log "Step 1e: ontology export to the Obsidian vault (non-fatal)"
PYTHONPATH=src venv/bin/python -m macmini.ontology.export >> "${LOG}" 2>&1 \
    || log "WARN: ontology export failed (non-fatal) — see ${LOG##*/}"

log "Step 2: deepseek daily briefing"
PYTHONPATH=src dl_venv/bin/python -m macmini.llm.commentary --generate >> "${LOG}" 2>&1 \
    || { log "ERROR: briefing generation failed"; notify high "Kronos nightly" "deepseek briefing failed; see ${LOG##*/}"; exit 32; }

log "Step 3: push DB to mini (so tomorrow's 09:00 brief sees tonight's"
log "        Kronos + candidates instead of the 19:50 pre-Kronos snapshot)"
bash scripts/push_to_macmini.sh >> "${LOG}" 2>&1 \
    || { log "WARN: post-Kronos push failed (non-fatal; 11:35 push will carry it)"; \
         notify default "Kronos nightly" "post-Kronos DB push failed; see ${LOG##*/}"; }

log "OK: Kronos forecast + briefing complete"
hc_ping "${HC_UUID_KRONOS:-}"
touch "${PROJ_ROOT}/logs/.hb_kronos"
