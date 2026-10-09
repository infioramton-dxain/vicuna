#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="${PROJ_ROOT}/macmini.db"
BACKUP="${PROJ_ROOT}/exports/macmini.db.backup"
LOG_DIR="${PROJ_ROOT}/logs"
DRY=0

usage() {
    cat <<'EOF'
Usage: push_to_macmini.sh [--dry-run]

Ships macmini.db to the Mac mini via Tailscale rsync + atomic replace.
Reads MACMINI_HOST / MACMINI_USER / MACMINI_DEST_DIR from .env.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n) DRY=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi

for var in MACMINI_HOST MACMINI_USER MACMINI_DEST_DIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} not set in .env" >&2
        exit 3
    fi
done

mkdir -p "${LOG_DIR}" "${PROJ_ROOT}/exports"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/push_${STAMP}.log"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$1" | tee -a "${LOG}"; }
run() {
    log "+ $*"
    if [[ "${DRY}" -eq 1 ]]; then return 0; fi
    "$@"
}

log "Phase 8 push starting (db=${DB})"
rm -f "${BACKUP}"
run python3 -c "
import sqlite3, sys
src = sqlite3.connect('${DB}')
dst = sqlite3.connect('${BACKUP}')
with dst:
    src.backup(dst)
src.close(); dst.close()
"

if [[ "${DRY}" -eq 0 && ! -s "${BACKUP}" ]]; then
    log "ERROR: backup file empty or missing"
    exit 4
fi

if [[ "${DRY}" -eq 0 ]]; then
    INTEG="$(python3 -c "
import sqlite3
c = sqlite3.connect('${BACKUP}')
print(c.execute('PRAGMA integrity_check').fetchone()[0])
c.close()
" | head -n 1)"
    log "integrity_check: ${INTEG}"
    if [[ "${INTEG}" != "ok" ]]; then
        log "ERROR: integrity check failed"
        exit 5
    fi
    LOCAL_SIZE="$(stat -c%s "${BACKUP}")"
    log "backup size: ${LOCAL_SIZE} bytes"
fi

source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
macmini_resolve || true
REMOTE="${MACMINI_USER}@${MACMINI_ADDR}"
DEST="${MACMINI_DEST_DIR%/}"

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o BatchMode=yes"

run ssh ${SSH_OPTS} "${REMOTE}" "mkdir -p '${DEST}'"

run rsync -avz --inplace --partial \
    -e "ssh ${SSH_OPTS}" \
    "${BACKUP}" \
    "${REMOTE}:${DEST}/macmini.db.staging"

if [[ "${DRY}" -eq 0 ]]; then
    REMOTE_SIZE="$(ssh ${SSH_OPTS} "${REMOTE}" "stat -f%z '${DEST}/macmini.db.staging' 2>/dev/null || stat -c%s '${DEST}/macmini.db.staging'")"
    if [[ "${REMOTE_SIZE}" != "${LOCAL_SIZE}" ]]; then
        log "ERROR: size mismatch (local=${LOCAL_SIZE} remote=${REMOTE_SIZE})"
        exit 6
    fi
    log "remote size matches: ${REMOTE_SIZE}"
fi

run ssh ${SSH_OPTS} "${REMOTE}" "mv '${DEST}/macmini.db.staging' '${DEST}/macmini.db'"

if [[ "${DRY}" -eq 0 ]]; then
    LOCAL_STAMP="$(python3 -c "
import sqlite3
c = sqlite3.connect('file:${BACKUP}?mode=ro', uri=True)
print(c.execute(\"SELECT value FROM mm_export_metadata WHERE key='generated_at'\").fetchone()[0])
c.close()
" 2>/dev/null || true)"
    REMOTE_STAMP="$(ssh ${SSH_OPTS} "${REMOTE}" "python3 -c \"
import sqlite3
c = sqlite3.connect('file:${DEST}/macmini.db?mode=ro', uri=True)
print(c.execute('SELECT value FROM mm_export_metadata WHERE key=\\'generated_at\\'').fetchone()[0])
c.close()
\" 2>/dev/null" | head -n 1)"
    if [[ -z "${REMOTE_STAMP}" ]]; then
        log "ERROR: remote DB did not open or has no export stamp AFTER replace"
        exit 7
    elif [[ -n "${LOCAL_STAMP}" && "${REMOTE_STAMP}" != "${LOCAL_STAMP}" ]]; then
        log "ERROR: remote export stamp ${REMOTE_STAMP} != pushed ${LOCAL_STAMP}"
        exit 7
    else
        log "remote open + export stamp ok: ${REMOTE_STAMP}"
    fi
fi

log "OK: pushed macmini.db → ${REMOTE}:${DEST}/macmini.db"
