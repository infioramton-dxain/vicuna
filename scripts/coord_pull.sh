#!/usr/bin/env bash
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"

source "${PROJ_ROOT}/scripts/notify.sh"
if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
MACMINI_USER="${MACMINI_USER:-agent}"
MACMINI_HOST="${MACMINI_HOST:-macmini.local}"
source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
macmini_resolve || true

LOG="${PROJ_ROOT}/logs/coord_pull.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" >> "${LOG}"; }

CHANGES="$(rsync -avi --timeout=30 \
    "${MACMINI_USER}@${MACMINI_HOST}:agent/coord/" \
    "${PROJ_ROOT}/coord/mini_files/" 2>>"${LOG}")" \
    || { log "skip: mini unreachable (rsync rc=$?)"; exit 0; }

NEW_MINI="$(printf '%s\n' "${CHANGES}" \
    | awk '$1 ~ /^>f/ { print $2 }' | grep '^MINI_' || true)"

if [[ -n "${NEW_MINI}" ]]; then
    COUNT="$(printf '%s\n' "${NEW_MINI}" | wc -l)"
    log "pulled ${COUNT} new/updated MINI file(s): $(echo "${NEW_MINI}" | tr '\n' ' ')"
    notify default "Vicuna coord: ${COUNT} new mini file(s)" \
        "$(printf '%s\n' "${NEW_MINI}" | head -10)"
else
    log "pull ok, nothing new"
fi
