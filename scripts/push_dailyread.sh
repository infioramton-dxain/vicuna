#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${PROJ_ROOT}/DailyRead.md"
STATE="${PROJ_ROOT}/logs/.dailyread_last_push"
REMOTE_DIR="agent/coord/dailyread"

source "${PROJ_ROOT}/scripts/notify.sh"
if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
HOST="${MACMINI_USER:?}@${MACMINI_HOST:?}"

if [[ -f "${PROJ_ROOT}/data/.vicuna_halt" ]]; then
    echo "$(date -u +%H:%M:%S) halt flag present; skipping dailyread push"
    exit 0
fi

fail() {
    echo "$(date -u +%H:%M:%S) ERROR: $1"
    notify high "DailyRead push" "$1"
    exit 1
}

[[ -f "${SRC}" ]] || fail "DailyRead.md missing at ${SRC}"

sum="$(sha256sum "${SRC}" | cut -d' ' -f1)"
if [[ -f "${STATE}" ]] && [[ "$(cat "${STATE}")" == "${sum}" ]]; then
    echo "$(date -u +%H:%M:%S) unchanged since last push (${sum:0:12}); nothing to do"
    exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
cp "${SRC}" "${tmp}"
sum2="$(sha256sum "${SRC}" | cut -d' ' -f1)"
if [[ "${sum}" != "${sum2}" ]]; then
    echo "$(date -u +%H:%M:%S) DailyRead.md changing under us; deferring to next slot"
    exit 0
fi

ssh "${HOST}" "mkdir -p ${REMOTE_DIR}" || fail "cannot reach mini / mkdir failed"
rsync -a "${tmp}" "${HOST}:${REMOTE_DIR}/DailyRead.md" || fail "rsync failed"

remote_sum="$(ssh "${HOST}" "sha256sum ${REMOTE_DIR}/DailyRead.md" | cut -d' ' -f1)"
[[ "${remote_sum}" == "${sum}" ]] \
    || fail "post-push verify mismatch: remote ${remote_sum:0:12} != local ${sum:0:12}"

echo "${sum}" > "${STATE}"
hc_ping "${HC_UUID_DAILYREAD:-}"
echo "$(date -u +%H:%M:%S) OK: pushed DailyRead.md (${sum:0:12}, $(stat -c%s "${SRC}") bytes)"
