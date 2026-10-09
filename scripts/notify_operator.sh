#!/usr/bin/env bash
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${PROJ_ROOT}/.env" ]]; then
    set -a; source "${PROJ_ROOT}/.env"; set +a
fi
MACMINI_USER="${MACMINI_USER:-agent}"
MACMINI_HOST="${MACMINI_HOST:-macmini.local}"
source "${PROJ_ROOT}/scripts/macmini_resolve.sh"
macmini_resolve || true

SUBJECT="${1:?usage: notify_operator.sh <subject> [body]}"
if [[ $# -ge 2 ]]; then
    BODY="$2"
else
    BODY="$(cat)"
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SLUG="$(printf '%s' "${SUBJECT}" | tr '[:upper:] ' '[:lower:]_' | tr -cd 'a-z0-9_-' | cut -c1-40)"
[[ -z "${SLUG}" ]] && SLUG="alert"
OUT="${PROJ_ROOT}/coord/LAPTOP_ALERT_${TS}_${SLUG}.md"

mkdir -p "${PROJ_ROOT}/coord"
cat > "${OUT}" <<EOF

**From:** laptop-CC · **UTC:** ${TS} · **Priority:** high · **Delivery:** relay-me

${BODY}

---
*Written by \`scripts/notify_operator.sh\`. The mini-side relay routine
is expected to forward this and then mark it delivered (rename to
\`LAPTOP_ALERT_${TS}_${SLUG}.delivered.md\` or move to
\`agent/coord/_delivered/\`).*
EOF

rsync -av --timeout=15 \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15" \
    "${OUT}" \
    "${MACMINI_USER}@${MACMINI_ADDR}:agent/coord/" >/dev/null 2>&1 \
    || echo "notify_operator: rsync of ${OUT##*/} to mini failed (file still on laptop)" >&2

source "${PROJ_ROOT}/scripts/notify.sh"
notify high "VICUNA: ${SUBJECT}" "${BODY}"

echo "notify_operator: wrote ${OUT##*/}"
exit 0
