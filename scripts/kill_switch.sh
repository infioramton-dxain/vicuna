#!/usr/bin/env bash
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HALT="${PROJ_ROOT}/data/.vicuna_halt"
COORD_MARKER="${PROJ_ROOT}/coord/LAPTOP_HALT_ACTIVE.md"
MACMINI_USER="${MACMINI_USER:-agent}"
MACMINI_HOST="${MACMINI_HOST:-macmini.local}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"

if [[ -f "${PROJ_ROOT}/.env" ]]; then set -a; source "${PROJ_ROOT}/.env"; set +a; fi
source "${PROJ_ROOT}/scripts/notify.sh"
source "${PROJ_ROOT}/scripts/macmini_resolve.sh" 2>/dev/null || true
type macmini_resolve >/dev/null 2>&1 && macmini_resolve >/dev/null 2>&1 || true
MACMINI_ADDR="${MACMINI_ADDR:-${MACMINI_HOST}}"

push_marker() {
    local ok=0
    rsync -a --timeout=20 -e "ssh ${SSH_OPTS}" "${COORD_MARKER}" \
        "${MACMINI_USER}@${MACMINI_ADDR}:agent/coord/" 2>/dev/null || ok=1
    ssh ${SSH_OPTS} "${MACMINI_USER}@${MACMINI_ADDR}" \
        "date -u +%Y-%m-%dT%H:%M:%SZ > agent/.halt" 2>/dev/null || ok=1
    if [[ ${ok} -eq 0 ]]; then
        echo "mini: agent/.halt written + coord halt marker pushed"
        return 0
    fi
    echo "WARN: could not fully halt the mini — laptop is halted;" \
         "halt the mini directly or retry when it is reachable"
    return 1
}

remove_marker() {
    rm -f "${COORD_MARKER}"
    ssh ${SSH_OPTS} "${MACMINI_USER}@${MACMINI_ADDR}" \
        "rm -f agent/coord/LAPTOP_HALT_ACTIVE.md agent/.halt" 2>/dev/null \
        && echo "mini: agent/.halt + coord halt marker removed" \
        || echo "WARN: could not clear the mini's halt — the mini stays halted" \
                "until agent/.halt is removed; retry when reachable"
}

case "${1:-halt}" in
--status)
    if [[ -f "${HALT}" ]]; then echo "HALTED since: $(cat "${HALT}")"; else echo "active (no halt flag)"; fi
    [[ -f "${COORD_MARKER}" ]] && echo "coord halt marker present (mini instructed to halt)"
    exit 0;;
--resume)
    rm -f "${HALT}"
    remove_marker
    notify default "Vicuna RESUMED" "halt cleared on laptop + mini; crons run at their next slot"
    echo "resumed: halt flag cleared — crons run at their next slot"
    exit 0;;
--cancel-orders|--cancel-orders-yes|halt|"")
    ;;
*)
    echo "usage: kill_switch.sh [--cancel-orders|--cancel-orders-yes|--resume|--status]"; exit 2;;
esac

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "${STAMP}" > "${HALT}"
echo "halt flag written: ${HALT}"

for pat in refresh_and_push.sh refresh_light.sh nightly_sweep.sh \
           nightly_dl_infer.sh kronos_nightly.sh train_zoo.sh; do
    for pid in $(pgrep -f "scripts.*${pat}" 2>/dev/null); do
        [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
        kill "${pid}" 2>/dev/null && echo "killed running: ${pat} (pid ${pid})"
    done
done

cat > "${COORD_MARKER}" <<EOF

**From:** laptop (kill_switch.sh) · **UTC:** ${STAMP}

While this file exists in agent/coord/, every mini-side routine
must NOT approve recommendations or place/modify orders (protective-leg
replacement by recs-protect is EXEMPT — protection stays on). The laptop has
also written agent/.halt — the flag recs/execute.py enforces. Resume = both files removed by the laptop.
EOF
if push_marker; then MARKER_STATE="sent"; else MARKER_STATE="PUSH FAILED — halt the mini directly"; fi
notify high "Vicuna KILL SWITCH" "HALT engaged ${STAMP} — laptop crons stopped, mini halt marker: ${MARKER_STATE}"

if [[ "${1:-}" == "--cancel-orders" || "${1:-}" == "--cancel-orders-yes" ]]; then
    if [[ "${1:-}" == "--cancel-orders" ]]; then
        echo "WARNING: this cancels ALL open orders on the paper account,"
        echo "including GTC protective stop/target legs. Positions stay open"
        echo "and UNPROTECTED until you re-place brackets or resume the mini."
        read -r -p "Type CANCEL to proceed: " ans
    else
        ans="CANCEL"   # --cancel-orders-yes: dashboard did type-to-confirm
    fi
    if [[ "${ans}" == "CANCEL" ]]; then
        cd "${PROJ_ROOT}" && "${PROJ_ROOT}/venv/bin/python" - <<'EOF'
import json, urllib.request
env = {}
with open('.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
base = env.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
req = urllib.request.Request(base + '/v2/orders', method='DELETE', headers={
    'APCA-API-KEY-ID': env['ALPACA_API_KEY'],
    'APCA-API-SECRET-KEY': env['ALPACA_SECRET_KEY']})
with urllib.request.urlopen(req, timeout=30) as r:
    print('cancel-all response:', json.dumps(json.load(r))[:400])
EOF
        notify high "Vicuna KILL SWITCH" "ALL open orders canceled — positions are UNPROTECTED"
    else
        echo "orders NOT canceled"
    fi
fi

echo "HALTED. Autonomy resumes only via: scripts/kill_switch.sh --resume or the dashboard"
