#!/usr/bin/env bash

notify() {
  [ -z "${NTFY_TOPIC:-}" ] && return 0   # no-op if unconfigured
  curl -fsS -m 10 --retry 2 -H "Title: $2" -H "Priority: $1" \
       -d "$3" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

hc_ping() { [ -z "${1:-}" ] && return 0; curl -fsS -m 10 --retry 2 "https://hc-ping.com/$1" >/dev/null 2>&1 || true; }
