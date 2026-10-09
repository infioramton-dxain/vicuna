#!/usr/bin/env bash

macmini_resolve() {
    local root="${PROJ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local cache="${root}/logs/.macmini_ip"
    local ip
    ip="$(getent hosts "${MACMINI_HOST}" 2>/dev/null | awk '{print $1; exit}')"
    if [[ -n "${ip}" ]]; then
        MACMINI_ADDR="${MACMINI_HOST}"
        mkdir -p "${root}/logs" 2>/dev/null || true
        printf '%s\n' "${ip}" > "${cache}" 2>/dev/null || true
        return 0
    fi
    if [[ -s "${cache}" ]]; then
        ip="$(head -n1 "${cache}")"
        if [[ -n "${ip}" ]]; then
            MACMINI_ADDR="${ip}"
            echo "macmini_resolve: WARN ${MACMINI_HOST} did not resolve; using cached IP ${ip}" >&2
            return 0
        fi
    fi
    if [[ -n "${MACMINI_FALLBACK_IP:-}" ]]; then
        MACMINI_ADDR="${MACMINI_FALLBACK_IP}"
        echo "macmini_resolve: WARN ${MACMINI_HOST} did not resolve; using .env fallback IP ${MACMINI_ADDR}" >&2
        return 0
    fi
    MACMINI_ADDR="${MACMINI_HOST}"
    echo "macmini_resolve: WARN ${MACMINI_HOST} did not resolve and no fallback IP available" >&2
    return 1
}
