""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

FORMULA_VERSION = "bar_v1"

SHADOW_RISK_BASE = 0.005
SHADOW_MULT_MIN = 0.5
SHADOW_MULT_MAX = 2.0
SHADOW_SIZE_FLOOR = 0.02
SHADOW_SIZE_CAP = 0.10


def conviction_multiplier(deflated_sharpe: float | None) -> float:
    ""
    if deflated_sharpe is None:
        return 1.0
    clamped = min(max(float(deflated_sharpe), 0.0), 1.0)
    return SHADOW_MULT_MIN + (SHADOW_MULT_MAX - SHADOW_MULT_MIN) * clamped


def shadow_size_pct(deflated_sharpe: float | None, stop_frac: float) -> float:
    ""
    mult = conviction_multiplier(deflated_sharpe)
    raw = SHADOW_RISK_BASE * mult / stop_frac
    return round(min(max(raw, SHADOW_SIZE_FLOOR), SHADOW_SIZE_CAP), 4)


def record_shadow_size(
    conn: sqlite3.Connection,
    *,
    rec_id: str,
    as_of_date: str,
    symbol: str,
    strategy_id: int,
    deflated_sharpe: float | None,
    real_size_pct: float,
    stop_frac: float,
) -> None:
    ""
    conn.execute(
        """INSERT OR REPLACE INTO mm_shadow_sizing
           (rec_id, as_of_date, symbol, strategy_id, deflated_sharpe,
            conviction_mult, real_size_pct, shadow_size_pct, stop_frac,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (rec_id, as_of_date, symbol, int(strategy_id), deflated_sharpe,
         round(conviction_multiplier(deflated_sharpe), 4), real_size_pct,
         shadow_size_pct(deflated_sharpe, stop_frac), stop_frac,
         datetime.now(timezone.utc).isoformat()),
    )
