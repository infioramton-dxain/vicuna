#!/usr/bin/env python3
""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("AGENT_SHIPPED_DB", "/home/agent/agent/db/macmini.db"))


def freshness_label(as_of_str: str) -> str:
    try:
        as_of = date.fromisoformat(as_of_str)
    except ValueError:
        return "unknown"
    days = (date.today() - as_of).days
    if days <= 1:
        return f"fresh ({days}d)"
    if days <= 3:
        return f"stale ({days}d) — flag"
    return f"very stale ({days}d) — HALT recommended"


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: shipped DB missing at {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    meta = dict(conn.execute("SELECT key, value FROM mm_export_metadata").fetchall())
    as_of = meta.get("as_of_date", "unknown")
    generated_at = meta.get("generated_at", "unknown")

    regime_row = conn.execute(
        "SELECT as_of_date, vix_close, vix_regime FROM mm_regime ORDER BY as_of_date DESC LIMIT 1"
    ).fetchone()

    picks = conn.execute(
        """
        SELECT rank, symbol, sector, strategy_name, deflated_sharpe,
               profit_factor, max_dd, suggested_size_pct,
               suggested_stop_loss_pct, suggested_target_pct, rationale
        FROM mm_active_picks
        ORDER BY rank
        """
    ).fetchall()

    print(f"DATE_LOCAL: {date.today().isoformat()}")
    print(f"DATA_AS_OF: {as_of}  ({freshness_label(as_of)})")
    print(f"GENERATED_AT: {generated_at}")
    if regime_row:
        print(f"VIX: {regime_row['vix_close']:.2f}  REGIME: {regime_row['vix_regime']}")
    else:
        print("VIX: unknown  REGIME: unknown")
    print(f"PICK_COUNT: {len(picks)}")
    print()

    if not picks:
        print("(No active picks today — leaderboard was empty at export time.)")
        return 0

    sectors: dict[str, int] = {}
    for p in picks:
        sec = p["sector"] or "Unknown"
        sectors[sec] = sectors.get(sec, 0) + 1
    concentrated = [f"{s}={n}" for s, n in sectors.items() if n >= 3]
    if concentrated:
        print(f"SECTOR_CONCENTRATION: {', '.join(concentrated)}")
        print()

    print("PICKS:")
    for p in picks:
        print(
            f"  #{p['rank']:>2}  {p['symbol']:<6}  {p['sector']:<12}  "
            f"{p['strategy_name']}"
        )
        print(
            f"        DSR={p['deflated_sharpe']:.3f}  PF={p['profit_factor']:.1f}  "
            f"MaxDD={p['max_dd']*100:.1f}%  "
            f"size={p['suggested_size_pct']*100:.0f}%  "
            f"stop={p['suggested_stop_loss_pct']*100:.0f}%  "
            f"target={p['suggested_target_pct']*100:.0f}%"
        )
        print(f"        rationale: {p['rationale']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
