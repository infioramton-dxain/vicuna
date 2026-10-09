""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.dl.data import compute_indicators_for_bars

log = logging.getLogger(__name__)

INDICATOR_COLUMNS = (
    "sma20", "sma50", "sma200",
    "ema12", "ema26", "ema200",
    "rsi14", "rsi2",
    "atr14", "adx14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "williams_r14", "ibs", "volume_zscore_50",
)


def _to_real(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _load_bars(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume "
        "FROM mm_bars WHERE symbol = ? ORDER BY date",
        conn, params=[symbol], parse_dates=["date"],
    )
    return df.set_index("date") if not df.empty else df


def _write_indicators(conn: sqlite3.Connection, symbol: str, indicators: pd.DataFrame) -> int:
    ""
    if indicators.empty:
        return 0
    conn.execute("DELETE FROM mm_indicators_daily WHERE symbol = ?", (symbol,))
    payload = []
    for d, row in indicators.iterrows():
        payload.append((
            symbol, d.strftime("%Y-%m-%d"),
            *(_to_real(row[c]) for c in INDICATOR_COLUMNS),
        ))
    placeholders = ", ".join("?" for _ in INDICATOR_COLUMNS)
    conn.executemany(
        f"""INSERT INTO mm_indicators_daily
            (symbol, date, {", ".join(INDICATOR_COLUMNS)})
            VALUES (?, ?, {placeholders})""",
        payload,
    )
    conn.commit()
    return len(payload)


def populate(
    symbols: list[str] | None = None,
    db_path: Path | str | None = None,
) -> dict:
    started = datetime.now(timezone.utc)
    counts = {"symbols_done": 0, "rows_written": 0, "symbols_skipped": 0}
    with connect(db_path) as c:
        if symbols is None:
            symbols = [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol"
            ).fetchall()]
        for sym in symbols:
            bars = _load_bars(c, sym)
            if bars.empty:
                counts["symbols_skipped"] += 1
                continue
            indicators = compute_indicators_for_bars(bars)
            n = _write_indicators(c, sym, indicators)
            if n > 0:
                counts["symbols_done"] += 1
                counts["rows_written"] += n
                log.info("  %s: %d rows", sym, n)
            else:
                counts["symbols_skipped"] += 1
    counts["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Populate mm_indicators_daily")
    p.add_argument("--db", default=None)
    p.add_argument("--symbols", help="comma-separated; default all in mm_bars")
    args = p.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    out = populate(symbols=syms, db_path=args.db)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
