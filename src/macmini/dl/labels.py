""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.config import resolve_db
from macmini.data._db import connect

log = logging.getLogger(__name__)

FLAT_LOWER = -0.02
FLAT_UPPER = 0.02


def compute_labels(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    if "close" not in bars.columns or bars.empty:
        return pd.DataFrame()
    close = bars["close"].astype(float).to_numpy()
    n = len(close)
    if n < 2:
        return pd.DataFrame()

    f1 = np.full(n, np.nan)
    f5 = np.full(n, np.nan)
    f10 = np.full(n, np.nan)
    if n >= 2:
        f1[: n - 1] = close[1:] / close[:-1] - 1
    if n >= 6:
        f5[: n - 5] = close[5:] / close[:-5] - 1
    if n >= 11:
        f10[: n - 10] = close[10:] / close[:-10] - 1

    f5_class = np.full(n, -1, dtype="int64")
    mask = ~np.isnan(f5)
    f5_class[mask] = np.where(
        f5[mask] < FLAT_LOWER, 0,
        np.where(f5[mask] > FLAT_UPPER, 2, 1),
    )

    f5_sharpe = np.full(n, np.nan)
    max_dd_5d = np.full(n, np.nan)
    for i in range(n - 5):
        window = close[i + 1 : i + 6]
        entry = close[i]
        daily_rets = window[1:] / window[:-1] - 1 if len(window) >= 2 else np.array([])
        if daily_rets.size >= 2 and daily_rets.std(ddof=1) > 0:
            f5_sharpe[i] = float((daily_rets.mean() / daily_rets.std(ddof=1)) * math.sqrt(252))
        else:
            f5_sharpe[i] = 0.0
        equity = window / entry
        peak = np.maximum.accumulate(equity)
        max_dd_5d[i] = float(((equity - peak) / peak).min())

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in bars.index],
        "forward_1d_return": f1,
        "forward_5d_return": f5,
        "forward_10d_return": f10,
        "forward_5d_class": f5_class,
        "forward_5d_sharpe": f5_sharpe,
        "max_dd_5d": max_dd_5d,
    })


def _load_bars(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, close FROM mm_bars WHERE symbol = ? ORDER BY date",
        conn, params=[symbol], parse_dates=["date"],
    )
    return df.set_index("date") if not df.empty else df


def _write_labels(conn: sqlite3.Connection, symbol: str, labels: pd.DataFrame) -> int:
    ""
    if labels.empty:
        return 0
    conn.execute("DELETE FROM mm_training_labels WHERE symbol = ?", (symbol,))
    payload = [
        (symbol, row["date"],
         _to_real(row["forward_1d_return"]),
         _to_real(row["forward_5d_return"]),
         _to_real(row["forward_10d_return"]),
         int(row["forward_5d_class"]) if row["forward_5d_class"] >= 0 else None,
         _to_real(row["forward_5d_sharpe"]),
         _to_real(row["max_dd_5d"]))
        for _, row in labels.iterrows()
    ]
    conn.executemany(
        """INSERT INTO mm_training_labels
           (symbol, date, forward_1d_return, forward_5d_return, forward_10d_return,
            forward_5d_class, forward_5d_sharpe, max_dd_5d)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        payload,
    )
    conn.commit()
    return len(payload)


def _to_real(v) -> float | None:
    ""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def populate(
    symbols: list[str] | None = None,
    db_path: Path | str | None = None,
) -> dict:
    ""
    started = datetime.now(timezone.utc)
    counts = {"symbols_done": 0, "rows_written": 0, "symbols_empty": 0}
    with connect(db_path) as c:
        if symbols is None:
            symbols = [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol"
            ).fetchall()]
        for sym in symbols:
            bars = _load_bars(c, sym)
            if bars.empty:
                counts["symbols_empty"] += 1
                continue
            labels = compute_labels(bars)
            if labels.empty:
                counts["symbols_empty"] += 1
                continue
            n = _write_labels(c, sym, labels)
            counts["symbols_done"] += 1
            counts["rows_written"] += n
            log.info("  %s: %d labels", sym, n)
    counts["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Populate mm_training_labels")
    p.add_argument("--db", default=None)
    p.add_argument("--symbols", help="comma-separated subset (default: all in mm_bars)")
    args = p.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    out = populate(symbols=syms, db_path=args.db)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
