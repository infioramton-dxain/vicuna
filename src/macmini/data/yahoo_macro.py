""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from macmini.config import resolve_db
from macmini.data._db import connect

log = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ("^VIX",)


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    ""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in df.index],
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "adj_close": df["Adj Close"].astype(float) if "Adj Close" in df.columns else df["Close"].astype(float),
        "volume": df["Volume"].fillna(0).astype("int64"),
    })
    return out.reset_index(drop=True)


def ingest_one(
    conn,
    symbol: str,
    start: str,
    end: str,
) -> int:
    df = _download(symbol, start, end)
    if df.empty:
        log.warning("%s: yahoo returned 0 rows", symbol)
        return 0
    payload = [
        (symbol, row["date"], row["open"], row["high"], row["low"],
         row["close"], row["adj_close"], int(row["volume"]), "yahoo")
        for _, row in df.iterrows()
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO mm_bars
           (symbol, date, open, high, low, close, adj_close, volume, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        payload,
    )
    conn.commit()
    return len(payload)


def run(
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    db_path: Path | str | None = None,
) -> dict:
    symbols = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counts = {"symbols_done": 0, "rows_written": 0, "symbols_failed": 0}
    run_id = f"yahoo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    started = datetime.now(timezone.utc).isoformat()

    with connect(db_path) as c:
        for sym in symbols:
            try:
                n = ingest_one(c, sym, start, end)
                if n > 0:
                    counts["symbols_done"] += 1
                    counts["rows_written"] += n
                else:
                    counts["symbols_failed"] += 1
            except Exception as exc:
                log.error("%s: %s", sym, exc)
                counts["symbols_failed"] += 1
        c.execute(
            """INSERT INTO mm_bars_ingest_log
               (run_id, started_at, ended_at, symbols_requested, symbols_ok,
                symbols_empty, symbols_failed, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, started, datetime.now(timezone.utc).isoformat(),
             len(symbols), counts["symbols_done"], 0, counts["symbols_failed"],
             f"yahoo macro: {','.join(symbols)}"),
        )
        c.commit()
    counts["run_id"] = run_id
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Yahoo macro/index ingester")
    p.add_argument("--db", default=None)
    p.add_argument("--symbols", help="comma-separated; default ^VIX")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args(argv)
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    out = run(symbols=syms, start=args.start, end=args.end, db_path=args.db)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
