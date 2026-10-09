""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from macmini.config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DB_PATH,
    SYMBOLS_CSV,
    resolve_db,
)
from macmini.data._window import clamp_feed_end

log = logging.getLogger(__name__)

BATCH_SIZE = 100
INTER_BATCH_SLEEP_SECONDS = 0.4


def _client():
    ""
    from alpaca.data.historical import StockHistoricalDataClient

    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Populate .env first."
        )
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _request(symbols: list[str], start: datetime, end: datetime):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed

    return StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
    )


def load_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def ingest(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    db_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict:
    ""
    db = resolve_db(db_path)
    symbols = list(symbols)
    run_id = f"alpaca-{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(timezone.utc).isoformat()

    clamped = clamp_feed_end(end)
    if clamped != end:
        log.info("end clamped %s → %s (recent-SIP / partial-bar guard, D71)", end, clamped)
        end = clamped
    if end <= start:
        log.info("window empty after clamp (start=%s end=%s); nothing to ingest", start, end)
        return {
            "run_id": run_id,
            "requested": len(symbols),
            "ok": 0,
            "empty": len(symbols),
            "failed": 0,
            "rows_written": 0,
        }

    client = _client()

    ok = 0
    empty = 0
    failed = 0
    rows_written = 0

    from macmini.data._db import connect as _db_connect
    with _db_connect(db) as conn:

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            try:
                resp = client.get_stock_bars(_request(batch, start, end))
            except Exception as e:
                log.warning("batch %d (%d syms) failed: %s", i // batch_size, len(batch), e)
                failed += len(batch)
                continue

            data = getattr(resp, "data", None) or {}

            for sym in batch:
                bars = data.get(sym) or []
                if not bars:
                    empty += 1
                    continue
                rows = [
                    (
                        sym,
                        b.timestamp.date().isoformat(),
                        float(b.open),
                        float(b.high),
                        float(b.low),
                        float(b.close),
                        float(b.close),
                        int(b.volume) if b.volume is not None else None,
                        "alpaca",
                    )
                    for b in bars
                ]
                conn.executemany(
                    """
                    INSERT INTO mm_bars
                        (symbol, date, open, high, low, close, adj_close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        open      = excluded.open,
                        high      = excluded.high,
                        low       = excluded.low,
                        close     = excluded.close,
                        adj_close = excluded.adj_close,
                        volume    = excluded.volume,
                        source    = excluded.source
                    """,
                    rows,
                )
                rows_written += len(rows)
                ok += 1

            conn.commit()

            if i + batch_size < len(symbols):
                time.sleep(INTER_BATCH_SLEEP_SECONDS)

        ended_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO mm_bars_ingest_log
                (run_id, started_at, ended_at,
                 symbols_requested, symbols_ok, symbols_empty, symbols_failed, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                started_at,
                ended_at,
                len(symbols),
                ok,
                empty,
                failed,
                f"start={start.date()} end={end.date()} rows_written={rows_written}",
            ),
        )
        conn.commit()

    return {
        "run_id": run_id,
        "requested": len(symbols),
        "ok": ok,
        "empty": empty,
        "failed": failed,
        "rows_written": rows_written,
    }


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest daily bars from Alpaca → mm_bars.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to macmini.db")
    parser.add_argument(
        "--symbols-file",
        default=str(SYMBOLS_CSV),
        help="CSV with a 'symbol' header column",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Override: explicit symbol list (skips --symbols-file)",
    )
    parser.add_argument("--start", default="2014-06-29", help="ISO date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="ISO date YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    syms = args.symbols if args.symbols else load_symbols(Path(args.symbols_file))
    report = ingest(
        syms,
        _parse_date(args.start),
        _parse_date(args.end),
        db_path=args.db,
        batch_size=args.batch_size,
    )
    print("alpaca bars ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if report["requested"] and report["failed"] / report["requested"] > 0.10:
        print(f"  FATAL: {report['failed']}/{report['requested']} symbols failed (>10%)")
        raise SystemExit(12)


if __name__ == "__main__":
    _cli()
