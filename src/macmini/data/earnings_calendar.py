""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from macmini.config import DB_PATH, FINNHUB_API_KEY, SYMBOLS_CSV, resolve_db

log = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 14


def _client():
    import finnhub
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in .env")
    return finnhub.Client(api_key=FINNHUB_API_KEY)


def _load_symbols(path: Path) -> set[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return {r[0].strip() for r in reader if r and r[0].strip()}


def fetch_calendar(client, start: date, end: date) -> list[dict]:
    ""
    resp = client.earnings_calendar(
        _from=start.isoformat(), to=end.isoformat(), symbol="", international=False
    ) or {}
    return resp.get("earningsCalendar") or []


def _upsert_forward(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO mm_earnings_events
            (symbol, report_date, eps_actual, eps_estimate, eps_surprise_pct,
             revenue_actual, revenue_estimate, revenue_surprise_pct, guidance_raised)
        VALUES (:symbol, :report_date, :eps_actual, :eps_estimate, NULL,
                :revenue_actual, :revenue_estimate, NULL, NULL)
        ON CONFLICT (symbol, report_date) DO UPDATE SET
            eps_actual       = COALESCE(mm_earnings_events.eps_actual,
                                        excluded.eps_actual),
            eps_estimate     = COALESCE(excluded.eps_estimate,
                                        mm_earnings_events.eps_estimate),
            revenue_actual   = COALESCE(mm_earnings_events.revenue_actual,
                                        excluded.revenue_actual),
            revenue_estimate = COALESCE(excluded.revenue_estimate,
                                        mm_earnings_events.revenue_estimate)
        """,
        rows,
    )
    return len(rows)


def ingest(
    db_path: Path | str | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    symbols: Iterable[str] | None = None,
    as_of: date | None = None,
) -> dict:
    db = resolve_db(db_path)
    universe = set(symbols) if symbols is not None else _load_symbols(SYMBOLS_CSV)
    start = as_of or date.today()
    end = start + timedelta(days=horizon_days)

    client = _client()
    calendar = fetch_calendar(client, start, end)

    rows = []
    for ev in calendar:
        sym = (ev.get("symbol") or "").strip()
        rdate = ev.get("date")
        if not sym or not rdate or sym not in universe:
            continue
        rows.append(
            {
                "symbol": sym,
                "report_date": rdate,
                "eps_actual": ev.get("epsActual"),
                "eps_estimate": ev.get("epsEstimate"),
                "revenue_actual": ev.get("revenueActual"),
                "revenue_estimate": ev.get("revenueEstimate"),
            }
        )

    from macmini.data._db import connect as _db_connect
    with _db_connect(db) as conn:
        n = _upsert_forward(conn, rows)
        conn.commit()

    return {
        "window": f"{start.isoformat()}..{end.isoformat()}",
        "calendar_events": len(calendar),
        "universe_matches": n,
        "symbols": sorted({r["symbol"] for r in rows}),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Pull the forward earnings calendar into mm_earnings_events."
    )
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = ingest(db_path=args.db, horizon_days=args.days)
    print("earnings_calendar ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
