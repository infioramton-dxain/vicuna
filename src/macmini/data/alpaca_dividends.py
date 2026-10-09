""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from macmini.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, SYMBOLS_CSV
from macmini.data._db import connect
from macmini.data.schema import MM_DIVIDENDS

log = logging.getLogger(__name__)

DATA_URL = "https://data.alpaca.markets/v1/corporate-actions"
BATCH_SIZE = 40
PER_REQUEST_SLEEP = 0.35


def _get(params: dict) -> dict:
    url = f"{DATA_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_dividends(symbols: list[str], start: str, end: str) -> list[dict]:
    ""
    out: list[dict] = []
    params = {
        "symbols": ",".join(symbols),
        "types": "cash_dividend",
        "start": start,
        "end": end,
        "limit": 1000,
    }
    page_token: str | None = None
    while True:
        if page_token:
            params["page_token"] = page_token
        data = _get(params)
        out.extend(data.get("corporate_actions", {}).get("cash_dividends", []))
        page_token = data.get("next_page_token")
        if not page_token:
            return out


def ingest(
    symbols: Iterable[str],
    start: str,
    end: str,
    db_path: Path | str | None = None,
) -> dict:
    symbols = list(symbols)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows_written = 0
    symbols_with_rows = 0

    with connect(db_path) as conn:
        conn.execute(MM_DIVIDENDS)
        conn.commit()

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            try:
                divs = fetch_dividends(batch, start, end)
            except Exception as e:
                log.warning("corporate-actions fetch failed for batch %d (%s...): %s",
                            i // BATCH_SIZE, batch[0], e)
                time.sleep(PER_REQUEST_SLEEP)
                continue

            by_symbol: dict[str, list[dict]] = {}
            for d in divs:
                by_symbol.setdefault(d["symbol"], []).append(d)

            for sym in batch:
                rows = by_symbol.get(sym, [])
                if not rows:
                    continue
                for d in rows:
                    conn.execute(
                        "INSERT OR REPLACE INTO mm_dividends "
                        "(symbol, ex_date, rate, special, is_foreign, source, fetched_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sym, d["ex_date"], float(d["rate"]),
                         1 if d.get("special") else 0,
                         1 if d.get("foreign") else 0,
                         "alpaca_corporate_actions", fetched_at),
                    )
                conn.commit()
                rows_written += len(rows)
                symbols_with_rows += 1
            log.info("batch %d/%d: %d dividends",
                     i // BATCH_SIZE + 1,
                     (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE, len(divs))
            time.sleep(PER_REQUEST_SLEEP)

    return {
        "requested_symbols": len(symbols),
        "symbols_with_dividends": symbols_with_rows,
        "rows_written": rows_written,
    }


def _load_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [r[0].strip() for r in reader if r and r[0].strip()]


def _cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Ingest cash dividends into mm_dividends.")
    p.add_argument("--db", default=None)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    args = p.parse_args(argv)

    symbols = args.symbols or _load_symbols(Path(args.symbols_file))
    report = ingest(symbols, args.start, args.end, db_path=args.db)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
