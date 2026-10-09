""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from macmini.config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DB_PATH,
)
from macmini.data._db import connect
from macmini.data._window import clamp_feed_end

log = logging.getLogger(__name__)

SECTOR_ETFS: list[str] = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLC",
    "XLRE",
    "ITA",
    "GDX",
    "GDXJ",
    "XME",
    "IGV",
    "SOXX",
]


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def _weekly_request(symbols: list[str], start: datetime, end: datetime):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    return StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Week),
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
    )


def ingest(
    etfs: list[str] = SECTOR_ETFS,
    start: datetime | None = None,
    end: datetime | None = None,
    db_path=None,
) -> dict:
    start = start or datetime(2018, 1, 1, tzinfo=timezone.utc)
    end = end or datetime.now(timezone.utc)
    end = clamp_feed_end(end)
    client = _client()
    resp = client.get_stock_bars(_weekly_request(etfs, start, end))
    data = getattr(resp, "data", None) or {}

    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        for etf in etfs:
            bars = data.get(etf) or []
            payload = []
            prev_close = None
            for b in bars:
                week_end = b.timestamp.date().isoformat()
                close = float(b.close)
                vol = int(b.volume) if b.volume is not None else 0
                if prev_close is None or close == prev_close:
                    sign = 0
                else:
                    sign = 1 if close > prev_close else -1
                net_flow = sign * close * vol
                aum = close * vol
                payload.append((etf, week_end, float(net_flow), float(aum)))
                prev_close = close

            conn.executemany(
                """
                INSERT INTO mm_sector_etf_flows_weekly
                    (etf_ticker, week_end, net_flow_usd, aum_usd)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(etf_ticker, week_end) DO UPDATE SET
                    net_flow_usd = excluded.net_flow_usd,
                    aum_usd      = excluded.aum_usd
                """,
                payload,
            )
            counts[etf] = len(payload)
            log.info("ingested %d weekly rows for %s", len(payload), etf)
        conn.commit()

    return counts


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ETF weekly flow proxy → mm_sector_etf_flows_weekly.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    report = ingest(start=start, end=end, db_path=args.db)
    print("etf_flows ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v} weekly rows")


if __name__ == "__main__":
    _cli()
