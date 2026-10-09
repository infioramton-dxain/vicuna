""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import requests

from macmini.config import DB_PATH
from macmini.data._db import connect

log = logging.getLogger(__name__)

SOCRATA_ENDPOINT = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

COMMODITIES: dict[str, str] = {
    "gold":         "GOLD - COMMODITY EXCHANGE",
    "silver":       "SILVER - COMMODITY EXCHANGE",
    "copper":       "COPPER- #1",
    "platinum":     "PLATINUM",
    "palladium":    "PALLADIUM",
    "wti_crude":    "CRUDE OIL, LIGHT SWEET",
    "natural_gas":  "NATURAL GAS - NEW YORK MERCANTILE",
}


def _fetch_for(market_like: str, since: str | None) -> list[dict]:
    ""
    where_clauses = [f"market_and_exchange_names like '%{market_like}%'"]
    if since:
        where_clauses.append(f"report_date_as_yyyy_mm_dd >= '{since}'")
    params = {
        "$where": " AND ".join(where_clauses),
        "$select": (
            "report_date_as_yyyy_mm_dd, market_and_exchange_names, "
            "comm_positions_long_all, comm_positions_short_all, "
            "noncomm_positions_long_all, noncomm_positions_short_all"
        ),
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": "5000",
    }
    r = requests.get(SOCRATA_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _to_int(s) -> int | None:
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def ingest(
    since: str | None = "2018-01-01",
    db_path=None,
    commodities: dict[str, str] = COMMODITIES,
) -> dict:
    ""
    report: dict[str, int] = {}

    with connect(db_path) as conn:
        for label, market_like in commodities.items():
            try:
                rows = _fetch_for(market_like, since)
            except Exception as e:
                log.warning("fetch %s failed: %s", label, e)
                report[label] = 0
                continue

            payload = []
            for row in rows:
                date = row.get("report_date_as_yyyy_mm_dd")
                if date and "T" in date:
                    date = date.split("T")[0]
                if not date:
                    continue
                long_c = _to_int(row.get("comm_positions_long_all"))
                short_c = _to_int(row.get("comm_positions_short_all"))
                net_c = (long_c - short_c) if (long_c is not None and short_c is not None) else None
                payload.append(
                    (
                        label,
                        date,
                        long_c,
                        short_c,
                        net_c,
                        _to_int(row.get("noncomm_positions_long_all")),
                        _to_int(row.get("noncomm_positions_short_all")),
                    )
                )

            conn.executemany(
                """
                INSERT INTO mm_cftc_cot_weekly
                    (commodity, report_date,
                     commercial_long, commercial_short, commercial_net,
                     noncommercial_long, noncommercial_short)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(commodity, report_date) DO UPDATE SET
                    commercial_long     = excluded.commercial_long,
                    commercial_short    = excluded.commercial_short,
                    commercial_net      = excluded.commercial_net,
                    noncommercial_long  = excluded.noncommercial_long,
                    noncommercial_short = excluded.noncommercial_short
                """,
                payload,
            )
            report[label] = len(payload)
            log.info("ingested %d rows for %s", len(payload), label)
        conn.commit()

    return report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest CFTC COT weekly → mm_cftc_cot_weekly.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--since", default="2018-01-01")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = ingest(since=args.since, db_path=args.db)
    print("cftc_cot ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v} rows")


if __name__ == "__main__":
    _cli()
