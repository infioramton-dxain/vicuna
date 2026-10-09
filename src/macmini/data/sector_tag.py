""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from macmini.config import DB_PATH, FINNHUB_API_KEY, SYMBOLS_CSV, resolve_db

log = logging.getLogger(__name__)

PER_REQUEST_SLEEP_SECONDS = 1.05

SECTOR_EXACT: dict[str, str] = {
    "Basic Materials":            "Materials",
    "Metals & Mining":            "Mining",
    "Aerospace & Defense":        "Defense",
    "Aerospace and Defense":      "Defense",
    "Technology":                 "Software",
    "Communication Services":     "Software",
    "Information Technology":     "Software",
    "Software":                   "Software",
}

SYMBOL_SECTOR_OVERRIDES: dict[str, str] = {s: "Consumer Staples" for s in (
    "WMT", "COST", "TGT", "KR", "DG", "DLTR", "BJ", "SFM", "SYY", "USFD",
    "PFGC", "ACI",
    "KO", "PEP", "MNST", "KDP", "STZ", "COKE", "TAP", "CELH", "FIZZ", "SAM",
    "PG", "CL", "KMB", "KVUE", "CHD", "CLX", "EL", "COTY", "EPC", "NWL",
    "PM", "MO", "BTI", "TPB",
    "MDLZ", "KHC", "HSY", "GIS", "CAG", "CPB", "SJM", "HRL", "MKC", "LW",
    "FLO", "JJSF", "POST", "BRBR",
    "ADM", "BG", "INGR", "DAR", "DOLE", "TSN", "CALM", "ANDE", "PPC", "SFD",
)}

SECTOR_SUBSTR: list[tuple[str, str]] = [
    ("gold",          "Mining"),
    ("silver",        "Mining"),
    ("mining",        "Mining"),
    ("metals",        "Mining"),
    ("minerals",      "Mining"),
    ("steel",         "Materials"),
    ("chemicals",     "Materials"),
    ("materials",     "Materials"),
    ("defense",       "Defense"),
    ("aerospace",     "Defense"),
    ("semiconductor", "Computers"),
    ("hardware",      "Computers"),
    ("storage",       "Computers"),
    ("computers",     "Computers"),
    ("electronic",    "Computers"),
    ("software",      "Software"),
    ("internet",      "Software"),
    ("media",         "Software"),
]


def map_sector(finnhub_industry: str | None, symbol: str | None = None) -> str:
    if symbol and symbol in SYMBOL_SECTOR_OVERRIDES:
        return SYMBOL_SECTOR_OVERRIDES[symbol]
    if not finnhub_industry:
        return "Other"
    if finnhub_industry in SECTOR_EXACT:
        return SECTOR_EXACT[finnhub_industry]
    lower = finnhub_industry.lower()
    for needle, bucket in SECTOR_SUBSTR:
        if needle in lower:
            return bucket
    return "Other"


def assign_archetype(sector: str, market_cap_usd: float | None) -> str | None:
    ""
    if market_cap_usd is None or market_cap_usd <= 0:
        return None
    cap_b = market_cap_usd / 1_000_000_000

    if sector in ("Software", "Computers") and cap_b > 200:
        return "mega_cap_tech"
    if sector == "Defense" and cap_b > 5:
        return "defense_prime"
    if sector == "Mining" and cap_b > 5:
        return "mining_senior"
    if sector == "Mining" and cap_b < 1:
        return "mining_junior"
    return None


def _client():
    import finnhub
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in .env")
    return finnhub.Client(api_key=FINNHUB_API_KEY)


def _load_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [r[0].strip() for r in reader if r and r[0].strip()]


def tag(
    symbols: Iterable[str],
    db_path: Path | str | None = None,
    sleep: float = PER_REQUEST_SLEEP_SECONDS,
) -> dict:
    ""
    db = resolve_db(db_path)
    symbols = list(symbols)
    client = _client()

    ok = empty = failed = 0
    counts_by_sector: dict[str, int] = {}

    from macmini.data._db import connect as _db_connect
    with _db_connect(db) as conn:

        for i, sym in enumerate(symbols):
            try:
                p = client.company_profile2(symbol=sym) or {}
            except Exception as e:
                log.warning("profile2(%s) failed: %s", sym, e)
                failed += 1
                time.sleep(sleep)
                continue

            if not p or not p.get("ticker"):
                log.info("profile2(%s) returned empty", sym)
                empty += 1
                time.sleep(sleep)
                continue

            finnhub_industry = p.get("finnhubIndustry")
            mcap_m = p.get("marketCapitalization")
            mcap_usd = float(mcap_m) * 1_000_000 if mcap_m else None
            sector = map_sector(finnhub_industry, symbol=sym)
            archetype = assign_archetype(sector, mcap_usd)
            counts_by_sector[sector] = counts_by_sector.get(sector, 0) + 1

            conn.execute(
                """
                INSERT INTO mm_sector_map (symbol, sector, finnhub_sector, market_cap_usd, archetype)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    sector          = excluded.sector,
                    finnhub_sector  = excluded.finnhub_sector,
                    market_cap_usd  = excluded.market_cap_usd,
                    archetype       = excluded.archetype
                """,
                (sym, sector, finnhub_industry, mcap_usd, archetype),
            )
            ok += 1
            if ok % 50 == 0:
                conn.commit()
                log.info("%d/%d tagged", i + 1, len(symbols))
            time.sleep(sleep)

        conn.commit()

    return {
        "requested": len(symbols),
        "ok": ok,
        "empty": empty,
        "failed": failed,
        "sector_counts": counts_by_sector,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Tag sectors + archetypes via Finnhub profile2.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    parser.add_argument("--symbols", nargs="*", help="Explicit symbols (skips --symbols-file)")
    parser.add_argument("--sleep", type=float, default=PER_REQUEST_SLEEP_SECONDS)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    syms = args.symbols if args.symbols else _load_symbols(Path(args.symbols_file))
    report = tag(syms, db_path=args.db, sleep=args.sleep)
    print("tag_sectors complete")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
