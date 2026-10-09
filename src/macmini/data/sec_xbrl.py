""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from typing import Iterable

import requests

from macmini.config import DB_PATH, SYMBOLS_CSV
from macmini.data._db import connect
from macmini.data.schema import MM_SEC_FACTS, MM_SEC_FACTS_INDEXES
from macmini.data.sec_form4 import HEADERS, SLEEP_SECONDS, _load_symbols, _load_tickers_map

log = logging.getLogger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}
DEFAULT_SINCE = "2015-01-01"

TAGS: dict[str, tuple[str, ...]] = {
    "us-gaap": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "CostOfRevenue",
        "GrossProfit",
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeTaxExpenseBenefit",
        "InterestExpense",
        "InterestExpenseNonoperating",
        "NetIncomeLoss",
        "EarningsPerShareDiluted",
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "ShareBasedCompensation",
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Assets",
        "AssetsCurrent",
        "Liabilities",
        "LiabilitiesCurrent",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "CommercialPaper",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "StockholdersEquity",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ),
    "dei": (
        "EntityCommonStockSharesOutstanding",
    ),
}

UPSERT = """
INSERT INTO mm_sec_facts
    (symbol, cik, tag, unit, period_end, period_start, fy, fp, form, filed,
     frame, value, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, tag, unit, period_end, period_start, filed) DO UPDATE SET
    fy = excluded.fy, fp = excluded.fp, form = excluded.form,
    frame = excluded.frame, value = excluded.value,
    fetched_at = excluded.fetched_at
"""


def _ensure_table(conn) -> None:
    conn.execute(MM_SEC_FACTS)
    for s in MM_SEC_FACTS_INDEXES:
        conn.execute(s)
    conn.commit()


def _fetch_facts(cik_padded: str) -> dict:
    r = requests.get(FACTS_URL.format(cik=cik_padded), headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def _rows_for(sym: str, cik: str, facts: dict, since: str, fetched_at: str) -> list[tuple]:
    out: list[tuple] = []
    taxonomies = facts.get("facts", {})
    for taxo, tags in TAGS.items():
        concepts = taxonomies.get(taxo, {})
        for tag in tags:
            units = concepts.get(tag, {}).get("units", {})
            for unit, entries in units.items():
                for e in entries:
                    if e.get("form") not in FORMS:
                        continue
                    end = e.get("end")
                    if not end or end < since or e.get("val") is None:
                        continue
                    out.append((
                        sym, cik, tag, unit, end, e.get("start") or "",
                        e.get("fy"), e.get("fp"), e.get("form"), e.get("filed"),
                        e.get("frame"), float(e["val"]), fetched_at,
                    ))
    return out


def ingest(symbols: Iterable[str], db_path=None, since: str = DEFAULT_SINCE,
           sleep: float = SLEEP_SECONDS) -> dict:
    symbols = list(symbols)
    log.info("loading SEC CIK map...")
    cik_map = _load_tickers_map()
    log.info("CIK map covers %d tickers", len(cik_map))
    time.sleep(sleep)

    ok = no_cik = failed = 0
    rows_written = 0
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    with connect(db_path) as conn:
        _ensure_table(conn)
        for i, sym in enumerate(symbols):
            cik = cik_map.get(sym.upper())
            if not cik:
                no_cik += 1
                log.debug("no CIK for %s", sym)
                continue
            try:
                facts = _fetch_facts(cik)
            except Exception as e:
                log.warning("companyfacts(%s/%s) failed: %s", sym, cik, e)
                failed += 1
                time.sleep(sleep)
                continue
            rows = _rows_for(sym, cik, facts, since, fetched_at)
            if rows:
                conn.executemany(UPSERT, rows)
                rows_written += len(rows)
            conn.commit()
            ok += 1
            if (i + 1) % 25 == 0:
                log.info("%d/%d symbols, %d rows so far", i + 1, len(symbols), rows_written)
            time.sleep(sleep)
        conn.commit()

    return {"requested": len(symbols), "ok": ok, "no_cik": no_cik,
            "failed": failed, "rows_written": rows_written, "since": since}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC XBRL company facts into mm_sec_facts.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    parser.add_argument("--symbols", nargs="*", help="Explicit symbols (skips --symbols-file)")
    parser.add_argument("--since", default=DEFAULT_SINCE, help="Keep periods ending on/after this date")
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    symbols = args.symbols or _load_symbols(args.symbols_file)
    report = ingest(symbols, db_path=args.db, since=args.since, sleep=args.sleep)
    print(report)


if __name__ == "__main__":
    _cli()
