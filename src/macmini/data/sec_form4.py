""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import requests

from macmini.config import DB_PATH, SYMBOLS_CSV
from macmini.data._db import connect

log = logging.getLogger(__name__)

USER_AGENT = "swing-trader research (contact@example.com)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SLEEP_SECONDS = 0.15
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

TXN_CODES: dict[str, str] = {
    "P": "P",
    "S": "S",
    "M": "M",
    "A": "A",
    "F": "F",
}

_ISO_DATE_HEAD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _normalize_txn_date(raw: str | None) -> str | None:
    ""
    if not raw:
        return None
    s = raw.strip()
    m = _ISO_DATE_HEAD.match(s)
    if not m:
        return None
    try:
        dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return s[:10]


def _load_tickers_map() -> dict[str, str]:
    ""
    r = requests.get(COMPANY_TICKERS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    out: dict[str, str] = {}
    for entry in raw.values():
        sym = (entry.get("ticker") or "").upper()
        cik = entry.get("cik_str")
        if sym and cik is not None:
            out[sym] = f"{int(cik):010d}"
    return out


def _fetch_submissions(cik_padded: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _form4_accessions(submissions: dict, limit: int = 20) -> list[dict]:
    ""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_documents = recent.get("primaryDocument") or []

    out: list[dict] = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        if i >= len(accession_numbers) or i >= len(filing_dates):
            continue
        out.append(
            {
                "accession": accession_numbers[i],
                "filing_date": filing_dates[i],
                "primary_doc": primary_documents[i] if i < len(primary_documents) else "",
            }
        )
        if len(out) >= limit:
            break
    return out


def _form4_xml_url(cik_padded: str, accession: str, primary_doc: str) -> str:
    ""
    acc_nodash = accession.replace("-", "")
    if "/" in primary_doc:
        head, tail = primary_doc.split("/", 1)
        if head.startswith("xsl"):
            primary_doc = tail
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik_padded)}/{acc_nodash}/{primary_doc}"


def _parse_form4(xml_bytes: bytes) -> list[dict]:
    ""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.debug("XML parse failed: %s", e)
        return []

    owner = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
    insider_name = owner.text.strip() if (owner is not None and owner.text) else None

    rel = root.find(".//reportingOwner/reportingOwnerRelationship")
    insider_title = None
    if rel is not None:
        for tag in ("officerTitle", "otherText"):
            el = rel.find(tag)
            if el is not None and el.text:
                insider_title = el.text.strip()
                break

    txns: list[dict] = []
    for t in root.findall(".//nonDerivativeTransaction"):
        code_el = t.find(".//transactionCoding/transactionCode")
        date_el = t.find(".//transactionDate/value")
        shares_el = t.find(".//transactionAmounts/transactionShares/value")
        price_el = t.find(".//transactionAmounts/transactionPricePerShare/value")
        if code_el is None or date_el is None:
            continue
        code = (code_el.text or "").strip()
        if code not in TXN_CODES:
            continue
        txn_date = _normalize_txn_date(date_el.text)
        if txn_date is None:
            continue
        try:
            shares = int(float((shares_el.text or "0")))
        except (TypeError, ValueError):
            shares = 0
        try:
            price = float(price_el.text) if price_el is not None and price_el.text else None
        except (TypeError, ValueError):
            price = None
        value_usd = price * shares if (price is not None and shares) else None
        txns.append(
            {
                "transaction_date": txn_date,
                "transaction_type": code,
                "shares": shares,
                "value_usd": value_usd,
                "insider_name": insider_name,
                "insider_title": insider_title,
            }
        )
    return txns


def _load_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [r[0].strip() for r in reader if r and r[0].strip()]


def ingest(
    symbols: Iterable[str],
    max_filings_per_symbol: int = 20,
    db_path=None,
    sleep: float = SLEEP_SECONDS,
) -> dict:
    symbols = list(symbols)
    log.info("loading SEC CIK map...")
    cik_map = _load_tickers_map()
    log.info("CIK map covers %d tickers", len(cik_map))
    time.sleep(sleep)

    ok = no_cik = no_filings = failed = 0
    txn_rows = 0

    with connect(db_path) as conn:
        for i, sym in enumerate(symbols):
            cik = cik_map.get(sym.upper())
            if not cik:
                no_cik += 1
                continue

            try:
                subs = _fetch_submissions(cik)
                time.sleep(sleep)
            except Exception as e:
                log.warning("submissions(%s) failed: %s", sym, e)
                failed += 1
                continue

            accessions = _form4_accessions(subs, limit=max_filings_per_symbol)
            if not accessions:
                no_filings += 1
                continue

            for acc in accessions:
                url = _form4_xml_url(cik, acc["accession"], acc["primary_doc"])
                try:
                    r = requests.get(url, headers=HEADERS, timeout=30)
                    r.raise_for_status()
                except Exception as e:
                    log.debug("form4 fetch %s failed: %s", url, e)
                    time.sleep(sleep)
                    continue

                content = r.content
                txns = _parse_form4(content)
                payload = [
                    (
                        sym,
                        t["transaction_date"],
                        t["insider_name"],
                        t["insider_title"],
                        t["transaction_type"],
                        t["shares"],
                        t["value_usd"],
                        url,
                    )
                    for t in txns
                    if t["transaction_date"]
                ]
                if payload:
                    conn.executemany(
                        """
                        INSERT INTO mm_insider_transactions
                            (symbol, transaction_date, insider_name, insider_title,
                             transaction_type, shares, value_usd, source_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, transaction_date, insider_name, transaction_type) DO UPDATE SET
                            shares      = excluded.shares,
                            value_usd   = excluded.value_usd,
                            source_url  = excluded.source_url
                        """,
                        payload,
                    )
                    txn_rows += len(payload)
                time.sleep(sleep)

            ok += 1
            conn.commit()
            if (i + 1) % 25 == 0:
                log.info("%d/%d symbols processed, %d txn rows so far", i + 1, len(symbols), txn_rows)
        conn.commit()

    return {
        "requested": len(symbols),
        "ok": ok,
        "no_cik": no_cik,
        "no_filings": no_filings,
        "failed": failed,
        "txn_rows": txn_rows,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC Form 4 insider transactions.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--max-filings", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    syms = args.symbols if args.symbols else _load_symbols(Path(args.symbols_file))
    report = ingest(syms, max_filings_per_symbol=args.max_filings, db_path=args.db, sleep=args.sleep)
    print("sec_form4 ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
