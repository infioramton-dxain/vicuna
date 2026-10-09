""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from macmini.config import DB_PATH, FINNHUB_API_KEY, SYMBOLS_CSV, resolve_db

log = logging.getLogger(__name__)

PER_REQUEST_SLEEP_SECONDS = 1.1


def _find_concept(items: list[dict] | None, *needles: str) -> float | None:
    ""
    if not items:
        return None
    for needle in needles:
        nl = needle.lower()
        for item in items:
            label = (item.get("label") or "").lower()
            concept = (item.get("concept") or "").lower()
            if nl in concept or nl in label:
                v = item.get("value")
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    continue
    return None


def _quarter_to_row(symbol: str, fr: dict) -> dict | None:
    ""
    end_date = fr.get("endDate") or fr.get("period")
    if not end_date:
        return None

    report = fr.get("report") or {}
    ic = report.get("ic") or []
    bs = report.get("bs") or []
    cf = report.get("cf") or []

    revenue = _find_concept(ic, "Revenues", "SalesRevenueNet", "Revenue")
    earnings = _find_concept(ic, "NetIncomeLoss")
    eps = _find_concept(ic, "EarningsPerShareDiluted", "EarningsPerShareBasic")
    ocf = _find_concept(cf, "NetCashProvidedByUsedInOperatingActivities")
    capex = _find_concept(cf, "PaymentsToAcquirePropertyPlantAndEquipment")
    if capex is not None:
        capex = abs(capex)
    free_cash_flow = (ocf - capex) if (ocf is not None and capex is not None) else None
    total_debt = _find_concept(bs, "LongTermDebt", "TotalDebt")
    total_equity = _find_concept(bs, "StockholdersEquity")
    shares = _find_concept(bs, "CommonStockSharesOutstanding", "SharesOutstanding")

    return {
        "symbol": symbol,
        "period_end": end_date,
        "revenue": revenue,
        "earnings": earnings,
        "eps": eps,
        "operating_cash_flow": ocf,
        "free_cash_flow": free_cash_flow,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "shares_outstanding": shares,
        "capex": capex,
    }


def _client():
    import finnhub
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in .env")
    return finnhub.Client(api_key=FINNHUB_API_KEY)


def _fetch_fundamentals(client, symbol: str, freq: str = "quarterly") -> list[dict]:
    resp = client.financials_reported(symbol=symbol, freq=freq)
    data = (resp or {}).get("data") or []
    rows: list[dict] = []
    for fr in data:
        r = _quarter_to_row(symbol, fr)
        if r:
            rows.append(r)
    return rows


def _fetch_analyst(client, symbol: str) -> dict | None:
    ""
    recs = client.recommendation_trends(symbol=symbol) or []
    if not recs:
        return None
    latest = recs[0]
    as_of = latest.get("period")
    if not as_of:
        return None

    fy0 = fy1 = None
    try:
        est = client.company_eps_estimates(symbol=symbol, freq="annual") or {}
        records = est.get("data") or []
        if records:
            records.sort(key=lambda r: r.get("period") or "")
            if len(records) >= 1:
                fy0 = records[0].get("epsAvg")
            if len(records) >= 2:
                fy1 = records[1].get("epsAvg")
    except Exception as e:
        log.debug("eps_estimates(%s) failed (often premium-gated): %s", symbol, e)

    return {
        "symbol": symbol,
        "as_of_date": as_of,
        "fy0_eps_consensus": fy0,
        "fy1_eps_consensus": fy1,
        "eps_revision_30d": None,
        "rating_buy": latest.get("buy", 0) + latest.get("strongBuy", 0),
        "rating_hold": latest.get("hold", 0),
        "rating_sell": latest.get("sell", 0) + latest.get("strongSell", 0),
    }


def _fetch_earnings(client, symbol: str) -> list[dict]:
    resp = client.company_earnings(symbol=symbol, limit=20) or []
    out: list[dict] = []
    for e in resp:
        rdate = e.get("period")
        if not rdate:
            continue
        eps_actual = e.get("actual")
        eps_estimate = e.get("estimate")
        surprise_pct = e.get("surprisePercent")
        if surprise_pct is None and eps_actual is not None and eps_estimate not in (None, 0):
            try:
                surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
            except Exception:
                surprise_pct = None
        out.append(
            {
                "symbol": symbol,
                "report_date": rdate,
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "eps_surprise_pct": surprise_pct,
                "revenue_actual": None,
                "revenue_estimate": None,
                "revenue_surprise_pct": None,
                "guidance_raised": None,
            }
        )
    return out


def _upsert_fundamentals(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO mm_fundamentals_quarterly
            (symbol, period_end, revenue, earnings, eps,
             operating_cash_flow, free_cash_flow,
             total_debt, total_equity,
             shares_outstanding, capex,
             source, fetched_at)
        VALUES (:symbol, :period_end, :revenue, :earnings, :eps,
                :operating_cash_flow, :free_cash_flow,
                :total_debt, :total_equity,
                :shares_outstanding, :capex,
                'finnhub', :fetched_at)
        ON CONFLICT (symbol, period_end) DO UPDATE SET
            revenue             = excluded.revenue,
            earnings            = excluded.earnings,
            eps                 = excluded.eps,
            operating_cash_flow = excluded.operating_cash_flow,
            free_cash_flow      = excluded.free_cash_flow,
            total_debt          = excluded.total_debt,
            total_equity        = excluded.total_equity,
            shares_outstanding  = excluded.shares_outstanding,
            capex               = excluded.capex,
            fetched_at          = excluded.fetched_at
        """,
        [{**r, "fetched_at": now} for r in rows],
    )
    return len(rows)


def _upsert_annual(conn: sqlite3.Connection, rows: list[dict]) -> int:
    ""
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO mm_fundamentals_annual
            (symbol, period_end, revenue, earnings, eps,
             operating_cash_flow, free_cash_flow,
             total_debt, total_equity,
             shares_outstanding, capex,
             source, fetched_at)
        VALUES (:symbol, :period_end, :revenue, :earnings, :eps,
                :operating_cash_flow, :free_cash_flow,
                :total_debt, :total_equity,
                :shares_outstanding, :capex,
                'finnhub', :fetched_at)
        ON CONFLICT (symbol, period_end) DO UPDATE SET
            revenue             = excluded.revenue,
            earnings            = excluded.earnings,
            eps                 = excluded.eps,
            operating_cash_flow = excluded.operating_cash_flow,
            free_cash_flow      = excluded.free_cash_flow,
            total_debt          = excluded.total_debt,
            total_equity        = excluded.total_equity,
            shares_outstanding  = excluded.shares_outstanding,
            capex               = excluded.capex,
            fetched_at          = excluded.fetched_at
        """,
        [{**r, "fetched_at": now} for r in rows],
    )
    return len(rows)


def _upsert_analyst(conn: sqlite3.Connection, row: dict | None) -> int:
    if not row:
        return 0
    conn.execute(
        """
        INSERT INTO mm_analyst_estimates
            (symbol, as_of_date, fy0_eps_consensus, fy1_eps_consensus,
             eps_revision_30d, rating_buy, rating_hold, rating_sell)
        VALUES (:symbol, :as_of_date, :fy0_eps_consensus, :fy1_eps_consensus,
                :eps_revision_30d, :rating_buy, :rating_hold, :rating_sell)
        ON CONFLICT (symbol, as_of_date) DO UPDATE SET
            fy0_eps_consensus = excluded.fy0_eps_consensus,
            fy1_eps_consensus = excluded.fy1_eps_consensus,
            eps_revision_30d  = excluded.eps_revision_30d,
            rating_buy        = excluded.rating_buy,
            rating_hold       = excluded.rating_hold,
            rating_sell       = excluded.rating_sell
        """,
        row,
    )
    return 1


def _upsert_earnings(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO mm_earnings_events
            (symbol, report_date, eps_actual, eps_estimate, eps_surprise_pct,
             revenue_actual, revenue_estimate, revenue_surprise_pct, guidance_raised)
        VALUES (:symbol, :report_date, :eps_actual, :eps_estimate, :eps_surprise_pct,
                :revenue_actual, :revenue_estimate, :revenue_surprise_pct, :guidance_raised)
        ON CONFLICT (symbol, report_date) DO UPDATE SET
            eps_actual           = excluded.eps_actual,
            eps_estimate         = excluded.eps_estimate,
            eps_surprise_pct     = excluded.eps_surprise_pct,
            revenue_actual       = excluded.revenue_actual,
            revenue_estimate     = excluded.revenue_estimate,
            revenue_surprise_pct = excluded.revenue_surprise_pct,
            guidance_raised      = excluded.guidance_raised
        """,
        rows,
    )
    return len(rows)


def _load_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [r[0].strip() for r in reader if r and r[0].strip()]


def ingest(
    symbols: Iterable[str],
    db_path: Path | str | None = None,
    sleep: float = PER_REQUEST_SLEEP_SECONDS,
    annual_only: bool = False,
) -> dict:
    db = resolve_db(db_path)
    symbols = list(symbols)
    client = _client()

    f_ok = a_ok = e_ok = n_ok = 0
    f_fail = a_fail = e_fail = n_fail = 0
    f_rows = e_rows = n_rows = 0

    from macmini.data._db import connect as _db_connect
    with _db_connect(db) as conn:
        from macmini.data.schema import MM_FUNDAMENTALS_ANNUAL
        conn.execute(MM_FUNDAMENTALS_ANNUAL)

        for i, sym in enumerate(symbols):
            try:
                rows = _fetch_fundamentals(client, sym, freq="annual")
                n_rows += _upsert_annual(conn, rows)
                n_ok += 1
            except Exception as e:
                log.warning("financials_reported annual(%s) failed: %s", sym, e)
                n_fail += 1
            time.sleep(sleep)

            if annual_only:
                conn.commit()
                if (i + 1) % 25 == 0:
                    log.info("%d/%d symbols processed (annual-only)", i + 1, len(symbols))
                continue

            try:
                rows = _fetch_fundamentals(client, sym)
                f_rows += _upsert_fundamentals(conn, rows)
                f_ok += 1
            except Exception as e:
                log.warning("financials_reported(%s) failed: %s", sym, e)
                f_fail += 1
            time.sleep(sleep)

            try:
                row = _fetch_analyst(client, sym)
                a_ok += _upsert_analyst(conn, row)
            except Exception as e:
                log.warning("analyst(%s) failed: %s", sym, e)
                a_fail += 1
            time.sleep(sleep)

            try:
                rows = _fetch_earnings(client, sym)
                e_rows += _upsert_earnings(conn, rows)
                e_ok += 1
            except Exception as e:
                log.warning("earnings(%s) failed: %s", sym, e)
                e_fail += 1
            time.sleep(sleep)

            conn.commit()
            if (i + 1) % 25 == 0:
                log.info("%d/%d symbols processed", i + 1, len(symbols))

        conn.commit()

    return {
        "requested": len(symbols),
        "annual":       {"ok": n_ok, "failed": n_fail, "rows": n_rows},
        "fundamentals": {"ok": f_ok, "failed": f_fail, "rows": f_rows},
        "analyst":      {"ok": a_ok, "failed": a_fail},
        "earnings":     {"ok": e_ok, "failed": e_fail, "rows": e_rows},
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest fundamentals/analyst/earnings via Finnhub.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    parser.add_argument("--symbols", nargs="*", help="Explicit symbols (skips --symbols-file)")
    parser.add_argument("--sleep", type=float, default=PER_REQUEST_SLEEP_SECONDS)
    parser.add_argument("--annual-only", action="store_true",
                        help="Fetch only the annual (10-K) statements — 1 request/symbol")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    syms = args.symbols if args.symbols else _load_symbols(Path(args.symbols_file))
    report = ingest(syms, db_path=args.db, sleep=args.sleep, annual_only=args.annual_only)
    print("finnhub_fund ingest complete")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
