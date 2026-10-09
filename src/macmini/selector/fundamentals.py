""
from __future__ import annotations

import sqlite3
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from macmini.recs.titan_notes import (CAPEX_TAGS, CASH_TAGS, LONG_DEBT_TAG_SETS,
                                      REVENUE_TAGS, SHARE_TAGS, SHORT_DEBT_TAGS,
                                      Fact, latest_instant, ttm)

FLOW_TAGS = {
    "rev": REVENUE_TAGS,
    "op": ("OperatingIncomeLoss",),
    "ni": ("NetIncomeLoss",),
    "cfo": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": CAPEX_TAGS,
    "sbc": ("ShareBasedCompensation",),
    "buyback": ("PaymentsForRepurchaseOfCommonStock",),
    "gross": ("GrossProfit",),
}
INSTANT_TAGS = {
    "cash": CASH_TAGS,
    "equity": ("StockholdersEquity",),
    "assets": ("Assets",),
}
FUND_FEATURES = [
    "fund_fcf_yield", "fund_earn_yield", "fund_sales_yield", "fund_op_margin",
    "fund_fcf_margin", "fund_gross_margin", "fund_rev_growth",
    "fund_earn_change", "fund_net_debt_to_mcap", "fund_buyback_yield",
    "fund_sbc_to_rev", "fund_accruals", "fund_roe", "fund_share_growth",
    "fund_days_since_filing",
]
MAX_FILING_AGE_DAYS = 200
_DB = ""


def _best(per_tag: dict, tags, fn):
    best = None
    for tag in tags:
        facts = per_tag.get(tag)
        if not facts:
            continue
        res = fn(list(facts.values()))
        if res is None:
            continue
        end = res.end if isinstance(res, Fact) else res[1]
        val = res.value if isinstance(res, Fact) else res[0]
        if best is None or end > best[0]:
            best = (end, val)
    return best[1] if best else np.nan


def filing_timeline(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    ""
    rows = conn.execute(
        "SELECT tag, unit, period_start, period_end, filed, value FROM mm_sec_facts "
        "WHERE symbol = ? AND unit IN ('USD', 'shares') ORDER BY filed", (symbol,)
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    per_tag: dict[str, dict[tuple, Fact]] = {}
    out, i, n = [], 0, len(rows)
    while i < n:
        filed = rows[i][4]
        while i < n and rows[i][4] == filed:
            tag, unit, start, end, _, value = rows[i]
            per_tag.setdefault(tag, {})[(start or "", end)] = Fact(
                tag, unit, start or "", end, filed, float(value))
            i += 1
        rec = {"filed": filed}
        for name, tags in FLOW_TAGS.items():
            rec[name] = _best(per_tag, tags, ttm)
        for name, tags in INSTANT_TAGS.items():
            rec[name] = _best(per_tag, tags, latest_instant)
        debt = np.nan
        for tags in LONG_DEBT_TAG_SETS:
            parts = [latest_instant(list(per_tag[t].values())) for t in tags if t in per_tag]
            parts = [p for p in parts if p is not None]
            if parts:
                debt = sum(p.value for p in parts)
                short = _best(per_tag, SHORT_DEBT_TAGS, latest_instant)
                debt += 0.0 if np.isnan(short) else short
                break
        rec["debt"] = debt
        rec["shares"] = _best(per_tag, SHARE_TAGS, lambda fs: max(
            fs, key=lambda f: (f.end, f.filed)) if fs else None)
        out.append(rec)
    t = pd.DataFrame(out)
    t["filed"] = pd.to_datetime(t["filed"])
    t = t.sort_values("filed").reset_index(drop=True)
    prev = t[["filed", "rev", "ni", "shares"]].rename(
        columns={"filed": "filed_prev", "rev": "rev_prev", "ni": "ni_prev",
                 "shares": "shares_prev"})
    t["lookup"] = t["filed"] - pd.Timedelta(days=350)
    t = pd.merge_asof(t, prev, left_on="lookup", right_on="filed_prev",
                      direction="backward", tolerance=pd.Timedelta(days=120))
    return t.drop(columns=["lookup", "filed_prev"])


def _one(args) -> pd.DataFrame | None:
    symbol, dates, closes, mcap_now, close_now = args
    conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=60)
    t = filing_timeline(conn, symbol)
    conn.close()
    if t.empty:
        return None
    s = pd.DataFrame({"date": pd.to_datetime(dates), "close": closes}).sort_values("date")
    m = pd.merge_asof(s, t, left_on="date", right_on="filed", direction="backward")
    age = (m["date"] - m["filed"]).dt.days
    mcap = mcap_now * m["close"] / close_now if mcap_now and close_now else np.nan
    fcf = m["cfo"] - m["capex"]
    rev = m["rev"].where(m["rev"] > 0)
    f = pd.DataFrame({"symbol": symbol, "date": m["date"]})
    f["fund_fcf_yield"] = fcf / mcap
    f["fund_earn_yield"] = m["ni"] / mcap
    f["fund_sales_yield"] = rev / mcap
    f["fund_op_margin"] = m["op"] / rev
    f["fund_fcf_margin"] = fcf / rev
    f["fund_gross_margin"] = m["gross"] / rev
    f["fund_rev_growth"] = rev / m["rev_prev"].where(m["rev_prev"] > 0) - 1
    f["fund_earn_change"] = (m["ni"] - m["ni_prev"]) / m["assets"].where(m["assets"] > 0)
    f["fund_net_debt_to_mcap"] = (m["debt"] - m["cash"]) / mcap
    f["fund_buyback_yield"] = m["buyback"] / mcap
    f["fund_sbc_to_rev"] = m["sbc"] / rev
    f["fund_accruals"] = (m["ni"] - m["cfo"]) / m["assets"].where(m["assets"] > 0)
    f["fund_roe"] = m["ni"] / m["equity"].where(m["equity"] > 0)
    f["fund_share_growth"] = m["shares"] / m["shares_prev"].where(m["shares_prev"] > 0) - 1
    f["fund_days_since_filing"] = age
    stale = age.isna() | (age > MAX_FILING_AGE_DAYS)
    f.loc[stale.values, FUND_FEATURES] = np.nan
    return f


def _init(db: str) -> None:
    global _DB
    _DB = db


def attach(panel: pd.DataFrame, db: str, workers: int = 12) -> pd.DataFrame:
    ""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    caps = dict(conn.execute("SELECT symbol, market_cap_usd FROM mm_sector_map"))
    last = dict(conn.execute(
        "SELECT b.symbol, b.close FROM mm_bars b JOIN (SELECT symbol, MAX(date) d "
        "FROM mm_bars GROUP BY symbol) m ON m.symbol = b.symbol AND m.d = b.date"))
    have = {s for (s,) in conn.execute("SELECT DISTINCT symbol FROM mm_sec_facts")}
    conn.close()
    jobs = [(sym, g["date"].tolist(), g["close"].tolist(), caps.get(sym), last.get(sym))
            for sym, g in panel.groupby("symbol") if sym in have]
    with Pool(workers, initializer=_init, initargs=(db,)) as pool:
        parts = [p for p in pool.imap_unordered(_one, jobs, chunksize=4) if p is not None]
    if not parts:
        return panel.assign(**{c: np.nan for c in FUND_FEATURES})
    return panel.merge(pd.concat(parts, ignore_index=True), on=["symbol", "date"], how="left")
