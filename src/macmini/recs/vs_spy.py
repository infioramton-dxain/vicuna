""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import requests

from macmini.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY
from macmini.data._db import connect

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
EPOCH_DATE = date(2026, 9, 3)
EPOCH_OPEN = time(9, 30)
_HTTP_TIMEOUT_S = 30

_DDL = """
CREATE TABLE IF NOT EXISTS mm_vs_spy_epoch (
    epoch_date       TEXT PRIMARY KEY,
    baseline_equity  REAL NOT NULL,
    spy_open         REAL NOT NULL,
    captured_at      TEXT NOT NULL,
    method           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mm_vs_spy (
    date        TEXT PRIMARY KEY,
    equity      REAL NOT NULL,
    spy_close   REAL NOT NULL,
    spy_div     REAL NOT NULL DEFAULT 0,
    spy_tr_idx  REAL NOT NULL,
    vicuna_ret  REAL NOT NULL,
    spy_ret     REAL NOT NULL,
    excess      REAL NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def _alpaca_get(path: str, params: dict | None = None):
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing; cannot read portfolio history.")
    r = requests.get(
        f"{ALPACA_BASE_URL.rstrip('/')}{path}",
        headers={"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params=params or {},
        timeout=_HTTP_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def _equity_by_day(start: date) -> dict[date, float]:
    ""
    ph = _alpaca_get(
        "/v2/account/portfolio/history",
        {"period": "1A", "timeframe": "1D", "intraday_reporting": "market_hours"},
    )
    out: dict[date, float] = {}
    for ts, eq in zip(ph.get("timestamp") or [], ph.get("equity") or []):
        if eq is None:
            continue
        d = datetime.fromtimestamp(ts, tz=ET).date()
        if d >= start:
            out[d] = float(eq)
    today = datetime.now(ET).date()
    if today >= start and today not in out:
        out[today] = float(_alpaca_get("/v2/account")["equity"])
    return out


def _spy_history(start: date):
    ""
    import yfinance as yf

    df = yf.Ticker("SPY").history(
        start=start.isoformat(), auto_adjust=False, actions=True
    )
    rows = []
    for idx, r in df.iterrows():
        rows.append((idx.date(), float(r["Open"]), float(r["Close"]),
                     float(r.get("Dividends", 0.0) or 0.0)))
    return rows


def capture_epoch(conn: sqlite3.Connection) -> dict | None:
    ""
    row = conn.execute("SELECT * FROM mm_vs_spy_epoch").fetchone()
    if row:
        return {"epoch_date": row[0], "baseline_equity": row[1], "spy_open": row[2]}

    now_et = datetime.now(ET)
    if now_et < datetime.combine(EPOCH_DATE, EPOCH_OPEN, tzinfo=ET):
        log.info("epoch %s not started yet (now %s ET) — nothing to capture",
                 EPOCH_DATE, now_et.strftime("%Y-%m-%d %H:%M"))
        return None

    spy = [r for r in _spy_history(EPOCH_DATE - timedelta(days=5)) if r[0] == EPOCH_DATE]
    if not spy:
        log.warning("SPY has no bar for %s yet — capture deferred to next run", EPOCH_DATE)
        return None
    spy_open = spy[0][1]

    baseline = None
    method = "intraday_first_point"
    try:
        ph = _alpaca_get(
            "/v2/account/portfolio/history",
            {"period": "1W", "timeframe": "5Min", "intraday_reporting": "market_hours"},
        )
        pts = [
            (datetime.fromtimestamp(ts, tz=ET), float(eq))
            for ts, eq in zip(ph.get("timestamp") or [], ph.get("equity") or [])
            if eq is not None
        ]
        day_pts = [p for p in pts if p[0].date() == EPOCH_DATE
                   and p[0].time() >= EPOCH_OPEN]
        if day_pts:
            baseline = day_pts[0][1]
    except requests.RequestException as e:
        log.warning("intraday portfolio history unavailable (%s); falling back", e)
    if baseline is None:
        eq = _equity_by_day(EPOCH_DATE)
        if EPOCH_DATE not in eq:
            log.warning("no equity datapoint for %s yet — capture deferred", EPOCH_DATE)
            return None
        baseline = eq[EPOCH_DATE]
        method = "daily_close_fallback"

    conn.execute(
        "INSERT INTO mm_vs_spy_epoch (epoch_date, baseline_equity, spy_open, captured_at, method) "
        "VALUES (?, ?, ?, ?, ?)",
        (EPOCH_DATE.isoformat(), baseline, spy_open,
         datetime.now(ET).isoformat(timespec="seconds"), method),
    )
    conn.commit()
    log.info("epoch captured: equity %.2f, SPY open %.2f (%s)", baseline, spy_open, method)
    return {"epoch_date": EPOCH_DATE.isoformat(), "baseline_equity": baseline, "spy_open": spy_open}


def rebuild(conn: sqlite3.Connection) -> dict | None:
    epoch = capture_epoch(conn)
    if epoch is None:
        return None
    baseline_equity = float(epoch["baseline_equity"])
    spy_open = float(epoch["spy_open"])

    equity = _equity_by_day(EPOCH_DATE)
    spy = [r for r in _spy_history(EPOCH_DATE - timedelta(days=5)) if r[0] >= EPOCH_DATE]
    if not equity or not spy:
        log.warning("missing inputs (equity days=%d, spy days=%d) — keeping previous rows",
                    len(equity), len(spy))
        return None

    tr_by_day: dict[date, tuple[float, float, float]] = {}
    idx = 1.0
    prev_close = None
    for d, _o, close, div in spy:
        base = spy_open if prev_close is None else prev_close
        idx *= (close + div) / base
        prev_close = close
        tr_by_day[d] = (close, div, idx)

    now_iso = datetime.now(ET).isoformat(timespec="seconds")
    rows = []
    for d in sorted(set(equity) & set(tr_by_day)):
        close, div, tr = tr_by_day[d]
        v_ret = equity[d] / baseline_equity - 1.0
        s_ret = tr - 1.0
        rows.append((d.isoformat(), equity[d], close, div, tr,
                     v_ret, s_ret, v_ret - s_ret, now_iso))
    if not rows:
        log.warning("no overlapping equity/SPY days yet — nothing written")
        return None

    with conn:
        conn.execute("DELETE FROM mm_vs_spy")
        conn.executemany(
            "INSERT INTO mm_vs_spy (date, equity, spy_close, spy_div, spy_tr_idx, "
            "vicuna_ret, spy_ret, excess, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    last = rows[-1]
    summary = {
        "as_of": last[0], "days": len(rows), "equity": last[1],
        "vicuna_ret": last[5], "spy_ret": last[6], "excess": last[7],
    }
    log.info(
        "vs SPY since %s (%d days): Vicuna %+.2f%%, SPY %+.2f%%, excess %+.2f pp",
        EPOCH_DATE, len(rows), 100 * last[5], 100 * last[6], 100 * last[7],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None, help="DB path override (default: canonical)")
    args = p.parse_args(argv)

    conn = connect(args.db)
    try:
        conn.executescript(_DDL)
        summary = rebuild(conn)
    finally:
        conn.close()
    if summary is None:
        print(f"vs_spy: epoch {EPOCH_DATE} not started or inputs missing; nothing written (ok)")
    else:
        print(
            f"vs_spy: as of {summary['as_of']} — Vicuna {100*summary['vicuna_ret']:+.2f}%, "
            f"SPY {100*summary['spy_ret']:+.2f}%, excess {100*summary['excess']:+.2f} pp"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
