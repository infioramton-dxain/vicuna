""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.data.schema import MM_OUTCOME_MEMORY

log = logging.getLogger(__name__)

FALLBACK_STOP = 0.08
BLOCKING = "bad_down"


def _classify(entry: float, stop_frac: float, bars: list[tuple]) -> dict:
    ""
    if not bars:
        return {"state": "pending", "stop_hit_date": None, "worst_low": None,
                "last_close": None, "last_date": None, "bars_since": 0}
    stop_level = entry * (1.0 - stop_frac)
    hit = next((d for d, low, _ in bars if low is not None and low <= stop_level), None)
    lows = [low for _, low, _ in bars if low is not None]
    last_date, _, last_close = bars[-1]
    if hit:
        state = BLOCKING if last_close < entry else "recovered"
    else:
        state = "good" if last_close >= entry else "open"
    return {"state": state, "stop_hit_date": hit,
            "worst_low": min(lows) if lows else None,
            "last_close": last_close, "last_date": last_date,
            "bars_since": len(bars)}


def evaluate(conn: sqlite3.Connection, as_of: date | None = None) -> list[dict]:
    ""
    cutoff = (as_of or date.today()).isoformat()
    recs = conn.execute(
        "SELECT rec_id, as_of_date, symbol, strategy_id, status, "
        "       suggested_stop_loss_pct, fill_status, actual_fill_price "
        "FROM mm_recs_ledger "
        "WHERE rec_id NOT LIKE 'fixture%' AND substr(as_of_date, 1, 10) < ? "
        "ORDER BY symbol, as_of_date, rec_id", (cutoff,)).fetchall()
    out: list[dict] = []
    bars_by_symbol: dict[str, list[tuple]] = {}
    for rec_id, rec_date, sym, sid, status, stop, fill_status, fill_px in recs:
        rec_date = rec_date[:10]
        if sym not in bars_by_symbol:
            bars_by_symbol[sym] = conn.execute(
                "SELECT substr(date, 1, 10), open, low, close FROM mm_bars "
                "WHERE symbol = ? AND substr(date, 1, 10) <= ? "
                "AND close IS NOT NULL ORDER BY date", (sym, cutoff)).fetchall()
        after = [b for b in bars_by_symbol[sym] if b[0] > rec_date]
        filled = fill_status == "filled" and fill_px
        entry = float(fill_px) if filled else (
            float(after[0][1] or after[0][3]) if after else None)
        stop_frac = float(stop) if stop else FALLBACK_STOP
        res = _classify(entry, stop_frac, [(d, lo, cl) for d, _, lo, cl in after]) \
            if entry else _classify(0.0, stop_frac, [])
        out.append({
            "rec_id": rec_id, "as_of_date": rec_date, "symbol": sym,
            "strategy_id": sid, "rec_status": status, "filled": int(bool(filled)),
            "entry_price": entry, "stop_frac": stop_frac,
            "return_to_date": (res["last_close"] / entry - 1.0)
            if entry and res["last_close"] else None,
            **res,
        })
    return out


def rebuild(conn: sqlite3.Connection, as_of: date | None = None) -> dict:
    ""
    conn.execute(MM_OUTCOME_MEMORY)
    rows = evaluate(conn, as_of)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM mm_outcome_memory")
    conn.commit()
    last_symbol = None
    for r in rows:
        if last_symbol is not None and r["symbol"] != last_symbol:
            conn.commit()
        last_symbol = r["symbol"]
        conn.execute(
            "INSERT INTO mm_outcome_memory (rec_id, as_of_date, symbol, "
            "strategy_id, rec_status, filled, entry_price, stop_frac, state, "
            "stop_hit_date, worst_low, last_close, last_date, return_to_date, "
            "bars_since, computed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["rec_id"], r["as_of_date"], r["symbol"], r["strategy_id"],
             r["rec_status"], r["filled"], r["entry_price"], r["stop_frac"],
             r["state"], r["stop_hit_date"], r["worst_low"], r["last_close"],
             r["last_date"], r["return_to_date"], r["bars_since"], now))
    conn.commit()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return {"recs": len(rows), "states": counts,
            "blocked": sorted(blocked_symbols(conn))}


def blocked_symbols(conn: sqlite3.Connection) -> dict[str, str]:
    ""
    try:
        rows = conn.execute(
            "SELECT symbol, as_of_date, entry_price, stop_hit_date, last_close, "
            "       return_to_date FROM mm_outcome_memory WHERE state = ? "
            "ORDER BY symbol, as_of_date", (BLOCKING,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, str] = {}
    for sym, rec_date, entry, hit, last, ret in rows:
        out.setdefault(sym, (
            f"recommended {rec_date} at {entry:.2f}, stop level breached "
            f"{hit}, still {ret:+.0%} at {last:.2f}"))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=None)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; default today")
    args = p.parse_args(argv)
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    with connect(resolve_db(args.db)) as conn:
        print(json.dumps(rebuild(conn, as_of)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
