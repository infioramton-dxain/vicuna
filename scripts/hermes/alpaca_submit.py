#!/usr/bin/env python3
""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("AGENT_SHIPPED_DB", "/home/agent/agent/db/macmini.db"))
TRADES_DB = Path(os.environ.get("AGENT_TRADES_DB", "/home/agent/agent/trades.db"))
DRY_RUN = os.environ.get("AGENT_DRY_RUN", "1") == "1"


def _read_pick(conn: sqlite3.Connection, symbol: str) -> dict | None:
    try:
        row = conn.execute(
            """SELECT rank, strategy_id, strategy_name, symbol, sector,
                      deflated_sharpe, suggested_size_pct,
                      suggested_stop_loss_pct, suggested_target_pct, rationale,
                      entry_type, exit_type, trail_pct
                 FROM mm_active_picks WHERE symbol = ? ORDER BY rank LIMIT 1""",
            (symbol,),
        ).fetchone()
        variant_entry = True
    except sqlite3.OperationalError:
        row = conn.execute(
            """SELECT rank, strategy_id, strategy_name, symbol, sector,
                      deflated_sharpe, suggested_size_pct,
                      suggested_stop_loss_pct, suggested_target_pct, rationale
                 FROM mm_active_picks WHERE symbol = ? ORDER BY rank LIMIT 1""",
            (symbol,),
        ).fetchone()
        variant_entry = False
    if not row:
        return None
    cols = ["rank", "strategy_id", "strategy_name", "symbol", "sector",
            "dsr", "size_pct", "stop_pct", "target_pct", "rationale"]
    if variant_entry:
        cols += ["entry_type", "exit_type", "trail_pct"]
    pick = dict(zip(cols, row))
    pick.setdefault("entry_type", None)
    pick.setdefault("exit_type", None)
    pick.setdefault("trail_pct", None)
    return pick


def _sector_map() -> dict[str, str]:
    ""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        m = dict(conn.execute(
            "SELECT symbol, COALESCE(sector, 'Other') FROM mm_sector_map"))
        conn.close()
        return m
    except sqlite3.Error:
        return {}


def _init_trades_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS submit_log (
        submitted_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        strategy_id INTEGER,
        strategy_name TEXT,
        size_pct REAL, stop_pct REAL, target_pct REAL,
        dry_run INTEGER NOT NULL,
        broker_order_id TEXT,
        detail_json TEXT
      )
    """)
    conn.commit()
    conn.close()


def _log_submit(path: Path, rec: dict) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO submit_log (submitted_at, symbol, strategy_id, strategy_name, "
        "size_pct, stop_pct, target_pct, dry_run, broker_order_id, detail_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rec["submitted_at"], rec["symbol"], rec["strategy_id"],
            rec["strategy_name"], rec["size_pct"], rec["stop_pct"],
            rec["target_pct"], int(rec["dry_run"]), rec["broker_order_id"],
            json.dumps(rec["detail"]),
        ),
    )
    conn.commit()
    conn.close()


def submit_one(pick: dict, dry_run: bool) -> dict:
    ""
    if dry_run:
        return {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "symbol": pick["symbol"],
            "strategy_id": pick["strategy_id"],
            "strategy_name": pick["strategy_name"],
            "size_pct": pick["size_pct"],
            "stop_pct": pick["stop_pct"],
            "target_pct": pick["target_pct"],
            "dry_run": True,
            "broker_order_id": None,
            "detail": {"reason": "DRY_RUN=1, no order submitted"},
        }

    try:
        sys.path.insert(0, "/home/agent/agent/src")
        from agent.alpaca_client import AlpacaClient, BracketOrder
        from agent.config import (
            MAX_POSITION_NOTIONAL_PCT,
            MAX_POSITION_NOTIONAL_USD,
            MAX_SECTOR_CONCENTRATION,
        )
    except Exception as e:
        raise SystemExit(f"cannot import agent modules: {e}")

    client = AlpacaClient()
    acct = client.get_account()
    quotes = client.get_quotes([pick["symbol"]])
    if pick["symbol"] not in quotes or quotes[pick["symbol"]].mid <= 0:
        return {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "symbol": pick["symbol"],
            "strategy_id": pick["strategy_id"],
            "strategy_name": pick["strategy_name"],
            "size_pct": pick["size_pct"],
            "stop_pct": pick["stop_pct"],
            "target_pct": pick["target_pct"],
            "dry_run": False,
            "broker_order_id": None,
            "detail": {"error": "no quote available"},
        }

    mid = quotes[pick["symbol"]].mid
    cap = min(MAX_POSITION_NOTIONAL_USD,
              MAX_POSITION_NOTIONAL_PCT * acct.equity)
    dollar_size = min(acct.equity * pick["size_pct"], cap)
    positions = client.get_positions()
    sector_of = _sector_map()
    sec = pick.get("sector") or "Other"
    held = sum(p.market_value for p in positions
               if sector_of.get(p.symbol, "Other") == sec)
    book = sum(p.market_value for p in positions) + acct.cash
    if book > 0 and (held + dollar_size) / book > MAX_SECTOR_CONCENTRATION:
        return {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "symbol": pick["symbol"],
            "strategy_id": pick["strategy_id"],
            "strategy_name": pick["strategy_name"],
            "size_pct": pick["size_pct"],
            "stop_pct": pick["stop_pct"],
            "target_pct": pick["target_pct"],
            "dry_run": False,
            "broker_order_id": None,
            "detail": {"vetoed": (
                f"sector {sec} would be "
                f"{(held + dollar_size) / book:.0%} of book "
                f"> {MAX_SECTOR_CONCENTRATION:.0%} cap (D56)")},
        }
    qty = max(int(dollar_size / mid), 1)
    entry = round(mid, 2)
    stop = round(entry * (1 - pick["stop_pct"]), 2)
    target = round(entry * (1 + pick["target_pct"]), 2)

    entry_type = pick.get("entry_type")
    exit_type = pick.get("exit_type")
    if entry_type or exit_type:
        from agent.alpaca_client import VariantOrder
        trigger = round(entry * 1.005, 2) if entry_type in ("stop", "stop_limit") else None
        cap = round((trigger or entry) * 1.005, 2) if entry_type == "stop_limit" else entry
        order = VariantOrder(
            symbol=pick["symbol"], qty=qty, side="buy",
            entry_type=entry_type or "market_open",
            exit_type=exit_type or "bracket",
            entry_price=cap if entry_type == "stop_limit" else entry,
            stop_trigger=trigger,
            stop_price=stop,
            target_price=None if exit_type == "trail" else target,
            trail_pct=pick.get("trail_pct"),
        )
        order_id = client.submit_variant_entry(order)
    else:
        order = BracketOrder(
            symbol=pick["symbol"], qty=qty, side="buy",
            entry_price=entry, stop_price=stop, target_price=target,
        )
        order_id = client.submit_bracket(order)
    return {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "symbol": pick["symbol"],
        "strategy_id": pick["strategy_id"],
        "strategy_name": pick["strategy_name"],
        "size_pct": pick["size_pct"],
        "stop_pct": pick["stop_pct"],
        "target_pct": pick["target_pct"],
        "dry_run": False,
        "broker_order_id": order_id,
        "detail": {
            "qty": qty, "entry": entry, "stop": stop, "target": target,
            "buying_power": acct.buying_power, "mid_at_submit": mid,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols", nargs="+", help="approved pick symbols (space-separated)")
    p.add_argument("--dry-run", action="store_true", help="override — force log-only")
    p.add_argument("--live", action="store_true", help="override — force real submit")
    args = p.parse_args()

    if args.dry_run and args.live:
        raise SystemExit("--dry-run and --live are mutually exclusive")

    dry = args.dry_run or (DRY_RUN and not args.live)
    print(f"mode: {'DRY-RUN' if dry else 'LIVE'}", flush=True)

    if not DB_PATH.exists():
        raise SystemExit(f"shipped DB missing at {DB_PATH}")

    _init_trades_db(TRADES_DB)
    conn = sqlite3.connect(DB_PATH)

    results = []
    for sym in args.symbols:
        pick = _read_pick(conn, sym.upper())
        if not pick:
            print(f"skip {sym}: not in mm_active_picks", flush=True)
            continue
        rec = submit_one(pick, dry_run=dry)
        _log_submit(TRADES_DB, rec)
        results.append(rec)
        print(f"  {sym}: {rec.get('broker_order_id') or '(dry)'} "
              f"size={pick['size_pct']*100:.0f}%", flush=True)

    print(f"done — {len(results)} recorded to {TRADES_DB}", flush=True)


if __name__ == "__main__":
    main()
