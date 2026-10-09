""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable

from macmini.config import DB_PATH, EXPORTS_DIR, resolve_db

log = logging.getLogger(__name__)


MAX_HOLD_DAYS = 20
CALENDAR_REF_SYMBOL = "SPY"
DECISIONS_JSONL_DEFAULT: Path = EXPORTS_DIR / "decisions.jsonl"

SCOREBOARD_WINDOWS: dict[str, int | None] = {
    "7d":  7,
    "30d": 30,
    "90d": 90,
    "all": None,
}

_BUCKET_MAP = {
    "approve":  "approved",
    "reject":   "rejected_or_expired",
    "expired":  "rejected_or_expired",
}

FIXTURE_PATTERNS = ("fixture", "paperfixture", "test-strat")


def _is_fixture(strategy_name: str | None, alpaca_order_id: str | None) -> bool:
    hay = f"{strategy_name or ''} {alpaca_order_id or ''}".lower()
    return any(p in hay for p in FIXTURE_PATTERNS)


@dataclass
class Decision:
    ""
    rec_id: str
    as_of_date: str
    symbol: str
    strategy_id: int
    strategy_name: str
    decision: str
    decided_at: str
    operator_reply: str
    suggested_stop_loss_pct: float
    suggested_target_pct: float
    suggested_size_pct: float
    alpaca_order_id: str | None
    notes: str


@dataclass
class FillUpdate:
    ""
    rec_id: str
    alpaca_order_id: str
    actual_fill_price: float
    actual_fill_at: str
    fill_qty: int


@dataclass
class ExitFillUpdate:
    ""
    rec_id: str
    alpaca_order_id: str
    actual_exit_price: float
    actual_exit_at: str
    exit_qty: int
    exit_reason: str | None = None


@dataclass
class Resolution:
    ""
    entry_date: str | None
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    exit_date: str | None
    exit_price: float | None
    exit_reason: str | None
    days_held: int | None
    status: str


def load_decisions_log(
    path: Path, allow_fixtures: bool = False
) -> tuple[dict[str, Decision], dict[str, FillUpdate], dict[str, ExitFillUpdate]]:
    ""
    decisions: dict[str, Decision] = {}
    fills: dict[str, FillUpdate] = {}
    exit_fills: dict[str, ExitFillUpdate] = {}
    if not path.exists():
        log.info("decisions.jsonl not present at %s; nothing to attribute", path)
        return decisions, fills, exit_fills

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("decisions.jsonl:%d JSON error: %s", lineno, exc)
                continue

            rec_id = rec.get("rec_id")
            if not isinstance(rec_id, str) or not rec_id:
                log.warning("decisions.jsonl:%d missing rec_id; skipping", lineno)
                continue

            kind = rec.get("decision")
            if kind is None and rec.get("type") == "pipeline_instruction":
                log.info("decisions.jsonl:%d pipeline_instruction %r (symbol=%s) — "
                         "not a decision; handled out-of-band",
                         lineno, rec.get("instruction"), rec.get("symbol"))
                continue
            if kind == "fill_update":
                leg = rec.get("leg", "entry")
                if leg == "entry":
                    try:
                        fills[rec_id] = FillUpdate(
                            rec_id=rec_id,
                            alpaca_order_id=rec["alpaca_order_id"],
                            actual_fill_price=float(rec["actual_fill_price"]),
                            actual_fill_at=rec["actual_fill_at"],
                            fill_qty=int(rec["fill_qty"]),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        log.warning("decisions.jsonl:%d malformed entry fill_update %s: %s",
                                    lineno, rec_id, exc)
                elif leg == "exit":
                    try:
                        exit_fills[rec_id] = ExitFillUpdate(
                            rec_id=rec_id,
                            alpaca_order_id=rec["alpaca_order_id"],
                            actual_exit_price=float(rec["exit_price"]),
                            actual_exit_at=rec["exit_at"],
                            exit_qty=int(rec["exit_qty"]),
                            exit_reason=rec.get("exit_reason"),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        log.warning("decisions.jsonl:%d malformed exit fill_update %s: %s",
                                    lineno, rec_id, exc)
                else:
                    log.warning("decisions.jsonl:%d fill_update %s has unknown leg=%r; skipping",
                                lineno, rec_id, leg)
            elif kind in ("approve", "reject", "expired"):
                if not allow_fixtures and _is_fixture(
                        rec.get("strategy_name"), rec.get("alpaca_order_id")):
                    log.warning("decisions.jsonl:%d skipping fixture record %s",
                                lineno, rec_id)
                    continue
                decided_at = rec.get("decided_at", "")
                current = decisions.get(rec_id)
                if current is None or decided_at > current.decided_at:
                    sid = rec.get("strategy_id")
                    if sid is None:
                        log.warning("decisions.jsonl:%d %s has strategy_id null; "
                                    "using -1 pending ledger backfill", lineno, rec_id)
                    decisions[rec_id] = Decision(
                        rec_id=rec_id,
                        as_of_date=rec["as_of_date"],
                        symbol=rec["symbol"],
                        strategy_id=int(sid) if sid is not None else -1,
                        strategy_name=rec.get("strategy_name", ""),
                        decision=kind,
                        decided_at=decided_at,
                        operator_reply=rec.get("operator_reply", ""),
                        suggested_stop_loss_pct=float(rec["suggested_stop_loss_pct"]),
                        suggested_target_pct=float(rec["suggested_target_pct"]),
                        suggested_size_pct=float(rec.get("suggested_size_pct") or 0.0),
                        alpaca_order_id=rec.get("alpaca_order_id"),
                        notes=rec.get("notes", ""),
                    )
            else:
                log.warning("decisions.jsonl:%d unknown decision kind %r", lineno, kind)
    return decisions, fills, exit_fills


def _bars_after(conn: sqlite3.Connection, symbol: str, after_date: str) -> list[dict]:
    ""
    rows = conn.execute(
        "SELECT date, open, high, low, close FROM mm_bars "
        "WHERE symbol = ? AND date > ? ORDER BY date ASC",
        (symbol, after_date),
    ).fetchall()
    return [{"date": r[0], "open": r[1], "high": r[2],
             "low": r[3], "close": r[4]} for r in rows]


def resolve_bracket(
    symbol_bars: list[dict],
    calendar_bars: list[dict],
    stop_pct: float,
    target_pct: float,
) -> Resolution:
    ""
    if not calendar_bars:
        return Resolution(None, None, None, None, None, None, None, None, "open")

    entry_date = calendar_bars[0]["date"]

    if not symbol_bars or symbol_bars[0]["date"] != entry_date:
        sessions_elapsed = len(calendar_bars) - 1
        if sessions_elapsed >= MAX_HOLD_DAYS:
            return Resolution(entry_date, None, None, None, entry_date, None,
                              "no_data_20d", MAX_HOLD_DAYS, "closed")
        return Resolution(entry_date, None, None, None, None, None, None, None, "open")

    entry_bar = symbol_bars[0]
    entry_price = float(entry_bar["open"])
    stop_price = entry_price * (1.0 - stop_pct)
    target_price = entry_price * (1.0 + target_pct)

    if entry_price <= stop_price:
        return Resolution(entry_date, entry_price, stop_price, target_price,
                          entry_date, entry_price, "gap_stop", 0, "closed")
    if entry_price >= target_price:
        return Resolution(entry_date, entry_price, stop_price, target_price,
                          entry_date, entry_price, "gap_target", 0, "closed")

    for i, bar in enumerate(symbol_bars[1:MAX_HOLD_DAYS + 1], start=1):
        if float(bar["low"]) <= stop_price:
            return Resolution(entry_date, entry_price, stop_price, target_price,
                              bar["date"], stop_price, "stop", i, "closed")
        if float(bar["high"]) >= target_price:
            return Resolution(entry_date, entry_price, stop_price, target_price,
                              bar["date"], target_price, "target", i, "closed")

    if len(symbol_bars) > MAX_HOLD_DAYS:
        last = symbol_bars[MAX_HOLD_DAYS]
        return Resolution(entry_date, entry_price, stop_price, target_price,
                          last["date"], float(last["close"]),
                          "time_20d", MAX_HOLD_DAYS, "closed")

    return Resolution(entry_date, entry_price, stop_price, target_price,
                      None, None, None, None, "open")


def _fetch_existing(conn: sqlite3.Connection, rec_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM mm_recommendation_performance WHERE rec_id = ?",
        (rec_id,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM mm_recommendation_performance LIMIT 0").description]
    return dict(zip(cols, row))


def compute_row(
    conn: sqlite3.Connection,
    decision: Decision,
    fill: FillUpdate | None,
    exit_fill: ExitFillUpdate | None,
    existing: dict | None,
    now_iso: str,
) -> dict:
    ""
    if existing is not None and existing.get("status") == "closed":
        resolution = Resolution(
            entry_date=existing["entry_date"],
            entry_price=existing["entry_price"],
            stop_price=existing["stop_price"],
            target_price=existing["target_price"],
            exit_date=existing["exit_date"],
            exit_price=existing["exit_price"],
            exit_reason=existing["exit_reason"],
            days_held=existing["days_held"],
            status="closed",
        )
    else:
        symbol_bars = _bars_after(conn, decision.symbol, decision.as_of_date)
        calendar_bars = _bars_after(conn, CALENDAR_REF_SYMBOL, decision.as_of_date)
        resolution = resolve_bracket(
            symbol_bars, calendar_bars,
            decision.suggested_stop_loss_pct,
            decision.suggested_target_pct,
        )

    dividends_per_share = 0.0
    if resolution.entry_date and resolution.exit_date:
        row = conn.execute(
            "SELECT COALESCE(SUM(rate), 0) FROM mm_dividends "
            "WHERE symbol = ? AND ex_date > ? AND ex_date <= ?",
            (decision.symbol, resolution.entry_date, resolution.exit_date),
        ).fetchone()
        dividends_per_share = float(row[0] or 0.0)

    if resolution.status == "closed" and resolution.entry_price and resolution.exit_price is not None:
        return_pct_cf = (resolution.exit_price + dividends_per_share
                         - resolution.entry_price) / resolution.entry_price
    else:
        return_pct_cf = None

    actual_fill_price = fill.actual_fill_price if fill else None
    actual_fill_at   = fill.actual_fill_at   if fill else None
    fill_qty         = fill.fill_qty         if fill else None
    slippage_pct     = None
    if decision.decision == "approve" and actual_fill_price and resolution.entry_price:
        slippage_pct = (actual_fill_price - resolution.entry_price) / resolution.entry_price

    actual_exit_price  = exit_fill.actual_exit_price  if exit_fill else None
    actual_exit_at     = exit_fill.actual_exit_at     if exit_fill else None
    actual_exit_qty    = exit_fill.exit_qty           if exit_fill else None
    actual_exit_reason = exit_fill.exit_reason        if exit_fill else None

    return_pct_actual = None
    if decision.decision == "approve" and actual_fill_price and actual_fill_price != 0:
        if actual_exit_price is not None:
            return_pct_actual = (actual_exit_price + dividends_per_share
                                 - actual_fill_price) / actual_fill_price
        elif resolution.exit_price is not None:
            return_pct_actual = (resolution.exit_price + dividends_per_share
                                 - actual_fill_price) / actual_fill_price

    return {
        "rec_id":                    decision.rec_id,
        "as_of_date":                decision.as_of_date,
        "symbol":                    decision.symbol,
        "strategy_id":               decision.strategy_id,
        "strategy_name":             decision.strategy_name,
        "decision":                  decision.decision,
        "decided_at":                decision.decided_at,
        "operator_reply":            decision.operator_reply,
        "entry_date":                resolution.entry_date,
        "entry_price":               resolution.entry_price,
        "stop_pct":                  decision.suggested_stop_loss_pct,
        "target_pct":                decision.suggested_target_pct,
        "size_pct":                  decision.suggested_size_pct,
        "stop_price":                resolution.stop_price,
        "target_price":              resolution.target_price,
        "alpaca_order_id":           decision.alpaca_order_id,
        "actual_fill_price":         actual_fill_price,
        "actual_fill_at":            actual_fill_at,
        "fill_qty":                  fill_qty,
        "slippage_pct":              slippage_pct,
        "exit_date":                 resolution.exit_date,
        "exit_price":                resolution.exit_price,
        "exit_reason":               resolution.exit_reason,
        "days_held":                 resolution.days_held,
        "actual_exit_price":         actual_exit_price,
        "actual_exit_at":            actual_exit_at,
        "actual_exit_qty":           actual_exit_qty,
        "actual_exit_reason":        actual_exit_reason,
        "return_pct_counterfactual": return_pct_cf,
        "return_pct_actual":         return_pct_actual,
        "dividends_per_share":       dividends_per_share,
        "status":                    resolution.status,
        "computed_at":               now_iso,
    }


_PERF_COLS = (
    "rec_id", "as_of_date", "symbol", "strategy_id", "strategy_name",
    "decision", "decided_at", "operator_reply",
    "entry_date", "entry_price", "stop_pct", "target_pct", "size_pct",
    "stop_price", "target_price",
    "alpaca_order_id", "actual_fill_price", "actual_fill_at", "fill_qty", "slippage_pct",
    "exit_date", "exit_price", "exit_reason", "days_held",
    "actual_exit_price", "actual_exit_at", "actual_exit_qty", "actual_exit_reason",
    "return_pct_counterfactual", "return_pct_actual",
    "dividends_per_share",
    "status", "computed_at",
)


def _upsert_perf(conn: sqlite3.Connection, row: dict) -> None:
    placeholders = ",".join("?" * len(_PERF_COLS))
    cols = ",".join(_PERF_COLS)
    conn.execute(
        f"INSERT OR REPLACE INTO mm_recommendation_performance ({cols}) "
        f"VALUES ({placeholders})",
        tuple(row[c] for c in _PERF_COLS),
    )


def rebuild_scoreboard(conn: sqlite3.Connection, as_of_date: str, now_iso: str) -> int:
    ""
    conn.execute("DELETE FROM mm_recommendation_scoreboard")

    rows = conn.execute(
        "SELECT decision, exit_date, return_pct_counterfactual, return_pct_actual, "
        "       slippage_pct, actual_fill_price "
        "FROM mm_recommendation_performance "
        "WHERE status = 'closed' AND exit_date IS NOT NULL "
        "  AND COALESCE(strategy_name, '') NOT LIKE '%fixture%' "
        "  AND COALESCE(alpaca_order_id, '') NOT LIKE 'paperfixture%'"
    ).fetchall()

    from datetime import date as date_type
    as_of_d = date_type.fromisoformat(as_of_date)

    n_written = 0
    for window, lookback_days in SCOREBOARD_WINDOWS.items():
        buckets: dict[str, list[float | None]] = {
            "approved": [], "rejected_or_expired": [],
        }
        exec_q = {"n_approved": 0, "n_filled": 0, "slippages": [],
                  "actual_returns": [], "shortfalls": []}
        for decision_kind, exit_date_s, ret_cf, ret_act, slip, fill_px in rows:
            bucket = _BUCKET_MAP.get(decision_kind)
            if bucket is None:
                continue
            if lookback_days is not None:
                exit_d = date_type.fromisoformat(exit_date_s)
                if (as_of_d - exit_d).days > lookback_days:
                    continue
                if exit_d > as_of_d:
                    continue
            eff = ret_act if (bucket == "approved" and ret_act is not None) else ret_cf
            buckets[bucket].append(eff)
            if bucket == "approved":
                exec_q["n_approved"] += 1
                if fill_px is not None:
                    exec_q["n_filled"] += 1
                if slip is not None:
                    exec_q["slippages"].append(slip)
                if ret_act is not None:
                    exec_q["actual_returns"].append(ret_act)
                if ret_act is not None and ret_cf is not None:
                    exec_q["shortfalls"].append(ret_act - ret_cf)

        for bucket in ("approved", "rejected_or_expired"):
            entries = buckets[bucket]
            returns = [r for r in entries if r is not None]
            n_recs = len(entries)
            n_closed = len(returns)
            win_rate = (sum(1 for r in returns if r > 0) / n_closed) if n_closed else None
            avg_return = (sum(returns) / n_closed) if n_closed else None
            median_return = median(returns) if returns else None
            total_return = sum(returns) if returns else None
            fill_rate = avg_slip = avg_actual = shortfall = None
            if bucket == "approved":
                if exec_q["n_approved"]:
                    fill_rate = exec_q["n_filled"] / exec_q["n_approved"]
                if exec_q["slippages"]:
                    avg_slip = sum(exec_q["slippages"]) / len(exec_q["slippages"])
                if exec_q["actual_returns"]:
                    avg_actual = (sum(exec_q["actual_returns"])
                                  / len(exec_q["actual_returns"]))
                if exec_q["shortfalls"]:
                    shortfall = sum(exec_q["shortfalls"]) / len(exec_q["shortfalls"])
            conn.execute(
                "INSERT INTO mm_recommendation_scoreboard "
                "(as_of_date, window, decision_bucket, n_recs, n_closed, "
                " win_rate, avg_return_pct, median_return_pct, total_return_pct, "
                " fill_rate, avg_slippage_pct, avg_return_pct_actual, "
                " implementation_shortfall_pct, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (as_of_date, window, bucket, n_recs, n_closed,
                 win_rate, avg_return, median_return, total_return,
                 fill_rate, avg_slip, avg_actual, shortfall, now_iso),
            )
            n_written += 1
    return n_written


def run_attribution(
    db_path: Path | str | None = None,
    decisions_path: Path | str | None = None,
    as_of_date: date | str | None = None,
    allow_fixtures: bool = False,
) -> dict:
    ""
    db = resolve_db(db_path)
    dec_path = Path(decisions_path) if decisions_path is not None else DECISIONS_JSONL_DEFAULT
    if as_of_date is None:
        as_of_str = date.today().isoformat()
    elif isinstance(as_of_date, date):
        as_of_str = as_of_date.isoformat()
    else:
        as_of_str = as_of_date

    decisions, fills, exit_fills = load_decisions_log(dec_path, allow_fixtures=allow_fixtures)
    now_iso = datetime.now(timezone.utc).isoformat()

    perf_written = 0
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for rec_id, decision in decisions.items():
            if decision.strategy_id == -1:
                row = conn.execute(
                    "SELECT strategy_id FROM mm_recs_ledger WHERE rec_id = ?",
                    (rec_id,),
                ).fetchone()
                if row is not None and row[0] is not None:
                    decision.strategy_id = int(row[0])
                    log.info("backfilled strategy_id=%d for %s from mm_recs_ledger",
                             decision.strategy_id, rec_id)
        for rec_id, decision in decisions.items():
            existing = _fetch_existing(conn, rec_id)
            row = compute_row(
                conn, decision, fills.get(rec_id), exit_fills.get(rec_id),
                existing, now_iso,
            )
            _upsert_perf(conn, row)
            perf_written += 1
        scoreboard_rows = rebuild_scoreboard(conn, as_of_str, now_iso)
        conn.commit()

    return {
        "decisions_read":    len(decisions),
        "fills_read":        len(fills),
        "exit_fills_read":   len(exit_fills),
        "perf_rows_written": perf_written,
        "scoreboard_rows":   scoreboard_rows,
        "as_of_date":        as_of_str,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Recommendation attribution engine")
    p.add_argument("--db", default=None)
    p.add_argument("--decisions", default=None,
                   help=f"Path to decisions.jsonl (default: {DECISIONS_JSONL_DEFAULT})")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    p.add_argument("--allow-fixtures", action="store_true",
                   help="Permit fixture records (manual parity workflow only)")
    args = p.parse_args(argv)

    out = run_attribution(db_path=args.db, decisions_path=args.decisions,
                          as_of_date=args.as_of, allow_fixtures=args.allow_fixtures)
    print("attribution done:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
