""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from pathlib import Path

from macmini.config import resolve_db

log = logging.getLogger(__name__)

LIVE_ORDER_STATUSES = (
    "new", "held", "accepted", "partially_filled", "pending_new",
    "pending_replace", "calculated",
)

STOP_ORDER_TYPES = ("stop", "stop_limit", "trailing_stop")

PRICE_TOLERANCE = 0.011
MIRROR_MAX_AGE_H = 36
DIV_POST_GRACE_D = 7

LEDGER_OPEN_WHERE = (
    "status = 'approved' AND fill_status = 'filled' "
    "AND (exit_status IS NULL OR exit_status = 'open')"
)


def _finding(severity: str, kind: str, symbol: str | None = None,
             rec_id: str | None = None, detail: str = "") -> dict:
    return {"severity": severity, "kind": kind, "symbol": symbol,
            "rec_id": rec_id, "detail": detail}


def check(db_path: Path | str | None = None) -> dict:
    ""
    path = resolve_db(db_path)
    findings: list[dict] = []

    with sqlite3.connect(path, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.row_factory = sqlite3.Row

        snapshot_at = c.execute(
            "SELECT MAX(snapshot_at) FROM ("
            "  SELECT snapshot_at FROM mm_alpaca_positions"
            "  UNION ALL SELECT snapshot_at FROM mm_alpaca_orders)"
        ).fetchone()[0]

        if snapshot_at is None:
            findings.append(_finding(
                "warn", "mirror_missing",
                detail="mm_alpaca_positions and mm_alpaca_orders are both "
                       "empty — broker mirror has never run; cannot check."))
        else:
            age_h = ((datetime.now(timezone.utc)
                      - datetime.fromisoformat(snapshot_at))
                     / timedelta(hours=1))
            if age_h > MIRROR_MAX_AGE_H:
                findings.append(_finding(
                    "warn", "mirror_stale",
                    detail=f"mirror snapshot is {age_h:.1f}h old "
                           f"(> {MIRROR_MAX_AGE_H}h) — findings below are "
                           f"against stale broker state."))

        positions = {r["symbol"]: r for r in c.execute(
            "SELECT * FROM mm_alpaca_positions")}

        ledger_open = c.execute(
            f"SELECT * FROM mm_recs_ledger WHERE {LEDGER_OPEN_WHERE} "
            f"ORDER BY updated_at").fetchall()

        by_symbol: dict[str, list[sqlite3.Row]] = {}
        for r in ledger_open:
            by_symbol.setdefault(r["symbol"], []).append(r)

        for sym, recs in by_symbol.items():
            if sym not in positions:
                findings.append(_finding(
                    "critical", "ledger_open_no_position", sym,
                    recs[-1]["rec_id"],
                    f"ledger holds {sum(r['fill_qty'] or 0 for r in recs)} "
                    f"shares but broker mirror shows no position. "
                    f"Triangulate positions + activities + "
                    f"portfolio history before believing either side; "
                    f"never force-sell while the broker says flat."))

        for r in c.execute(
            "SELECT * FROM mm_recs_ledger WHERE status = 'approved' "
            "AND alpaca_order_id IS NULL "
            "AND datetime(decided_at) < datetime('now', '-2 hours')"):
            findings.append(_finding(
                "critical", "approved_no_order", r["symbol"], r["rec_id"],
                f"rec approved at {r['decided_at']} but alpaca_order_id is "
                f"still NULL — order was never placed (AVPT 2026-07-10 "
                f"shape: partial ledger write, execution skipped). Confirm "
                f"at the broker, then either place the order via the "
                f"mini's agent.recs.execute or expire the rec."))

        for r in c.execute(
            "SELECT l.rec_id, l.symbol, l.fill_status, o.status AS o_status, "
            "       o.filled_qty, o.filled_avg_price, o.filled_at "
            "FROM mm_recs_ledger l JOIN mm_alpaca_orders o ON o.id = l.alpaca_order_id "
            "WHERE l.status = 'approved' "
            "AND (l.fill_status IS NULL OR l.fill_status NOT IN ('filled', 'shadow')) "
            "AND o.status IN ('filled', 'partially_filled') "
            "AND COALESCE(o.filled_qty, 0) > 0"):
            findings.append(_finding(
                "critical", "order_filled_ledger_stale", r["symbol"], r["rec_id"],
                f"broker order {r['o_status']} {r['filled_qty']:g} @ "
                f"{r['filled_avg_price']} at {r['filled_at']} but ledger "
                f"fill_status is {r['fill_status']!r} — the mini's poller never "
                f"picked this row up ('pending' shape). Backfill "
                f"the row in trades.db from the broker; until then the trade is "
                f"missing from the dashboard, attribution and recs-protect."))

        ph_live = ",".join("?" * len(LIVE_ORDER_STATUSES))
        et = ZoneInfo("America/New_York")
        snap_day = (datetime.fromisoformat(snapshot_at).astimezone(et).date()
                    if snapshot_at else None)
        for o in c.execute(
            f"SELECT o.id, o.symbol, o.qty, o.limit_price, o.submitted_at, "
            f"       l.rec_id FROM mm_alpaca_orders o "
            f"LEFT JOIN mm_recs_ledger l ON l.alpaca_order_id = o.id "
            f"WHERE o.side = 'buy' AND o.status IN ({ph_live}) "
            f"AND (o.parent_id IS NULL OR o.parent_id = '')",
            LIVE_ORDER_STATUSES):
            rec_id = o["rec_id"] or ""
            if rec_id.startswith("OPERATOR_") or "-op-" in rec_id:
                continue
            if not o["submitted_at"] or snap_day is None:
                continue
            placed_day = datetime.fromisoformat(
                o["submitted_at"].replace("Z", "+00:00")).astimezone(et).date()
            if placed_day < snap_day:
                findings.append(_finding(
                    "warn", "stale_resting_entry", o["symbol"], o["rec_id"],
                    f"entry parent {o['id'][:8]} ({o['qty']:g} @ limit "
                    f"{o['limit_price']}) placed {placed_day} is still resting "
                    f"on {snap_day} — v1.6: unfilled entries are canceled at the "
                    f"16:05 sweep and re-priced next morning from fresh data. "
                    f"The sweep missed it (or is not yet live on the mini)."))

        for sym, pos in positions.items():
            if sym not in by_symbol:
                manual = c.execute(
                    "SELECT rec_id FROM mm_recs_ledger WHERE symbol = ? "
                    "AND fill_status = 'filled' "
                    "AND exit_status = 'operator_manual' "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (sym,)).fetchone()
                if manual:
                    findings.append(_finding(
                        "info", "operator_manual_position", sym,
                        manual["rec_id"],
                        f"broker holds {pos['qty']:g} shares under a "
                        f"manually managed rec — expected; excluded from "
                        f"ledger-open checks."))
                else:
                    findings.append(_finding(
                        "warn", "position_not_in_ledger", sym, None,
                        f"broker holds {pos['qty']:g} shares with no "
                        f"ledger-open rec (untracked position)."))

        for sym, recs in by_symbol.items():
            pos = positions.get(sym)
            if pos is None:
                continue
            ledger_qty = sum(r["fill_qty"] or 0 for r in recs)
            if abs(ledger_qty - (pos["qty"] or 0)) > 1e-9:
                findings.append(_finding(
                    "warn", "qty_mismatch", sym, recs[-1]["rec_id"],
                    f"ledger fill_qty {ledger_qty:g} != broker qty "
                    f"{pos['qty']:g}."))

        ph = ",".join("?" * len(LIVE_ORDER_STATUSES))
        for sym, pos in positions.items():
            legs = c.execute(
                f"SELECT * FROM mm_alpaca_orders WHERE symbol = ? "
                f"AND side = 'sell' AND status IN ({ph})",
                (sym, *LIVE_ORDER_STATUSES)).fetchall()
            stop_legs = [o for o in legs if o["order_type"] in STOP_ORDER_TYPES]
            tp_legs = [o for o in legs if o["order_type"] == "limit"]
            recs = by_symbol.get(sym, [])
            rec = recs[-1] if recs else None
            rec_id = rec["rec_id"] if rec else None

            rec_exit_type_early = (
                rec["exit_type"] if rec is not None and "exit_type" in rec.keys()
                else None
            )
            if rec_exit_type_early == "operator_manual":
                continue
            if rec is None and c.execute(
                "SELECT 1 FROM mm_recs_ledger WHERE symbol = ? "
                "AND fill_status = 'filled' "
                "AND (exit_status = 'operator_manual' OR exit_type = 'operator_manual') "
                "LIMIT 1", (sym,)).fetchone():
                continue

            if not stop_legs:
                findings.append(_finding(
                    "critical", "missing_stop_leg", sym, rec_id,
                    f"broker position ({pos['qty']:g} shares) has no live "
                    f"sell-stop leg in the mirror (position is "
                    f"unprotected on the downside)."))
            rec_exit_type = (
                rec["exit_type"] if rec is not None and "exit_type" in rec.keys()
                else None
            )
            if not tp_legs and rec_exit_type != "trail":
                findings.append(_finding(
                    "warn", "missing_target_leg", sym, rec_id,
                    f"broker position has no live sell-limit (take-profit) "
                    f"leg in the mirror."))

            for leg_kind, leg_list in (("stop", stop_legs), ("target", tp_legs)):
                for o in leg_list:
                    if o["qty"] is not None and abs(o["qty"] - (pos["qty"] or 0)) > 1e-9:
                        findings.append(_finding(
                            "warn", "leg_qty_mismatch", sym, rec_id,
                            f"{leg_kind} leg {o['id'][:8]}… qty {o['qty']:g} "
                            f"!= position qty {pos['qty']:g}."))

            if rec is not None:
                if stop_legs and rec["stop_price"] is not None \
                        and stop_legs[0]["order_type"] != "trailing_stop":
                    live = stop_legs[0]["stop_price"]
                    if live is not None and abs(live - rec["stop_price"]) > PRICE_TOLERANCE:
                        findings.append(_finding(
                            "info", "leg_price_drift", sym, rec_id,
                            f"ledger stop {rec['stop_price']} vs live stop "
                            f"leg {live} — re-peg not yet synced, or stale "
                            f"ledger."))
                if tp_legs and rec["target_price"] is not None:
                    live = tp_legs[0]["limit_price"]
                    if live is not None and abs(live - rec["target_price"]) > PRICE_TOLERANCE:
                        findings.append(_finding(
                            "info", "leg_price_drift", sym, rec_id,
                            f"ledger target {rec['target_price']} vs live "
                            f"TP leg {live} — re-peg not yet synced, or "
                            f"stale ledger."))

        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=DIV_POST_GRACE_D)).strftime("%Y-%m-%d")
        for rec in c.execute(
                """SELECT rec_id, symbol, fill_qty,
                          COALESCE(date(actual_fill_at), entry_date) AS held_from
                   FROM mm_recs_ledger
                   WHERE fill_status='filled'
                     AND (exit_status IS NULL
                          OR exit_status IN ('open', 'manual_override'))"""):
            divs = c.execute(
                """SELECT ex_date, rate FROM mm_dividends
                   WHERE symbol = ? AND ex_date > ? AND ex_date <= ?""",
                (rec["symbol"], rec["held_from"], cutoff)).fetchall()
            for d in divs:
                credited = c.execute(
                    """SELECT COUNT(*) FROM mm_alpaca_activities
                       WHERE symbol = ? AND activity_type LIKE 'DIV%'
                         AND COALESCE(transaction_time,
                                      substr(id, 1, 4) || '-' ||
                                      substr(id, 5, 2) || '-' ||
                                      substr(id, 7, 2)) >= ?""",
                    (rec["symbol"], d["ex_date"])).fetchone()[0]
                if not credited:
                    owed = (rec["fill_qty"] or 0) * (d["rate"] or 0)
                    findings.append(_finding(
                        "warn", "dividend_not_credited", rec["symbol"],
                        rec["rec_id"],
                        f"ex-date {str(d['ex_date'])[:10]} rate {d['rate']} "
                        f"x {rec['fill_qty']} sh = ${owed:.2f} — no broker "
                        f"DIV activity within {DIV_POST_GRACE_D}d "
                        f"(paper accounts may never post; equity is "
                        f"understated by this amount)."))

        snap_for_rows = snapshot_at or datetime.now(timezone.utc).isoformat()
        c.execute("DELETE FROM mm_broker_divergences")
        c.executemany(
            "INSERT INTO mm_broker_divergences "
            "(snapshot_at, severity, kind, symbol, rec_id, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(snap_for_rows, f["severity"], f["kind"], f["symbol"],
              f["rec_id"], f["detail"]) for f in findings])
        c.commit()

    counts = {s: sum(1 for f in findings if f["severity"] == s)
              for s in ("critical", "warn", "info")}
    result = {"snapshot_at": snapshot_at, "findings": findings, **counts}
    log.info("divergence check: %d critical, %d warn, %d info (mirror %s)",
             counts["critical"], counts["warn"], counts["info"], snapshot_at)
    return result


def summary_text(result: dict, limit: int = 6) -> str:
    ""
    sev_order = {"critical": 0, "warn": 1, "info": 2}
    fs = sorted(result["findings"], key=lambda f: sev_order[f["severity"]])
    fs = [f for f in fs if f["severity"] != "info"]
    lines = [f"{f['severity'].upper()} {f['kind']}"
             + (f" {f['symbol']}" if f["symbol"] else "")
             for f in fs[:limit]]
    if len(fs) > limit:
        lines.append(f"(+{len(fs) - limit} more)")
    lines.append("Triangulate before acting — see dashboard.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(
        description="Cross-check mm_recs_ledger vs mm_alpaca_* broker mirror.")
    p.add_argument("--db", default=None)
    p.add_argument("--summary-file", default=None,
                   help="write a short alert body here when findings exist")
    args = p.parse_args(argv)

    result = check(args.db)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}))
    for f in result["findings"]:
        print(f"  [{f['severity']}] {f['kind']}"
              + (f" {f['symbol']}" if f["symbol"] else "")
              + f" — {f['detail']}")

    actionable = result["critical"] + result["warn"]
    if args.summary_file and actionable:
        Path(args.summary_file).write_text(summary_text(result))
    return 3 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
