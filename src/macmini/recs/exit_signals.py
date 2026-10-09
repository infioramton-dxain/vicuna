""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from macmini.config import resolve_db
from macmini.data._db import connect

log = logging.getLogger(__name__)

CHANDELIER_MULT = 2.5
CHANDELIER_LOOKBACK = 20
STOP_ATR_MULT = 3.0
STOP_FRAC_MIN, STOP_FRAC_MAX = 0.05, 0.15

TIGHTEN_CAVEAT = (
    "Caveat: the backtested version of tighten-on-rollover reduced "
    "deflated Sharpe across all qualifying cells; treat as a discussion "
    "prompt, not a default."
)


def _entry_date(c, symbol: str) -> str | None:
    row = c.execute(
        "SELECT MIN(transaction_time) FROM mm_alpaca_activities "
        "WHERE symbol = ? AND activity_type = 'FILL' AND side = 'buy'",
        (symbol,),
    ).fetchone()
    return row[0][:10] if row and row[0] else None


def _ledger_exit_prefs(c, symbol: str) -> tuple[str | None, float | None]:
    ""
    row = c.execute(
        "SELECT exit_type, trail_pct FROM mm_recs_ledger "
        "WHERE symbol = ? ORDER BY updated_at DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _kronos_path(c, symbol: str) -> tuple[int | None, float | None]:
    ""
    as_of = c.execute(
        "SELECT MAX(as_of_date) FROM mm_kronos_paths WHERE symbol = ?",
        (symbol,),
    ).fetchone()[0]
    if not as_of:
        return None, None
    flag = c.execute(
        "SELECT quality_flag FROM mm_kronos_signals "
        "WHERE symbol = ? AND date = ? ORDER BY created_at DESC LIMIT 1",
        (symbol, as_of),
    ).fetchone()
    if flag and flag[0] == "degenerate":
        return None, None
    rows = c.execute(
        "SELECT step, close FROM mm_kronos_paths "
        "WHERE symbol = ? AND as_of_date = ? ORDER BY step",
        (symbol, as_of),
    ).fetchall()
    if not rows:
        return None, None
    anchor = c.execute(
        "SELECT close FROM mm_bars WHERE symbol = ? AND date = ?",
        (symbol, as_of),
    ).fetchone()
    steps = [(s, cl) for s, cl in rows]
    if anchor and anchor[0] is not None:
        steps.insert(0, (0, anchor[0]))
    peak_step, peak_price = max(steps, key=lambda t: (t[1] if t[1] is not None else float("-inf")))
    return int(peak_step), float(peak_price) if peak_price is not None else None


def _kronos_5d(c, symbol: str) -> float | None:
    row = c.execute(
        "SELECT predicted_5d_return FROM mm_kronos_signals "
        "WHERE symbol = ? AND COALESCE(quality_flag, 'ok') = 'ok' "
        "ORDER BY date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return row[0] if row else None


def _momentum_flags(ind: list) -> tuple[int, int, str]:
    ""
    if len(ind) < 3:
        return 0, 0, "flat"
    (r2, h2), (r1, h1), (r0, h0) = [(r["rsi14"], r["macd_hist"]) for r in ind[-3:]]
    rsi_roll = int(
        r2 is not None and r1 is not None and r0 is not None
        and r2 > 65 and r0 < r1 < r2
    )
    hist_shrink = int(
        h2 is not None and h1 is not None and h0 is not None
        and h2 > 0 and h0 < h1 < h2
    )
    if rsi_roll or hist_shrink:
        state = "decelerating"
    elif (
        h0 is not None and h1 is not None and h2 is not None
        and h0 > h1 > h2 and r0 is not None and r1 is not None and r0 > r1
    ):
        state = "accelerating"
    else:
        state = "flat"
    return rsi_roll, hist_shrink, state


def _suggest(
    exit_type: str | None,
    peak_step: int | None,
    state: str,
    rsi_roll: int,
    hist_shrink: int,
    hwm: float | None,
    entry: float | None,
) -> tuple[str, str | None]:
    if exit_type == "operator_manual":
        return "operator_managed", "manually managed exit; no stop advice."
    if peak_step is not None and peak_step <= 1 and state == "decelerating":
        return "take_profit", None
    if rsi_roll and hist_shrink:
        return "tighten_trail", TIGHTEN_CAVEAT
    if (
        hwm is not None and entry is not None
        and hwm <= entry * 1.01 and state == "decelerating"
    ):
        return "exit", None
    return "hold", None


def refresh(db_path: Path | str | None = None) -> dict:
    ""
    path = resolve_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    out = {"rows": 0, "symbols": [], "as_of_date": None}

    with connect(path) as c:
        c.row_factory = __import__("sqlite3").Row
        from macmini.data.schema import MM_EXIT_SIGNALS
        c.execute(MM_EXIT_SIGNALS)
        c.commit()
        positions = c.execute(
            "SELECT symbol, qty, avg_entry_price FROM mm_alpaca_positions "
            "WHERE side = 'long' AND qty > 0"
        ).fetchall()

        for pos in positions:
            sym = pos["symbol"]
            entry_date = _entry_date(c, sym)
            bars = c.execute(
                "SELECT date, high, low, close FROM mm_bars "
                "WHERE symbol = ? ORDER BY date DESC LIMIT 90",
                (sym,),
            ).fetchall()[::-1]
            if not bars:
                log.warning("%s: no bars; skipping", sym)
                continue
            as_of = bars[-1]["date"]

            since_entry = [b for b in bars if entry_date and b["date"] >= entry_date]
            hwm = max((b["high"] for b in since_entry), default=None)
            anchor_bars = (since_entry or bars)[-CHANDELIER_LOOKBACK:]
            high20 = max(b["high"] for b in anchor_bars)

            ind = c.execute(
                "SELECT rsi14, macd_hist, atr14, adx14 FROM mm_indicators_daily "
                "WHERE symbol = ? ORDER BY date DESC LIMIT 3",
                (sym,),
            ).fetchall()[::-1]
            latest = ind[-1] if ind else None
            atr14 = latest["atr14"] if latest else None
            close = bars[-1]["close"]

            exit_type, ledger_trail = _ledger_exit_prefs(c, sym)
            broker = c.execute(
                "SELECT stop_price FROM mm_alpaca_orders "
                "WHERE symbol = ? AND order_type = 'trailing_stop' "
                "AND side = 'sell' AND status IN ('new','accepted','held') "
                "ORDER BY updated_at DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if broker and broker[0]:
                trail_stop = float(broker[0])
                trail_pct = (100.0 * (1.0 - trail_stop / hwm)) if hwm else None
                trail_source = "broker"
            else:
                if ledger_trail:
                    trail_pct = float(ledger_trail)
                    trail_source = "ledger"
                elif atr14 and close:
                    trail_pct = 100.0 * min(
                        max(STOP_ATR_MULT * atr14 / close, STOP_FRAC_MIN),
                        STOP_FRAC_MAX,
                    )
                    trail_source = "atr_model"
                else:
                    trail_pct = 8.0
                    trail_source = "atr_model"
                trail_stop = hwm * (1.0 - trail_pct / 100.0) if hwm else None
            chandelier = (high20 - CHANDELIER_MULT * atr14) if atr14 else None

            peak_step, peak_price = _kronos_path(c, sym)
            rsi_roll, hist_shrink, state = _momentum_flags(ind)
            roc5 = 0
            if len(bars) >= 11:
                roc_now = bars[-1]["close"] / bars[-6]["close"] - 1.0
                roc_prev = bars[-2]["close"] / bars[-7]["close"] - 1.0
                roc5 = int(roc_now < 0 <= roc_prev)

            action, note = _suggest(
                exit_type, peak_step, state, rsi_roll, hist_shrink,
                hwm, pos["avg_entry_price"],
            )

            c.execute(
                """INSERT OR REPLACE INTO mm_exit_signals
                   (as_of_date, symbol, entry_date, avg_entry_price,
                    hwm_price, trail_stop, trail_pct, trail_source,
                    chandelier, kronos_path_peak_step, kronos_path_peak_price,
                    kronos_5d, rsi14, rsi_rollover, macd_hist_shrink,
                    roc5_cross_zero, adx14, momentum_state, suggested_action,
                    note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (as_of, sym, entry_date, pos["avg_entry_price"],
                 hwm, trail_stop, trail_pct, trail_source, chandelier,
                 peak_step, peak_price, _kronos_5d(c, sym),
                 latest["rsi14"] if latest else None, rsi_roll, hist_shrink,
                 roc5, latest["adx14"] if latest else None, state, action,
                 note, now),
            )
            c.commit()
            out["rows"] += 1
            out["symbols"].append(sym)
            out["as_of_date"] = as_of
            log.info("%s as_of=%s hwm=%s trail=%.4g pct=%.1f action=%s",
                     sym, as_of, hwm, trail_stop or 0, trail_pct, action)

    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(
        description="Compute advisory per-position exit intelligence.")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    print(json.dumps(refresh(args.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
