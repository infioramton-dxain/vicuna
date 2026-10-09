""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from agent.alpaca_client import AlpacaClient, BracketOrder, StubAlpacaClient
from agent.config import DRY_RUN, LOG_DIR, MAX_PICKS_TO_SUBMIT, SHIPPED_DB
from agent.db import open_shipped_ro, open_trades
from agent.llm_router import rank_picks
from agent.risk_overlay import filter_and_size

log = logging.getLogger(__name__)


def _load_picks_and_regime(as_of: str | None = None) -> tuple[list[dict], dict | None, str, dict]:
    ""
    with open_shipped_ro() as c:
        meta_rows = c.execute("SELECT key, value FROM mm_export_metadata").fetchall()
        meta = dict(meta_rows)
        as_of_eff = as_of or meta.get("as_of_date") or date.today().isoformat()
        picks = [
            {
                "rank": r[0], "strategy_id": r[1], "strategy_name": r[2],
                "symbol": r[3], "sector": r[4], "archetype": r[5],
                "deflated_sharpe": r[6], "profit_factor": r[7], "max_dd": r[8],
                "suggested_size_pct": r[9], "suggested_stop_loss_pct": r[10],
                "suggested_target_pct": r[11], "rationale": r[12],
            }
            for r in c.execute(
                "SELECT rank, strategy_id, strategy_name, symbol, sector, archetype, "
                "deflated_sharpe, profit_factor, max_dd, suggested_size_pct, "
                "suggested_stop_loss_pct, suggested_target_pct, rationale "
                "FROM mm_active_picks ORDER BY rank"
            ).fetchall()
        ]
        regime_row = c.execute(
            "SELECT as_of_date, spy_50d_above_200d, vix_close, vix_regime, "
            "fomc_within_3d, earnings_blackout_symbols FROM mm_regime "
            "WHERE as_of_date = ?", (as_of_eff,)
        ).fetchone()
    regime = None
    if regime_row:
        regime = {
            "as_of_date": regime_row[0], "spy_50d_above_200d": regime_row[1],
            "vix_close": regime_row[2], "vix_regime": regime_row[3],
            "fomc_within_3d": regime_row[4],
            "earnings_blackout_symbols": regime_row[5] or "",
        }
    return picks, regime, as_of_eff, meta


def _sentiment_context(symbols: list[str], as_of_str: str) -> dict[str, list]:
    if not symbols:
        return {}
    with open_shipped_ro() as c:
        out: dict[str, list] = {}
        for sym in symbols:
            rows = c.execute(
                """SELECT substr(window_start, 1, 10) AS date,
                          AVG(sentiment) AS s, MAX(materiality) AS m
                   FROM sentiment
                   WHERE symbol = ? AND status='ok'
                     AND window_start >= date(?, '-30 days')
                   GROUP BY date ORDER BY date""",
                (sym, as_of_str),
            ).fetchall()
            out[sym] = [{"date": r[0], "sentiment_mean": r[1], "materiality_max": r[2]}
                        for r in rows]
    return out


def _log_decision(
    trades_conn, as_of_str: str, snapshot_path: Path,
    picks_considered: int, picks_taken: int,
    llm_raw: str, regime_read: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    trades_conn.execute(
        """INSERT INTO agent_decisions
           (as_of_date, snapshot_path, picks_considered, picks_taken,
            llm_response_json, regime_read, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (as_of_str, str(snapshot_path), picks_considered, picks_taken,
         llm_raw, regime_read, now),
    )
    trades_conn.commit()


def _log_trade(trades_conn, sized, broker_id: str | None, status: str, status_reason: str, llm_rationale: str, as_of_str: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    trades_conn.execute(
        """INSERT INTO trades
           (as_of_date, symbol, strategy_id, side, qty, entry_price, stop_price,
            target_price, broker_order_id, status, status_reason, llm_rationale,
            created_at, updated_at)
           VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (as_of_str, sized.symbol, sized.strategy_id, sized.qty,
         sized.entry_price, sized.stop_price, sized.target_price,
         broker_id, status, status_reason, llm_rationale, now, now),
    )
    trades_conn.commit()


def run_once(dry_run: bool | None = None, as_of: str | None = None) -> dict:
    if dry_run is None:
        dry_run = DRY_RUN
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    picks, regime, as_of_str, meta = _load_picks_and_regime(as_of)
    if not picks:
        log.warning("mm_active_picks is empty for as_of=%s — nothing to do", as_of_str)
        return {"as_of": as_of_str, "picks": 0, "submitted": 0, "dry_run": dry_run}

    picks = picks[:MAX_PICKS_TO_SUBMIT]
    pick_meta = {p["symbol"]: p for p in picks}
    sentiment = _sentiment_context([p["symbol"] for p in picks], as_of_str)

    alpaca = StubAlpacaClient() if dry_run else AlpacaClient()
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    quotes = alpaca.get_quotes([p["symbol"] for p in picks])

    ranking = rank_picks(picks, quotes, regime, positions, sentiment)

    earnings_blackout = set(
        (regime or {}).get("earnings_blackout_symbols", "").split(",")
    ) - {""}

    sized = filter_and_size(
        ranked_picks=ranking.ranked_picks,
        pick_meta=pick_meta,
        overrides=ranking.overrides,
        account=account,
        positions=positions,
        quotes=quotes,
        regime=regime,
        earnings_blackout=earnings_blackout,
    )

    submitted = 0
    with open_trades() as tdb:
        _log_decision(tdb, as_of_str, SHIPPED_DB, len(picks), len(sized),
                      ranking.raw_response, ranking.regime_read)
        for order in sized:
            if dry_run:
                broker_id, status, reason = "DRYRUN", "dry_run", ""
            else:
                try:
                    broker_id = alpaca.submit_bracket(BracketOrder(
                        symbol=order.symbol, qty=order.qty, side="buy",
                        entry_price=order.entry_price, stop_price=order.stop_price,
                        target_price=order.target_price,
                    ))
                    status, reason = "submitted", ""
                except Exception as exc:
                    broker_id, status, reason = None, "rejected", str(exc)
            _log_trade(tdb, order, broker_id, status, reason, order.rationale, as_of_str)
            log.info("%s × %d @ $%.2f → broker_id=%s status=%s",
                     order.symbol, order.qty, order.entry_price, broker_id, status)
            if status == "submitted":
                submitted += 1

    return {
        "as_of": as_of_str, "picks": len(picks), "sized": len(sized),
        "submitted": submitted, "dry_run": dry_run,
        "regime_read": ranking.regime_read,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Mac mini trading agent loop")
    p.add_argument("--dry-run", action="store_true", help="don't hit Alpaca; print + log only")
    p.add_argument("--as-of", default=None, help="override as_of_date for testing")
    args = p.parse_args(argv)
    out = run_once(dry_run=args.dry_run, as_of=args.as_of)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
