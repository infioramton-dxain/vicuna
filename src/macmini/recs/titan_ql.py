""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from macmini.config import resolve_db
from macmini.data._db import connect

log = logging.getLogger(__name__)

STRATEGY_ID = "TITAN-1"
APPROVED_SECTORS = ("Materials", "Mining", "Defense", "Computers", "Software")

REV_GROWTH_MIN_PCT = 15.0
MARGIN_MIN = 0.15
MARGIN_ALT = 0.10
DE_MAX = 0.5
CAP_PCTL = 0.90
KRONOS_5D_MIN = 0.03
KRONOS_30D_MIN = 0.0
KRONOS_MAX_STALE_DAYS = 5
DIP_52W_FACTOR = 1.15
GAP_DROP = 0.05
GAP_BARS = 3
RSI_MAX = 45.0
SMA_KEY = "sma50"
PRE_EARNINGS_CAL_DAYS = 7
STOP_ATR_MULT = 2.0
STOP_MIN_FRAC = 0.08


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sector_cap_floor(c) -> dict[str, float]:
    ""
    floors: dict[str, float] = {}
    for sector in APPROVED_SECTORS:
        caps = sorted(v for (v,) in c.execute(
            "SELECT market_cap_usd FROM mm_sector_map "
            "WHERE sector = ? AND market_cap_usd IS NOT NULL", (sector,)))
        if caps:
            floors[sector] = caps[int(CAP_PCTL * (len(caps) - 1))]
    return floors


def _quality(c, symbol: str) -> dict | None:
    ""
    growth = c.execute(
        "SELECT rev_chg_ttm FROM mm_fundamentals_growth "
        "WHERE symbol = ? ORDER BY as_of_date DESC LIMIT 1", (symbol,)).fetchone()
    fq = c.execute(
        "SELECT revenue, earnings, free_cash_flow, total_debt, total_equity "
        "FROM mm_fundamentals_quarterly WHERE symbol = ? "
        "ORDER BY period_end DESC LIMIT 1", (symbol,)).fetchone()
    if not growth or growth[0] is None or not fq:
        return None
    rev_growth = float(growth[0])
    revenue, earnings, fcf, debt, equity = fq
    if not revenue or earnings is None or fcf is None or debt is None or not equity:
        return None
    revenue, earnings, fcf = float(revenue), float(earnings), float(fcf)
    debt, equity = float(debt), float(equity)
    if equity <= 0 or revenue <= 0:
        return None
    margin = earnings / revenue
    de = debt / equity
    ok = (rev_growth >= REV_GROWTH_MIN_PCT
          and (margin >= MARGIN_MIN or (margin >= MARGIN_ALT and fcf > 0))
          and fcf > 0 and de <= DE_MAX)
    if not ok:
        return None
    return {"rev_growth_ttm": rev_growth, "net_margin": margin,
            "free_cash_flow": fcf, "debt_to_equity": de}


def _analyst(c, symbol: str) -> dict | None:
    row = c.execute(
        "SELECT rating_buy, rating_hold, rating_sell FROM mm_analyst_estimates "
        "WHERE symbol = ? ORDER BY as_of_date DESC LIMIT 1", (symbol,)).fetchone()
    if not row or any(v is None for v in row):
        return None
    buy, hold, sell = (int(v) for v in row)
    if sell != 0 or buy < 2 * hold:
        return None
    return {"rating_buy": buy, "rating_hold": hold, "rating_sell": sell}


def _kronos(c, symbol: str, as_of: str) -> dict | None:
    row = c.execute(
        "SELECT date, predicted_5d_return, predicted_30d_return "
        "FROM mm_kronos_signals WHERE symbol = ? "
        "AND COALESCE(quality_flag, 'ok') = 'ok' "
        "ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
    if not row or row[1] is None or row[2] is None:
        return None
    date, r5, r30 = row[0], float(row[1]), float(row[2])
    stale = (datetime.fromisoformat(as_of) - datetime.fromisoformat(date)).days
    if stale > KRONOS_MAX_STALE_DAYS:
        return None
    if r5 < KRONOS_5D_MIN or r30 < KRONOS_30D_MIN:
        return None
    return {"kronos_5d": r5, "kronos_30d": r30}


def _dip_triggers(c, symbol: str, bars: list) -> list[str]:
    ""
    triggers: list[str] = []
    close = bars[-1][1]
    low_52w = min(b[2] for b in bars)
    if close <= DIP_52W_FACTOR * low_52w:
        triggers.append("near_52w_low")

    recent_dates = [b[0] for b in bars[-(GAP_BARS + 1):]]
    events = c.execute(
        "SELECT report_date FROM mm_earnings_events WHERE symbol = ? "
        "AND eps_surprise_pct > 0 AND report_date >= ? AND report_date <= ?",
        (symbol, recent_dates[0], bars[-1][0])).fetchall()
    for (rd,) in events:
        pre = [b[1] for b in bars if b[0] < rd]
        if pre and close <= (1.0 - GAP_DROP) * pre[-1]:
            triggers.append("post_earnings_gap")
            break

    ind = c.execute(
        f"SELECT rsi14, {SMA_KEY} FROM mm_indicators_daily "
        "WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
    if (ind and ind[0] is not None and ind[1] is not None
            and close < float(ind[1]) and float(ind[0]) < RSI_MAX):
        triggers.append("sma50_rsi")
    return triggers


def _blocked_by_earnings(c, symbol: str, as_of: str, blackout: set[str]) -> bool:
    if symbol in blackout:
        return True
    horizon = (datetime.fromisoformat(as_of)
               + timedelta(days=PRE_EARNINGS_CAL_DAYS)).strftime("%Y-%m-%d")
    nxt = c.execute(
        "SELECT 1 FROM mm_earnings_events WHERE symbol = ? "
        "AND report_date > ? AND report_date <= ? LIMIT 1",
        (symbol, as_of, horizon)).fetchone()
    return nxt is not None


def scan(db_path: Path | str | None = None) -> dict:
    ""
    path = resolve_db(db_path)
    now = _now_iso()
    out = {"as_of_date": None, "scanned": 0, "candidates": []}

    with connect(path) as c:
        from macmini.data.schema import MM_QUALITY_CANDIDATES
        c.execute(MM_QUALITY_CANDIDATES)
        c.commit()

        floors = _sector_cap_floor(c)
        regime = c.execute(
            "SELECT earnings_blackout_symbols FROM mm_regime "
            "ORDER BY as_of_date DESC LIMIT 1").fetchone()
        blackout = set((regime[0] or "").split(",")) if regime else set()
        held = {s for (s,) in c.execute(
            "SELECT symbol FROM mm_alpaca_positions WHERE qty > 0")}
        picks = {s for (s,) in c.execute("SELECT symbol FROM mm_active_picks")}

        leaders = c.execute(
            "SELECT symbol, sector, market_cap_usd FROM mm_sector_map "
            f"WHERE sector IN ({','.join('?' * len(APPROVED_SECTORS))}) "
            "AND market_cap_usd IS NOT NULL", APPROVED_SECTORS).fetchall()

        for symbol, sector, cap in leaders:
            if cap < floors.get(sector, float("inf")):
                continue
            out["scanned"] += 1
            bars = c.execute(
                "SELECT date, close, low FROM mm_bars WHERE symbol = ? "
                "ORDER BY date DESC LIMIT 252", (symbol,)).fetchall()[::-1]
            if len(bars) < 60:
                continue
            as_of = bars[-1][0]
            out["as_of_date"] = max(out["as_of_date"] or as_of, as_of)

            quality = _quality(c, symbol)
            if not quality:
                continue
            analyst = _analyst(c, symbol)
            if not analyst:
                continue
            kronos = _kronos(c, symbol, as_of)
            if not kronos:
                continue
            triggers = _dip_triggers(c, symbol, bars)
            if not triggers:
                continue
            if _blocked_by_earnings(c, symbol, as_of, blackout):
                continue

            close = float(bars[-1][1])
            atr = c.execute(
                "SELECT atr14 FROM mm_indicators_daily WHERE symbol = ? "
                "ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
            stop_frac = max(STOP_ATR_MULT * float(atr[0]) / close, STOP_MIN_FRAC) \
                if atr and atr[0] else STOP_MIN_FRAC
            c.execute(
                """INSERT OR REPLACE INTO mm_quality_candidates
                   (strategy_id, as_of_date, symbol, sector, close,
                    market_cap_usd, rev_growth_ttm, net_margin, free_cash_flow,
                    debt_to_equity, rating_buy, rating_hold, rating_sell,
                    kronos_5d, kronos_30d, dip_triggers, suggested_stop,
                    held, in_picks, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (STRATEGY_ID, as_of, symbol, sector, close, cap,
                 quality["rev_growth_ttm"], quality["net_margin"],
                 quality["free_cash_flow"], quality["debt_to_equity"],
                 analyst["rating_buy"], analyst["rating_hold"],
                 analyst["rating_sell"], kronos["kronos_5d"],
                 kronos["kronos_30d"], ",".join(triggers),
                 close * (1.0 - stop_frac), int(symbol in held),
                 int(symbol in picks), now))
            c.commit()
            out["candidates"].append(symbol)
            log.info("%s: candidate (triggers=%s kronos_5d=%.3f)",
                     symbol, ",".join(triggers), kronos["kronos_5d"])

    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="TITAN-1 quality-leader candidate scan.")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    print(json.dumps(scan(args.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
