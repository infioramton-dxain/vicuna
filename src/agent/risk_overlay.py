""
from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.alpaca_client import AccountSnapshot, BracketOrder, Position, Quote
from agent.config import (
    EARNINGS_BLACKOUT_DAYS,
    MAX_CONCURRENT_POSITIONS,
    MAX_DAILY_DD_PCT,
    MAX_POSITION_NOTIONAL_PCT,
    MAX_POSITION_NOTIONAL_USD,
    MAX_RISK_PER_TRADE_PCT,
    MAX_SECTOR_CONCENTRATION,
    MIN_NOTIONAL,
)

log = logging.getLogger(__name__)

PEAD_STRATEGY_ID = 35


@dataclass
class SizedOrder:
    symbol: str
    strategy_id: int
    qty: int
    entry_price: float
    stop_price: float
    target_price: float
    sector: str
    rationale: str


def size_position(equity: float, quote: Quote, stop_price: float) -> int:
    ""
    risk_per_share = max(1e-6, quote.mid - stop_price)
    max_risk = equity * MAX_RISK_PER_TRADE_PCT
    qty = int(max_risk // risk_per_share)
    entry = quote.ask or quote.mid
    if entry > 0:
        cap = min(MAX_POSITION_NOTIONAL_USD,
                  MAX_POSITION_NOTIONAL_PCT * equity)
        qty = min(qty, int(cap // entry))
    return max(qty, 0)


def filter_and_size(
    ranked_picks: list[dict],
    pick_meta: dict[str, dict],
    overrides: dict | None,
    account: AccountSnapshot,
    positions: list[Position],
    quotes: dict[str, Quote],
    regime: dict | None,
    earnings_blackout: set[str] | None = None,
) -> list[SizedOrder]:
    ""
    overrides = overrides or {}
    earnings_blackout = earnings_blackout or set()
    skip = set(overrides.get("skip_today") or [])
    halve = set(overrides.get("halve_size") or [])

    open_symbols = {p.symbol for p in positions}
    sector_value = _sector_book(positions, pick_meta)
    total_book = sum(p.market_value for p in positions) + account.cash
    fomc_halve = bool((regime or {}).get("fomc_within_3d"))
    vix_high = (regime or {}).get("vix_regime") == "high"

    accepted: list[SizedOrder] = []
    for pick in ranked_picks:
        if len(accepted) + len(positions) >= MAX_CONCURRENT_POSITIONS:
            log.info("max concurrent positions reached; stopping")
            break
        symbol = pick["symbol"]
        if symbol in skip:
            log.info("skipping %s — LLM override", symbol); continue
        if symbol in open_symbols:
            log.info("skipping %s — already long", symbol); continue
        meta = pick_meta.get(symbol)
        if not meta:
            log.warning("skipping %s — missing pick_meta", symbol); continue
        strategy_id = int(pick.get("strategy_id") or meta["strategy_id"])
        if symbol in earnings_blackout and strategy_id != PEAD_STRATEGY_ID:
            log.info("skipping %s — earnings blackout", symbol); continue
        quote = quotes.get(symbol)
        if not quote or quote.mid <= 0:
            log.info("skipping %s — no live quote", symbol); continue
        entry = quote.ask or quote.mid
        stop_pct = float(meta.get("suggested_stop_loss_pct") or 0.08)
        target_pct = float(meta.get("suggested_target_pct") or 0.20)
        stop_price = round(entry * (1 - stop_pct), 2)
        target_price = round(entry * (1 + target_pct), 2)
        qty = size_position(account.equity, quote, stop_price)
        if symbol in halve or fomc_halve or vix_high:
            qty = qty // 2
        if qty < 1:
            log.info("skipping %s — qty < 1 after sizing", symbol); continue
        notional = qty * entry
        if entry < MIN_NOTIONAL:
            log.info("skipping %s — share price $%.2f below notional floor", symbol, entry); continue
        if notional > account.buying_power:
            log.info("skipping %s — notional $%.2f > buying power $%.2f",
                     symbol, notional, account.buying_power); continue
        sector = meta.get("sector") or "Other"
        sector_after = (sector_value.get(sector, 0.0) + notional) / max(total_book, 1.0)
        if sector_after > MAX_SECTOR_CONCENTRATION:
            log.info("skipping %s — sector %s would be %.0f%% > %.0f%% cap",
                     symbol, sector, sector_after * 100, MAX_SECTOR_CONCENTRATION * 100)
            continue
        accepted.append(SizedOrder(
            symbol=symbol, strategy_id=strategy_id, qty=qty,
            entry_price=entry, stop_price=stop_price, target_price=target_price,
            sector=sector, rationale=pick.get("rationale") or meta.get("rationale") or "",
        ))
        sector_value[sector] = sector_value.get(sector, 0.0) + notional

    return accepted


def _sector_book(positions: list[Position], pick_meta: dict[str, dict]) -> dict[str, float]:
    ""
    out: dict[str, float] = {}
    for p in positions:
        sector = (pick_meta.get(p.symbol) or {}).get("sector") or "Other"
        out[sector] = out.get(sector, 0.0) + p.market_value
    return out
