""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import rsi, sma
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import (
    STAPLES, bar_dates, quarterly_history, yoy_growth)

MIN_QUALITY = 4
RSI2_ENTRY = 10.0
HOLD_DAYS = 5


def _quality_score(fundamentals: pd.DataFrame, hist4: pd.DataFrame) -> int:
    last = hist4.iloc[-1]
    score = 0
    score += int(hist4["operating_cash_flow"].astype(float).fillna(0).sum() > 0)
    score += int(hist4["free_cash_flow"].astype(float).fillna(0).sum() > 0)
    eps = last.get("eps")
    score += int(eps is not None and not pd.isna(eps) and float(eps) > 0)
    equity = float(last.get("total_equity") or 0)
    debt = float(last.get("total_debt") or 0)
    score += int(equity > 0 and debt / equity < 1.0)
    rev_yoy = yoy_growth(fundamentals, "revenue")
    score += int(rev_yoy is not None and rev_yoy > 0)
    return score


@register
class StaplesQualityDip(Strategy):
    id = 57
    name = "Staples Quality Dip (Quality-5 + RSI2)"
    category = "mean_reversion"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 220:
            return []
        hist4 = quarterly_history(fundamentals, 4)
        if hist4 is None:
            return []
        quality = _quality_score(fundamentals, hist4)
        if quality < MIN_QUALITY:
            return []

        close = bars["close"]
        s20 = sma(close, 20)
        s200 = sma(close, 200)
        r2 = rsi(close, 2)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(200, len(bars)):
            if open_idx is None:
                if close.iloc[i] > s200.iloc[i] and r2.iloc[i] < RSI2_ENTRY:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.6,
                        reason=f"quality {quality}/5, RSI2 {r2.iloc[i]:.0f} above SMA200",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if close.iloc[i] >= s20.iloc[i] or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="SMA20 tag" if close.iloc[i] >= s20.iloc[i] else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
