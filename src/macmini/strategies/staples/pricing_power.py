""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import rsi, sma
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates, yoy_growth

MIN_REV_YOY = 0.03
RSI_ENTRY = 45.0
RSI_EXIT = 60.0
HOLD_DAYS = 10


@register
class StaplesPricingPower(Strategy):
    id = 61
    name = "Inflation Pricing Power (Margin Trend)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 120:
            return []
        rev_yoy = yoy_growth(fundamentals, "revenue")
        eps_yoy = yoy_growth(fundamentals, "eps")
        if rev_yoy is None or eps_yoy is None:
            return []
        if rev_yoy < MIN_REV_YOY or eps_yoy < rev_yoy:
            return []

        close = bars["close"]
        s100 = sma(close, 100)
        r14 = rsi(close, 14)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(100, len(bars)):
            if open_idx is None:
                if close.iloc[i] > s100.iloc[i] and r14.iloc[i] < RSI_ENTRY:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.6,
                        reason=f"EPS {eps_yoy:+.0%} >= rev {rev_yoy:+.0%} YoY, pullback",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if r14.iloc[i] > RSI_EXIT or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="RSI recovered" if r14.iloc[i] > RSI_EXIT else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
