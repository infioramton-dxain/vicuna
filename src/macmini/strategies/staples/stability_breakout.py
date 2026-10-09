""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import donchian, sma
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates, yoy_pairs

YOY_LO, YOY_HI = -0.02, 0.15
HOLD_DAYS = 10


@register
class StaplesStabilityBreakout(Strategy):
    id = 60
    name = "Shelf-Stability Breakout (Stable YoY + Donchian)"
    category = "breakout"
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
        pairs = yoy_pairs(fundamentals, "revenue", n_pairs=4)
        if len(pairs) < 4 or not all(YOY_LO <= g <= YOY_HI for g in pairs):
            return []

        close = bars["close"]
        s20 = sma(close, 20)
        s200 = sma(close, 200)
        upper, _ = donchian(bars["high"], bars["low"], 20)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(200, len(bars)):
            if open_idx is None:
                if close.iloc[i] > s200.iloc[i] and bars["high"].iloc[i] >= upper.iloc[i - 1]:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.6,
                        reason=f"stable rev YoY {pairs[-1]:+.1%}, 20d Donchian high above SMA200",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if close.iloc[i] < s20.iloc[i] or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="closed below SMA20" if close.iloc[i] < s20.iloc[i] else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
