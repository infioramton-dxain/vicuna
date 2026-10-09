""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, donchian
from macmini.strategies.base import Signal, Strategy, register

ENTRY_N, EXIT_N = 20, 10
ATR_N = 20
ATR_MULT = 3.0


@register
class Donchian20(Strategy):
    id = 3
    name = "Donchian 20-Day Breakout (swing)"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < ENTRY_N + ATR_N + 2:
            return []
        close = bars["close"]
        high, low = bars["high"], bars["low"]
        entry_high, _ = donchian(high, low, ENTRY_N)
        _, exit_low = donchian(high, low, EXIT_N)
        a = atr(high, low, close, ATR_N)
        prev_entry = entry_high.shift(1)
        entry = close > prev_entry
        exit_break = close < exit_low.shift(1)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if entry.iloc[i]:
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.6, reason=f"close > {ENTRY_N}d high",
                                   entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
            elif exit_break.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.55, reason=f"close < {EXIT_N}d low"))
        return sigs
