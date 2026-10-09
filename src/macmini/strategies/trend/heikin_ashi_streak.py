""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, ema, heikin_ashi
from macmini.strategies.base import Signal, Strategy, register

ATR_MULT = 2.0


@register
class HeikinAshiStreak(Strategy):
    id = 7
    name = "Heikin-Ashi Color Streak"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 30:
            return []
        ha = heikin_ashi(bars["open"], bars["high"], bars["low"], bars["close"])
        e20 = ema(bars["close"], 20)
        a = atr(bars["high"], bars["low"], bars["close"], 14)
        green = ha["close"] > ha["open"]
        three_green = green & green.shift(1) & green.shift(2)
        entry = three_green & (ha["close"] > e20)
        red = ~green

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if entry.iloc[i] and not (i > 0 and entry.iloc[i - 1]):
                px = float(bars["close"].iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55, reason="3 consecutive green HA bars",
                                   entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
            elif red.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.45, reason="HA flipped red"))
        return sigs
