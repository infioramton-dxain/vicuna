""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import supertrend
from macmini.strategies.base import Signal, Strategy, register


@register
class SupertrendFlip(Strategy):
    id = 6
    name = "Supertrend Flip (ATR x3)"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 30:
            return []
        st_line, direction = supertrend(bars["high"], bars["low"], bars["close"], n=10, mult=3.0)
        flipped_up = (direction == 1) & (direction.shift(1) == -1)
        flipped_dn = (direction == -1) & (direction.shift(1) == 1)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if flipped_up.iloc[i]:
                px = float(bars["close"].iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55, reason="Supertrend flipped long",
                                   entry_price=px, stop_price=float(st_line.iloc[i])))
            elif flipped_dn.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.55, reason="Supertrend flipped short"))
        return sigs
