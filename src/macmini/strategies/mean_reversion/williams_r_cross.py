""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma, williams_r
from macmini.strategies.base import Signal, Strategy, register

TIME_STOP = 5


@register
class WilliamsRCross(Strategy):
    id = 13
    name = "Williams %R Extreme Cross"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 70:
            return []
        close = bars["close"]
        w = williams_r(bars["high"], bars["low"], close, 14)
        s50 = sma(close, 50)
        crossed_up = (w > -80) & (w.shift(1) <= -80)
        entry = crossed_up & (close > s50)
        exit_sig = w > -20

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.5, reason="Williams %R reclaimed -80",
                                       entry_price=px))
                    open_idx = i
            else:
                if bool(exit_sig.iloc[i]) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="Williams %R > -20" if exit_sig.iloc[i] else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
