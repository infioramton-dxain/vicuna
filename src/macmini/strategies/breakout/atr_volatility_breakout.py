""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, sma
from macmini.strategies.base import Signal, Strategy, register

ATR_MULT = 3.0
TIME_STOP = 5


@register
class AtrVolatilityBreakout(Strategy):
    id = 17
    name = "Volatility Breakout (Close > Prior High + ATR)"
    category = "breakout"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 70:
            return []
        close = bars["close"]
        a = atr(bars["high"], bars["low"], close, 14)
        s50 = sma(close, 50)
        prev_high = bars["high"].shift(1)
        entry = (close > prev_high + a) & (close > s50)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.6, reason="close > prior high + ATR",
                                       entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
                    open_idx = i
            else:
                if i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason=f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
