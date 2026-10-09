""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register


@register
class FiveDayPullback(Strategy):
    id = 11
    name = "5-Day Pullback in Uptrend (3-of-5 down)"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 220:
            return []
        close = bars["close"]
        s50, s200 = sma(close, 50), sma(close, 200)
        down = (close < close.shift(1)).astype(int)
        downs_5d = down.rolling(5).sum()
        entry = (close > s50) & (s50 > s200) & (downs_5d >= 3)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        entry_close: float | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.5, reason="3-of-5 down in uptrend",
                                       entry_price=px, stop_price=float(s50.iloc[i])))
                    open_idx = i
                    entry_close = px
            else:
                px = float(close.iloc[i])
                if px > (entry_close or 0) and close.iloc[i] > close.iloc[i - 1]:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason="up day close above entry"))
                    open_idx = None
                    entry_close = None
                elif px < float(s50.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason="close below SMA50 stop"))
                    open_idx = None
                    entry_close = None
        return sigs
