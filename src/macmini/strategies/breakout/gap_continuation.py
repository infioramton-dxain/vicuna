""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

GAP_PCT = 0.02
VOL_MULT = 1.5
TIME_STOP = 5


@register
class GapContinuation(Strategy):
    id = 18
    name = "Gap Continuation"
    category = "breakout"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 60:
            return []
        open_, close = bars["open"], bars["close"]
        prev_close = close.shift(1)
        vol_avg = bars["volume"].rolling(50).mean()
        entry = (open_ > prev_close * (1 + GAP_PCT)) & (close > open_) & (bars["volume"] > VOL_MULT * vol_avg)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        entry_open: float | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55, reason="gap up + volume + green close",
                                       entry_price=px, stop_price=float(open_.iloc[i])))
                    open_idx = i
                    entry_open = float(open_.iloc[i])
            else:
                if float(close.iloc[i]) < (entry_open or 0) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="filled gap" if float(close.iloc[i]) < (entry_open or 0) else f"{TIME_STOP}d stop"))
                    open_idx = None
                    entry_open = None
        return sigs
