""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma, zscore
from macmini.strategies.base import Signal, Strategy, register

Z_FLOOR = 3.0
TIME_STOP = 3


@register
class VolumeSpike(Strategy):
    id = 19
    name = "Volume Z-Score Spike + Price Up"
    category = "breakout"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 70:
            return []
        close = bars["close"]
        z = zscore(bars["volume"], 50)
        s50 = sma(close, 50)
        s10 = sma(close, 10)
        entry = (z > Z_FLOOR) & (close > bars["open"]) & (close > s50)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55, reason=f"volume z={float(z.iloc[i]):.1f}",
                                       entry_price=px))
                    open_idx = i
            else:
                if float(close.iloc[i]) < float(s10.iloc[i]) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="close < SMA10" if float(close.iloc[i]) < float(s10.iloc[i]) else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
