""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import bollinger, sma
from macmini.strategies.base import Signal, Strategy, register

TIME_STOP = 3


@register
class BollingerReversion(Strategy):
    id = 9
    name = "Bollinger Band Reversion (2σ pullback in uptrend)"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 220:
            return []
        close = bars["close"]
        _, mid, lower, _ = bollinger(close, 20, 2.0)
        s200 = sma(close, 200)
        entry = (close < lower) & (close > s200)
        exit_above_mid = close >= mid

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55, reason="close below lower BB in uptrend",
                                       entry_price=px, target_price=float(mid.iloc[i])))
                    open_idx = i
            else:
                if bool(exit_above_mid.iloc[i]) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="reverted to SMA20" if exit_above_mid.iloc[i] else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
