""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import rsi, sma
from macmini.strategies.base import Signal, Strategy, register

TIME_STOP = 5


@register
class Rsi2Connors(Strategy):
    id = 10
    name = "RSI(2) Connors Mean Reversion"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 220:
            return []
        close = bars["close"]
        r2 = rsi(close, 2)
        s5, s200 = sma(close, 5), sma(close, 200)
        entry = (r2 < 5) & (close > s200) & (close < s5)
        exit_sig = close > s5

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55, reason=f"RSI(2)={float(r2.iloc[i]):.1f} below 5",
                                       entry_price=px))
                    open_idx = i
            else:
                if bool(exit_sig.iloc[i]) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="close > SMA5" if exit_sig.iloc[i] else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
