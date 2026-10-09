""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, sma
from macmini.strategies.base import Signal, Strategy, register

NR_N = 7
ATR_MULT = 2.0
TIME_STOP = 3


@register
class Nr7Expansion(Strategy):
    id = 27
    name = "NR7 Narrowest Range Expansion"
    category = "volatility"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 80:
            return []
        close = bars["close"]
        rng = bars["high"] - bars["low"]
        nr_min = rng.rolling(NR_N).min()
        is_nr7 = rng == nr_min
        a = atr(bars["high"], bars["low"], close, 14)
        s50 = sma(close, 50)
        prior_nr7 = is_nr7.shift(1, fill_value=False)
        prior_high = bars["high"].shift(1)
        entry = prior_nr7 & (close > prior_high) & (close > s50)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55, reason="NR7 expansion + uptrend",
                                       entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
                    open_idx = i
            else:
                if i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason=f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
