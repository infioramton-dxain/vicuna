""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ibs, sma
from macmini.strategies.base import Signal, Strategy, register

IBS_FLOOR = 0.2
HOLD_DAYS = 2


@register
class IbsLow(Strategy):
    id = 14
    name = "Internal Bar Strength (IBS < 0.2 in uptrend)"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 20:
            return []
        close = bars["close"]
        i_val = ibs(bars["open"], bars["high"], bars["low"], close)
        s10 = sma(close, 10)
        entry = (i_val < IBS_FLOOR) & (close > s10)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.5, reason=f"IBS {float(i_val.iloc[i]):.2f} in uptrend",
                                       entry_price=px))
                    open_idx = i
            else:
                if i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason=f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
