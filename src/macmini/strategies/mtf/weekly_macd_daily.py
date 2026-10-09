""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import macd
from macmini.strategies.base import Signal, Strategy, register


@register
class WeeklyMacdDaily(Strategy):
    id = 29
    name = "Higher-TF MACD + Daily MACD Trigger"
    category = "mtf"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 200:
            return []
        close = bars["close"]
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        d_line, d_sig, _ = macd(close)
        weekly = close.copy()
        weekly.index = dts
        wclose = weekly.resample("W-FRI").last()
        w_line, w_sig, _ = macd(wclose)
        weekly_bull = (w_line > w_sig).reindex(dts, method="ffill", fill_value=False).astype(bool)
        cross_up = (d_line > d_sig) & (d_line.shift(1) <= d_sig.shift(1))
        cross_dn = (d_line < d_sig) & (d_line.shift(1) >= d_sig.shift(1))
        entry = weekly_bull & cross_up

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                       conviction=0.55, reason="weekly MACD bull + daily cross up",
                                       entry_price=px))
                    open_idx = i
            else:
                if bool(cross_dn.iloc[i]) or not bool(weekly_bull.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                       conviction=0.5, reason="daily MACD cross down / weekly trend broke"))
                    open_idx = None
        return sigs
