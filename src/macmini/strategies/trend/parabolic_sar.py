""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ema, parabolic_sar
from macmini.strategies.base import Signal, Strategy, register


@register
class ParabolicSarEma200(Strategy):
    id = 8
    name = "Parabolic SAR + EMA200 Filter"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 220:
            return []
        close = bars["close"]
        psar = parabolic_sar(bars["high"], bars["low"], close)
        e200 = ema(close, 200)
        below = psar < close
        flip_up = below & (~below.shift(1, fill_value=False))
        flip_dn = (~below) & below.shift(1, fill_value=False)
        entry = flip_up & (close > e200)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if entry.iloc[i]:
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55, reason="SAR flipped bullish above EMA200",
                                   entry_price=px, stop_price=float(psar.iloc[i])))
            elif flip_dn.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.55, reason="SAR flipped bearish"))
        return sigs
