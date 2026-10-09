""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ichimoku
from macmini.strategies.base import Signal, Strategy, register


@register
class IchimokuCloud(Strategy):
    id = 5
    name = "Ichimoku Cloud Breakout"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 60:
            return []
        close = bars["close"]
        tenkan, kijun, _, senkou_b = ichimoku(bars["high"], bars["low"], close)
        long_setup = (close > senkou_b) & (tenkan > kijun)
        prev_setup = long_setup.shift(1, fill_value=False)
        entry = long_setup & (~prev_setup)
        below_kijun = close < kijun
        tenkan_under = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        exit_sig = below_kijun | tenkan_under

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if entry.iloc[i]:
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55, reason="close above cloud + Tenkan>Kijun",
                                   entry_price=px, stop_price=float(kijun.iloc[i])))
            elif exit_sig.iloc[i] and prev_setup.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.5, reason="cloud structure broke"))
        return sigs
