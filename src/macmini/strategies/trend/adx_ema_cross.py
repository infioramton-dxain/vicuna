""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import adx, ema
from macmini.strategies.base import Signal, Strategy, register

FAST, SLOW = 12, 26
ADX_FLOOR = 25
ADX_N = 14


@register
class AdxEmaCross(Strategy):
    id = 4
    name = "ADX-Filtered EMA Crossover (12/26 + ADX>25)"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < SLOW + ADX_N + 2:
            return []
        close = bars["close"]
        e_fast, e_slow = ema(close, FAST), ema(close, SLOW)
        a = adx(bars["high"], bars["low"], close, ADX_N)

        prev_above = (e_fast.shift(1) > e_slow.shift(1)).fillna(False)
        now_above = (e_fast > e_slow).fillna(False)
        cross_up = (~prev_above) & now_above
        cross_dn = prev_above & (~now_above)
        adx_filter = (a > ADX_FLOOR) & (a > a.shift(1))

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if cross_up.iloc[i] and adx_filter.iloc[i]:
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.6, reason=f"EMA{FAST}>EMA{SLOW} with ADX {float(a.iloc[i]):.1f}",
                                   entry_price=px))
            elif cross_dn.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.55, reason=f"EMA{FAST} crossed below EMA{SLOW}"))
        return sigs
