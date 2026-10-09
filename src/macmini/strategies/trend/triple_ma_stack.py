""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ema
from macmini.strategies.base import Signal, Strategy, register

FAST, MID, SLOW = 5, 20, 50
UNSTACKED_LOOKBACK = 10


@register
class TripleMaStack(Strategy):
    id = 2
    name = "Triple Moving Average Stack (5/20/50)"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < SLOW + UNSTACKED_LOOKBACK + 2:
            return []
        close = bars["close"]
        e_fast, e_mid, e_slow = ema(close, FAST), ema(close, MID), ema(close, SLOW)
        stacked = (e_fast > e_mid) & (e_mid > e_slow)
        prior_unstacked = (~stacked.shift(1).rolling(UNSTACKED_LOOKBACK).max().fillna(False).astype(bool))
        fresh = stacked & prior_unstacked & (close > e_fast)
        stack_break = (~stacked) & stacked.shift(1, fill_value=False)
        below_mid = close < e_mid

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if fresh.iloc[i]:
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55, reason="fresh 5/20/50 EMA stack",
                                   entry_price=px, stop_price=float(e_mid.iloc[i])))
            elif stack_break.iloc[i] or below_mid.iloc[i]:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.5,
                                   reason="stack broke" if stack_break.iloc[i] else f"close < EMA{MID}"))
        return sigs
