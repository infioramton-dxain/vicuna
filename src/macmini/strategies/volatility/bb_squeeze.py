""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, bollinger, sma
from macmini.strategies.base import Signal, Strategy, register

ATR_MULT = 3.0
SQUEEZE_LOOKBACK = 126
SQUEEZE_DAYS = 5


@register
class BollingerSqueeze(Strategy):
    id = 25
    name = "Bollinger Squeeze + Direction"
    category = "volatility"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < SQUEEZE_LOOKBACK + 30:
            return []
        close = bars["close"]
        upper, mid, _, bb_w = bollinger(close, 20, 2.0)
        bb_w_min = bb_w.rolling(SQUEEZE_LOOKBACK).min()
        a = atr(bars["high"], bars["low"], close, 14)
        in_squeeze = (bb_w <= bb_w_min * 1.05).rolling(SQUEEZE_DAYS).min().fillna(0).astype(bool)
        breakout = close > upper
        entry = in_squeeze.shift(1, fill_value=False) & breakout
        below_mid = close < mid

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.6, reason="Bollinger squeeze + upper break",
                                       entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
                    open_idx = i
            else:
                if bool(below_mid.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5, reason="close < SMA20"))
                    open_idx = None
        return sigs
