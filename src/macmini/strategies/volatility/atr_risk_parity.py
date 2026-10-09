""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, donchian
from macmini.strategies.base import Signal, Strategy, register

ATR_MULT = 2.0


@register
class AtrRiskParityBreakout(Strategy):
    id = 26
    name = "ATR Breakout with Risk-Parity Sizing"
    category = "volatility"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 40:
            return []
        close = bars["close"]
        entry_high, _ = donchian(bars["high"], bars["low"], 20)
        _, exit_low = donchian(bars["high"], bars["low"], 10)
        a = atr(bars["high"], bars["low"], close, 14)
        entry = close > entry_high.shift(1)
        exit_break = close < exit_low.shift(1)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        for i in range(len(bars)):
            if bool(entry.iloc[i]):
                px = float(close.iloc[i])
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                   conviction=0.55,
                                   reason=f"20d break; size = risk$ / (2*ATR={float(a.iloc[i]):.2f})",
                                   entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
            elif bool(exit_break.iloc[i]):
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                   conviction=0.5, reason="close < 10d low"))
        return sigs
