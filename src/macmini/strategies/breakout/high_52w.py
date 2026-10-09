""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

LOOKBACK = 252
VOLUME_LOOKBACK = 50
VOLUME_MULT = 1.5
TRAILING_STOP_PCT = 0.08
EXIT_SMA = 20


@register
class High52WeekBreakout(Strategy):
    id = 15
    name = "52-Week High Breakout with Volume"
    category = "breakout"
    data_required = ("mm_bars",)

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        **kwargs,
    ) -> list[Signal]:
        if len(bars) < LOOKBACK + 2:
            return []

        close = bars["close"]
        high = bars["high"] if "high" in bars else close
        volume = bars["volume"]

        rolling_high = high.rolling(LOOKBACK).max()
        vol_avg = volume.rolling(VOLUME_LOOKBACK).mean()
        sma_exit = close.rolling(EXIT_SMA).mean()

        prev_high = rolling_high.shift(1)
        is_breakout = (close > prev_high) & (volume >= VOLUME_MULT * vol_avg)
        is_exit = close < sma_exit

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        signals: list[Signal] = []
        for i in range(len(bars)):
            if is_breakout.iloc[i]:
                px = float(close.iloc[i])
                vmult = float(volume.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] else 0.0
                signals.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="enter",
                        conviction=min(1.0, 0.5 + 0.1 * (vmult - VOLUME_MULT)),
                        reason=f"new 52w high {px:.2f}; volume {vmult:.2f}x avg",
                        entry_price=px,
                        stop_price=px * (1 - TRAILING_STOP_PCT),
                    )
                )
            elif is_exit.iloc[i]:
                signals.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="exit",
                        conviction=0.5,
                        reason=f"close {float(close.iloc[i]):.2f} fell below SMA{EXIT_SMA}",
                    )
                )
        return signals
