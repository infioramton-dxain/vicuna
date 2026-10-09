""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

FAST = 50
SLOW = 200
TRAILING_STOP_PCT = 0.08


@register
class GoldenCross(Strategy):
    id = 1
    name = "50/200 SMA Golden Cross"
    category = "trend"
    data_required = ("mm_bars",)

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        **kwargs,
    ) -> list[Signal]:
        if len(bars) < SLOW + 2:
            return []

        close = bars["close"]
        sma_fast = close.rolling(FAST).mean()
        sma_slow = close.rolling(SLOW).mean()

        prev_above = (sma_fast.shift(1) > sma_slow.shift(1)).fillna(False)
        now_above = (sma_fast > sma_slow).fillna(False)
        golden = (~prev_above) & now_above & (close > sma_slow)
        death = prev_above & (~now_above)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        signals: list[Signal] = []
        for i, (is_golden, is_death) in enumerate(zip(golden, death)):
            if not (is_golden or is_death):
                continue
            d = dates[i]
            px = float(close.iloc[i])
            if is_golden:
                signals.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(d),
                        action="enter",
                        conviction=0.6,
                        reason=f"SMA{FAST} crossed above SMA{SLOW}; close {px:.2f} > SMA{SLOW}",
                        entry_price=px,
                        stop_price=px * (1 - TRAILING_STOP_PCT),
                    )
                )
            else:
                signals.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(d),
                        action="exit",
                        conviction=0.6,
                        reason=f"death cross: SMA{FAST} fell below SMA{SLOW}",
                    )
                )
        return signals
