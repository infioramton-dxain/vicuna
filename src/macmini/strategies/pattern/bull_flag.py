""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

POLE_MIN_PCT = 0.05
POLE_MAX_PCT = 0.15
POLE_MAX_DAYS = 5
FLAG_DAYS = 5
STOP_PCT = 0.05


@register
class BullFlag(Strategy):
    id = 22
    name = "Bull Flag Continuation"
    category = "pattern"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < POLE_MAX_DAYS + FLAG_DAYS + 5:
            return []
        close = bars["close"]
        high = bars["high"]
        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])

        sigs: list[Signal] = []
        for i in range(POLE_MAX_DAYS + FLAG_DAYS, len(bars)):
            pole_end = i - FLAG_DAYS
            pole_start = pole_end - POLE_MAX_DAYS
            if pole_start < 0:
                continue
            pole_low = float(close.iloc[pole_start])
            pole_high = float(close.iloc[pole_end])
            advance = (pole_high - pole_low) / pole_low if pole_low else 0
            if not (POLE_MIN_PCT <= advance <= POLE_MAX_PCT):
                continue
            flag_window = high.iloc[pole_end : i]
            flag_high = float(flag_window.max())
            flag_low = float(flag_window.min())
            if pole_high <= 0:
                continue
            flag_range = (flag_high - flag_low) / pole_high
            if flag_range > 0.5 * advance:
                continue
            cur_close = float(close.iloc[i])
            if cur_close > flag_high:
                flagpole_len = pole_high - pole_low
                sigs.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="enter",
                        conviction=0.55,
                        reason=f"flag break above {flag_high:.2f}, pole +{advance*100:.1f}%",
                        entry_price=cur_close,
                        stop_price=cur_close * (1 - STOP_PCT),
                        target_price=cur_close + flagpole_len,
                    )
                )
        return sigs
