""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register


@register
class CalendarSeasonality(Strategy):
    id = 55
    name = "Calendar / Seasonality Composite"
    category = "composite"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 5:
            return []
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        dt_idx = pd.DatetimeIndex(dts).normalize()
        sigs: list[Signal] = []
        in_window: str | None = None
        for i, d in enumerate(dt_idx):
            window = None
            if d.month == 12 and d.day >= 23:
                window = "santa"
            elif d.month == 1 and d.day <= 10:
                window = "january_effect"
            elif d.month in (3, 6, 9, 12) and d.day >= 27:
                window = "window_dressing"
            if window and window != in_window:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                   conviction=0.5, reason=f"seasonal window: {window}",
                                   entry_price=float(close.iloc[i])))
                in_window = window
            elif in_window and not window:
                sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                   conviction=0.5, reason=f"seasonal window {in_window} ended"))
                in_window = None
        return sigs
