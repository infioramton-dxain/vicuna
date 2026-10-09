""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register


@register
class SalesAcceleration(Strategy):
    id = 36
    name = "Sales Acceleration (3Q Sequential)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if fundamentals is None or fundamentals.empty or len(bars) < 80:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end").reset_index(drop=True)
        if len(f) < 7:
            return []

        def yoy(idx: int) -> float | None:
            if idx < 4 or pd.isna(f.iloc[idx].get("revenue")) or pd.isna(f.iloc[idx - 4].get("revenue")) or f.iloc[idx - 4]["revenue"] == 0:
                return None
            return (f.iloc[idx]["revenue"] - f.iloc[idx - 4]["revenue"]) / f.iloc[idx - 4]["revenue"]

        g0 = yoy(len(f) - 1)
        g1 = yoy(len(f) - 2)
        g2 = yoy(len(f) - 3)
        if None in (g0, g1, g2):
            return []
        if not (g2 < g1 < g0 and g0 > 0):
            return []

        close = bars["close"]
        s50 = sma(close, 50)
        if pd.isna(s50.iloc[-1]) or close.iloc[-1] <= s50.iloc[-1]:
            return []

        d = pd.Timestamp(bars.index[-1]) if isinstance(bars.index, pd.DatetimeIndex) else pd.Timestamp(pd.to_datetime(bars["date"]).iloc[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.6,
                reason=f"YoY revenue accel: {g2*100:.1f}%→{g1*100:.1f}%→{g0*100:.1f}%",
                entry_price=float(close.iloc[-1]),
            )
        ]
