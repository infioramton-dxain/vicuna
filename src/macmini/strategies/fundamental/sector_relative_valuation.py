""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

BOTTOM_DECILE = 0.1


@register
class SectorRelativeValuation(Strategy):
    id = 42
    name = "Sector-Relative Valuation Rotation"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        valuation_pct: float | None = None,
        sector_rs_positive: bool = False,
        **kwargs,
    ) -> list[Signal]:
        if valuation_pct is None or not sector_rs_positive or len(bars) < 5:
            return []
        if valuation_pct > BOTTOM_DECILE:
            return []
        d = pd.Timestamp(bars.index[-1]) if isinstance(bars.index, pd.DatetimeIndex) else pd.Timestamp(pd.to_datetime(bars["date"]).iloc[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"valuation pct {valuation_pct*100:.1f}% in sector with positive RS",
                entry_price=float(bars["close"].iloc[-1]),
            )
        ]
