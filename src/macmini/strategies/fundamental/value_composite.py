""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register


@register
class ValueComposite(Strategy):
    id = 32
    name = "Value Composite (Sector-Relative)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        sector_z: float | None = None,
        **kwargs,
    ) -> list[Signal]:
        ""
        if fundamentals is None or fundamentals.empty or sector_z is None or len(bars) < 5:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end")
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        eps_positive = f["earnings"].iloc[-1] > 0 if "earnings" in f and f["earnings"].notna().any() else False
        if not eps_positive:
            return []
        if sector_z > -1:
            return []
        d = pd.Timestamp(dts[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"value composite Z={sector_z:.2f} (bottom decile of sector)",
                entry_price=float(bars["close"].iloc[-1]),
            )
        ]
