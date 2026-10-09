""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

DECLINE_QUARTERS = 4


@register
class CapexSalesDecline(Strategy):
    id = 40
    name = "Capex / Sales Decline"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if fundamentals is None or fundamentals.empty or len(bars) < 220:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end").reset_index(drop=True)
        if len(f) < DECLINE_QUARTERS + 1:
            return []
        ratios = []
        for _, r in f.tail(DECLINE_QUARTERS + 1).iterrows():
            cap = r.get("capex"); rev = r.get("revenue")
            if cap is None or rev is None or rev == 0:
                return []
            ratios.append(abs(cap) / rev)
        if not all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1)):
            return []
        close = bars["close"]
        s200 = sma(close, 200)
        if pd.isna(s200.iloc[-1]) or close.iloc[-1] <= s200.iloc[-1]:
            return []
        d = pd.Timestamp(bars.index[-1]) if isinstance(bars.index, pd.DatetimeIndex) else pd.Timestamp(pd.to_datetime(bars["date"]).iloc[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"capex/sales declined {ratios[0]:.2%} → {ratios[-1]:.2%} over {DECLINE_QUARTERS}Q",
                entry_price=float(close.iloc[-1]),
            )
        ]
