""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

MIN_YIELD = 0.08


@register
class FcfYield(Strategy):
    id = 39
    name = "Free Cash Flow Yield (FCF/EV)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        market_cap: float | None = None,
        **kwargs,
    ) -> list[Signal]:
        if fundamentals is None or fundamentals.empty or len(bars) < 220:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end")
        fcf = f["free_cash_flow"].dropna().tail(4).sum() if "free_cash_flow" in f else None
        if fcf is None or fcf <= 0:
            return []
        if market_cap is None or market_cap <= 0:
            return []
        ev = market_cap + (f.iloc[-1].get("total_debt") or 0)
        if ev <= 0:
            return []
        yld = fcf / ev
        if yld < MIN_YIELD:
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
                reason=f"FCF/EV {yld*100:.1f}%",
                entry_price=float(close.iloc[-1]),
            )
        ]
