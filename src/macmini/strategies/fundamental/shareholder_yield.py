""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

MIN_YIELD = 0.06


@register
class ShareholderYield(Strategy):
    id = 37
    name = "Shareholder Yield (Buybacks + Dividends)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if fundamentals is None or fundamentals.empty or len(bars) < 5:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end")
        bb = f["buyback_yield"].dropna().iloc[-1] if "buyback_yield" in f and f["buyback_yield"].notna().any() else None
        dv = f["dividend_yield"].dropna().iloc[-1] if "dividend_yield" in f and f["dividend_yield"].notna().any() else None
        if bb is None or dv is None:
            return []
        yld = bb + dv
        if yld < MIN_YIELD:
            return []
        fcf = f["free_cash_flow"].dropna()
        if fcf.empty or fcf.iloc[-1] <= 0:
            return []
        d = pd.Timestamp(bars.index[-1]) if isinstance(bars.index, pd.DatetimeIndex) else pd.Timestamp(pd.to_datetime(bars["date"]).iloc[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"shareholder yield {yld*100:.1f}% (FCF positive)",
                entry_price=float(bars["close"].iloc[-1]),
            )
        ]
