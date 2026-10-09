""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

IMPROVEMENT_TURNS = 0.5


@register
class DebtEbitdaImprovement(Strategy):
    id = 38
    name = "Net Debt / EBITDA Improvement"
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
        if len(f) < 5:
            return []
        last, year_ago = f.iloc[-1], f.iloc[-5]
        denom_last = last.get("ebitda") if last.get("ebitda") else last.get("earnings")
        denom_prev = year_ago.get("ebitda") if year_ago.get("ebitda") else year_ago.get("earnings")
        td_l, td_p = last.get("total_debt"), year_ago.get("total_debt")
        if not (denom_last and denom_prev and td_l is not None and td_p is not None and denom_last > 0 and denom_prev > 0):
            return []
        leverage_last = td_l / denom_last
        leverage_prev = td_p / denom_prev
        if (leverage_prev - leverage_last) < IMPROVEMENT_TURNS:
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
                reason=f"leverage {leverage_prev:.1f}x → {leverage_last:.1f}x in uptrend",
                entry_price=float(close.iloc[-1]),
            )
        ]
