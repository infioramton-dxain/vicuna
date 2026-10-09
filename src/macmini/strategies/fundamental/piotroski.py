""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register


@register
class PiotroskiF(Strategy):
    id = 33
    name = "Piotroski F-Score + Uptrend"
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
        last, prev_year = f.iloc[-1], f.iloc[-5]

        score = 0
        possible = 0
        ni = last.get("earnings")
        if ni is not None:
            possible += 1
            if ni > 0: score += 1
        ocf = last.get("operating_cash_flow")
        if ocf is not None:
            possible += 1
            if ocf > 0: score += 1
        ni_prev = prev_year.get("earnings")
        if ni is not None and ni_prev is not None:
            possible += 1
            if ni > ni_prev: score += 1
        if ni is not None and ocf is not None:
            possible += 1
            if ocf > ni: score += 1
        td = last.get("total_debt"); td_prev = prev_year.get("total_debt")
        te = last.get("total_equity"); te_prev = prev_year.get("total_equity")
        if all(x is not None for x in (td, td_prev, te, te_prev)) and te > 0 and te_prev > 0:
            possible += 1
            if (td / te) < (td_prev / te_prev): score += 1

        if possible < 3 or score < max(3, int(0.7 * possible)):
            return []

        close = bars["close"]
        s200 = sma(close, 200)
        if pd.isna(s200.iloc[-1]) or close.iloc[-1] <= s200.iloc[-1]:
            return []
        if close.iloc[-1] <= close.iloc[-22]:
            return []

        d = pd.Timestamp(bars.index[-1]) if isinstance(bars.index, pd.DatetimeIndex) else pd.Timestamp(pd.to_datetime(bars["date"]).iloc[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"F-score {score}/{possible} + uptrend",
                entry_price=float(close.iloc[-1]),
            )
        ]
