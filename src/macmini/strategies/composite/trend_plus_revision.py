""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.fundamental.eps_revision import EpsRevisionMomentum
from macmini.strategies.trend.golden_cross import GoldenCross


@register
class TrendPlusRevision(Strategy):
    id = 53
    name = "Trend + Earnings Revision"
    category = "composite"
    data_required = ("mm_bars", "mm_analyst_estimates")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        estimates: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        trend_sigs = [s for s in GoldenCross().generate_signals(symbol, bars)]
        rev_sigs = [s for s in EpsRevisionMomentum().generate_signals(symbol, bars, estimates=estimates)]
        if not trend_sigs or not rev_sigs:
            return []
        trend_enters = [(s.date, s) for s in trend_sigs if s.action == "enter"]
        rev_enters = [s.date for s in rev_sigs if s.action == "enter"]
        rev_enters_idx = pd.DatetimeIndex(rev_enters)
        out: list[Signal] = []
        for d, ts in trend_enters:
            window = (rev_enters_idx >= d - pd.Timedelta(days=60)) & (rev_enters_idx <= d + pd.Timedelta(days=60))
            if window.any():
                out.append(Signal(symbol=symbol, date=d, action="enter", conviction=min(1.0, ts.conviction + 0.2),
                                  reason=f"golden cross + EPS revision active",
                                  entry_price=ts.entry_price))
        return out
