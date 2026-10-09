""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import williams_r
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import (
    STAPLES, bar_dates, quarterly_history, ttm_yield_series)

MIN_YIELD_PCT = 4.5
NEAR_LOW_PCT = 0.15
WR_ENTRY = -80.0
WR_EXIT = -20.0
HOLD_DAYS = 10


@register
class StaplesHighYieldSustainability(Strategy):
    id = 63
    name = "High-Yield Staples Sustainability"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_dividends", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        dividends: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 260:
            return []
        hist4 = quarterly_history(fundamentals, 4)
        if hist4 is None:
            return []
        fcf_ttm = hist4["free_cash_flow"].astype(float).fillna(0).sum()
        if fcf_ttm <= 0:
            return []

        close = bars["close"]
        dts = bar_dates(bars)
        yld = ttm_yield_series(dividends, close, dts)
        if yld is None:
            return []
        low252 = close.rolling(252).min()
        wr = williams_r(bars["high"], bars["low"], close, 14)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(252, len(bars)):
            if open_idx is None:
                y = yld.iloc[i]
                near_low = close.iloc[i] <= low252.iloc[i] * (1 + NEAR_LOW_PCT)
                wr_cross_up = wr.iloc[i] > WR_ENTRY >= wr.iloc[i - 1]
                if not pd.isna(y) and y >= MIN_YIELD_PCT and near_low and wr_cross_up:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.55,
                        reason=f"ttm yield {float(y):.1f}% FCF-covered, near 52w low, %R turn",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if wr.iloc[i] > WR_EXIT or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="%R recovered" if wr.iloc[i] > WR_EXIT else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
