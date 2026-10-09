""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import macd
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import (
    STAPLES, bar_dates, quarterly_history, trailing_eps_annual, ttm_yield_series)

MAX_PE = 18.0
MIN_YIELD_PCT = 2.5
HOLD_DAYS = 10


@register
class StaplesValueYield(Strategy):
    id = 59
    name = "Staples Value + Yield (PE/DivYield/FCF)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly", "mm_fundamentals_annual",
                     "mm_dividends", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        fundamentals_annual: pd.DataFrame | None = None,
        dividends: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 60:
            return []
        eps4 = trailing_eps_annual(fundamentals_annual)
        hist4 = quarterly_history(fundamentals, 4)
        if eps4 is None or eps4 <= 0 or hist4 is None:
            return []
        fcf_ttm = hist4["free_cash_flow"].astype(float).fillna(0).sum()
        if fcf_ttm <= 0:
            return []

        close = bars["close"]
        dts = bar_dates(bars)
        yld = ttm_yield_series(dividends, close, dts)
        if yld is None:
            return []
        _, _, hist = macd(close)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(35, len(bars)):
            if open_idx is None:
                pe = float(close.iloc[i]) / eps4
                y = yld.iloc[i]
                if (0 < pe <= MAX_PE and not pd.isna(y) and y >= MIN_YIELD_PCT
                        and hist.iloc[i] > 0 and hist.iloc[i - 1] <= 0):
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.6,
                        reason=f"PE {pe:.1f}, ttm yield {float(y):.1f}%, MACD turn",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if (hist.iloc[i] < 0 and hist.iloc[i - 1] >= 0) or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="MACD rolled over" if hist.iloc[i] < 0 else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
