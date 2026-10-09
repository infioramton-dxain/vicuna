""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import rsi, sma
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates, ttm_yield_series

MIN_YIELD_PCT = 3.0
RSI_ENTRY = 40.0
RSI_EXIT = 55.0
HOLD_DAYS = 10


@register
class DogsOfStaples(Strategy):
    id = 56
    name = "Dogs of the Staples"
    category = "fundamental"
    data_required = ("mm_bars", "mm_dividends", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        dividends: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 60:
            return []
        close = bars["close"]
        dts = bar_dates(bars)
        yld = ttm_yield_series(dividends, close, dts)
        if yld is None:
            return []
        s50 = sma(close, 50)
        r14 = rsi(close, 14)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(50, len(bars)):
            if open_idx is None:
                y = yld.iloc[i]
                if (not pd.isna(y) and y >= MIN_YIELD_PCT
                        and close.iloc[i] < s50.iloc[i] and r14.iloc[i] < RSI_ENTRY):
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=min(0.75, 0.55 + float(y) / 40.0),
                        reason=f"ttm yield {float(y):.1f}%, below SMA50, RSI {r14.iloc[i]:.0f}",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if r14.iloc[i] > RSI_EXIT or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="RSI recovered" if r14.iloc[i] > RSI_EXIT else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
