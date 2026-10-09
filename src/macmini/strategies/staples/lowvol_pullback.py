""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, sma
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates

ATR_MULT = 2.5
SLOPE_LOOKBACK = 10
HOLD_DAYS = 7


@register
class StaplesLowVolPullback(Strategy):
    id = 58
    name = "Staples Low-Vol Pullback (2.5 ATR)"
    category = "mean_reversion"
    data_required = ("mm_bars", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 80:
            return []
        close = bars["close"]
        s20 = sma(close, 20)
        s50 = sma(close, 50)
        a14 = atr(bars["high"], bars["low"], close, 14)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(60, len(bars)):
            if open_idx is None:
                slope_up = s50.iloc[i] > s50.iloc[i - SLOPE_LOOKBACK]
                deep_dip = close.iloc[i] <= s20.iloc[i] - ATR_MULT * a14.iloc[i]
                if slope_up and deep_dip:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.55,
                        reason=f"{ATR_MULT}x ATR below SMA20 in an uptrend",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if close.iloc[i] >= s20.iloc[i] or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="SMA20 tag" if close.iloc[i] >= s20.iloc[i] else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
