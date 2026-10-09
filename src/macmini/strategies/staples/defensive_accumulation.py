""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ibs, rsi, sma, zscore
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates

VOL_Z_FLOOR = 0.5
IBS_ENTRY = 0.3
IBS_EXIT = 0.8
RSI_LO, RSI_HI = 35.0, 55.0
HOLD_DAYS = 7


@register
class StaplesDefensiveAccumulation(Strategy):
    id = 65
    name = "Defensive Accumulation Composite"
    category = "composite"
    data_required = ("mm_bars", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or len(bars) < 220:
            return []
        close = bars["close"]
        s200 = sma(close, 200)
        r14 = rsi(close, 14)
        bar_ibs = ibs(bars["open"], bars["high"], bars["low"], close)
        vol_z = zscore(bars["volume"].astype(float), 50)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(200, len(bars)):
            if open_idx is None:
                if (close.iloc[i] > s200.iloc[i]
                        and bar_ibs.iloc[i] < IBS_ENTRY
                        and vol_z.iloc[i] > VOL_Z_FLOOR
                        and RSI_LO <= r14.iloc[i] <= RSI_HI):
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                        conviction=0.55,
                        reason=f"weak close (IBS {bar_ibs.iloc[i]:.2f}) on volume z {vol_z.iloc[i]:.1f} in uptrend",
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if bar_ibs.iloc[i] > IBS_EXIT or i - open_idx >= HOLD_DAYS:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                        conviction=0.5,
                        reason="strong close" if bar_ibs.iloc[i] > IBS_EXIT else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
