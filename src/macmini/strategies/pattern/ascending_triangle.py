""
from __future__ import annotations

import numpy as np
import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

WINDOW = 40
HIGH_FLAT_TOL = 0.02
VOL_MULT = 1.4


@register
class AscendingTriangle(Strategy):
    id = 24
    name = "Ascending Triangle Breakout"
    category = "pattern"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < WINDOW + 5:
            return []
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        vol = bars["volume"]
        vol_avg = vol.rolling(50).mean()
        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])

        sigs: list[Signal] = []
        for i in range(WINDOW, len(bars)):
            highs = high.iloc[i - WINDOW : i].values
            lows = low.iloc[i - WINDOW : i].values
            if len(highs) < WINDOW:
                continue
            mid = float(np.median(highs))
            flat = (highs.max() - highs.min()) / mid if mid else 0
            if flat > HIGH_FLAT_TOL:
                continue
            x = np.arange(len(lows))
            slope = float(np.polyfit(x, lows, 1)[0])
            if slope <= 0:
                continue
            resistance = float(np.median(highs))
            cur_close = float(close.iloc[i])
            cur_vol = float(vol.iloc[i])
            vavg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else 0
            if vavg > 0 and cur_close > resistance and cur_vol >= VOL_MULT * vavg:
                triangle_height = resistance - float(lows.min())
                sigs.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="enter",
                        conviction=0.55,
                        reason=f"ascending triangle break above {resistance:.2f}",
                        entry_price=cur_close,
                        stop_price=float(lows.min()),
                        target_price=cur_close + triangle_height,
                    )
                )
        return sigs
