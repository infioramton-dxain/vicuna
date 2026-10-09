""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma, zscore
from macmini.strategies.base import Signal, Strategy, register

Z_FLOOR = -2.0
TIME_STOP = 7


@register
class ZScoreReversion(Strategy):
    id = 12
    name = "Z-Score Reversion (20-day, -2σ)"
    category = "mean_reversion"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 120:
            return []
        close = bars["close"]
        s20 = sma(close, 20)
        s100 = sma(close, 100)
        std20 = close.rolling(20).std(ddof=0)
        z = (close - s20) / std20.replace(0, pd.NA)
        entry = (z < Z_FLOOR) & (close > s100)

        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="enter",
                                       conviction=0.55,
                                       reason=f"z-score {float(z.iloc[i]):.2f} in uptrend",
                                       entry_price=px, target_price=float(s20.iloc[i])))
                    open_idx = i
            else:
                z_i = z.iloc[i]
                if (not pd.isna(z_i) and z_i >= 0) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="z reverted" if not pd.isna(z_i) and z_i >= 0 else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
