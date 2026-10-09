""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import ema
from macmini.strategies.base import Signal, Strategy, register


@register
class MonthlyHigherLow(Strategy):
    id = 30
    name = "Monthly Higher-Low + Weekly EMA8"
    category = "mtf"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 400:
            return []
        close = bars["close"]
        low = bars["low"]
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])

        c_idx = close.copy()
        c_idx.index = dts
        l_idx = low.copy()
        l_idx.index = dts

        monthly_low = l_idx.resample("ME").min()
        higher_lows = monthly_low > monthly_low.shift(1)
        weekly_close = c_idx.resample("W-FRI").last()
        w_e8 = ema(weekly_close, 8)
        weekly_above = weekly_close > w_e8
        w_e21 = ema(weekly_close, 21)
        weekly_under_21 = weekly_close < w_e21

        m_at_bar = higher_lows.reindex(dts, method="ffill", fill_value=False).astype(bool)
        w_at_bar = weekly_above.reindex(dts, method="ffill", fill_value=False).astype(bool)
        w_exit_at_bar = weekly_under_21.reindex(dts, method="ffill", fill_value=False).astype(bool)

        entry = m_at_bar & w_at_bar

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]) and (i == 0 or not bool(entry.iloc[i - 1])):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                       conviction=0.55, reason="monthly higher-low + weekly > EMA8",
                                       entry_price=px))
                    open_idx = i
            else:
                if bool(w_exit_at_bar.iloc[i]) or not bool(m_at_bar.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                       conviction=0.5, reason="weekly < EMA21 or monthly broke"))
                    open_idx = None
        return sigs
