""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import rsi, sma
from macmini.strategies.base import Signal, Strategy, register


@register
class WeeklyDailyPullback(Strategy):
    id = 28
    name = "Weekly Trend + Daily Pullback Entry"
    category = "mtf"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 250:
            return []
        close = bars["close"]
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])

        daily_rsi = rsi(close, 14)
        daily_sma20 = sma(close, 20)
        weekly = close.copy()
        weekly.index = dts
        weekly_close = weekly.resample("W-FRI").last()
        weekly_sma20 = sma(weekly_close, 20)
        weekly_trend = weekly_close > weekly_sma20

        trend_at_bar = weekly_trend.reindex(dts, method="ffill", fill_value=False).astype(bool)
        up_today = close > close.shift(1)
        entry = trend_at_bar & (daily_rsi < 35) & up_today
        exit_sig = (close < daily_sma20) | (~trend_at_bar)

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                       conviction=0.55, reason="weekly uptrend + daily RSI<35 + up close",
                                       entry_price=px))
                    open_idx = i
            else:
                if bool(exit_sig.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                       conviction=0.5, reason="daily SMA20 breach / weekly broke"))
                    open_idx = None
        return sigs
