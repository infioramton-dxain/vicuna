""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import bollinger
from macmini.strategies.base import Signal, Strategy, register

CUP_WINDOW = 60
HANDLE_WINDOW = 7
VOL_MULT = 1.4
HANDLE_RETRACE_MAX = 0.08
STOP_PCT = 0.07


@register
class CupHandle(Strategy):
    id = 16
    name = "Cup-with-Handle / VCP Breakout"
    category = "breakout"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < CUP_WINDOW + 10:
            return []
        close = bars["close"]
        high = bars["high"]
        volume = bars["volume"]
        _, _, _, bb_w = bollinger(close, 20, 2.0)
        bb_w_med = bb_w.rolling(CUP_WINDOW).median()
        vol_avg = volume.rolling(50).mean()

        sigs: list[Signal] = []
        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])

        for i in range(CUP_WINDOW, len(bars)):
            cup = high.iloc[i - CUP_WINDOW : i - HANDLE_WINDOW]
            handle = high.iloc[i - HANDLE_WINDOW : i]
            cup_high = float(cup.max())
            handle_high = float(handle.max())
            if cup_high <= 0:
                continue
            retrace = (cup_high - handle_high) / cup_high
            if retrace > HANDLE_RETRACE_MAX:
                continue
            cur_close = float(close.iloc[i])
            cur_vol = float(volume.iloc[i])
            vavg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else 0
            cur_bbw = float(bb_w.iloc[i]) if not pd.isna(bb_w.iloc[i]) else 0
            bbw_med = float(bb_w_med.iloc[i]) if not pd.isna(bb_w_med.iloc[i]) else 0
            if (
                cur_close > handle_high
                and vavg > 0
                and cur_vol >= VOL_MULT * vavg
                and bbw_med > 0
                and cur_bbw < bbw_med
            ):
                sigs.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="enter",
                        conviction=0.6,
                        reason=f"cup-handle breakout above {handle_high:.2f}, vol {cur_vol/vavg:.2f}x",
                        entry_price=cur_close,
                        stop_price=cur_close * (1 - STOP_PCT),
                    )
                )
        return sigs
