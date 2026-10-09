""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

WINDOW = 60
VOL_MULT = 1.2


@register
class InverseHeadAndShoulders(Strategy):
    id = 23
    name = "Inverse Head-and-Shoulders Breakout"
    category = "pattern"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < WINDOW + 5:
            return []
        close = bars["close"]
        low = bars["low"]
        high = bars["high"]
        vol = bars["volume"]
        vol_avg = vol.rolling(50).mean()
        dates = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])

        sigs: list[Signal] = []
        for i in range(WINDOW, len(bars)):
            window = slice(i - WINDOW, i)
            lows = low.iloc[window].values
            highs = high.iloc[window].values
            if len(lows) < WINDOW:
                continue
            head_idx = int(lows.argmin())
            if not (WINDOW * 0.3 <= head_idx <= WINDOW * 0.7):
                continue
            ls_lows = lows[: head_idx - 5]
            rs_lows = lows[head_idx + 5 :]
            if len(ls_lows) < 3 or len(rs_lows) < 3:
                continue
            ls_idx = int(ls_lows.argmin())
            rs_idx = head_idx + 5 + int(rs_lows.argmin())
            head = float(lows[head_idx])
            ls = float(lows[ls_idx])
            rs = float(lows[rs_idx])
            if not (ls > head and rs > head):
                continue
            if abs(ls - rs) / max(ls, rs) > 0.15:
                continue
            neckline = max(float(highs[ls_idx:head_idx].max()), float(highs[head_idx:rs_idx].max()))
            cur_close = float(close.iloc[i])
            cur_vol = float(vol.iloc[i])
            vavg = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else 0
            if vavg > 0 and cur_close > neckline and cur_vol >= VOL_MULT * vavg:
                target = neckline + (neckline - head)
                sigs.append(
                    Signal(
                        symbol=symbol,
                        date=pd.Timestamp(dates[i]),
                        action="enter",
                        conviction=0.55,
                        reason=f"iH&S neckline {neckline:.2f} broken on {cur_vol/vavg:.2f}x volume",
                        entry_price=cur_close,
                        stop_price=rs,
                        target_price=target,
                    )
                )
        return sigs
