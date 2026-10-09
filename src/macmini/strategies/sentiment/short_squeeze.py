""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import atr, rsi
from macmini.strategies.base import Signal, Strategy, register

SI_FLOOR = 0.15
DTC_FLOOR = 5
VOL_MULT = 2.0
ATR_MULT = 5.0


@register
class ShortSqueeze(Strategy):
    id = 45
    name = "Short Interest Squeeze Setup"
    category = "sentiment"
    data_required = ("mm_bars", "mm_short_interest")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        short_interest: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if short_interest is None or short_interest.empty or len(bars) < 30:
            return []
        si = short_interest.copy()
        si["date"] = pd.to_datetime(si["date"])
        si = si.sort_values("date")
        close = bars["close"]
        high = bars["high"]
        vol = bars["volume"]
        vol_avg = vol.rolling(50).mean()
        twenty_high = high.rolling(20).max()
        r = rsi(close, 14)
        a = atr(bars["high"], bars["low"], close, 14)

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(20, len(bars)):
            d = pd.Timestamp(dts[i]).normalize()
            recent = si[si["date"] <= d]
            if recent.empty:
                continue
            last = recent.iloc[-1]
            si_pct = last.get("short_interest_pct")
            dtc = last.get("days_to_cover")
            if si_pct is None or dtc is None or si_pct < SI_FLOOR or dtc < DTC_FLOOR:
                continue
            if open_idx is None:
                if close.iloc[i] >= twenty_high.iloc[i] and vol.iloc[i] >= VOL_MULT * vol_avg.iloc[i]:
                    px = float(close.iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                       conviction=0.6, reason=f"SI {si_pct*100:.1f}% + breakout",
                                       entry_price=px, stop_price=px - ATR_MULT * float(a.iloc[i])))
                    open_idx = i
            else:
                if r.iloc[i] > 80:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                       conviction=0.55, reason="RSI(14) > 80"))
                    open_idx = None
        return sigs
