""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

LOOKBACK = 252
RECENT_SKIP = 21
VOL_WINDOW = 30


@register
class LowVolMomentum(Strategy):
    id = 54
    name = "Defensive Low-Volatility Momentum (12-1)"
    category = "composite"
    data_required = ("mm_bars",)

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        momentum_pct: float | None = None,
        vol_pct: float | None = None,
        **kwargs,
    ) -> list[Signal]:
        ""
        if momentum_pct is None or vol_pct is None or len(bars) < LOOKBACK + RECENT_SKIP + 5:
            return []
        if momentum_pct < 0.8 or vol_pct > 0.5:
            return []
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        d = pd.Timestamp(dts[-1])
        return [
            Signal(
                symbol=symbol,
                date=d,
                action="enter",
                conviction=0.55,
                reason=f"12-1 momentum top {(1-momentum_pct)*100:.0f}%, vol bottom {vol_pct*100:.0f}%",
                entry_price=float(bars["close"].iloc[-1]),
            )
        ]
