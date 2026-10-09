""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

DROP_WINDOW = 3
DROP_THRESHOLD = -0.10
VOLUME_MULT = 1.5
VIX_PANIC_LEVEL = 20.0
VIX_SPIKE_WINDOW = 5
VIX_SPIKE_PCT = 0.25
TIME_STOP = 7


@register
class PanicDip(Strategy):
    id = 66
    name = "Panic Dip (Crash Mean Reversion)"
    category = "mean_reversion"
    data_required = ("mm_bars", "market")

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        market = kwargs.get("market")
        if len(bars) < 220 or market is None or len(market) == 0:
            return []
        if "vix_close" not in market.columns:
            return []

        close, volume = bars["close"], bars["volume"]
        dates = (bars.index if isinstance(bars.index, pd.DatetimeIndex)
                 else pd.to_datetime(bars["date"]))

        vix = market.copy()
        vix.index = pd.to_datetime(vix["date"].astype(str).str.slice(0, 10))
        v = vix["vix_close"].reindex(pd.DatetimeIndex(dates)).ffill(limit=5)
        v_spike = v / v.shift(VIX_SPIKE_WINDOW) - 1.0
        panic = ((v >= VIX_PANIC_LEVEL) | (v_spike >= VIX_SPIKE_PCT)).fillna(False)

        drop = close / close.shift(DROP_WINDOW) - 1.0
        s5, s200 = sma(close, 5), sma(close, 200)
        vol_avg = sma(volume, 20)
        entry = ((drop <= DROP_THRESHOLD)
                 & (close > s200)
                 & (volume >= VOLUME_MULT * vol_avg)
                 & panic.to_numpy())
        exit_sig = close > s5

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(entry.iloc[i]):
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dates[i]),
                        action="enter", conviction=0.6,
                        reason=(f"{DROP_WINDOW}d drop {float(drop.iloc[i]):+.1%} "
                                f"above SMA200, VIX {float(v.iloc[i]):.1f}"),
                        entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if bool(exit_sig.iloc[i]) or i - open_idx >= TIME_STOP:
                    sigs.append(Signal(
                        symbol=symbol, date=pd.Timestamp(dates[i]),
                        action="exit", conviction=0.5,
                        reason=("close > SMA5"
                                if exit_sig.iloc[i] else f"{TIME_STOP}d stop")))
                    open_idx = None
        return sigs
