""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

HOLD_DAYS = 5
TOP_PCTILE = 0.9
EXIT_PCTILE = 0.5


@register
class RsLeadership(Strategy):
    id = 20
    name = "Relative Strength Leadership Rotation"
    category = "breakout"
    data_required = ("mm_bars", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        peers_close: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if peers_close is None or len(peers_close.columns) < 5 or len(bars) < 60:
            return []
        if symbol not in peers_close.columns:
            return []
        weekly = peers_close.resample("W-FRI").last()
        rets = weekly.pct_change(5)
        ranks = rets.rank(axis=1, pct=True)
        my_rank = ranks[symbol]

        if isinstance(bars.index, pd.DatetimeIndex):
            bar_dates = bars.index
        else:
            bar_dates = pd.to_datetime(bars["date"])

        rank_at_bar = my_rank.reindex(bar_dates, method="ffill")
        in_top = rank_at_bar >= TOP_PCTILE
        below_exit = rank_at_bar < EXIT_PCTILE

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            if open_idx is None:
                if bool(in_top.iloc[i]):
                    px = float(bars["close"].iloc[i])
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(bar_dates[i]), action="enter",
                                       conviction=0.55, reason=f"weekly RS top {TOP_PCTILE*100:.0f}%",
                                       entry_price=px))
                    open_idx = i
            else:
                if i - open_idx >= HOLD_DAYS or bool(below_exit.iloc[i]):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(bar_dates[i]), action="exit",
                                       conviction=0.5,
                                       reason="RS faded" if bool(below_exit.iloc[i]) else f"{HOLD_DAYS}d hold"))
                    open_idx = None
        return sigs
