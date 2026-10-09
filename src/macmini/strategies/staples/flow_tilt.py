""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma, zscore
from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates

Z_FLOOR = 1.0
HOLD_DAYS = 5


@register
class StaplesFlowTilt(Strategy):
    id = 62
    name = "XLP Flow Tilt (13wk Z-Score)"
    category = "sentiment"
    data_required = ("mm_bars", "mm_sector_etf_flows_weekly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        etf_flows: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or etf_flows is None or etf_flows.empty or len(bars) < 60:
            return []
        f = etf_flows.copy()
        f["week_end"] = pd.to_datetime(f["week_end"])
        f = f.sort_values("week_end")
        f["z"] = zscore(f["net_flow_usd"], 13)
        hot = f[f["z"] > Z_FLOOR]
        if hot.empty:
            return []

        close = bars["close"]
        s50 = sma(close, 50)
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        last_exit = -1
        for _, row in hot.iterrows():
            mask = dts >= row["week_end"]
            if not mask.any():
                continue
            idx = int(mask.argmax())
            if idx <= last_exit or idx >= len(bars):
                continue
            if pd.isna(s50.iloc[idx]) or close.iloc[idx] <= s50.iloc[idx]:
                continue
            sigs.append(Signal(
                symbol=symbol, date=pd.Timestamp(dts[idx]), action="enter",
                conviction=0.55, reason=f"XLP flow z={float(row['z']):.1f}",
                entry_price=float(close.iloc[idx])))
            exit_idx = min(idx + HOLD_DAYS, len(bars) - 1)
            sigs.append(Signal(
                symbol=symbol, date=pd.Timestamp(dts[exit_idx]), action="exit",
                conviction=0.5, reason=f"{HOLD_DAYS}d flow-tilt hold"))
            last_exit = exit_idx
        return sigs
