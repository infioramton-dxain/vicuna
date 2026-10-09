""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import zscore
from macmini.strategies.base import Signal, Strategy, register

Z_FLOOR = 2.0
HOLD_WEEKS = 4


@register
class SectorEtfFlow(Strategy):
    id = 50
    name = "Sector ETF Flow / AUM Z-Score"
    category = "sentiment"
    data_required = ("mm_bars", "mm_sector_etf_flows_weekly", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        etf_flows: pd.DataFrame | None = None,
        rs_pctile: float | None = None,
        **kwargs,
    ) -> list[Signal]:
        if etf_flows is None or etf_flows.empty or rs_pctile is None or rs_pctile < 0.9 or len(bars) < 20:
            return []
        f = etf_flows.copy()
        f["week_end"] = pd.to_datetime(f["week_end"])
        f = f.sort_values("week_end")
        f["z"] = zscore(f["net_flow_usd"], 13)
        recent = f[f["z"] > Z_FLOOR]
        if recent.empty:
            return []

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]

        sigs: list[Signal] = []
        for _, row in recent.iterrows():
            week = row["week_end"]
            mask = pd.DatetimeIndex(dts) >= week
            if not mask.any():
                continue
            idx = int(mask.argmax())
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[idx]), action="enter",
                               conviction=0.55, reason=f"sector flow z={float(row['z']):.1f}",
                               entry_price=float(close.iloc[idx])))
            exit_idx = min(idx + HOLD_WEEKS * 5, len(bars) - 1)
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[exit_idx]), action="exit",
                               conviction=0.5, reason=f"{HOLD_WEEKS}-week hold complete"))
        return sigs
