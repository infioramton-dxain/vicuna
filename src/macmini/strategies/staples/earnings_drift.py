""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.staples._universe import STAPLES, bar_dates

MIN_SURPRISE_PCT = 2.0
HOLD_DAYS = 8


@register
class StaplesEarningsDrift(Strategy):
    id = 64
    name = "Staples Earnings-Stability Drift (PEAD)"
    category = "pattern"
    data_required = ("mm_bars", "mm_earnings_events", "mm_sector_map")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        earnings: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if symbol not in STAPLES or earnings is None or earnings.empty or len(bars) < 20:
            return []
        e = earnings.copy()
        e["report_date"] = pd.to_datetime(e["report_date"])
        e = e.sort_values("report_date")
        beats = e[e["eps_surprise_pct"].astype(float) >= MIN_SURPRISE_PCT]
        if beats.empty:
            return []

        close = bars["close"]
        dts = bar_dates(bars)

        sigs: list[Signal] = []
        last_exit = -1
        for _, row in beats.iterrows():
            mask = dts >= row["report_date"]
            if not mask.any():
                continue
            idx = int(mask.argmax())
            if idx <= last_exit:
                continue
            sigs.append(Signal(
                symbol=symbol, date=pd.Timestamp(dts[idx]), action="enter",
                conviction=min(0.7, 0.55 + float(row["eps_surprise_pct"]) / 100.0),
                reason=f"EPS beat +{float(row['eps_surprise_pct']):.1f}%",
                entry_price=float(close.iloc[idx])))
            exit_idx = min(idx + HOLD_DAYS, len(bars) - 1)
            sigs.append(Signal(
                symbol=symbol, date=pd.Timestamp(dts[exit_idx]), action="exit",
                conviction=0.5, reason=f"{HOLD_DAYS}d drift hold"))
            last_exit = exit_idx
        return sigs
