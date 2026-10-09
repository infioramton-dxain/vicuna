""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

EPS_BEAT_PCT = 0.05
HOLD_DAYS = 30
TRAILING_STOP_PCT = 0.10


@register
class Pead(Strategy):
    id = 35
    name = "Post-Earnings Announcement Drift"
    category = "fundamental"
    data_required = ("mm_bars", "mm_earnings_events")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        earnings: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if earnings is None or earnings.empty or len(bars) < 5:
            return []
        e = earnings.copy()
        e["report_date"] = pd.to_datetime(e["report_date"])
        e = e.sort_values("report_date")
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        bar_set = pd.Series(range(len(bars)), index=pd.DatetimeIndex(dts).normalize())

        sigs: list[Signal] = []
        for _, row in e.iterrows():
            sp = row.get("eps_surprise_pct")
            if sp is None or sp < EPS_BEAT_PCT * 100:
                continue
            rev_actual = row.get("revenue_actual")
            rev_est = row.get("revenue_estimate")
            if rev_actual is not None and rev_est is not None and rev_actual <= rev_est:
                continue
            rd = row["report_date"].normalize()
            mask = bar_set.index > rd
            if not mask.any():
                continue
            entry_idx = int(bar_set[mask].iloc[0])
            entry_px = float(close.iloc[entry_idx])
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[entry_idx]),
                    action="enter",
                    conviction=min(1.0, 0.5 + float(sp) / 100),
                    reason=f"EPS surprise +{float(sp):.1f}%",
                    entry_price=entry_px,
                    stop_price=entry_px * (1 - TRAILING_STOP_PCT),
                )
            )
            exit_idx = min(entry_idx + HOLD_DAYS, len(bars) - 1)
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[exit_idx]),
                    action="exit",
                    conviction=0.5,
                    reason=f"{HOLD_DAYS}-day PEAD window end",
                )
            )
        return sigs
