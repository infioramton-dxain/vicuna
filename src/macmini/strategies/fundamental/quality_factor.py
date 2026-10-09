""
from __future__ import annotations

import numpy as np
import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

ROIC_FLOOR = 0.15
DE_CEILING = 0.5
EPS_STABILITY_MAX = 0.3
LOOKBACK_QUARTERS = 20
FILING_LAG_DAYS = 45


@register
class QualityFactor(Strategy):
    id = 31
    name = "Quality Factor (ROIC + Leverage + EPS Stability)"
    category = "fundamental"
    data_required = ("mm_bars", "mm_fundamentals_quarterly")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if fundamentals is None or fundamentals.empty or len(bars) < 30:
            return []
        f = fundamentals.copy()
        f["period_end"] = pd.to_datetime(f["period_end"])
        f = f.sort_values("period_end")
        f["_visible_from"] = f["period_end"] + pd.Timedelta(days=FILING_LAG_DAYS)
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])

        sigs: list[Signal] = []
        for i in range(len(bars)):
            d = pd.Timestamp(dts[i]).normalize()
            history = f[f["_visible_from"] <= d]
            if len(history) < LOOKBACK_QUARTERS:
                continue
            recent = history.tail(LOOKBACK_QUARTERS)
            roic = recent["roic"].dropna().iloc[-1] if "roic" in recent and recent["roic"].notna().any() else None
            if roic is None:
                last = recent.iloc[-1]
                cap = (last.get("total_debt") or 0) + (last.get("total_equity") or 0)
                roic = (last.get("earnings") / cap) if cap else None
            if roic is None or roic < ROIC_FLOOR:
                continue
            de = None
            last = recent.iloc[-1]
            te, td = last.get("total_equity"), last.get("total_debt")
            if te and te > 0 and td is not None:
                de = td / te
            if de is None or de > DE_CEILING:
                continue
            eps_series = recent["eps"].dropna()
            if len(eps_series) < 8 or eps_series.mean() == 0:
                continue
            stability = eps_series.std() / abs(eps_series.mean())
            if stability > EPS_STABILITY_MAX:
                continue
            if i > 0:
                prev_d = pd.Timestamp(dts[i - 1]).normalize()
                if (history["_visible_from"].max() <= prev_d):
                    continue
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=d,
                    action="enter",
                    conviction=0.6,
                    reason=f"ROIC≈{roic:.2f}, D/E={de:.2f}, EPS σ/μ={stability:.2f}",
                    entry_price=float(bars["close"].iloc[i]),
                )
            )
        return sigs
