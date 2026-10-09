""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

UPGRADE_FLOOR = 3
WINDOW_DAYS = 5
HOLD_DAYS = 30


@register
class AnalystCascade(Strategy):
    id = 46
    name = "Analyst Rating Cascade"
    category = "sentiment"
    data_required = ("mm_bars", "mm_analyst_estimates", "sentiment")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        estimates: pd.DataFrame | None = None,
        sentiment_daily: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if estimates is None or estimates.empty or len(bars) < 70:
            return []
        est = estimates.copy()
        est["as_of_date"] = pd.to_datetime(est["as_of_date"])
        est = est.sort_values("as_of_date").reset_index(drop=True)
        if len(est) < 2:
            return []
        est["buy_delta"] = est["rating_buy"].diff().fillna(0)
        est["sell_delta"] = est["rating_sell"].diff().fillna(0)

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        s50 = sma(close, 50)

        sigs: list[Signal] = []
        for _, row in est.iterrows():
            if row["buy_delta"] < UPGRADE_FLOOR or row["sell_delta"] > 0:
                continue
            d = row["as_of_date"]
            mask = pd.DatetimeIndex(dts) >= d
            if not mask.any():
                continue
            idx = int(mask.argmax())
            if pd.isna(s50.iloc[idx]) or close.iloc[idx] <= s50.iloc[idx]:
                continue
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[idx]),
                    action="enter",
                    conviction=0.6,
                    reason=f"+{int(row['buy_delta'])} analyst upgrades, no downgrades",
                    entry_price=float(close.iloc[idx]),
                )
            )
            exit_idx = min(idx + HOLD_DAYS, len(bars) - 1)
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[exit_idx]),
                    action="exit",
                    conviction=0.5,
                    reason=f"{HOLD_DAYS}-day hold complete",
                )
            )
        return sigs
