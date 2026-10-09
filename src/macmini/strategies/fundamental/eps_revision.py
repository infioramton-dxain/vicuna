""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

REV_FLOOR = 0.05
TREND_SMA = 50
MAX_HOLD_DAYS = 30


@register
class EpsRevisionMomentum(Strategy):
    id = 34
    name = "EPS Revision Momentum"
    category = "fundamental"
    data_required = ("mm_bars", "mm_analyst_estimates")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        estimates: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if estimates is None or estimates.empty or len(bars) < TREND_SMA + 1:
            return []

        if isinstance(bars.index, pd.DatetimeIndex):
            bar_dates = bars.index
        else:
            bar_dates = pd.to_datetime(bars["date"])

        close = bars["close"]
        sma_trend = close.rolling(TREND_SMA).mean()

        est = estimates.copy()
        est["as_of_date"] = pd.to_datetime(est["as_of_date"])
        est = est.sort_values("as_of_date")
        rev = est[["as_of_date", "eps_revision_30d"]].dropna()
        if rev.empty:
            return []

        merged = pd.merge_asof(
            pd.DataFrame({"date": bar_dates}).sort_values("date"),
            rev.rename(columns={"as_of_date": "date"}),
            on="date",
            direction="backward",
        )
        revision = merged["eps_revision_30d"].to_numpy()

        signals: list[Signal] = []
        position_open_idx: int | None = None
        for i in range(len(bars)):
            rev_today = revision[i]
            if pd.isna(rev_today):
                continue
            px = float(close.iloc[i])
            sma = float(sma_trend.iloc[i]) if not pd.isna(sma_trend.iloc[i]) else None

            if position_open_idx is None:
                if rev_today > REV_FLOOR and sma is not None and px > sma:
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=pd.Timestamp(bar_dates[i]),
                            action="enter",
                            conviction=min(1.0, 0.5 + min(rev_today, 0.5)),
                            reason=f"EPS revision +{rev_today*100:.1f}% (30d); close > SMA{TREND_SMA}",
                            entry_price=px,
                        )
                    )
                    position_open_idx = i
            else:
                days_held = i - position_open_idx
                if rev_today < 0 or days_held >= MAX_HOLD_DAYS:
                    why = "EPS revision flipped negative" if rev_today < 0 else f"{MAX_HOLD_DAYS}-day max hold"
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=pd.Timestamp(bar_dates[i]),
                            action="exit",
                            conviction=0.5,
                            reason=why,
                        )
                    )
                    position_open_idx = None
        return signals
