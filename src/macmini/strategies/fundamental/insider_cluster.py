""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

WINDOW_DAYS = 30
MIN_DISTINCT_INSIDERS = 3
MIN_TOTAL_USD = 250_000.0
TREND_SMA = 200
MAX_HOLD_DAYS = 90


@register
class InsiderCluster(Strategy):
    id = 41
    name = "Insider Cluster Buying"
    category = "fundamental"
    data_required = ("mm_bars", "mm_insider_transactions")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        insider_tx: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if insider_tx is None or insider_tx.empty or len(bars) < TREND_SMA + 1:
            return []

        if isinstance(bars.index, pd.DatetimeIndex):
            bar_dates = bars.index
        else:
            bar_dates = pd.to_datetime(bars["date"])

        close = bars["close"]
        sma_trend = close.rolling(TREND_SMA).mean()

        tx = insider_tx.copy()
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
        tx = tx.sort_values("transaction_date")

        buys = tx[tx["transaction_type"] == "P"]
        sales = tx[tx["transaction_type"] == "S"]
        if buys.empty:
            return []

        sale_dates = set(sales["transaction_date"].dt.normalize())

        signals: list[Signal] = []
        position_open_idx: int | None = None
        for i in range(len(bars)):
            d = pd.Timestamp(bar_dates[i]).normalize()
            sma = sma_trend.iloc[i]
            if pd.isna(sma):
                continue
            window_start = d - pd.Timedelta(days=WINDOW_DAYS)
            window = buys[
                (buys["transaction_date"] > window_start)
                & (buys["transaction_date"] <= d)
            ]
            if window.empty:
                pass

            distinct = window["insider_name"].nunique()
            total_usd = float(window["value_usd"].fillna(0.0).sum())
            px = float(close.iloc[i])

            if position_open_idx is None:
                if (
                    distinct >= MIN_DISTINCT_INSIDERS
                    and total_usd >= MIN_TOTAL_USD
                    and px > float(sma)
                ):
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=d,
                            action="enter",
                            conviction=min(1.0, 0.6 + 0.05 * (distinct - MIN_DISTINCT_INSIDERS)),
                            reason=(
                                f"{distinct} insiders bought ${total_usd:,.0f} in last "
                                f"{WINDOW_DAYS}d; close > SMA{TREND_SMA}"
                            ),
                            entry_price=px,
                        )
                    )
                    position_open_idx = i
            else:
                days_held = i - position_open_idx
                if d in sale_dates or days_held >= MAX_HOLD_DAYS:
                    why = (
                        "insider sale recorded"
                        if d in sale_dates
                        else f"{MAX_HOLD_DAYS}-day max hold"
                    )
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=d,
                            action="exit",
                            conviction=0.5,
                            reason=why,
                        )
                    )
                    position_open_idx = None
        return signals
