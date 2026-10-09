""
from __future__ import annotations

import math

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

WINDOW_DAYS = 30
HISTORY_WINDOW = 252
SCORE_PCTILE = 0.9


@register
class InsiderComposite(Strategy):
    id = 47
    name = "Insider Transaction Composite"
    category = "sentiment"
    data_required = ("mm_bars", "mm_insider_transactions")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        insider_tx: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if insider_tx is None or insider_tx.empty or len(bars) < 220:
            return []
        tx = insider_tx.copy()
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
        tx = tx.sort_values("transaction_date")
        buys = tx[tx["transaction_type"] == "P"]
        if buys.empty:
            return []

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        s200 = sma(close, 200)

        def role_weight(title: str | None) -> float:
            if title is None:
                return 1.0
            t = str(title).lower()
            if "ceo" in t or "cfo" in t or "chief executive" in t or "chief financial" in t:
                return 2.0
            return 1.0

        scores = []
        for i in range(len(bars)):
            d = pd.Timestamp(dts[i]).normalize()
            wstart = d - pd.Timedelta(days=WINDOW_DAYS)
            window = buys[(buys["transaction_date"] > wstart) & (buys["transaction_date"] <= d)]
            if window.empty:
                scores.append(0.0); continue
            distinct = window["insider_name"].nunique()
            value = window["value_usd"].fillna(0).sum()
            w = window["insider_title"].apply(role_weight).max() if "insider_title" in window else 1.0
            scores.append(distinct * math.log1p(float(value)) * w)
        scores_s = pd.Series(scores, index=dts)
        pct = scores_s.rolling(HISTORY_WINDOW).rank(pct=True)

        sigs: list[Signal] = []
        for i in range(len(bars)):
            if scores[i] <= 0:
                continue
            if pd.isna(pct.iloc[i]) or pct.iloc[i] < SCORE_PCTILE:
                continue
            if pd.isna(s200.iloc[i]) or close.iloc[i] <= s200.iloc[i]:
                continue
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[i]),
                    action="enter",
                    conviction=0.55,
                    reason=f"insider composite score top {(1-pct.iloc[i])*100:.1f}%",
                    entry_price=float(close.iloc[i]),
                )
            )
        return sigs
