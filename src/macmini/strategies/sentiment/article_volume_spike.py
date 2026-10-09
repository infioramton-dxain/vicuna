""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma, zscore
from macmini.strategies.base import Signal, Strategy, register

Z_FLOOR = 3.0
SENT_FLOOR = 0.5
TIME_STOP = 3


@register
class ArticleVolumeSpike(Strategy):
    id = 48
    name = "Article-Volume Sentiment Spike"
    category = "sentiment"
    data_required = ("mm_bars", "sentiment")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        sentiment_daily: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if sentiment_daily is None or sentiment_daily.empty or len(bars) < 40:
            return []
        s = sentiment_daily.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.set_index("date").sort_index()
        s["article_z"] = zscore(s["article_count_sum"], 30)

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        s20 = sma(close, 20)

        aligned = s.reindex(pd.to_datetime(dts))

        sigs: list[Signal] = []
        open_idx: int | None = None
        for i in range(len(bars)):
            z = aligned["article_z"].iloc[i]
            sm = aligned["sentiment_mean"].iloc[i]
            if open_idx is None:
                if not pd.isna(z) and z > Z_FLOOR and not pd.isna(sm) and sm > SENT_FLOOR and not pd.isna(s20.iloc[i]) and close.iloc[i] > s20.iloc[i]:
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="enter",
                                       conviction=0.5, reason=f"article z={float(z):.1f}, sentiment={float(sm):.2f}",
                                       entry_price=float(close.iloc[i])))
                    open_idx = i
            else:
                if i - open_idx >= TIME_STOP or (not pd.isna(z) and z < 1):
                    sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[i]), action="exit",
                                       conviction=0.5,
                                       reason="article volume normalized" if (not pd.isna(z) and z < 1) else f"{TIME_STOP}d stop"))
                    open_idx = None
        return sigs
