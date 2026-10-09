""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

SENTIMENT_FLOOR = 0.6
MATERIALITY_FLOOR = 0.5
PRICE_CONFIRM_PCT = 0.02
HOLD_DAYS = 5


@register
class NewsSentimentShock(Strategy):
    id = 43
    name = "News Sentiment Shock + Price Confirmation"
    category = "sentiment"
    data_required = ("mm_bars", "sentiment")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        sentiment_daily: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if sentiment_daily is None or sentiment_daily.empty or len(bars) < 2:
            return []

        if isinstance(bars.index, pd.DatetimeIndex):
            bar_dates = pd.Series(bars.index)
        else:
            bar_dates = pd.to_datetime(bars["date"])

        close = bars["close"].reset_index(drop=True)
        prior_close = close.shift(1)

        sd = sentiment_daily.copy()
        sd["date"] = pd.to_datetime(sd["date"])
        sd = sd.set_index("date")[["sentiment_mean", "materiality_max"]]

        aligned = sd.reindex(pd.to_datetime(bar_dates.values))

        signals: list[Signal] = []
        position_open_idx: int | None = None
        entry_close: float | None = None
        for i in range(len(bars)):
            px = float(close.iloc[i])
            prev = prior_close.iloc[i]
            sent = aligned["sentiment_mean"].iloc[i]
            mat = aligned["materiality_max"].iloc[i]

            if position_open_idx is None:
                if (
                    not pd.isna(prev)
                    and not pd.isna(sent)
                    and not pd.isna(mat)
                    and sent > SENTIMENT_FLOOR
                    and mat > MATERIALITY_FLOOR
                    and px >= float(prev) * (1 + PRICE_CONFIRM_PCT)
                ):
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=pd.Timestamp(bar_dates.iloc[i]),
                            action="enter",
                            conviction=min(1.0, 0.5 + float(sent - SENTIMENT_FLOOR)),
                            reason=(
                                f"sentiment {float(sent):.2f}, materiality {float(mat):.2f}, "
                                f"day +{(px/float(prev)-1)*100:.1f}%"
                            ),
                            entry_price=px,
                        )
                    )
                    position_open_idx = i
                    entry_close = px
            else:
                days_held = i - position_open_idx
                if days_held >= HOLD_DAYS or (entry_close is not None and px < entry_close):
                    why = (
                        f"{HOLD_DAYS}-day time stop"
                        if days_held >= HOLD_DAYS
                        else f"close {px:.2f} below entry {entry_close:.2f}"
                    )
                    signals.append(
                        Signal(
                            symbol=symbol,
                            date=pd.Timestamp(bar_dates.iloc[i]),
                            action="exit",
                            conviction=0.5,
                            reason=why,
                        )
                    )
                    position_open_idx = None
                    entry_close = None
        return signals
