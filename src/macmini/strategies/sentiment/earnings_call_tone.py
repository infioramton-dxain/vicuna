""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

TONE_DELTA = 0.3


@register
class EarningsCallToneShift(Strategy):
    id = 44
    name = "Earnings Call Tone Shift"
    category = "sentiment"
    data_required = ("mm_bars", "mm_earnings_events", "sentiment")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        earnings: pd.DataFrame | None = None,
        sentiment_daily: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if earnings is None or earnings.empty or sentiment_daily is None or sentiment_daily.empty or len(bars) < 5:
            return []
        e = earnings.copy()
        e["report_date"] = pd.to_datetime(e["report_date"])
        e = e.sort_values("report_date").reset_index(drop=True)
        s = sentiment_daily.copy()
        s["date"] = pd.to_datetime(s["date"])
        s = s.sort_values("date").set_index("date")

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        bar_set = pd.Series(range(len(bars)), index=pd.DatetimeIndex(dts).normalize())

        def post_tone(report_date: pd.Timestamp) -> float | None:
            window = s.loc[s.index >= report_date]
            window = window[window.index <= report_date + pd.Timedelta(days=1)]
            if window.empty:
                return None
            return float(window["sentiment_mean"].mean())

        sigs: list[Signal] = []
        for i in range(1, len(e)):
            prev_tone = post_tone(e.iloc[i - 1]["report_date"])
            this_tone = post_tone(e.iloc[i]["report_date"])
            if prev_tone is None or this_tone is None:
                continue
            sp = e.iloc[i].get("eps_surprise_pct") or 0
            if (this_tone - prev_tone) <= TONE_DELTA or sp <= 0:
                continue
            rd = e.iloc[i]["report_date"].normalize()
            mask = bar_set.index > rd
            if not mask.any():
                continue
            idx = int(bar_set[mask].iloc[0])
            sigs.append(
                Signal(
                    symbol=symbol,
                    date=pd.Timestamp(dts[idx]),
                    action="enter",
                    conviction=0.6,
                    reason=f"tone Δ +{this_tone-prev_tone:.2f}, EPS surprise +{float(sp):.1f}%",
                    entry_price=float(bars["close"].iloc[idx]),
                )
            )
        return sigs
