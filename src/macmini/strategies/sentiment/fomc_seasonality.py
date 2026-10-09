""
from __future__ import annotations

from datetime import date

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register

FOMC_DATES: tuple[date, ...] = (
    date(2020, 1, 29), date(2020, 3, 18), date(2020, 4, 29), date(2020, 6, 10),
    date(2020, 7, 29), date(2020, 9, 16), date(2020, 11, 5), date(2020, 12, 16),
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
    date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4), date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3), date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 11, 5), date(2025, 12, 17),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 11, 4), date(2026, 12, 16),
)


@register
class FomcSeasonality(Strategy):
    id = 52
    name = "FOMC / Macro-Event Seasonality"
    category = "sentiment"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        if len(bars) < 5:
            return []
        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        bar_set = pd.DatetimeIndex(dts).normalize()

        sigs: list[Signal] = []
        for f in FOMC_DATES:
            f_ts = pd.Timestamp(f)
            before = bar_set < f_ts
            if not before.any():
                continue
            entry_idx = int(before.argmax() if (~before).any() else len(bar_set) - 1)
            entry_idx = int(pd.Series(before).cumsum().idxmax())
            on_or_after = bar_set >= f_ts
            if not on_or_after.any():
                continue
            exit_idx = int(on_or_after.argmax())
            if exit_idx <= entry_idx:
                continue
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[entry_idx]), action="enter",
                               conviction=0.5, reason=f"FOMC seasonality, decision {f.isoformat()}",
                               entry_price=float(close.iloc[entry_idx])))
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[exit_idx]), action="exit",
                               conviction=0.5, reason="FOMC day close"))
        return sigs
