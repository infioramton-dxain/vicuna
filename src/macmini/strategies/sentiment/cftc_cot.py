""
from __future__ import annotations

import pandas as pd

from macmini.strategies._indicators import sma
from macmini.strategies.base import Signal, Strategy, register

LOOKBACK_WEEKS = 52
HOLD_DAYS = 90

SYMBOL_TO_COMMODITY = {
    "NEM": "gold", "AEM": "gold", "GOLD": "gold", "AGI": "gold",
    "AG": "silver", "PAAS": "silver", "WPM": "silver",
    "FCX": "copper", "SCCO": "copper",
}


@register
class CftcCot(Strategy):
    id = 51
    name = "CFTC COT — Commercials at 52w High"
    category = "sentiment"
    data_required = ("mm_bars", "mm_cftc_cot_weekly")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        cot: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        if cot is None or cot.empty or len(bars) < 70:
            return []
        c = cot.copy()
        c["report_date"] = pd.to_datetime(c["report_date"])
        c = c.sort_values("report_date").reset_index(drop=True)
        c["commercial_net_52w_high"] = c["commercial_net"] >= c["commercial_net"].rolling(LOOKBACK_WEEKS).max()
        c["commercial_net_short"] = c["commercial_net"] < 0

        if isinstance(bars.index, pd.DatetimeIndex):
            dts = bars.index
        else:
            dts = pd.to_datetime(bars["date"])
        close = bars["close"]
        s50 = sma(close, 50)

        sigs: list[Signal] = []
        for i, row in c.iterrows():
            if not row["commercial_net_52w_high"]:
                continue
            rd = row["report_date"]
            mask = pd.DatetimeIndex(dts) >= rd
            if not mask.any():
                continue
            idx = int(mask.argmax())
            if pd.isna(s50.iloc[idx]) or close.iloc[idx] <= s50.iloc[idx]:
                continue
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[idx]), action="enter",
                               conviction=0.55, reason="commercial net long at 52w high",
                               entry_price=float(close.iloc[idx])))
            exit_idx = min(idx + HOLD_DAYS, len(bars) - 1)
            sigs.append(Signal(symbol=symbol, date=pd.Timestamp(dts[exit_idx]), action="exit",
                               conviction=0.5, reason=f"{HOLD_DAYS}-day hold complete"))
        return sigs
