""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register
from macmini.strategies.breakout.cup_handle import CupHandle


@register
class CupHandlePattern(Strategy):
    id = 21
    name = "Cup-with-Handle (pattern view)"
    category = "pattern"
    data_required = ("mm_bars",)

    def generate_signals(self, symbol: str, bars: pd.DataFrame, **kwargs) -> list[Signal]:
        return [Signal(**{**s.__dict__, "reason": f"[pattern] {s.reason}"}) for s in CupHandle().generate_signals(symbol, bars, **kwargs)]
