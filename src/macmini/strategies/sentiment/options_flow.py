""
from __future__ import annotations

import pandas as pd

from macmini.strategies.base import Signal, Strategy, register


@register
class OptionsFlow(Strategy):
    id = 49
    name = "Unusual Call Volume (Options Flow)"
    category = "sentiment"
    data_required = ("mm_bars", "mm_options_flow")

    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        options_flow: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[Signal]:
        return []
