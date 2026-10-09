""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import pandas as pd


@dataclass
class Signal:
    ""

    symbol: str
    date: pd.Timestamp
    action: str
    conviction: float
    reason: str
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None


STRATEGY_REGISTRY: dict[int, type["Strategy"]] = {}


def register(cls: type["Strategy"]) -> type["Strategy"]:
    ""
    if not hasattr(cls, "id") or cls.id is None:
        raise TypeError(f"{cls.__name__} must declare a class-level `id`")
    if cls.id in STRATEGY_REGISTRY and STRATEGY_REGISTRY[cls.id] is not cls:
        existing = STRATEGY_REGISTRY[cls.id].__name__
        raise ValueError(
            f"duplicate strategy id {cls.id}: {existing} vs {cls.__name__}"
        )
    STRATEGY_REGISTRY[cls.id] = cls
    return cls


class Strategy(ABC):
    ""

    id: ClassVar[int]
    name: ClassVar[str]
    category: ClassVar[str]
    timeframe: ClassVar[str] = "swing"
    data_required: ClassVar[tuple[str, ...]] = ("mm_bars",)

    @abstractmethod
    def generate_signals(
        self,
        symbol: str,
        bars: pd.DataFrame,
        **kwargs,
    ) -> list[Signal]:
        ""
