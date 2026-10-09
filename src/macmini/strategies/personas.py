""
from __future__ import annotations

TITAN, QUANT, AXE = "Titan", "Quant", "Axe"

_BY_CATEGORY = {"fundamental": TITAN, "sentiment": AXE}


def persona_of(strategy_cls) -> str:
    return _BY_CATEGORY.get(getattr(strategy_cls, "category", ""), QUANT)
