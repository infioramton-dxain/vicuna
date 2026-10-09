""
from __future__ import annotations

import importlib
import pkgutil

from macmini.strategies.base import STRATEGY_REGISTRY, Signal, Strategy, register

__all__ = ["STRATEGY_REGISTRY", "Signal", "Strategy", "register"]


def _autoload() -> None:
    ""
    package_path = __path__
    for module_info in pkgutil.walk_packages(package_path, prefix=f"{__name__}."):
        if module_info.name.endswith(".base"):
            continue
        importlib.import_module(module_info.name)


_autoload()
