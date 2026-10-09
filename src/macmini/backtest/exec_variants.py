""
from __future__ import annotations

import numpy as np
import pandas as pd

from macmini.strategies._indicators import atr, macd, rsi

ENTRY_TYPES = ("market_open", "limit", "stop", "stop_limit")
EXIT_TYPES = (
    "signal", "bracket", "trail", "trail_tp", "time_stop",
    "trail_reentry", "ratchet_trail", "ratchet_trail_reentry",
)
WALKER_EXIT_TYPES = ("trail_reentry", "ratchet_trail", "ratchet_trail_reentry")

REFERENCE_VARIANT = ("market_open", "bracket")
BASELINE_VARIANT = ("market_open", "signal")

ENTRY_TTL = 3
STOP_LIMIT_CAP = 0.005
TIME_STOP_HORIZON = 20

RE_ENTRY_TTL = 10
MAX_REENTRIES = 2
RATCHET_TIGHTEN = 0.5

STOP_ATR_MULT = 3.0
STOP_FRAC_MIN, STOP_FRAC_MAX = 0.05, 0.15
TARGET_R_MULT = 2.0


def variant_grid() -> list[tuple[str, str]]:
    ""
    return [(e, x) for e in ENTRY_TYPES for x in EXIT_TYPES if x != "signal"]


def stop_frac_series(bars: pd.DataFrame) -> pd.Series:
    ""
    a = atr(bars["high"], bars["low"], bars["close"], 14)
    frac = (STOP_ATR_MULT * a / bars["close"]).clip(STOP_FRAC_MIN, STOP_FRAC_MAX)
    return frac.fillna(0.08)


def apply_entry_model(
    entries: pd.Series,
    bars: pd.DataFrame,
    entry_type: str,
    ttl: int = ENTRY_TTL,
) -> tuple[pd.Series, pd.Series]:
    ""
    if entry_type == "market_open":
        raise ValueError("market_open entries are handled in engine.py")
    if entry_type not in ENTRY_TYPES:
        raise ValueError(f"unknown entry_type {entry_type!r}")

    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    raw = entries.to_numpy(dtype=bool)
    n = len(raw)

    filled = np.zeros(n, dtype=bool)
    price = opens.copy()

    for t in np.flatnonzero(raw):
        if entry_type == "limit":
            level = closes[t]
        else:
            level = highs[t]
        cap = level * (1.0 + STOP_LIMIT_CAP)
        triggered = False
        for k in range(t + 1, min(t + 1 + ttl, n)):
            fill = None
            if entry_type == "limit":
                if opens[k] <= level:
                    fill = opens[k]
                elif lows[k] <= level:
                    fill = level
            elif entry_type == "stop":
                if opens[k] >= level:
                    fill = opens[k]
                elif highs[k] >= level:
                    fill = level
            else:
                if not triggered:
                    if opens[k] >= level:
                        triggered = True
                        if opens[k] <= cap:
                            fill = opens[k]
                    elif highs[k] >= level:
                        triggered = True
                        fill = level
                if triggered and fill is None:
                    if opens[k] <= cap:
                        fill = opens[k]
                    elif lows[k] <= cap:
                        fill = cap
            if fill is not None:
                if not filled[k]:
                    filled[k] = True
                    price[k] = fill
                break

    return (
        pd.Series(filled, index=entries.index),
        pd.Series(price, index=entries.index),
    )


def time_stop_exits(
    filled_entries: pd.Series,
    signal_exits: pd.Series,
    horizon: int = TIME_STOP_HORIZON,
) -> pd.Series:
    ""
    fe = filled_entries.to_numpy(dtype=bool)
    se = signal_exits.to_numpy(dtype=bool)
    n = len(fe)
    out = se.copy()
    in_pos = False
    deadline = -1
    for k in range(n):
        if in_pos and (se[k] or k >= deadline):
            out[k] = True
            in_pos = False
        if not in_pos and fe[k] and not out[k]:
            in_pos = True
            deadline = k + horizon
    return pd.Series(out, index=filled_entries.index)


def momentum_decel(bars: pd.DataFrame) -> pd.Series:
    ""
    r = rsi(bars["close"], 14)
    _, _, hist = macd(bars["close"])
    rsi_roll = (r.shift(2) > 65) & (r < r.shift(1)) & (r.shift(1) < r.shift(2))
    hist_shrink = (
        (hist < hist.shift(1))
        & (hist.shift(1) < hist.shift(2))
        & (hist.shift(2) > 0)
    )
    return (rsi_roll | hist_shrink).fillna(False)


def walker_exits(
    filled_entries: pd.Series,
    signal_exits: pd.Series,
    bars: pd.DataFrame,
    exit_type: str,
    price: pd.Series,
    max_reentries: int = MAX_REENTRIES,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ""
    if exit_type not in WALKER_EXIT_TYPES:
        raise ValueError(f"unknown walker exit_type {exit_type!r}")
    reentry_enabled = exit_type in ("trail_reentry", "ratchet_trail_reentry")
    ratchet_enabled = exit_type in ("ratchet_trail", "ratchet_trail_reentry")

    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    frac_all = stop_frac_series(bars).to_numpy(dtype=float)
    decel = (
        momentum_decel(bars).to_numpy(dtype=bool)
        if ratchet_enabled
        else np.zeros(len(bars), dtype=bool)
    )

    fe = filled_entries.to_numpy(dtype=bool)
    se = signal_exits.to_numpy(dtype=bool)
    in_price = price.to_numpy(dtype=float)
    n = len(fe)

    out_entries = np.zeros(n, dtype=bool)
    out_exits = np.zeros(n, dtype=bool)
    out_price = in_price.copy()

    in_pos = False
    watching = False
    hwm = trail = frac0 = 0.0
    reentries_left = 0
    watch_level = 0.0
    watch_deadline = -1
    pending_reentry = False

    for k in range(n):
        d = decel[k - 1] if k > 0 else False
        if in_pos:
            if se[k]:
                out_exits[k] = True
                out_price[k] = opens[k]
                in_pos = False
                continue
            eff = frac0 * (RATCHET_TIGHTEN if d else 1.0)
            fill = None
            trail = max(trail, hwm * (1.0 - eff))
            if opens[k] <= trail:
                fill = opens[k]
            else:
                hwm = max(hwm, highs[k])
                trail = max(trail, hwm * (1.0 - eff))
                if lows[k] <= trail:
                    fill = trail
            if fill is not None:
                out_exits[k] = True
                out_price[k] = fill
                in_pos = False
                if reentry_enabled and reentries_left > 0:
                    watching = True
                    watch_level = fill
                    watch_deadline = k + RE_ENTRY_TTL
            continue

        if pending_reentry:
            pending_reentry = False
            if not fe[k]:
                out_entries[k] = True
                out_price[k] = opens[k]
                in_pos = True
                watching = False
                reentries_left -= 1
                hwm = max(opens[k], highs[k])
                trail = hwm * (1.0 - frac0 * (RATCHET_TIGHTEN if d else 1.0))
                continue
        if fe[k]:
            out_entries[k] = True
            out_price[k] = in_price[k]
            in_pos = True
            watching = False
            frac0 = frac_all[k]
            reentries_left = max_reentries
            hwm = max(in_price[k], highs[k])
            trail = hwm * (1.0 - frac0 * (RATCHET_TIGHTEN if d else 1.0))
            continue
        if watching:
            if se[k] or k > watch_deadline:
                watching = False
            elif closes[k] > watch_level:
                pending_reentry = True
                watching = False

    return (
        pd.Series(out_entries, index=filled_entries.index),
        pd.Series(out_exits, index=filled_entries.index),
        pd.Series(out_price, index=filled_entries.index),
    )


def exit_overlay_kwargs(exit_type: str, bars: pd.DataFrame) -> dict:
    ""
    if exit_type in ("signal", "time_stop") or exit_type in WALKER_EXIT_TYPES:
        return {}
    if exit_type not in EXIT_TYPES:
        raise ValueError(f"unknown exit_type {exit_type!r}")
    from vectorbt.portfolio.enums import StopExitPrice

    frac = stop_frac_series(bars)
    kwargs: dict = {
        "open": bars["open"],
        "high": bars["high"],
        "low": bars["low"],
        "sl_stop": frac.to_numpy(),
        "stop_exit_price": StopExitPrice.StopMarket,
    }
    if exit_type in ("trail", "trail_tp"):
        kwargs["sl_trail"] = True
    if exit_type in ("bracket", "trail_tp"):
        kwargs["tp_stop"] = (TARGET_R_MULT * frac).to_numpy()
    return kwargs
