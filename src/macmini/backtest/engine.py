""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from macmini.backtest import exec_variants
from macmini.backtest.loader import slice_aux
from macmini.backtest.metrics import (
    max_drawdown,
    profit_factor,
    sharpe_pvalue,
    sharpe_ratio,
)
from macmini.strategies.base import Signal, Strategy

PERIODS_PER_YEAR = 252


@dataclass
class WalkForwardConfig:
    ""

    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21
    min_history_days: int = 504
    transaction_cost_bps: int = 5
    slippage_bps: int = 10


def signals_to_arrays(
    signals: list[Signal],
    bars: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    ""
    idx = bars.index if isinstance(bars.index, pd.DatetimeIndex) else pd.to_datetime(bars["date"])
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    for s in signals:
        d = pd.Timestamp(s.date).normalize()
        try:
            pos = idx.searchsorted(d)
        except Exception:
            continue
        if pos >= len(idx):
            continue
        if s.action == "enter":
            entries.iloc[pos] = True
        elif s.action == "exit":
            exits.iloc[pos] = True
    return entries, exits


def _dividend_tr_window(bars_window: pd.DataFrame,
                        dividends: pd.DataFrame | None) -> pd.DataFrame:
    ""
    if dividends is None or dividends.empty:
        return bars_window
    idx = bars_window.index
    divs = dividends[(dividends["ex_date"] > idx[0]) & (dividends["ex_date"] <= idx[-1])]
    if divs.empty:
        return bars_window
    factor = pd.Series(1.0, index=idx)
    closes = bars_window["close"]
    for _, row in divs.iterrows():
        pos = idx.searchsorted(row["ex_date"])
        if pos <= 0 or pos >= len(idx):
            continue
        prev_close = closes.iloc[pos - 1]
        if prev_close and prev_close > 0 and row["rate"]:
            factor.iloc[pos:] *= 1.0 + float(row["rate"]) / float(prev_close)
    if (factor == 1.0).all():
        return bars_window
    out = bars_window.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    return out


def _window_stats(
    bars_window: pd.DataFrame,
    signals: list[Signal],
    config: WalkForwardConfig,
    entry_type: str = "market_open",
    exit_type: str = "signal",
    dividends: pd.DataFrame | None = None,
) -> dict:
    ""
    if len(bars_window) < 5:
        return _empty_stats(bars_window.index[0], bars_window.index[-1])
    bars_window = _dividend_tr_window(bars_window, dividends)
    entries, exits = signals_to_arrays(signals, bars_window)
    exits = exits.shift(1, fill_value=False)
    if entry_type == "market_open":
        entries = entries.shift(1, fill_value=False)
        price = bars_window["open"]
    else:
        entries, price = exec_variants.apply_entry_model(
            entries, bars_window, entry_type
        )
    close = bars_window["close"]
    if not entries.any():
        return _empty_stats(bars_window.index[0], bars_window.index[-1])

    overlay = exec_variants.exit_overlay_kwargs(exit_type, bars_window)
    if exit_type == "time_stop":
        exits = exec_variants.time_stop_exits(entries, exits)
    elif exit_type in exec_variants.WALKER_EXIT_TYPES:
        entries, exits, price = exec_variants.walker_exits(
            entries, exits, bars_window, exit_type, price
        )
        if not entries.any():
            return _empty_stats(bars_window.index[0], bars_window.index[-1])

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        price=price,
        fees=config.transaction_cost_bps / 10_000,
        slippage=config.slippage_bps / 10_000,
        init_cash=10_000,
        freq="1D",
        **overlay,
    )
    trades = pf.trades.records_readable
    returns = pf.returns().to_numpy()
    if len(trades) == 0:
        return {
            "window_start": bars_window.index[0].strftime("%Y-%m-%d"),
            "window_end": bars_window.index[-1].strftime("%Y-%m-%d"),
            "n_trades": 0, "n_returns": int(len(returns)),
            "sharpe": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "hit_rate": 0.0, "total_return": 0.0,
            "avg_holding_days": 0.0, "skew": 0.0, "kurt": 3.0,
            "p_value": 1.0,
        }
    trade_returns = trades["Return"].to_numpy()
    hit_rate = float((trade_returns > 0).mean())
    avg_hold = float(trades["Duration"].mean()) if "Duration" in trades.columns else 0.0
    sr = sharpe_ratio(returns, PERIODS_PER_YEAR)
    pf_val = profit_factor(trade_returns)
    dd = max_drawdown(pf.value().to_numpy())
    total_ret = float(pf.total_return())
    ret_ser = pd.Series(returns)
    skew = float(ret_ser.skew()) if len(returns) > 2 else 0.0
    kurt = float(ret_ser.kurtosis()) + 3.0 if len(returns) > 3 else 3.0
    p = sharpe_pvalue(sr, len(returns))
    return {
        "window_start": bars_window.index[0].strftime("%Y-%m-%d"),
        "window_end": bars_window.index[-1].strftime("%Y-%m-%d"),
        "n_trades": int(len(trades)),
        "n_returns": int(len(returns)),
        "sharpe": float(sr),
        "profit_factor": min(float(pf_val), 20.0) if np.isfinite(pf_val) else 20.0,
        "max_drawdown": float(dd),
        "hit_rate": float(hit_rate),
        "total_return": float(total_ret),
        "avg_holding_days": float(avg_hold),
        "skew": float(skew),
        "kurt": float(kurt),
        "p_value": float(p),
    }


def _empty_stats(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return {
        "window_start": start.strftime("%Y-%m-%d"),
        "window_end": end.strftime("%Y-%m-%d"),
        "n_trades": 0, "n_returns": 0,
        "sharpe": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "hit_rate": 0.0, "total_return": 0.0,
        "avg_holding_days": 0.0, "skew": 0.0, "kurt": 3.0, "p_value": 1.0,
    }


def walk_forward_backtest(
    strategy: Strategy,
    symbol: str,
    bars: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
    aux: dict | None = None,
    entry_type: str = "market_open",
    exit_type: str = "signal",
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ""
    if aux is None:
        aux = {}
    bars = bars.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.get("date", bars.index))
    n = len(bars)
    if n < config.min_history_days:
        return pd.DataFrame()

    train, test, step = config.train_window_days, config.test_window_days, config.step_days
    rows: list[dict] = []
    start = config.min_history_days
    while start + test <= n:
        test_end = start + test
        slice_bars = bars.iloc[:test_end]
        window_aux = slice_aux(aux, slice_bars.index[-1])
        try:
            signals = strategy.generate_signals(symbol, slice_bars, **window_aux)
        except Exception:
            signals = []
        test_window = bars.iloc[start:test_end]
        test_start_ts = test_window.index[0]
        test_end_ts = test_window.index[-1]
        test_signals = [
            s for s in signals
            if pd.Timestamp(s.date) >= test_start_ts and pd.Timestamp(s.date) <= test_end_ts
        ]
        rows.append(_window_stats(
            test_window, test_signals, config,
            entry_type=entry_type, exit_type=exit_type,
            dividends=dividends,
        ))
        start += step

    return pd.DataFrame(rows)
