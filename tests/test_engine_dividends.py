""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macmini.backtest.engine import WalkForwardConfig, _dividend_tr_window, _window_stats
from macmini.strategies.base import Signal


def _flat_bars(n=30, px=100.0, start="2025-01-02"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px,
         "adj_close": px, "volume": 1e6},
        index=idx)


def _div_frame(ex_ts, rate):
    return pd.DataFrame({"ex_date": [pd.Timestamp(ex_ts)], "rate": [rate]})


def test_tr_window_noop_without_dividends():
    bars = _flat_bars()
    assert _dividend_tr_window(bars, None) is bars
    assert _dividend_tr_window(bars, pd.DataFrame(columns=["ex_date", "rate"])) is bars
    outside = _div_frame(bars.index[0] - pd.Timedelta(days=30), 1.0)
    assert _dividend_tr_window(bars, outside) is bars


def test_tr_window_scales_from_ex_date():
    bars = _flat_bars()
    ex = bars.index[10]
    out = _dividend_tr_window(bars, _div_frame(ex, 1.0))
    assert out["close"].iloc[9] == pytest.approx(100.0)
    assert out["close"].iloc[10] == pytest.approx(101.0)
    assert out["close"].iloc[-1] == pytest.approx(101.0)
    assert out["open"].iloc[10] == pytest.approx(101.0)
    assert bars["close"].iloc[10] == 100.0


def test_window_stats_total_return_includes_dividend():
    bars = _flat_bars(n=30)
    config = WalkForwardConfig()
    sig = [Signal(symbol="TST", date=bars.index[1], action="enter",
                  conviction=0.5, reason="test")]
    raw = _window_stats(bars, sig, config)
    tr = _window_stats(bars, sig, config,
                       dividends=_div_frame(bars.index[10], 1.0))
    assert raw["n_trades"] == tr["n_trades"] == 1
    uplift = tr["total_return"] - raw["total_return"]
    assert uplift == pytest.approx(0.01, rel=0.05)


def test_window_stats_identical_when_no_dividends():
    bars = _flat_bars(n=30)
    config = WalkForwardConfig()
    sig = [Signal(symbol="TST", date=bars.index[1], action="enter",
                  conviction=0.5, reason="test")]
    a = _window_stats(bars, sig, config)
    b = _window_stats(bars, sig, config, dividends=None)
    assert a == b
