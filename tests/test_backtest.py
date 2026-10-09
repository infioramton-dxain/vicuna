""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from macmini.backtest.engine import WalkForwardConfig, walk_forward_backtest, signals_to_arrays
from macmini.backtest.loader import load_bars
from macmini.backtest.metrics import (
    bonferroni_pvalue,
    deflated_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
)
from macmini.config import DB_PATH
from macmini.strategies import STRATEGY_REGISTRY
from macmini.strategies.base import Signal


class TestMetrics:
    def test_sharpe_constant_returns_is_finite(self) -> None:
        r = np.full(252, 0.001)
        sr = sharpe_ratio(r)
        assert sr == 0.0

    def test_sharpe_positive_drift_is_positive(self) -> None:
        rng = np.random.default_rng(42)
        r = 0.0005 + rng.normal(0, 0.01, 504)
        assert sharpe_ratio(r) > 0

    def test_profit_factor_handles_no_losses(self) -> None:
        assert profit_factor(np.array([0.01, 0.02])) == float("inf")
        assert profit_factor(np.array([0.0, 0.0])) == 0.0

    def test_bonferroni_clamps_at_1(self) -> None:
        assert bonferroni_pvalue(0.5, 55) == 1.0
        assert bonferroni_pvalue(0.001, 55) == pytest.approx(0.055)

    def test_deflated_sharpe_zero_when_no_returns(self) -> None:
        assert deflated_sharpe_ratio(1.0, 0, 55, 0, 0) == 0.0

    def test_deflated_sharpe_in_unit_interval(self) -> None:
        ds = deflated_sharpe_ratio(2.0, 252, 55, -0.1, 3.0)
        assert 0.0 <= ds <= 1.0


class TestEngine:
    @pytest.fixture(scope="class")
    def aapl_bars(self) -> pd.DataFrame:
        return load_bars("AAPL", db_path=DB_PATH)

    def test_signals_to_arrays_alignment(self, aapl_bars: pd.DataFrame) -> None:
        ts0 = aapl_bars.index[100]
        ts1 = aapl_bars.index[200]
        ts2 = aapl_bars.index[300]
        sigs = [
            Signal("AAPL", ts0, "enter", 0.5, "x", entry_price=10.0),
            Signal("AAPL", ts1, "exit", 0.5, "y"),
            Signal("AAPL", ts2, "enter", 0.5, "z", entry_price=12.0),
        ]
        entries, exits = signals_to_arrays(sigs, aapl_bars)
        assert entries.sum() == 2
        assert exits.sum() == 1
        assert entries.iloc[100] and entries.iloc[300]
        assert exits.iloc[200]

    def test_walk_forward_schedule(self, aapl_bars: pd.DataFrame) -> None:
        cfg = WalkForwardConfig()
        cls = STRATEGY_REGISTRY[3]
        df = walk_forward_backtest(cls(), "AAPL", aapl_bars, config=cfg)
        assert not df.empty
        first_start_pos = aapl_bars.index.get_loc(pd.Timestamp(df.iloc[0]["window_start"]))
        assert first_start_pos == cfg.min_history_days
        if len(df) >= 2:
            second_start_pos = aapl_bars.index.get_loc(pd.Timestamp(df.iloc[1]["window_start"]))
            assert second_start_pos - first_start_pos == cfg.step_days

    def test_columns_present(self, aapl_bars: pd.DataFrame) -> None:
        cls = STRATEGY_REGISTRY[3]
        df = walk_forward_backtest(cls(), "AAPL", aapl_bars)
        expected = {"window_start", "window_end", "n_trades", "sharpe", "profit_factor",
                    "max_drawdown", "hit_rate", "total_return", "avg_holding_days",
                    "skew", "kurt", "p_value"}
        assert expected.issubset(df.columns)


class TestCanonicalRuns:
    ""

    def test_table_has_rows(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_backtest_runs").fetchone()[0]
        assert n >= 1000, f"mm_backtest_runs too sparse — only {n} rows; did Phase 5 smoke run?"

    def test_deflated_sharpe_and_bonferroni_populated(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            row = c.execute(
                "SELECT COUNT(deflated_sharpe), COUNT(p_value_bonferroni) FROM mm_backtest_runs"
            ).fetchone()
        ds_n, pb_n = row
        assert ds_n > 0 and pb_n > 0

    def test_metrics_in_plausible_ranges(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            df = pd.read_sql_query(
                "SELECT sharpe, deflated_sharpe, p_value, p_value_bonferroni, max_drawdown "
                "FROM mm_backtest_runs",
                c,
            )
        assert df["sharpe"].between(-20, 20).all(), "Sharpe outside sane range"
        assert df["deflated_sharpe"].between(0, 1).all(), "DSR must be a probability"
        assert df["p_value"].between(0, 1).all()
        assert df["p_value_bonferroni"].between(0, 1).all()
        assert df["max_drawdown"].between(-1, 0).all(), "DD must be in [-1, 0]"


def _flat_bars(n: int = 40, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price,
         "adj_close": price, "volume": 1_000_000},
        index=idx,
    )


class TestNextBarOpenFills:
    ""

    def test_entry_fills_at_next_open(self) -> None:
        from macmini.backtest.engine import _window_stats

        bars = _flat_bars(40)
        bars.iloc[4, bars.columns.get_loc("open")] = 50.0
        sig = [Signal("TEST", bars.index[3], "enter", 0.5, "x", entry_price=100.0)]
        row = _window_stats(bars, sig, WalkForwardConfig())
        assert row["n_trades"] == 1
        assert row["total_return"] > 0.5, (
            f"expected ~+100% from next-open fill, got {row['total_return']:.4f}"
        )

    def test_last_bar_signal_never_fills(self) -> None:
        from macmini.backtest.engine import _window_stats

        bars = _flat_bars(40)
        sig = [Signal("TEST", bars.index[-1], "enter", 0.5, "x", entry_price=100.0)]
        row = _window_stats(bars, sig, WalkForwardConfig())
        assert row["n_trades"] == 0


class TestDsrInputs:
    ""

    def test_window_stats_carry_n_returns_and_raw_kurt(self) -> None:
        from macmini.backtest.engine import _window_stats

        bars = _flat_bars(40)
        rng = np.random.default_rng(7)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(bars)))
        bars["close"] = closes
        bars["open"] = np.roll(closes, 1); bars.iloc[0, bars.columns.get_loc("open")] = 100.0
        sig = [Signal("TEST", bars.index[3], "enter", 0.5, "x", entry_price=100.0)]
        row = _window_stats(bars, sig, WalkForwardConfig())
        assert row["n_returns"] == len(bars)
        assert row["kurt"] > 1.0, f"kurt={row['kurt']} looks like excess, not raw"

    def test_zero_trade_row_defaults_to_normal_kurt(self) -> None:
        from macmini.backtest.engine import _empty_stats

        row = _empty_stats(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-31"))
        assert row["kurt"] == 3.0 and row["n_returns"] == 0

    def test_bigger_trial_count_is_stricter(self) -> None:
        common = dict(sharpe=2.0, n_returns=63, skew=0.0, kurt=3.0)
        lenient = deflated_sharpe_ratio(n_trials=55, **common)
        strict = deflated_sharpe_ratio(n_trials=21_670, **common)
        assert strict < lenient


class TestSliceAux:
    ""

    def test_fundamentals_get_45d_filing_lag(self) -> None:
        from macmini.backtest.loader import slice_aux

        fundamentals = pd.DataFrame({
            "period_end": ["2026-03-31", "2025-12-31", "2025-09-30"],
            "eps": [1.0, 2.0, 3.0],
        })
        out = slice_aux({"fundamentals": fundamentals}, "2026-04-30")
        assert len(out["fundamentals"]) == 2
        assert "2026-03-31" not in out["fundamentals"]["period_end"].values
        out = slice_aux({"fundamentals": fundamentals}, "2026-05-16")
        assert len(out["fundamentals"]) == 3

    def test_estimates_are_same_day_visible(self) -> None:
        from macmini.backtest.loader import slice_aux

        estimates = pd.DataFrame({
            "as_of_date": ["2026-06-01"], "eps_revision_30d": [0.07],
        })
        out = slice_aux({"estimates": estimates}, "2026-06-01")
        assert len(out["estimates"]) == 1

    def test_future_rows_never_visible(self) -> None:
        from macmini.backtest.loader import slice_aux

        estimates = pd.DataFrame({
            "as_of_date": ["2026-06-01"], "eps_revision_30d": [0.07],
        })
        out = slice_aux({"estimates": estimates}, "2026-05-31")
        assert out["estimates"].empty

    def test_unknown_and_empty_frames_pass_through(self) -> None:
        from macmini.backtest.loader import slice_aux

        empty = pd.DataFrame()
        weird = pd.DataFrame({"foo": [1]})
        out = slice_aux({"short_interest": empty, "mystery": weird}, "2026-01-01")
        assert out["short_interest"] is empty
        assert out["mystery"] is weird

    def test_dirty_date_values_dont_crash(self) -> None:
        from macmini.backtest.loader import slice_aux

        insider = pd.DataFrame({
            "transaction_date": ["2026-05-22-05:00", "2026-05-01", "garbage"],
            "shares": [1, 2, 3],
        })
        out = slice_aux({"insider_tx": insider}, "2026-06-01")
        assert list(out["insider_tx"]["shares"]) == [1, 2]

    def test_datetime_suffixed_period_end_parses(self) -> None:
        from macmini.backtest.loader import slice_aux

        fundamentals = pd.DataFrame({
            "period_end": ["2025-12-31 00:00:00"], "eps": [1.0],
        })
        out = slice_aux({"fundamentals": fundamentals}, "2026-03-01")
        assert len(out["fundamentals"]) == 1
