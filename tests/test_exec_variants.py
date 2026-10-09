""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from macmini.backtest import exec_variants as ev
from macmini.backtest.engine import WalkForwardConfig, _window_stats, signals_to_arrays
from macmini.strategies.base import Signal


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    ""
    idx = pd.bdate_range("2025-01-06", periods=len(rows))
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": 1_000_000},
        index=idx,
    )


def _flat(n: int, price: float = 100.0) -> pd.DataFrame:
    return _bars([(price, price, price, price)] * n)


def _entries_at(bars: pd.DataFrame, *positions: int) -> pd.Series:
    s = pd.Series(False, index=bars.index)
    for p in positions:
        s.iloc[p] = True
    return s


ZERO_COST = WalkForwardConfig(transaction_cost_bps=0, slippage_bps=0)


class TestLimitEntry:
    def test_intrabar_touch_fills_at_limit(self):
        rows = [(100, 100, 100, 100)] * 8
        rows[4] = (101, 102, 98, 101)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "limit")
        assert filled.iloc[4] and filled.sum() == 1
        assert price.iloc[4] == pytest.approx(100.0)

    def test_gap_down_fills_at_open(self):
        rows = [(100, 100, 100, 100)] * 8
        rows[4] = (97, 99, 96, 98)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "limit")
        assert filled.iloc[4]
        assert price.iloc[4] == pytest.approx(97.0)

    def test_never_touched_is_a_miss(self):
        rows = [(101, 102, 100.5, 101)] * 10
        rows[3] = (100, 100, 100, 100)
        bars = _bars(rows)
        filled, _ = ev.apply_entry_model(_entries_at(bars, 3), bars, "limit")
        assert not filled.any()

    def test_ttl_expires(self):
        rows = [(101, 102, 100.5, 101)] * 10
        rows[3] = (100, 100, 100, 100)
        rows[9] = (99, 99, 98, 99)
        bars = _bars(rows)
        filled, _ = ev.apply_entry_model(_entries_at(bars, 3), bars, "limit")
        assert not filled.any()


class TestStopEntry:
    def test_intrabar_trigger_fills_at_stop(self):
        rows = [(100, 101, 99, 100)] * 8
        rows[4] = (100.5, 102, 100, 101.5)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop")
        assert filled.iloc[4]
        assert price.iloc[4] == pytest.approx(101.0)

    def test_gap_up_fills_at_open(self):
        rows = [(100, 101, 99, 100)] * 8
        rows[4] = (103, 104, 102.5, 103.5)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop")
        assert price.iloc[4] == pytest.approx(103.0)

    def test_no_breakout_is_a_miss(self):
        rows = [(100, 101, 99, 100)] * 10
        for k in range(4, 10):
            rows[k] = (99, 100.5, 98, 99)
        bars = _bars(rows)
        filled, _ = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop")
        assert not filled.any()


class TestStopLimitEntry:
    def test_gap_above_cap_waits_then_fills_at_cap(self):
        rows = [(100, 101, 99, 100)] * 8
        rows[4] = (103, 104, 102.5, 103)
        rows[5] = (102, 102.5, 101.2, 101.8)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop_limit")
        assert filled.iloc[5] and not filled.iloc[4]
        assert price.iloc[5] == pytest.approx(101.0 * 1.005)

    def test_gap_into_band_fills_at_open(self):
        rows = [(100, 101, 99, 100)] * 8
        rows[4] = (101.2, 102, 101, 101.5)
        bars = _bars(rows)
        filled, price = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop_limit")
        assert price.iloc[4] == pytest.approx(101.2)

    def test_runaway_breakout_never_fills(self):
        rows = [(100, 101, 99, 100)] * 8
        for k in range(4, 8):
            rows[k] = (105 + k, 106 + k, 104 + k, 105.5 + k)
        bars = _bars(rows)
        filled, _ = ev.apply_entry_model(_entries_at(bars, 3), bars, "stop_limit")
        assert not filled.any()


class TestTimeStopExits:
    def test_forced_exit_at_horizon(self):
        bars = _flat(30)
        fe = _entries_at(bars, 5)
        out = ev.time_stop_exits(fe, pd.Series(False, index=bars.index))
        assert out.iloc[25] and out.sum() == 1

    def test_signal_exit_wins_and_reentry_resets(self):
        bars = _flat(40)
        fe = _entries_at(bars, 5, 12)
        se = _entries_at(bars, 8)
        out = ev.time_stop_exits(fe, se)
        assert out.iloc[8]
        assert out.iloc[32]
        assert not out.iloc[25]


class TestExitOverlays:
    def test_baseline_and_time_stop_add_nothing(self):
        bars = _flat(30)
        assert ev.exit_overlay_kwargs("signal", bars) == {}
        assert ev.exit_overlay_kwargs("time_stop", bars) == {}

    def test_bracket_has_sl_and_tp(self):
        bars = _flat(30)
        kw = ev.exit_overlay_kwargs("bracket", bars)
        assert "sl_stop" in kw and "tp_stop" in kw and "sl_trail" not in kw
        np.testing.assert_allclose(kw["tp_stop"], 2.0 * kw["sl_stop"])

    def test_trail_variants(self):
        bars = _flat(30)
        assert ev.exit_overlay_kwargs("trail", bars)["sl_trail"] is True
        assert "tp_stop" not in ev.exit_overlay_kwargs("trail", bars)
        assert "tp_stop" in ev.exit_overlay_kwargs("trail_tp", bars)

    def test_stop_frac_clamped(self):
        bars = _flat(30)
        frac = ev.stop_frac_series(bars)
        assert (frac.iloc[15:] == 0.05).all()


def _one_enter(bars: pd.DataFrame, pos: int) -> list[Signal]:
    return [Signal(symbol="TST", date=bars.index[pos].date(),
                   action="enter", conviction=1.0, reason="test")]


class TestEngineIntegration:
    def test_default_args_reproduce_pre_d36_call(self):
        ""
        rows = [(100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5, 100.5 + i * 0.5)
                for i in range(40)]
        bars = _bars(rows)
        sigs = _one_enter(bars, 10) + [
            Signal(symbol="TST", date=bars.index[20].date(),
                   action="exit", conviction=1.0, reason="test")
        ]
        got = _window_stats(bars, sigs, ZERO_COST)

        entries, exits = signals_to_arrays(sigs, bars)
        pf = vbt.Portfolio.from_signals(
            close=bars["close"],
            entries=entries.shift(1, fill_value=False),
            exits=exits.shift(1, fill_value=False),
            price=bars["open"],
            fees=0.0, slippage=0.0, init_cash=10_000, freq="1D",
        )
        assert got["n_trades"] == len(pf.trades.records_readable) == 1
        assert got["total_return"] == pytest.approx(float(pf.total_return()))

    def test_bracket_stop_loss_bounds_the_loss(self):
        rows = [(100, 100.5, 99.5, 100)] * 15
        rows += [(90, 91, 80, 82), (75, 76, 70, 72), (70, 71, 65, 66)]
        rows += [(66, 67, 65, 66)] * 10
        bars = _bars(rows)
        sigs = _one_enter(bars, 10)
        base = _window_stats(bars, sigs, ZERO_COST, exit_type="signal")
        brk = _window_stats(bars, sigs, ZERO_COST, exit_type="bracket")
        assert brk["n_trades"] == 1
        assert brk["total_return"] > base["total_return"]

    def test_trailing_stop_locks_in_gains(self):
        up = [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i) for i in range(20)]
        down = [(134 - 4 * i, 135 - 4 * i, 129 - 4 * i, 130 - 4 * i) for i in range(6)]
        bars = _bars(up + down)
        sigs = _one_enter(bars, 2)
        base = _window_stats(bars, sigs, ZERO_COST, exit_type="signal")
        trail = _window_stats(bars, sigs, ZERO_COST, exit_type="trail")
        assert trail["n_trades"] == 1
        assert trail["total_return"] > base["total_return"]
        assert trail["total_return"] > 0.05

    def test_limit_entry_variant_runs_end_to_end(self):
        rows = [(100, 101, 99, 100)] * 12
        rows[6] = (101, 101.5, 97, 99)
        rows += [(102 + i, 103 + i, 101 + i, 102.5 + i) for i in range(10)]
        bars = _bars(rows)
        sigs = _one_enter(bars, 5)
        out = _window_stats(bars, sigs, ZERO_COST,
                            entry_type="limit", exit_type="bracket")
        assert out["n_trades"] >= 1

    def test_missed_entry_is_empty_not_error(self):
        rows = [(100, 101, 99, 100)] * 30
        for k in range(6, 30):
            rows[k] = (99, 100.5, 98, 99)
        bars = _bars(rows)
        sigs = _one_enter(bars, 5)
        out = _window_stats(bars, sigs, ZERO_COST,
                            entry_type="stop", exit_type="bracket")
        assert out["n_trades"] == 0

    def test_variant_grid_shape(self):
        grid = ev.variant_grid()
        assert len(grid) == 28
        assert ev.REFERENCE_VARIANT in grid
        assert ev.BASELINE_VARIANT not in grid


def _walker_bars_whipsaw() -> pd.DataFrame:
    ""
    rows = [(100, 101, 99, 100)] * 3
    rows += [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i)
             for i in range(8)]
    rows += [(113, 113, 103, 104)]
    rows += [(105, 106, 104, 105.5)]
    rows += [(106, 107, 105, 106.5)]
    rows += [(107 + 2 * i, 108.5 + 2 * i, 106 + 2 * i, 108 + 2 * i)
             for i in range(8)]
    return _bars(rows)


class TestWalkerExits:
    def _run(self, bars, exit_type, entry_pos=2, se_pos=(), **kw):
        fe = _entries_at(bars, entry_pos + 1)
        se = _entries_at(bars, *se_pos) if se_pos else pd.Series(False, index=bars.index)
        price = bars["open"]
        return ev.walker_exits(fe, se, bars, exit_type, price, **kw)

    def test_trail_out_then_reentry(self):
        bars = _walker_bars_whipsaw()
        entries, exits, price = self._run(bars, "trail_reentry")
        assert entries.sum() == 2
        assert exits.sum() == 1
        assert exits.iloc[11]
        fill = price.iloc[11]
        assert 103 < fill < 113
        cross = next(j for j in range(12, len(bars)) if bars["close"].iloc[j] > fill)
        assert entries.iloc[cross + 1]
        assert price.iloc[cross + 1] == pytest.approx(bars["open"].iloc[cross + 1])

    def test_reentry_captures_recovery_plain_trail_misses_it(self):
        bars = _walker_bars_whipsaw()
        sigs = _one_enter(bars, 2)
        trail = _window_stats(bars, sigs, ZERO_COST, exit_type="trail")
        rent = _window_stats(bars, sigs, ZERO_COST, exit_type="trail_reentry")
        assert rent["n_trades"] == trail["n_trades"] + 1
        assert rent["total_return"] > trail["total_return"]

    def test_no_recovery_no_reentry(self):
        rows = [(100, 101, 99, 100)] * 3
        rows += [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i)
                 for i in range(8)]
        rows += [(113, 113, 100, 101)]
        rows += [(100, 101, 98, 99)] * 12
        bars = _bars(rows)
        entries, exits, _ = self._run(bars, "trail_reentry")
        assert entries.sum() == 1 and exits.sum() == 1

    def test_watch_expires_after_ttl(self):
        rows = [(100, 101, 99, 100)] * 3
        rows += [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i)
                 for i in range(8)]
        rows += [(113, 113, 103, 104)]
        rows += [(104, 105, 103, 104)] * (ev.RE_ENTRY_TTL + 1)
        rows += [(120, 121, 119, 120)] * 3
        bars = _bars(rows)
        entries, exits, _ = self._run(bars, "trail_reentry")
        assert entries.sum() == 1

    def test_signal_exit_kills_watch(self):
        bars = _walker_bars_whipsaw()
        entries, exits, _ = self._run(bars, "trail_reentry", se_pos=(12,))
        assert entries.sum() == 1
        assert exits.sum() == 1

    def test_reentry_budget_bounds_churn(self):
        chunk = [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i)
                 for i in range(6)]
        whip = [(110, 110, 99, 100), (101, 113, 100, 112.5)]
        rows = [(100, 101, 99, 100)] * 3 + chunk + whip * 5 + [(113, 114, 112, 113)] * 3
        bars = _bars(rows)
        entries, _, _ = self._run(bars, "trail_reentry")
        assert entries.sum() == 1 + ev.MAX_REENTRIES
        entries0, _, _ = self._run(bars, "trail_reentry", max_reentries=0)
        assert entries0.sum() == 1

    def test_ratchet_exits_tighter_into_deceleration(self):
        rows = [(100 + i, 100.8 + i, 99.4 + i, 100.5 + i) for i in range(30)]
        rows += [(130, 130.5, 129.5, 129.8), (129.6, 130, 128.8, 129.2)]
        rows += [(129, 129.5, 125.6, 126)]
        rows += [(126, 127, 125, 126.5)] * 5
        bars = _bars(rows)
        r_ent, r_ex, _ = self._run(bars, "ratchet_trail")
        p_ent, p_ex, _ = self._run(bars, "trail_reentry", max_reentries=0)
        assert r_ex.sum() == 1
        d = ev.momentum_decel(bars)
        assert d.iloc[31] or d.iloc[32]
        assert r_ex.iloc[32]
        assert not p_ex.iloc[32]

    def test_trail_level_never_decreases(self):
        rows = [(100 + i, 100.8 + i, 99.4 + i, 100.5 + i) for i in range(30)]
        rows += [(130, 130.5, 129.5, 129.8), (129.6, 130, 128.8, 129.2)]
        rows += [(129.5, 132, 129.4, 131.8)]
        rows += [(131, 131.5, 124, 124.5)]
        rows += [(124, 125, 123, 124)] * 3
        bars = _bars(rows)
        entries, exits, price = self._run(bars, "ratchet_trail")
        assert exits.sum() == 1
        ex_bar = int(np.flatnonzero(exits.to_numpy())[0])
        peak = bars["high"].iloc[: ex_bar + 1].max()
        assert price.iloc[ex_bar] >= peak * (1 - 0.06)

    def test_walker_matches_vbt_trail_when_reentry_moot(self):
        up = [(100 + 2 * i, 101 + 2 * i, 99 + 2 * i, 100 + 2 * i) for i in range(20)]
        down = [(134 - 4 * i, 135 - 4 * i, 129 - 4 * i, 130 - 4 * i) for i in range(6)]
        bars = _bars(up + down)
        sigs = _one_enter(bars, 2)
        vbt_trail = _window_stats(bars, sigs, ZERO_COST, exit_type="trail")
        walker = _window_stats(bars, sigs, ZERO_COST, exit_type="trail_reentry")
        assert walker["n_trades"] == vbt_trail["n_trades"]
        assert walker["total_return"] == pytest.approx(
            vbt_trail["total_return"], rel=1e-9)

    def test_walker_kwargs_are_empty(self):
        bars = _flat(30)
        for xt in ev.WALKER_EXIT_TYPES:
            assert ev.exit_overlay_kwargs(xt, bars) == {}
