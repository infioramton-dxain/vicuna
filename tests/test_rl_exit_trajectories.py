""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macmini.backtest import exec_variants
from macmini.rl.exit_policy import trajectories as tj

COST = 0.0015
ENTRY_COST = float(np.log1p(-COST))
EXIT_COST = float(np.log1p(-COST))


def _bars(closes, lows=None, highs=None, opens=None, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes.copy()
    highs = np.asarray(highs, dtype=float) if highs is not None else np.maximum(opens, closes)
    lows = np.asarray(lows, dtype=float) if lows is not None else np.minimum(opens, closes)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "adj_close": closes, "volume": np.full(n, 1e6)},
        index=idx,
    )


def _feats(bars, div_rate=None):
    f = pd.DataFrame(index=bars.index)
    f["atr14"] = 1.0
    f["div_rate"] = div_rate if div_rate is not None else 0.0
    f["ret_5d"] = 0.0
    f["ret_10d"] = 0.0
    f["ret_20d"] = 0.0
    f["atr_percentile"] = 0.5
    f["vix_low"] = 0.0
    f["vix_normal"] = 1.0
    f["vix_high"] = 0.0
    return f


def _roll(bars, variant, entry_bar=0, exit_bars=(), stats=None, div_rate=None):
    entries = pd.Series(False, index=bars.index)
    entries.iloc[entry_bar] = True
    exits = pd.Series(False, index=bars.index)
    for b in exit_bars:
        exits.iloc[b] = True
    stats = stats if stats is not None else {
        "episodes": 0, "same_bar_exits": 0, "episode_return_sum": 0.0}
    static = tj._static_onehots("Software", "trend")
    return tj._roll_episodes(bars, entries, exits, variant,
                             _feats(bars, div_rate=div_rate),
                             static, COST, stats), stats


def _episode_sum(rows):
    return sum(r["reward"] for r in rows)


def test_bracket_stop_hit_all_hold():
    closes = [100] * 6 + [96, 92]
    bars = _bars(closes)
    rows, stats = _roll(bars, "bracket")
    assert stats["episodes"] == 1
    assert all(r["action"] == tj.ACTIONS["HOLD"] for r in rows)
    assert rows[-1]["terminal"]
    expect = np.log(92 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)


def test_bracket_target_hit():
    closes = [100] * 5 + [109]
    bars = _bars(closes, highs=[100] * 5 + [112], opens=[100] * 6)
    rows, _ = _roll(bars, "bracket")
    expect = np.log(110 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)
    assert all(r["action"] == tj.ACTIONS["HOLD"] for r in rows)


def test_stop_beats_target_same_bar():
    closes = [100, 100, 100]
    bars = _bars(closes, highs=[100, 100, 118], lows=[100, 100, 91],
                 opens=[100, 100, 100])
    rows, _ = _roll(bars, "bracket")
    expect = np.log(95 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)


def test_trail_ratchets_and_stops_above_static():
    closes = [100, 100, 106, 112, 118, 108, 100]
    highs = [100, 100, 107, 113, 120, 108, 100]
    bars = _bars(closes, highs=highs, opens=closes)
    rows, _ = _roll(bars, "trail")
    tightens = [r for r in rows if r["action"] == tj.ACTIONS["TIGHTEN_STOP"]]
    assert len(tightens) >= 2
    expect = np.log(108 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)


def test_time_stop_exits_at_horizon():
    n = exec_variants.TIME_STOP_HORIZON + 10
    bars = _bars([100.0] * n)
    rows, _ = _roll(bars, "time_stop")
    assert len(rows) == exec_variants.TIME_STOP_HORIZON
    assert rows[-1]["action"] == tj.ACTIONS["EXIT"]
    assert rows[-1]["terminal"]
    assert all(r["action"] == tj.ACTIONS["HOLD"] for r in rows[:-1])


def test_hard_cap_forces_exit_all_variants():
    n = tj.MAX_EPISODE_BARS + 15
    bars = _bars([100.0] * n)
    rows, _ = _roll(bars, "bracket")
    assert len(rows) == tj.MAX_EPISODE_BARS
    assert rows[-1]["action"] == tj.ACTIONS["EXIT"]


def test_signal_exit_fills_next_open():
    closes = [100, 100, 100, 100, 104, 105]
    opens = [100, 100, 100, 100, 100, 105]
    bars = _bars(closes, opens=opens)
    rows, _ = _roll(bars, "bracket", exit_bars=[4])
    assert rows[-1]["action"] == tj.ACTIONS["EXIT"]
    expect = np.log(105 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)


def test_same_bar_env_exit_is_skipped_and_counted():
    closes = [100, 90, 90, 90]
    bars = _bars(closes, opens=[100, 100, 90, 90], lows=[100, 89, 90, 90])
    rows, stats = _roll(bars, "bracket")
    assert rows == []
    assert stats["same_bar_exits"] == 1
    assert stats["episodes"] == 0


def test_obs_layout_and_onehots():
    closes = [100.0] * 10
    bars = _bars(closes)
    rows, _ = _roll(bars, "bracket")
    obs = rows[0]["obs"]
    assert obs.shape == (len(tj.OBS_FIELDS),)
    assert obs[8:11].sum() == pytest.approx(1.0)
    n_sec = len(tj.SECTORS)
    assert obs[11:11 + n_sec].sum() == pytest.approx(1.0)
    assert obs[11 + n_sec:].sum() == pytest.approx(1.0)
    assert obs[0] == pytest.approx(0.0)
    assert obs[1] == pytest.approx(1 / tj.MAX_EPISODE_BARS)


def test_dividend_credited_in_rewards():
    n = exec_variants.TIME_STOP_HORIZON + 10
    bars = _bars([100.0] * n)
    div = pd.Series(0.0, index=bars.index)
    div.iloc[4] = 1.0
    rows, _ = _roll(bars, "time_stop", div_rate=div)
    expect = np.log(101 / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)

    div2 = pd.Series(0.0, index=bars.index)
    div2.iloc[1] = 1.0
    rows2, _ = _roll(bars, "time_stop", div_rate=div2)
    expect2 = ENTRY_COST + EXIT_COST
    assert _episode_sum(rows2) == pytest.approx(expect2, abs=1e-9)


def test_dividend_credited_on_terminal_bar():
    closes = [100] * 6 + [96, 92]
    bars = _bars(closes)
    div = pd.Series(0.0, index=bars.index)
    div.iloc[7] = 1.0
    rows, _ = _roll(bars, "bracket", div_rate=div)
    expect = np.log((92 + 1) / 100) + ENTRY_COST + EXIT_COST
    assert _episode_sum(rows) == pytest.approx(expect, abs=1e-9)


def test_no_overlapping_positions():
    n = exec_variants.TIME_STOP_HORIZON + 12
    bars = _bars([100.0] * n)
    entries = pd.Series(False, index=bars.index)
    entries.iloc[0] = True
    entries.iloc[3] = True
    exits = pd.Series(False, index=bars.index)
    stats = {"episodes": 0, "same_bar_exits": 0, "episode_return_sum": 0.0}
    rows = tj._roll_episodes(bars, entries, exits, "time_stop", _feats(bars),
                             tj._static_onehots("Other", "trend"), COST, stats)
    assert stats["episodes"] == 1
    ids = {r["episode_id"] for r in rows}
    assert len(ids) == 1
