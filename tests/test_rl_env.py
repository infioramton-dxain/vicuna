""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from macmini.config import DB_PATH
from macmini.rl.env import (
    ACTION_DIM,
    EPISODE_LEN,
    K_SLOTS,
    MacminiSwingEnv,
    OBS_DIM,
)


@pytest.fixture(scope="module")
def db_copy(tmp_path_factory) -> Path:
    dst = tmp_path_factory.mktemp("rl_db") / "macmini.db"
    src = sqlite3.connect(DB_PATH, timeout=30)
    d = sqlite3.connect(dst)
    with d:
        src.backup(d)
    src.close()
    d.close()
    return dst


@pytest.fixture(scope="module")
def env(db_copy) -> MacminiSwingEnv:
    return MacminiSwingEnv(db_path=db_copy, start_date="2023-01-01", end_date="2025-06-30",
                           seed=42)


def test_action_space_dim(env: MacminiSwingEnv) -> None:
    assert env.action_space.n == ACTION_DIM == 1 + 2 * K_SLOTS


def test_observation_shape(env: MacminiSwingEnv) -> None:
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "date" in info and "port_value" in info


def test_hold_step_is_free(env: MacminiSwingEnv) -> None:
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["reward_parts"]["r_cost"] == 0.0
    assert not terminated
    assert not truncated


def test_buy_reduces_cash(env: MacminiSwingEnv) -> None:
    env.reset(seed=42)
    initial_cash = env.state.cash
    obs, reward, term, trunc, info = env.step(1)
    assert env.state.cash <= initial_cash


def test_episode_finishes(env: MacminiSwingEnv) -> None:
    ""
    obs, _ = env.reset(seed=42)
    rng = np.random.default_rng(42)
    terminated = truncated = False
    step_count = 0
    while not (terminated or truncated):
        action = int(rng.integers(0, ACTION_DIM))
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(reward)
        assert step_count <= EPISODE_LEN + 1
    assert terminated or truncated
    assert step_count >= 1


def test_reward_parts_present(env: MacminiSwingEnv) -> None:
    env.reset(seed=42)
    _, _, _, _, info = env.step(0)
    parts = info["reward_parts"]
    for k in ("r_ret", "r_vol", "r_dd", "r_conc", "r_cost"):
        assert k in parts
        assert np.isfinite(parts[k])


def test_terminal_bonus_only_at_end(env: MacminiSwingEnv) -> None:
    ""
    env.reset(seed=42)
    _, _, _, _, info = env.step(0)
    assert "r_term" not in info["reward_parts"]


def test_deterministic_with_seed(db_copy: Path) -> None:
    e1 = MacminiSwingEnv(db_path=db_copy, start_date="2023-01-01",
                         end_date="2025-06-30", seed=123)
    e2 = MacminiSwingEnv(db_path=db_copy, start_date="2023-01-01",
                         end_date="2025-06-30", seed=123)
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    np.testing.assert_allclose(o1, o2)


def test_universe_changes_across_dates(env: MacminiSwingEnv) -> None:
    ""
    env.reset(seed=42)
    slots_over_time = []
    for _ in range(30):
        slots_over_time.append(tuple(env.symbols))
        env.step(0)
    distinct = set(slots_over_time)
    assert len(distinct) >= 2, "universe did not change across 30 steps"


def test_fixed_universe_fallback(db_copy: Path) -> None:
    ""
    e = MacminiSwingEnv(db_path=db_copy, start_date="2023-01-01",
                       end_date="2025-06-30", seed=1,
                       universe_symbols=["AAPL", "MSFT", "SPY"])
    obs, _ = e.reset(seed=1)
    assert e.symbols[:3] == ["AAPL", "MSFT", "SPY"]
    assert len(e.symbols) == K_SLOTS
