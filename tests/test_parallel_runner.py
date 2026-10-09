""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from macmini.backtest.engine import WalkForwardConfig
from macmini.backtest.parallel_runner import PARALLEL_MIN_SYMBOLS, run_parallel
from macmini.backtest.runner import run as run_sequential
from macmini.data.schema import init_schema


STRATEGY_IDS = [1, 9, 15]

N_SYMBOLS = max(10, PARALLEL_MIN_SYMBOLS + 2)

TEST_CONFIG = WalkForwardConfig(
    train_window_days=63,
    test_window_days=21,
    step_days=21,
    min_history_days=126,
    transaction_cost_bps=5,
    slippage_bps=10,
)

METRIC_COLS = (
    "n_trades", "sharpe", "deflated_sharpe", "profit_factor", "max_drawdown",
    "hit_rate", "total_return", "avg_holding_days", "skew", "kurt",
    "p_value", "p_value_bonferroni",
)


def _weekday_dates(start: str, n: int) -> list[str]:
    out: list[str] = []
    d = date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _synthetic_bars(symbol: str, n: int, seed: int) -> list[tuple]:
    ""
    rng = np.random.default_rng(seed)
    dates = _weekday_dates("2023-01-02", n)
    price = 100.0
    rows: list[tuple] = []
    for d in dates:
        drift = 0.0005
        shock = rng.normal(0, 0.015)
        price *= (1 + drift + shock)
        hi = price * (1 + abs(rng.normal(0, 0.005)))
        lo = price * (1 - abs(rng.normal(0, 0.005)))
        op = price * (1 + rng.normal(0, 0.002))
        vol = int(rng.integers(100_000, 5_000_000))
        rows.append(
            (symbol, d, op, hi, lo, price, price, vol, "test")
        )
    return rows


def _seed_universe(db_path: Path) -> list[str]:
    ""
    symbols = [f"SYM{i:02d}" for i in range(N_SYMBOLS)]
    sectors = ["Software", "Mining", "Defense", "Other"]
    with sqlite3.connect(db_path) as c:
        for idx, sym in enumerate(symbols):
            bars = _synthetic_bars(sym, n=260, seed=1000 + idx)
            c.executemany(
                "INSERT INTO mm_bars (symbol, date, open, high, low, close, "
                "adj_close, volume, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bars,
            )
            c.execute(
                "INSERT INTO mm_sector_map (symbol, sector) VALUES (?, ?)",
                (sym, sectors[idx % len(sectors)]),
            )
        c.commit()
    return symbols


def _rows_for_run(db_path: Path, run_id: str) -> dict:
    ""
    with sqlite3.connect(db_path) as c:
        cur = c.execute(
            f"SELECT strategy_id, symbol, window_start, window_end, sector, "
            f"       {','.join(METRIC_COLS)} "
            f"FROM mm_backtest_runs WHERE run_id = ?",
            (run_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    out: dict = {}
    for row in rows:
        rec = dict(zip(cols, row))
        key = (rec["strategy_id"], rec["symbol"], rec["window_start"], rec["window_end"])
        payload = {k: rec[k] for k in ("sector", *METRIC_COLS)}
        out[key] = payload
    return out


@pytest.fixture
def universe_db(tmp_path: Path) -> tuple[Path, list[str]]:
    db_path = tmp_path / "macmini.db"
    sqlite3.connect(db_path).close()
    init_schema(db_path)
    symbols = _seed_universe(db_path)
    return db_path, symbols


def _assert_rowsets_equal(seq: dict, par: dict, tag: str) -> None:
    seq_keys = set(seq.keys())
    par_keys = set(par.keys())
    assert seq_keys == par_keys, (
        f"[{tag}] row-key mismatch: "
        f"seq_only={sorted(seq_keys - par_keys)[:5]} "
        f"par_only={sorted(par_keys - seq_keys)[:5]}"
    )
    for key, seq_payload in seq.items():
        par_payload = par[key]
        for col, seq_val in seq_payload.items():
            par_val = par_payload[col]
            if isinstance(seq_val, float) and isinstance(par_val, float):
                assert seq_val == par_val or (
                    (seq_val != seq_val) and (par_val != par_val)
                ), f"[{tag}] {key} {col}: seq={seq_val} par={par_val}"
            else:
                assert seq_val == par_val, (
                    f"[{tag}] {key} {col}: seq={seq_val!r} par={par_val!r}"
                )


def test_parallel_matches_sequential(universe_db: tuple[Path, list[str]]) -> None:
    ""
    db_path, symbols = universe_db

    seq_out = run_sequential(symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG)
    par_out = run_parallel(
        symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG, max_workers=4,
    )

    assert par_out["rows_written"] > 0, "parallel run wrote no rows"
    assert par_out["run_id"] != seq_out["run_id"], (
        "distinct runs must have distinct run_ids"
    )
    assert seq_out["rows_written"] == par_out["rows_written"]
    assert seq_out["pairs"] == par_out["pairs"]

    seq_rows = _rows_for_run(db_path, seq_out["run_id"])
    par_rows = _rows_for_run(db_path, par_out["run_id"])
    _assert_rowsets_equal(seq_rows, par_rows, tag="seq_vs_par4")


def test_two_vs_four_workers(universe_db: tuple[Path, list[str]]) -> None:
    ""
    db_path, symbols = universe_db

    par2 = run_parallel(
        symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG, max_workers=2,
    )
    par4 = run_parallel(
        symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG, max_workers=4,
    )

    assert par2["rows_written"] == par4["rows_written"]
    assert par2["pairs"] == par4["pairs"]

    rows_2 = _rows_for_run(db_path, par2["run_id"])
    rows_4 = _rows_for_run(db_path, par4["run_id"])
    _assert_rowsets_equal(rows_2, rows_4, tag="par2_vs_par4")


def test_single_worker_falls_through_to_sequential(
    universe_db: tuple[Path, list[str]],
) -> None:
    ""
    db_path, symbols = universe_db

    seq_out = run_sequential(symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG)
    par_out = run_parallel(
        symbols, STRATEGY_IDS, db_path, config=TEST_CONFIG, max_workers=1,
    )

    assert seq_out["rows_written"] == par_out["rows_written"]
    seq_rows = _rows_for_run(db_path, seq_out["run_id"])
    par_rows = _rows_for_run(db_path, par_out["run_id"])
    _assert_rowsets_equal(seq_rows, par_rows, tag="seq_vs_workers1")


def test_tiny_universe_falls_through_to_sequential(
    universe_db: tuple[Path, list[str]],
) -> None:
    ""
    db_path, symbols = universe_db
    small = symbols[:PARALLEL_MIN_SYMBOLS - 1]

    seq_out = run_sequential(small, STRATEGY_IDS, db_path, config=TEST_CONFIG)
    par_out = run_parallel(
        small, STRATEGY_IDS, db_path, config=TEST_CONFIG, max_workers=4,
    )

    assert seq_out["rows_written"] == par_out["rows_written"]
    seq_rows = _rows_for_run(db_path, seq_out["run_id"])
    par_rows = _rows_for_run(db_path, par_out["run_id"])
    _assert_rowsets_equal(seq_rows, par_rows, tag="seq_vs_tiny_par")
