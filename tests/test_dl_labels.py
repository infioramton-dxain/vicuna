""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macmini.config import DB_PATH
from macmini.dl.labels import (
    FLAT_LOWER, FLAT_UPPER,
    compute_labels, populate,
)


def _synthetic_bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes}, index=idx)


class TestComputeLabels:
    def test_returns_match_manual_calc(self) -> None:
        bars = _synthetic_bars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])
        out = compute_labels(bars)
        assert out.iloc[0]["forward_5d_return"] == pytest.approx(0.05)
        assert out.iloc[0]["forward_1d_return"] == pytest.approx(0.01)
        assert out.iloc[0]["forward_10d_return"] == pytest.approx(0.10)

    def test_last_bars_are_nan(self) -> None:
        bars = _synthetic_bars(list(range(100, 130)))
        out = compute_labels(bars)
        assert pd.isna(out.iloc[-1]["forward_1d_return"])
        for i in range(-5, 0):
            assert pd.isna(out.iloc[i]["forward_5d_return"])
        for i in range(-10, 0):
            assert pd.isna(out.iloc[i]["forward_10d_return"])

    def test_class_boundaries(self) -> None:
        bars = _synthetic_bars([100, 100, 100, 100, 100, 103, 99, 95])
        out = compute_labels(bars)
        assert out.iloc[0]["forward_5d_class"] == 2
        assert out.iloc[1]["forward_5d_class"] == 1
        assert out.iloc[2]["forward_5d_class"] == 0

    def test_max_dd_negative_when_window_dips(self) -> None:
        bars = _synthetic_bars([100, 101, 102, 103, 90, 91] + list(range(100, 130)))
        out = compute_labels(bars)
        assert out.iloc[0]["max_dd_5d"] < -0.10

    def test_single_bar_returns_empty(self) -> None:
        bars = _synthetic_bars([100])
        assert compute_labels(bars).empty

    def test_short_history_columns_mostly_nan(self) -> None:
        ""
        out = compute_labels(_synthetic_bars([100, 101, 102]))
        assert len(out) == 3
        assert not pd.isna(out.iloc[0]["forward_1d_return"])
        assert pd.isna(out.iloc[-1]["forward_1d_return"])
        assert out["forward_5d_return"].isna().all()
        assert out["forward_10d_return"].isna().all()


class TestLookaheadDiscipline:
    def test_no_label_uses_features_dated_after_self(self) -> None:
        ""
        n = 30
        base = list(np.linspace(100, 130, n))
        variant = base.copy()
        for i in range(15):
            variant[i] = base[i] + 5.0
        out_base = compute_labels(_synthetic_bars(base))
        out_variant = compute_labels(_synthetic_bars(variant))

        for col in ("forward_1d_return", "forward_5d_return", "forward_10d_return",
                    "forward_5d_class", "forward_5d_sharpe", "max_dd_5d"):
            for i in range(15, 20):
                b = out_base.iloc[i][col]
                v = out_variant.iloc[i][col]
                if pd.isna(b) and pd.isna(v):
                    continue
                assert b == pytest.approx(v), \
                    f"{col} at row {i} changed when ONLY pre-row bars were mutated → look-ahead leak"


class TestPopulateRoundtrip:
    @pytest.fixture
    def tmp_db(self, tmp_path: Path) -> Path:
        ""
        db = tmp_path / "tiny.db"
        with sqlite3.connect(db) as c:
            c.execute("""CREATE TABLE mm_bars (
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume INTEGER, source TEXT,
                PRIMARY KEY (symbol, date))""")
            c.execute("""CREATE TABLE mm_training_labels (
                symbol TEXT, date TEXT,
                forward_1d_return REAL, forward_5d_return REAL, forward_10d_return REAL,
                forward_5d_class INTEGER, forward_5d_sharpe REAL, max_dd_5d REAL,
                PRIMARY KEY (symbol, date))""")
            base = pd.bdate_range("2024-01-02", periods=30)
            for i, d in enumerate(base):
                c.execute(
                    "INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("FAKE", d.strftime("%Y-%m-%d"), 100, 101, 99,
                     100.0 + i * 0.5, 100.0 + i * 0.5, 1_000_000, "synthetic"),
                )
            c.commit()
        return db

    def test_populate_writes_rows(self, tmp_db: Path) -> None:
        out = populate(symbols=["FAKE"], db_path=tmp_db)
        assert out["symbols_done"] == 1
        assert out["rows_written"] >= 10
        with sqlite3.connect(tmp_db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_training_labels WHERE symbol='FAKE'").fetchone()[0]
        assert n >= 10

    def test_populate_idempotent(self, tmp_db: Path) -> None:
        first = populate(symbols=["FAKE"], db_path=tmp_db)
        second = populate(symbols=["FAKE"], db_path=tmp_db)
        assert first["rows_written"] == second["rows_written"]
        with sqlite3.connect(tmp_db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_training_labels WHERE symbol='FAKE'").fetchone()[0]
        assert n == first["rows_written"]
