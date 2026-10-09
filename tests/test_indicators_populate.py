""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macmini.data.indicators import INDICATOR_COLUMNS, populate


def _synthetic_bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = 100 + rng.normal(0, 1, n).cumsum() * 0.5
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.uniform(0, 0.5, n),
        "low": close - rng.uniform(0, 0.5, n),
        "close": close,
        "volume": rng.integers(500_000, 2_000_000, n),
    }, index=idx)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "tiny.db"
    with sqlite3.connect(db) as c:
        c.executescript("""
            CREATE TABLE mm_bars (
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume INTEGER, source TEXT,
                PRIMARY KEY (symbol, date));
            CREATE TABLE mm_indicators_daily (
                symbol TEXT, date TEXT,
                sma20 REAL, sma50 REAL, sma200 REAL,
                ema12 REAL, ema26 REAL, ema200 REAL,
                rsi14 REAL, rsi2 REAL,
                atr14 REAL,
                macd_line REAL, macd_signal REAL, macd_hist REAL,
                bb_upper REAL, bb_middle REAL, bb_lower REAL, bb_width REAL,
                adx14 REAL,
                williams_r14 REAL,
                ibs REAL,
                volume_zscore_50 REAL,
                PRIMARY KEY (symbol, date));
        """)
        bars = _synthetic_bars(250)
        for d, r in bars.iterrows():
            c.execute("INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      ("AAPL", d.strftime("%Y-%m-%d"),
                       r["open"], r["high"], r["low"], r["close"],
                       r["close"], int(r["volume"]), "synthetic"))
        c.commit()
    return db


def test_populate_writes_rows(tmp_db: Path) -> None:
    out = populate(symbols=["AAPL"], db_path=tmp_db)
    assert out["symbols_done"] == 1
    assert out["rows_written"] == 250

    with sqlite3.connect(tmp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM mm_indicators_daily WHERE symbol='AAPL'").fetchone()[0]
        cols = [r[1] for r in c.execute("PRAGMA table_info(mm_indicators_daily)").fetchall()]
    assert n == 250
    for col in INDICATOR_COLUMNS:
        assert col in cols, f"{col} missing from mm_indicators_daily schema"


def test_populate_idempotent(tmp_db: Path) -> None:
    first = populate(symbols=["AAPL"], db_path=tmp_db)
    second = populate(symbols=["AAPL"], db_path=tmp_db)
    assert first["rows_written"] == second["rows_written"]
    with sqlite3.connect(tmp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM mm_indicators_daily WHERE symbol='AAPL'").fetchone()[0]
    assert n == 250


def test_populate_warmup_nans_present(tmp_db: Path) -> None:
    ""
    populate(symbols=["AAPL"], db_path=tmp_db)
    with sqlite3.connect(tmp_db) as c:
        rows = c.execute(
            "SELECT date, sma20, sma200 FROM mm_indicators_daily "
            "WHERE symbol='AAPL' ORDER BY date LIMIT 25"
        ).fetchall()
    assert rows[0][1] is None
    assert rows[19][1] is None or rows[20][1] is not None
    assert all(r[2] is None for r in rows[:25])


def test_caret_symbol_stored_verbatim(tmp_db: Path) -> None:
    ""
    with sqlite3.connect(tmp_db) as c:
        for d in pd.bdate_range("2024-01-02", periods=50):
            c.execute("INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      ("^VIX", d.strftime("%Y-%m-%d"), 15, 16, 14, 15.5, 15.5, 0, "synthetic"))
        c.commit()
    populate(symbols=["^VIX"], db_path=tmp_db)
    with sqlite3.connect(tmp_db) as c:
        n = c.execute("SELECT COUNT(*) FROM mm_indicators_daily WHERE symbol='^VIX'").fetchone()[0]
    assert n == 50
