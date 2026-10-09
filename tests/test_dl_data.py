""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from macmini.dl.data import (
    compute_indicators_for_bars,
    materialize_all,
    materialize_symbol,
)


def _synthetic_bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = 100 + rng.normal(0, 1, n).cumsum() * 0.5
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.uniform(0, 0.5, n),
        "low": close - rng.uniform(0, 0.5, n),
        "close": close,
        "adj_close": close,
        "volume": rng.integers(500_000, 2_000_000, n),
    }, index=idx)


class TestIndicators:
    def test_expected_columns(self) -> None:
        bars = _synthetic_bars(50)
        out = compute_indicators_for_bars(bars)
        expected = {"sma20", "sma50", "sma200", "ema12", "ema26", "ema200",
                    "rsi14", "atr14", "macd_line", "macd_signal", "macd_hist",
                    "bb_upper", "bb_middle", "bb_lower", "bb_width",
                    "williams_r14", "ibs", "volume_zscore_50"}
        assert expected.issubset(out.columns)
        assert list(out.index) == list(bars.index)

    def test_empty_bars_returns_empty(self) -> None:
        assert compute_indicators_for_bars(pd.DataFrame()).empty

    def test_no_lookahead_in_indicators(self) -> None:
        ""
        bars_a = _synthetic_bars(120)
        bars_b = bars_a.copy()
        bars_b.iloc[-30:, bars_b.columns.get_loc("close")] += 50.0
        ind_a = compute_indicators_for_bars(bars_a)
        ind_b = compute_indicators_for_bars(bars_b)
        for col in ind_a.columns:
            a = ind_a.iloc[50][col]
            b = ind_b.iloc[50][col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b, nan_ok=True), \
                f"{col} at row 50 changed when only rows ≥90 were perturbed → leakage"


class TestMaterializeSymbol:
    @pytest.fixture
    def tmp_db_with_bars(self, tmp_path: Path) -> Path:
        ""
        db = tmp_path / "tiny.db"
        with sqlite3.connect(db) as c:
            c.executescript("""
                CREATE TABLE mm_bars (
                    symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, adj_close REAL, volume INTEGER, source TEXT,
                    PRIMARY KEY (symbol, date));
                CREATE TABLE mm_sector_map (
                    symbol TEXT PRIMARY KEY, sector TEXT, finnhub_sector TEXT,
                    market_cap_usd REAL, archetype TEXT);
                CREATE TABLE mm_fundamentals_quarterly (
                    symbol TEXT, period_end TEXT, revenue REAL, earnings REAL,
                    eps REAL, operating_cash_flow REAL, free_cash_flow REAL,
                    total_debt REAL, total_equity REAL, ebitda REAL,
                    market_cap REAL, pe_ratio REAL, pb_ratio REAL,
                    ev_ebitda REAL, roic REAL, shares_outstanding REAL,
                    dividend_yield REAL, buyback_yield REAL, capex REAL,
                    f_score INTEGER, source TEXT, fetched_at TEXT,
                    PRIMARY KEY (symbol, period_end));
                CREATE TABLE mm_analyst_estimates (
                    symbol TEXT, as_of_date TEXT, fy0_eps_consensus REAL,
                    fy1_eps_consensus REAL, eps_revision_30d REAL,
                    rating_buy INTEGER, rating_hold INTEGER, rating_sell INTEGER,
                    PRIMARY KEY (symbol, as_of_date));
                CREATE TABLE mm_training_labels (
                    symbol TEXT, date TEXT, forward_1d_return REAL,
                    forward_5d_return REAL, forward_10d_return REAL,
                    forward_5d_class INTEGER, forward_5d_sharpe REAL,
                    max_dd_5d REAL, PRIMARY KEY (symbol, date));
                CREATE TABLE sentiment (
                    symbol TEXT, window_start TEXT, window_end TEXT,
                    published_at_max TEXT, scored_at TEXT, score_source TEXT,
                    article_count INTEGER, source_count INTEGER,
                    truncated INTEGER, status TEXT,
                    sentiment REAL, sentiment_confidence REAL,
                    materiality REAL, geo_risk INTEGER,
                    contract_award INTEGER, guidance INTEGER, mna INTEGER,
                    regulatory_export INTEGER, litigation INTEGER,
                    analyst_action INTEGER, commodity_move INTEGER,
                    PRIMARY KEY (symbol, window_start));
            """)
            bars = _synthetic_bars(250)
            for d, row in bars.iterrows():
                c.execute("INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          ("AAPL", d.strftime("%Y-%m-%d"),
                           row["open"], row["high"], row["low"], row["close"],
                           row["adj_close"], int(row["volume"]), "synthetic"))
            c.execute("INSERT INTO mm_sector_map VALUES (?, ?, ?, ?, ?)",
                      ("AAPL", "Software", "Tech", 3e12, "mega_cap_tech"))
            c.commit()
        return db

    def test_materialize_creates_parquet(self, tmp_db_with_bars: Path, tmp_path: Path) -> None:
        out = materialize_symbol("AAPL", tmp_path / "shards", db_path=tmp_db_with_bars)
        assert out is not None
        assert out.exists()
        assert out.suffix == ".parquet"
        df = pq.read_table(out).to_pandas()
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert "sma200" in df.columns
        assert "sector" in df.columns
        assert (df["symbol"] == "AAPL").all()
        assert (df["sector"] == "Software").all()

    def test_materialize_unknown_symbol_returns_none(self, tmp_db_with_bars: Path, tmp_path: Path) -> None:
        out = materialize_symbol("NOEXIST", tmp_path / "shards", db_path=tmp_db_with_bars)
        assert out is None

    def test_caret_symbol_safe_filename(self, tmp_db_with_bars: Path, tmp_path: Path) -> None:
        ""
        with sqlite3.connect(tmp_db_with_bars) as c:
            for d in pd.bdate_range("2024-01-02", periods=30):
                c.execute("INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          ("^VIX", d.strftime("%Y-%m-%d"), 15, 16, 14, 15.5, 15.5, 0, "synthetic"))
            c.commit()
        out = materialize_symbol("^VIX", tmp_path / "shards", db_path=tmp_db_with_bars)
        assert out is not None
        assert "^" not in out.name

    def test_materialize_all_loops(self, tmp_db_with_bars: Path, tmp_path: Path) -> None:
        out = materialize_all(symbols=["AAPL"], output_dir=tmp_path / "shards", db_path=tmp_db_with_bars)
        assert out["written"] == 1
        assert out["skipped"] == 0
