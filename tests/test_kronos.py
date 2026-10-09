""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts" / "dl"))

from macmini.data import schema
from macmini.dl.kronos_infer import (
    DEFAULT_MODEL_ID, FLAG_DEGENERATE, FLAG_OK, classify_forecast,
    get_cached_forecast, load_symbol_bars, persist_forecast,
)


@pytest.fixture()
def db(tmp_path) -> Path:
    ""
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for sql in (schema.MM_BARS, schema.MM_DL_SIGNALS, schema.MM_KRONOS_SIGNALS,
                    schema.MM_KRONOS_PATHS, schema.MM_LLM_COMMENTARY):
            c.execute(sql)
        c.commit()
    return path


def _fake_forecast(as_of: str, pred_len: int = 30) -> dict:
    dates = pd.bdate_range(as_of, periods=pred_len + 1)[1:]
    closes = np.linspace(100.0, 110.0, pred_len)
    return {
        "as_of_date": as_of,
        "y_dates": pd.Series(dates).dt.strftime("%Y-%m-%d"),
        "mean_path": pd.DataFrame({
            "open": closes - 0.5, "high": closes + 1.0,
            "low": closes - 1.0, "close": closes,
            "volume": np.full(pred_len, 5000.0)}),
        "close_lo": closes - 2.0,
        "close_hi": closes + 2.0,
        "sample_5d_returns": np.array([0.01, 0.02, 0.03]),
        "sample_30d_returns": np.array([0.05, 0.10, 0.15]),
    }


class TestPersistRoundTrip:
    def test_signal_and_paths_written(self, db):
        persist_forecast(db, "TEST", _fake_forecast("2026-07-23"))
        sig, paths = get_cached_forecast(db, "TEST")
        assert sig["model_id"] == DEFAULT_MODEL_ID
        assert sig["date"] == "2026-07-23"
        assert sig["predicted_5d_return"] == pytest.approx(0.02)
        assert sig["predicted_30d_return"] == pytest.approx(0.10)
        assert sig["predicted_5d_vol"] == pytest.approx(np.std([0.01, 0.02, 0.03]))
        assert len(paths) == 30
        assert (paths["close_hi"] >= paths["close_lo"]).all()

    def test_reforecast_same_asof_replaces_not_duplicates(self, db):
        persist_forecast(db, "TEST", _fake_forecast("2026-07-23"))
        persist_forecast(db, "TEST", _fake_forecast("2026-07-23"))
        with sqlite3.connect(db) as c:
            n_sig = c.execute("SELECT COUNT(*) FROM mm_kronos_signals").fetchone()[0]
            n_path = c.execute("SELECT COUNT(*) FROM mm_kronos_paths").fetchone()[0]
        assert (n_sig, n_path) == (1, 30)

    def test_cache_miss_returns_none(self, db):
        sig, paths = get_cached_forecast(db, "NOPE")
        assert sig is None and paths is None

    def test_newer_asof_wins(self, db):
        persist_forecast(db, "TEST", _fake_forecast("2026-07-22"))
        persist_forecast(db, "TEST", _fake_forecast("2026-07-23"))
        sig, _ = get_cached_forecast(db, "TEST")
        assert sig["date"] == "2026-07-23"


class TestQualityGuard:
    ""

    def test_classify_sane(self):
        assert classify_forecast(0.069, 0.042, 0.03, 3) == FLAG_OK

    def test_classify_5d_explosion(self):
        assert classify_forecast(14.98, 0.5, 4.0, 3) == FLAG_DEGENERATE

    def test_classify_30d_reversion(self):
        assert classify_forecast(0.4, -1.2, 0.05, 3) == FLAG_DEGENERATE

    def test_classify_collapsed_sampler(self):
        assert classify_forecast(0.1, 0.2, 0.0, 3) == FLAG_DEGENERATE

    def test_classify_single_sample_zero_vol_ok(self):
        assert classify_forecast(0.1, 0.2, 0.0, 1) == FLAG_OK

    def test_persist_writes_flag(self, db):
        persist_forecast(db, "SANE", _fake_forecast("2026-07-23"))
        bad = _fake_forecast("2026-07-23")
        bad["sample_5d_returns"] = np.array([14.0, 15.0, 16.0])
        persist_forecast(db, "DEGEN", bad)
        with sqlite3.connect(db) as c:
            flags = dict(c.execute(
                "SELECT symbol, quality_flag FROM mm_kronos_signals"))
        assert flags == {"SANE": FLAG_OK, "DEGEN": FLAG_DEGENERATE}

    def test_rank_ic_excludes_degenerate_rows(self, db):
        ""
        TestRankICEval()._build(db)
        with sqlite3.connect(db) as c:
            c.execute(
                "UPDATE mm_kronos_signals SET predicted_5d_return = -577929.5, "
                "quality_flag = 'degenerate' WHERE symbol = 'S039'")
            c.commit()
        from eval_dl_gate import daily_rank_ic
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ic = daily_rank_ic(conn, "k_test", table="mm_kronos_signals")
        assert len(ic) > 0
        assert (ic > 0.99).all()


class TestLoadSymbolBars:
    def test_ascending_and_limited(self, db):
        with sqlite3.connect(db) as c:
            for i, d in enumerate(pd.bdate_range("2026-01-01", periods=40)):
                c.execute(
                    "INSERT INTO mm_bars (symbol, date, open, high, low, close, "
                    "adj_close, volume, source) VALUES (?,?,?,?,?,?,?,?,?)",
                    ("TEST", d.strftime("%Y-%m-%d"), 1 + i, 2 + i, 0.5 + i,
                     1.5 + i, 1.5 + i, 1000, "test"))
            c.commit()
        bars = load_symbol_bars(sqlite3.connect(f"file:{db}?mode=ro", uri=True),
                                "TEST", max_context=30)
        assert len(bars) == 30
        assert bars["date"].is_monotonic_increasing
        assert bars["date"].iloc[-1] == "2026-02-25"


class TestRankICEval:
    ""

    N_SYMS = 40

    def _build(self, db):
        dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-06-01", periods=20)]
        with sqlite3.connect(db) as c:
            for si in range(self.N_SYMS):
                sym = f"S{si:03d}"
                for di, d in enumerate(dates):
                    price = 100.0 * (1 + 0.001 * si) ** di
                    c.execute(
                        "INSERT INTO mm_bars (symbol, date, open, high, low, "
                        "close, adj_close, volume, source) VALUES (?,?,?,?,?,?,?,?,?)",
                        (sym, d, price, price, price, price, price, 1000, "t"))
                for d in dates[:10]:
                    c.execute(
                        "INSERT INTO mm_kronos_signals (model_id, symbol, date, "
                        "predicted_5d_return, predicted_30d_return, predicted_5d_vol, "
                        "sample_count, created_at) VALUES (?,?,?,?,?,?,?,?)",
                        ("k_test", sym, d, 0.001 * si, 0.01, 0.001, 3, "now"))
                    c.execute(
                        "INSERT INTO mm_dl_signals (model_id, symbol, date, "
                        "predicted_5d_return, predicted_class, confidence, created_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        ("p_test", sym, d, -0.001 * si, 0, 0.5, "now"))
            c.commit()

    def test_signs_recovered_and_paired_diff(self, db):
        self._build(db)
        from eval_dl_gate import daily_rank_ic
        from eval_kronos_vs_patchtst import compare

        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ic_k = daily_rank_ic(conn, "k_test", table="mm_kronos_signals")
        ic_p = daily_rank_ic(conn, "p_test", table="mm_dl_signals")
        assert len(ic_k) > 0
        assert (ic_k > 0.99).all()
        assert (ic_p < -0.99).all()

        result = compare(str(db), "k_test", "p_test")
        pdiff = result["paired_diff_kronos_minus_patchtst"]
        assert pdiff["n_common_days"] == len(ic_k)
        assert pdiff["mean_ic_diff"] > 1.9

    def test_table_param_rejects_unknown_table(self, db):
        from eval_dl_gate import daily_rank_ic
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        with pytest.raises(ValueError):
            daily_rank_ic(conn, "x", table="mm_active_picks; DROP TABLE mm_bars")
