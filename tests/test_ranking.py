""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from macmini.config import DB_PATH
from macmini.ranking.exporter import (
    ExporterConfig,
    _meta_one_liner,
    compute_rec_id,
    refresh_rankings,
)
from macmini.strategies import STRATEGY_REGISTRY


@pytest.fixture(scope="module")
def copy_of_live_db(tmp_path_factory) -> Path:
    ""
    dst = tmp_path_factory.mktemp("ranking_db") / "macmini.db"
    src = sqlite3.connect(DB_PATH, timeout=30)
    dst_conn = sqlite3.connect(dst)
    with dst_conn:
        src.backup(dst_conn)
    src.close()
    dst_conn.close()
    return dst


@pytest.fixture(scope="module")
def exporter_run(copy_of_live_db: Path) -> dict:
    ""
    return refresh_rankings(db_path=copy_of_live_db, as_of=date(2026, 6, 26))


class TestExporterRuns:
    def test_returns_summary(self, exporter_run: dict) -> None:
        assert exporter_run["as_of_date"] == "2026-06-26"
        assert exporter_run["leaderboard_rows"] >= 1
        assert "active_picks" in exporter_run


class TestLeaderboardShape:
    def test_columns_filled(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT COUNT(deflated_sharpe_avg), COUNT(qualifies) FROM mm_leaderboard"
            ).fetchone()
        assert row[0] > 0
        assert row[1] > 0

    def test_qualifies_in_zero_or_one(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            vals = {r[0] for r in c.execute("SELECT DISTINCT qualifies FROM mm_leaderboard")}
        assert vals.issubset({0, 1})


class TestRegime:
    def test_vix_close_and_regime_populated(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        ""
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT vix_close, vix_regime FROM mm_regime WHERE as_of_date = ?",
                (exporter_run["as_of_date"],),
            ).fetchone()
        assert row is not None
        vix_close, vix_regime = row
        assert vix_close is not None and 5 <= vix_close <= 100
        assert vix_regime in ("low", "normal", "high")
        if vix_close < 15:
            assert vix_regime == "low"
        elif vix_close > 25:
            assert vix_regime == "high"
        else:
            assert vix_regime == "normal"

    def test_row_for_as_of_exists(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            r = c.execute(
                "SELECT spy_50d_above_200d, fomc_within_3d, earnings_blackout_symbols "
                "FROM mm_regime WHERE as_of_date = ?",
                (exporter_run["as_of_date"],),
            ).fetchone()
        assert r is not None
        assert r[0] in (0, 1)
        assert r[1] in (0, 1)
        assert isinstance(r[2], str)


class TestStrategyMeta:
    def test_all_66_rows(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_strategy_meta").fetchone()[0]
        assert n == 66

    def test_entry_rule_non_empty(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            empties = c.execute(
                "SELECT strategy_id FROM mm_strategy_meta WHERE entry_rule = '' OR entry_rule IS NULL"
            ).fetchall()
        assert not empties, f"strategies with blank entry_rule: {empties}"

    def test_one_liner_helper_handles_minimal_class(self) -> None:
        ""
        class _NoDoc:
            name = "fallback"
            __module__ = __name__
            __doc__ = None
        out = _meta_one_liner(_NoDoc)
        assert out


class TestExportMetadata:
    def test_metadata_keys(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            meta = dict(c.execute("SELECT key, value FROM mm_export_metadata").fetchall())
        for k in ("as_of_date", "generated_at", "backtest_run_id",
                  "schema_version", "macmini_min_version"):
            assert k in meta and meta[k]


class TestLeaderboardSurvivalWindows:
    ""

    @staticmethod
    def _scratch_db(tmp_path: Path) -> Path:
        db = tmp_path / "leaderboard.db"
        with sqlite3.connect(db) as c:
            c.execute(
                """CREATE TABLE mm_backtest_runs (
                     run_id TEXT, strategy_id INTEGER, symbol TEXT, sector TEXT,
                     window_start TEXT, deflated_sharpe REAL, profit_factor REAL,
                     max_drawdown REAL, n_trades INTEGER, hit_rate REAL,
                     created_at TEXT)"""
            )
            c.execute(
                """CREATE TABLE mm_leaderboard (
                     strategy_id INTEGER, symbol TEXT, sector TEXT,
                     last_run_id TEXT, deflated_sharpe_avg REAL,
                     profit_factor_avg REAL, max_dd_worst REAL,
                     n_windows_passed INTEGER, qualifies INTEGER,
                     updated_at TEXT)"""
            )
            c.execute(
                """CREATE TABLE mm_sector_map (
                     symbol TEXT, sector TEXT, finnhub_sector TEXT,
                     market_cap_usd REAL, archetype TEXT)"""
            )
        return db

    def test_out_of_scope_cells_never_qualify(self, tmp_path: Path) -> None:
        ""
        from macmini.ranking.exporter import _refresh_leaderboard

        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            for sym, fs in (("LAR", "Metals & Mining"), ("XOM", "Energy"),
                            ("MRNA", "Biotechnology"), ("BAC", "Banking")):
                c.execute("INSERT INTO mm_sector_map VALUES (?, ?, ?, ?, ?)",
                          (sym, "Other", fs, None, None))
            for i in range(12):
                win = f"2025-{1 + i:02d}-01"
                for sym in ("LAR", "XOM", "MRNA", "BAC", "SPY"):
                    c.execute(
                        "INSERT INTO mm_backtest_runs (run_id, strategy_id, symbol, "
                        "sector, window_start, deflated_sharpe, profit_factor, "
                        "max_drawdown, n_trades, hit_rate, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ("run_test", 2, sym, "Other", win, 0.5, 1.5, -0.05, 5,
                         0.6, "2026-01-01T00:00:00+00:00"),
                    )
            c.commit()
            _refresh_leaderboard(c)
            q = dict(c.execute("SELECT symbol, qualifies FROM mm_leaderboard"))
        assert q == {"LAR": 1, "XOM": 1, "MRNA": 0, "BAC": 1, "SPY": 1}

    def test_recent_windows_decide_qualification(self, tmp_path: Path) -> None:
        from macmini.ranking.exporter import _refresh_leaderboard

        db = self._scratch_db(tmp_path)
        good = dict(deflated_sharpe=0.5, profit_factor=1.5, max_drawdown=-0.05,
                    n_trades=5, hit_rate=0.6)
        bad = dict(deflated_sharpe=-0.5, profit_factor=0.7, max_drawdown=-0.05,
                   n_trades=5, hit_rate=0.3)
        with sqlite3.connect(db) as c:
            for i in range(24):
                stats_stale_hero = good if i < 12 else bad
                stats_fresh_hero = bad if i < 12 else good
                win = f"2024-{1 + i // 2:02d}-{1 + 14 * (i % 2):02d}"
                for sym, stats in (("STALE", stats_stale_hero),
                                   ("FRESH", stats_fresh_hero)):
                    c.execute(
                        "INSERT INTO mm_backtest_runs (run_id, strategy_id, symbol, "
                        "sector, window_start, deflated_sharpe, profit_factor, "
                        "max_drawdown, n_trades, hit_rate, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ("run_test", 2, sym, "Other", win,
                         stats["deflated_sharpe"], stats["profit_factor"],
                         stats["max_drawdown"], stats["n_trades"],
                         stats["hit_rate"], "2026-01-01T00:00:00+00:00"),
                    )
            c.commit()
            n = _refresh_leaderboard(c)
            q = dict(c.execute("SELECT symbol, qualifies FROM mm_leaderboard"))
        assert n == 2
        assert q["STALE"] == 0, "old-windows-only performer must NOT qualify"
        assert q["FRESH"] == 1, "recent-windows performer must qualify"


class TestD70Screens:
    ""

    @staticmethod
    def _scratch_db(tmp_path: Path) -> Path:
        db = tmp_path / "d70.db"
        with sqlite3.connect(db) as c:
            c.execute("CREATE TABLE mm_bars (symbol TEXT, date TEXT, open REAL,"
                      " high REAL, low REAL, close REAL, adj_close REAL,"
                      " volume INTEGER, source TEXT)")
            c.execute("CREATE TABLE mm_indicators_daily (symbol TEXT, date TEXT,"
                      " atr14 REAL)")
        return db

    @staticmethod
    def _add_bars(conn, symbol, closes, volume, atr14=None):
        ""
        from datetime import date as _d, timedelta as _td
        for i, close in enumerate(closes):
            day = (_d(2026, 7, 1) + _td(days=i)).isoformat()
            conn.execute(
                "INSERT INTO mm_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')",
                (symbol, day, close, close, close, close, close, volume))
            if atr14 is not None:
                conn.execute("INSERT INTO mm_indicators_daily VALUES (?, ?, ?)",
                             (symbol, day, atr14))
        conn.commit()

    def _veto(self, db, symbol):
        from macmini.ranking.exporter import _veto_veto
        with sqlite3.connect(db) as c:
            return _veto_veto(c, symbol, date(2026, 9, 1))

    def test_liquid_steady_name_passes(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_bars(c, "OK", [100.0] * 60, volume=500_000, atr14=2.0)
        assert self._veto(db, "OK") is None

    def test_idr_class_dollar_adv_vetoed(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_bars(c, "IDR", [32.41] * 60, volume=8_000, atr14=1.5)
        reason = self._veto(db, "IDR")
        assert reason is not None and "ADV" in reason

    def test_penny_price_floor_vetoed(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_bars(c, "PNY", [4.0] * 60, volume=10_000_000, atr14=0.2)
        reason = self._veto(db, "PNY")
        assert reason is not None and "price" in reason

    def test_cvlt_class_falling_knife_vetoed(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            closes = [143.0] * 57 + [142.95, 137.40, 131.62]
            self._add_bars(c, "CVLT", closes, volume=600_000, atr14=5.06)
        reason = self._veto(db, "CVLT")
        assert reason is not None and "falling knife" in reason

    def test_knife_recovers_after_stabilizing(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            closes = [143.0] * 55 + [142.95, 137.40, 131.62, 131.5, 131.6]
            self._add_bars(c, "CVLT", closes, volume=600_000, atr14=5.06)
        assert self._veto(db, "CVLT") is None

    def test_missing_history_fails_closed(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        assert self._veto(db, "GHOST") is not None
        with sqlite3.connect(db) as c:
            self._add_bars(c, "NULLVOL", [50.0] * 60, volume=None)
        assert self._veto(db, "NULLVOL") is not None

    def test_missing_atr_skips_knife_only(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            closes = [143.0] * 57 + [142.95, 137.40, 131.62]
            self._add_bars(c, "NOATR", closes, volume=600_000, atr14=None)
        assert self._veto(db, "NOATR") is None


class TestComputeRecId:
    ""

    def test_length_and_hex(self) -> None:
        r = compute_rec_id("2026-07-02", "QQQI", 7)
        assert len(r) == 16
        assert all(ch in "0123456789abcdef" for ch in r)

    def test_deterministic(self) -> None:
        assert (
            compute_rec_id("2026-07-02", "QQQI", 7)
            == compute_rec_id("2026-07-02", "QQQI", 7)
        )

    def test_input_changes_change_id(self) -> None:
        base = compute_rec_id("2026-07-02", "QQQI", 7)
        assert compute_rec_id("2026-07-03", "QQQI", 7) != base
        assert compute_rec_id("2026-07-02", "AAPL", 7) != base
        assert compute_rec_id("2026-07-02", "QQQI", 3) != base

    def test_pinned_value(self) -> None:
        ""
        assert compute_rec_id("2026-07-02", "QQQI", 7) == "e81b08e759568762"


class TestActivePicksRecId:
    ""

    def test_rec_id_populated(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute(
                "SELECT rec_id, as_of_date FROM mm_active_picks"
            ).fetchall()
        if not rows:
            pytest.skip("no active picks in this run — coverage assertions n/a")
        for rec_id, as_of in rows:
            assert rec_id is not None and len(rec_id) == 16
            assert as_of == exporter_run["as_of_date"]

    def test_rec_id_matches_formula(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        ""
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute(
                "SELECT rec_id, symbol, strategy_id, as_of_date FROM mm_active_picks"
            ).fetchall()
        if not rows:
            pytest.skip("no active picks in this run — coverage assertions n/a")
        for rec_id, symbol, sid, as_of in rows:
            assert rec_id == compute_rec_id(as_of, symbol, sid)

    def test_one_pick_per_symbol_and_in_scope(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        from macmini.ranking.exporter import _out_of_scope_symbols
        with sqlite3.connect(copy_of_live_db) as c:
            syms = [r[0] for r in c.execute("SELECT symbol FROM mm_active_picks")]
            barred = _out_of_scope_symbols(c)
            leaked = [r[0] for r in c.execute(
                "SELECT symbol FROM mm_leaderboard WHERE qualifies = 1")
                if r[0] in barred]
        assert len(syms) == len(set(syms)), f"duplicate symbols shipped: {syms}"
        assert not set(syms) & barred
        assert not leaked, f"out-of-scope cells qualify: {leaked}"

    def test_rec_id_unique(self, exporter_run: dict, copy_of_live_db: Path) -> None:
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute("SELECT rec_id FROM mm_active_picks").fetchall()
        ids = [r[0] for r in rows]
        assert len(ids) == len(set(ids))


class TestDlSoftFilter:
    ""

    _MODEL_ID = "patchtst_test_20260704"

    @staticmethod
    def _seed_dl_signals(db: Path, as_of_str: str, preds: dict[str, float]) -> None:
        ""
        with sqlite3.connect(db) as c:
            c.execute("DELETE FROM mm_dl_signals WHERE model_id = ?",
                      (TestDlSoftFilter._MODEL_ID,))
            c.execute("DELETE FROM mm_dl_runs WHERE run_id = ?",
                      ("test_infer_dl_soft_filter",))
            for sym, pred in preds.items():
                c.execute(
                    "INSERT INTO mm_dl_signals (model_id, symbol, date, "
                    "predicted_5d_return, predicted_class, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (TestDlSoftFilter._MODEL_ID, sym, as_of_str, pred,
                     1 if pred > 0 else 0, min(1.0, abs(pred) / 0.05),
                     "2026-07-04T00:00:00+00:00"),
                )
            import json
            c.execute(
                "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
                "epochs_or_timesteps, best_val_metric, best_metric_name, "
                "config_json, checkpoint_path, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test_infer_dl_soft_filter", "patchtst_inference",
                 "2099-01-01T00:00:00+00:00", "2099-01-01T00:00:01+00:00",
                 None, 0.548, "val_dir_acc",
                 json.dumps({"model_id": TestDlSoftFilter._MODEL_ID}),
                 "/tmp/fake.ckpt", "test seed"),
            )
            c.commit()

    def test_load_dl_predictions_uses_latest_run(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        from macmini.ranking.exporter import _load_dl_predictions
        as_of = exporter_run["as_of_date"]
        self._seed_dl_signals(copy_of_live_db, as_of, {"AAPL": 0.03, "MSFT": -0.02})
        with sqlite3.connect(copy_of_live_db) as c:
            preds, model_id = _load_dl_predictions(c, as_of, None)
        assert model_id == self._MODEL_ID
        assert preds.get("AAPL") == pytest.approx(0.03)
        assert preds.get("MSFT") == pytest.approx(-0.02)

    def test_no_penalty_when_pred_missing(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        as_of = exporter_run["as_of_date"]
        self._seed_dl_signals(copy_of_live_db, as_of, {})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(dl_soft_filter=True))
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute("SELECT rationale FROM mm_active_picks").fetchall()
        if not rows:
            pytest.skip("no active picks — coverage n/a")
        assert all("| DL=" not in r[0] for r in rows), \
            "no DL rows seeded → no rationale should carry a DL suffix"

    def test_penalty_shows_in_rationale(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        target_symbol = picks[0][0]
        self._seed_dl_signals(copy_of_live_db, as_of, {target_symbol: -0.04})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(dl_soft_filter=True, dl_penalty=0.5))
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT rationale FROM mm_active_picks WHERE symbol = ?", (target_symbol,),
            ).fetchone()
        if row is None:
            pytest.skip(f"{target_symbol} dropped after re-rank — pick set shifted")
        assert "| DL=-4.00%" in row[0]
        assert "×0.5" in row[0]

    def test_positive_pred_annotated_but_not_penalized(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        target_symbol = picks[0][0]
        self._seed_dl_signals(copy_of_live_db, as_of, {target_symbol: 0.025})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(dl_soft_filter=True))
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT rationale FROM mm_active_picks WHERE symbol = ?", (target_symbol,),
            ).fetchone()
        if row is None:
            pytest.skip(f"{target_symbol} dropped after re-rank")
        assert "| DL=+2.50%" in row[0]
        assert "×" not in row[0], "positive prediction must NOT show a penalty marker"

    def test_uses_latest_prediction_before_as_of(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        from macmini.ranking.exporter import _load_dl_predictions
        as_of = exporter_run["as_of_date"]
        from datetime import date as _date, timedelta
        as_of_d = _date.fromisoformat(as_of)
        near = (as_of_d - timedelta(days=1)).isoformat()
        far  = (as_of_d - timedelta(days=3)).isoformat()
        with sqlite3.connect(copy_of_live_db) as c:
            c.execute("DELETE FROM mm_dl_signals WHERE model_id = ?",
                      (self._MODEL_ID,))
            for d, val in [(far, 0.01), (near, 0.09)]:
                c.execute(
                    "INSERT INTO mm_dl_signals (model_id, symbol, date, "
                    "predicted_5d_return, predicted_class, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self._MODEL_ID, "AAPL", d, val, 1, 0.5, "2026-07-04T00:00:00+00:00"),
                )
            c.commit()
            preds, _ = _load_dl_predictions(c, as_of, self._MODEL_ID)
        assert preds["AAPL"] == pytest.approx(0.09)

    def test_drops_stale_predictions(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        from macmini.ranking.exporter import _load_dl_predictions, DL_MAX_STALENESS_DAYS
        from datetime import date as _date, timedelta
        as_of = exporter_run["as_of_date"]
        stale = (_date.fromisoformat(as_of) - timedelta(days=DL_MAX_STALENESS_DAYS + 3)).isoformat()
        with sqlite3.connect(copy_of_live_db) as c:
            c.execute("DELETE FROM mm_dl_signals WHERE model_id = ?",
                      (self._MODEL_ID,))
            c.execute(
                "INSERT INTO mm_dl_signals (model_id, symbol, date, "
                "predicted_5d_return, predicted_class, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._MODEL_ID, "AAPL", stale, -0.05, 0, 0.5, "2026-07-04T00:00:00+00:00"),
            )
            c.commit()
            preds, _ = _load_dl_predictions(c, as_of, self._MODEL_ID)
        assert "AAPL" not in preds

    def test_off_switch_disables_filter(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        ""
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        self._seed_dl_signals(copy_of_live_db, as_of, {picks[0][0]: -0.05})
        result = refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                                  cfg=ExporterConfig(dl_soft_filter=False))
        assert result["dl_soft_filter"] is False
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute("SELECT rationale FROM mm_active_picks").fetchall()
        assert all("| DL=" not in r[0] for r in rows)


class TestRlAnnotation:
    ""

    _POLICY_ID = "test_ppo_20260704"

    @staticmethod
    def _seed_rl_signals(db: Path, as_of_str: str, actions: dict[str, int]) -> None:
        with sqlite3.connect(db) as c:
            c.execute("DELETE FROM mm_rl_signals WHERE policy_id = ?",
                      (TestRlAnnotation._POLICY_ID,))
            c.execute("DELETE FROM mm_dl_runs WHERE run_id = ?",
                      ("test_infer_rl_annotation",))
            for sym, act in actions.items():
                c.execute(
                    "INSERT INTO mm_rl_signals (policy_id, symbol, date, action, "
                    "suggested_size_pct, policy_logits_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (TestRlAnnotation._POLICY_ID, sym, as_of_str, act,
                     0.05 if act == 1 else 0.0, "{}",
                     "2026-07-04T00:00:00+00:00"),
                )
            import json
            c.execute(
                "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
                "epochs_or_timesteps, best_val_metric, best_metric_name, "
                "config_json, checkpoint_path, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test_infer_rl_annotation", "maskable_ppo_inference",
                 "2099-01-01T00:00:00+00:00", "2099-01-01T00:00:01+00:00",
                 None, None, None,
                 json.dumps({"policy_id": TestRlAnnotation._POLICY_ID}),
                 "/tmp/fake.zip", "test seed"),
            )
            c.commit()

    def test_load_rl_actions_uses_latest_run(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        from macmini.ranking.exporter import _load_rl_actions
        as_of = exporter_run["as_of_date"]
        self._seed_rl_signals(copy_of_live_db, as_of, {"AAPL": 1, "MSFT": 0})
        with sqlite3.connect(copy_of_live_db) as c:
            acts, policy_id = _load_rl_actions(c, as_of, None)
        assert policy_id == self._POLICY_ID
        assert acts.get("AAPL") == 1
        assert acts.get("MSFT") == 0

    def test_buy_action_shows_in_rationale(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        target = picks[0][0]
        self._seed_rl_signals(copy_of_live_db, as_of, {target: 1})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(rl_annotate=True))
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT rationale FROM mm_active_picks WHERE symbol = ?", (target,),
            ).fetchone()
        if row is None:
            pytest.skip(f"{target} dropped after re-rank")
        assert "| RL=BUY" in row[0]

    def test_hold_action_shows_in_rationale(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        target = picks[0][0]
        self._seed_rl_signals(copy_of_live_db, as_of, {target: 0})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(rl_annotate=True))
        with sqlite3.connect(copy_of_live_db) as c:
            row = c.execute(
                "SELECT rationale FROM mm_active_picks WHERE symbol = ?", (target,),
            ).fetchone()
        if row is None:
            pytest.skip(f"{target} dropped after re-rank")
        assert "| RL=HOLD" in row[0]

    def test_no_annotation_when_row_missing(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        as_of = exporter_run["as_of_date"]
        self._seed_rl_signals(copy_of_live_db, as_of, {})
        refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                         cfg=ExporterConfig(rl_annotate=True))
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute("SELECT rationale FROM mm_active_picks").fetchall()
        if not rows:
            pytest.skip("no active picks — coverage n/a")
        assert all("| RL=" not in r[0] for r in rows)

    def test_off_switch_disables_annotation(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        as_of = exporter_run["as_of_date"]
        with sqlite3.connect(copy_of_live_db) as c:
            picks = c.execute("SELECT symbol FROM mm_active_picks").fetchall()
        if not picks:
            pytest.skip("no active picks — coverage n/a")
        self._seed_rl_signals(copy_of_live_db, as_of, {picks[0][0]: 1})
        result = refresh_rankings(db_path=copy_of_live_db, as_of=date.fromisoformat(as_of),
                                  cfg=ExporterConfig(rl_annotate=False))
        assert result["rl_annotate"] is False
        with sqlite3.connect(copy_of_live_db) as c:
            rows = c.execute("SELECT rationale FROM mm_active_picks").fetchall()
        assert all("| RL=" not in r[0] for r in rows)

    def test_drops_stale_rl_actions(
        self, exporter_run: dict, copy_of_live_db: Path,
    ) -> None:
        from macmini.ranking.exporter import _load_rl_actions, RL_MAX_STALENESS_DAYS
        from datetime import date as _date, timedelta
        as_of = exporter_run["as_of_date"]
        stale = (_date.fromisoformat(as_of) - timedelta(days=RL_MAX_STALENESS_DAYS + 3)).isoformat()
        with sqlite3.connect(copy_of_live_db) as c:
            c.execute("DELETE FROM mm_rl_signals WHERE policy_id = ?", (self._POLICY_ID,))
            c.execute(
                "INSERT INTO mm_rl_signals (policy_id, symbol, date, action, "
                "suggested_size_pct, policy_logits_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._POLICY_ID, "AAPL", stale, 1, 0.05, "{}",
                 "2026-07-04T00:00:00+00:00"),
            )
            c.commit()
            acts, _ = _load_rl_actions(c, as_of, self._POLICY_ID)
        assert "AAPL" not in acts


class TestD63QualityScreen:
    ""

    @staticmethod
    def _scratch_db(tmp_path: Path) -> Path:
        db = tmp_path / "d63.db"
        with sqlite3.connect(db) as c:
            c.execute("CREATE TABLE mm_bars (symbol TEXT, date TEXT, close REAL)")
            c.execute("CREATE TABLE mm_kronos_signals (symbol TEXT, date TEXT,"
                      " predicted_5d_return REAL, predicted_30d_return REAL,"
                      " quality_flag TEXT)")
            c.execute("CREATE TABLE sentiment (symbol TEXT, window_end TEXT,"
                      " sentiment REAL, materiality REAL)")
            c.execute("CREATE TABLE mm_earnings_events (symbol TEXT,"
                      " report_date TEXT, eps_surprise_pct REAL)")
        return db

    @staticmethod
    def _add_closes(conn, symbol, daily_ret, n=61, start_price=100.0):
        ""
        from datetime import date as _d, timedelta as _td
        px = start_price
        for i in range(n):
            day = (_d(2026, 8, 31) - _td(days=n - 1 - i)).isoformat()
            px *= 1.0 + (daily_ret if i % 2 else -daily_ret)
            conn.execute("INSERT INTO mm_bars VALUES (?, ?, ?)", (symbol, day, px))
        conn.commit()

    def _stack_and_reason(self, db, symbol):
        from macmini.ranking.exporter import _qs_disqualify, _model_stack
        with sqlite3.connect(db) as c:
            stack = _model_stack(c, symbol, date(2026, 8, 31))
        return stack, _qs_disqualify(stack)

    def test_lar_class_profile_disqualified(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "LAR", 0.044)
            for i, d in enumerate(("2026-08-10", "2026-05-10", "2026-02-10",
                                   "2025-11-10", "2025-08-10")):
                c.execute("INSERT INTO mm_earnings_events VALUES (?, ?, ?)",
                          ("LAR", d, -25.0))
            c.execute("INSERT INTO sentiment VALUES (?, ?, ?, ?)",
                      ("LAR", "2026-08-28", -1.0, 0.8))
            c.commit()
        stack, reason = self._stack_and_reason(db, "LAR")
        assert stack["realized_vol_60d_ann"] > 0.65
        assert stack["earnings_miss_streak"] == 5
        assert reason is not None and "vol" in reason

    def test_miss_streak_alone_disqualifies(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "MISS", 0.01)
            c.execute("INSERT INTO mm_earnings_events VALUES (?, ?, ?)",
                      ("MISS", "2026-08-10", -5.0))
            c.execute("INSERT INTO mm_earnings_events VALUES (?, ?, ?)",
                      ("MISS", "2026-05-10", -12.0))
            c.commit()
        _, reason = self._stack_and_reason(db, "MISS")
        assert reason is not None and "misses" in reason

    def test_meta_class_profile_passes(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "META", 0.03)
            c.execute("INSERT INTO mm_earnings_events VALUES (?, ?, ?)",
                      ("META", "2026-07-30", -3.0))
            c.execute("INSERT INTO mm_earnings_events VALUES (?, ?, ?)",
                      ("META", "2026-04-30", +8.0))
            c.execute("INSERT INTO sentiment VALUES (?, ?, ?, ?)",
                      ("META", "2026-08-28", -1.0, 0.5))
            c.execute("INSERT INTO mm_kronos_signals VALUES (?, ?, ?, ?, ?)",
                      ("META", "2026-08-28", 0.117, 0.005, "ok"))
            c.commit()
        stack, reason = self._stack_and_reason(db, "META")
        assert reason is None
        assert stack["kronos_5d"] == 0.117

    def test_extreme_material_sentiment_disqualifies(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "BAD", 0.01)
            c.execute("INSERT INTO sentiment VALUES (?, ?, ?, ?)",
                      ("BAD", "2026-08-28", -0.9, 0.8))
            c.commit()
        _, reason = self._stack_and_reason(db, "BAD")
        assert reason is not None and "sentiment" in reason

    def test_no_model_coverage_passes_open(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "PLAIN", 0.01)
            c.commit()
        stack, reason = self._stack_and_reason(db, "PLAIN")
        assert reason is None
        assert stack["kronos_5d"] is None and stack["sentiment_last7d"] is None

    def test_negative_kronos_is_not_a_veto(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "TWLO", 0.02)
            c.execute("INSERT INTO mm_kronos_signals VALUES (?, ?, ?, ?, ?)",
                      ("TWLO", "2026-08-28", -0.34, -0.51, "ok"))
            c.commit()
        stack, reason = self._stack_and_reason(db, "TWLO")
        assert reason is None
        assert stack["kronos_5d"] == -0.34

    def test_stale_or_flagged_kronos_ignored(self, tmp_path: Path) -> None:
        db = self._scratch_db(tmp_path)
        with sqlite3.connect(db) as c:
            self._add_closes(c, "OLDK", 0.01)
            c.execute("INSERT INTO mm_kronos_signals VALUES (?, ?, ?, ?, ?)",
                      ("OLDK", "2026-08-10", 0.05, 0.10, "ok"))
            c.execute("INSERT INTO mm_kronos_signals VALUES (?, ?, ?, ?, ?)",
                      ("OLDK", "2026-08-28", 0.02, 0.04, "stale_bars_d71"))
            c.commit()
        stack, _ = self._stack_and_reason(db, "OLDK")
        assert stack["kronos_5d"] is None


class TestRecencyAggregation:
    ""

    def test_weighted_median_downweights_old_capped_windows(self) -> None:
        from macmini.ranking.exporter import _winsor_wmedian
        vals = [999.0, 999.0, 999.0, 999.0, 999.0, 2.71, 1.23, 1.10]
        ages = [11, 10, 9, 8, 7, 2, 1, 0]
        from macmini.ranking.exporter import _recency_weight
        w = [_recency_weight(a) for a in ages]
        assert _winsor_wmedian(vals, w) < 3.0
        assert _winsor_wmedian(vals, [1.0] * len(vals)) == 10.0

    def test_weighted_median_empty_and_nan(self) -> None:
        from macmini.ranking.exporter import _winsor_wmedian
        assert _winsor_wmedian([], []) == 0.0
        assert _winsor_wmedian([float("nan"), 2.0], [1.0, 1.0]) == 2.0

    def test_recency_weight_decay(self) -> None:
        from macmini.ranking.exporter import _recency_weight
        assert _recency_weight(0) == 1.0
        assert 0.05 < _recency_weight(11) < 0.15


class TestRetiredStrategies:
    ""

    def test_retired_strategy_never_qualifies(self, tmp_path: Path) -> None:
        from macmini.ranking.exporter import RETIRED_STRATEGY_IDS, _refresh_leaderboard

        db = TestLeaderboardSurvivalWindows()._scratch_db(tmp_path)
        retired = min(RETIRED_STRATEGY_IDS)
        live = next(i for i in range(1, 67) if i not in RETIRED_STRATEGY_IDS)
        funnel: list = []
        with sqlite3.connect(db) as c:
            for sid in (retired, live):
                for i in range(12):
                    c.execute(
                        "INSERT INTO mm_backtest_runs (run_id, strategy_id, symbol, "
                        "sector, window_start, deflated_sharpe, profit_factor, "
                        "max_drawdown, n_trades, hit_rate, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        ("run_test", sid, "BAC", "Other", f"2025-{i + 1:02d}-01",
                         0.5, 1.5, -0.05, 5, 0.6, "2026-01-01T00:00:00+00:00"))
            c.commit()
            _refresh_leaderboard(c, funnel)
            q = dict(c.execute("SELECT strategy_id, qualifies FROM mm_leaderboard"))
        assert q == {retired: 0, live: 1}
        assert funnel[0][1] == 2 and funnel[-1][1] == 1
