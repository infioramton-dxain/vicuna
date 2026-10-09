""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from macmini.recs.exit_signals import TIGHTEN_CAVEAT, refresh

NOW = datetime.now(timezone.utc).isoformat()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    from macmini.data.schema import init_schema
    p = tmp_path / "macmini.db"
    p.touch()
    init_schema(str(p))
    return p


def _conn(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def _add_position(c, symbol="TST", qty=100.0, avg=100.0):
    c.execute(
        "INSERT INTO mm_alpaca_positions "
        "(symbol, side, qty, avg_entry_price, snapshot_at) "
        "VALUES (?, 'long', ?, ?, ?)", (symbol, qty, avg, NOW))


def _add_entry_fill(c, symbol="TST", when="2026-07-10T14:00:00Z"):
    c.execute(
        "INSERT INTO mm_alpaca_activities "
        "(id, activity_type, transaction_time, symbol, side, qty, price, "
        " snapshot_at) VALUES (?, 'FILL', ?, ?, 'buy', 100, 100, ?)",
        (f"fixture-{symbol}-entry", when, symbol, NOW))


def _add_bars(c, symbol="TST", rows=None):
    ""
    for d, h, l, cl in rows:
        c.execute(
            "INSERT INTO mm_bars (symbol, date, open, high, low, close, "
            "volume, source) VALUES (?, ?, ?, ?, ?, ?, 1000, 'fixture')",
            (symbol, d, cl, h, l, cl))


def _add_indicators(c, symbol="TST", rows=None):
    ""
    for d, r, mh, a, adx in rows:
        c.execute(
            "INSERT INTO mm_indicators_daily "
            "(symbol, date, rsi14, macd_hist, atr14, adx14) "
            "VALUES (?, ?, ?, ?, ?, ?)", (symbol, d, r, mh, a, adx))


BASE_BARS = [
    ("2026-07-10", 101, 99, 100),
    ("2026-07-13", 104, 100, 103),
    ("2026-07-14", 108, 103, 107),
    ("2026-07-15", 112, 107, 111),
    ("2026-07-16", 115, 110, 114),
]
FLAT_IND = [
    ("2026-07-14", 55.0, 0.5, 2.0, 20.0),
    ("2026-07-15", 56.0, 0.6, 2.0, 20.0),
    ("2026-07-16", 57.0, 0.7, 2.0, 20.0),
]


class TestLevels:
    def test_row_written_with_atr_trail(self, db):
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=FLAT_IND)
            c.commit()
        out = refresh(db)
        assert out["rows"] == 1 and out["as_of_date"] == "2026-07-16"
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["hwm_price"] == 115
        assert row["trail_source"] == "atr_model"
        assert row["trail_pct"] == pytest.approx(100 * 6 / 114, rel=1e-6)
        assert row["trail_stop"] == pytest.approx(115 * (1 - 6 / 114), rel=1e-6)
        assert row["chandelier"] == pytest.approx(110.0)
        assert row["suggested_action"] == "hold"

    def test_broker_trailing_stop_wins(self, db):
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=FLAT_IND)
            c.execute(
                "INSERT INTO mm_alpaca_orders (id, symbol, side, order_type, "
                "status, stop_price, updated_at, snapshot_at) "
                "VALUES ('fixture-trail', 'TST', 'sell', 'trailing_stop', "
                "'new', 105.8, ?, ?)", (NOW, NOW))
            c.commit()
        refresh(db)
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["trail_source"] == "broker"
        assert row["trail_stop"] == pytest.approx(105.8)
        assert row["trail_pct"] == pytest.approx(100 * (1 - 105.8 / 115))

    def test_hwm_ignores_pre_entry_highs(self, db):
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c, when="2026-07-15T14:00:00Z")
            _add_bars(c, rows=[("2026-07-08", 200, 195, 198)] + BASE_BARS)
            _add_indicators(c, rows=FLAT_IND)
            c.commit()
        refresh(db)
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["entry_date"] == "2026-07-15"
        assert row["hwm_price"] == 115
        assert row["chandelier"] == pytest.approx(115 - 2.5 * 2)


class TestSuggestedAction:
    def test_operator_manual_is_exempt(self, db):
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=FLAT_IND)
            c.execute(
                "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, "
                "symbol, strategy_id, status, exit_type, created_at, "
                "updated_at) VALUES ('fixture-om', '2026-07-10', "
                "'2026-07-10', 'TST', 1, 'approved', 'operator_manual', "
                "?, ?)", (NOW, NOW))
            c.commit()
        refresh(db)
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["suggested_action"] == "operator_managed"
        assert "manually managed" in row["note"]

    def test_double_rollover_suggests_tighten_with_caveat(self, db):
        ind = [
            ("2026-07-14", 75.0, 1.0, 2.0, 20.0),
            ("2026-07-15", 70.0, 0.7, 2.0, 20.0),
            ("2026-07-16", 65.0, 0.4, 2.0, 20.0),
        ]
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=ind)
            c.commit()
        refresh(db)
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["rsi_rollover"] == 1 and row["macd_hist_shrink"] == 1
        assert row["momentum_state"] == "decelerating"
        assert row["suggested_action"] == "tighten_trail"
        assert row["note"] == TIGHTEN_CAVEAT

    def test_kronos_peaked_plus_decel_suggests_take_profit(self, db):
        ind = [
            ("2026-07-14", 75.0, 1.0, 2.0, 20.0),
            ("2026-07-15", 70.0, 1.0, 2.0, 20.0),
            ("2026-07-16", 65.0, 1.0, 2.0, 20.0),
        ]
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=ind)
            for step, cl in ((1, 112.0), (2, 110.0), (3, 108.0)):
                c.execute(
                    "INSERT INTO mm_kronos_paths (model_id, symbol, "
                    "as_of_date, step, date, close, created_at) VALUES "
                    "('kronos_base_zs_t10p90s3', 'TST', '2026-07-16', ?, "
                    "'2026-07-17', ?, ?)", (step, cl, NOW))
            c.commit()
        refresh(db)
        with _conn(db) as c:
            row = c.execute("SELECT * FROM mm_exit_signals").fetchone()
        assert row["kronos_path_peak_step"] == 0
        assert row["kronos_path_peak_price"] == pytest.approx(114.0)
        assert row["suggested_action"] == "take_profit"

    def test_no_positions_is_a_noop(self, db):
        out = refresh(db)
        assert out["rows"] == 0


class TestAdvisoryContract:
    def test_module_never_touches_order_tables(self, db):
        ""
        with _conn(db) as c:
            _add_position(c)
            _add_entry_fill(c)
            _add_bars(c, rows=BASE_BARS)
            _add_indicators(c, rows=FLAT_IND)
            c.commit()
            before = {
                t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("mm_alpaca_orders", "mm_recs_ledger")
            }
        refresh(db)
        with _conn(db) as c:
            after = {
                t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("mm_alpaca_orders", "mm_recs_ledger")
            }
        assert before == after
