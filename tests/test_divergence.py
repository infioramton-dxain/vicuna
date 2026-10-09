""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from macmini.recs.divergence import check, summary_text


NOW = datetime.now(timezone.utc)
FRESH = NOW.isoformat()
STALE = (NOW - timedelta(hours=48)).isoformat()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    from macmini.data.schema import init_schema
    p = tmp_path / "macmini.db"
    p.touch()
    init_schema(str(p))
    return p


def _add_position(c, symbol="HPQ", qty=222.0, snapshot_at=FRESH):
    c.execute(
        "INSERT INTO mm_alpaca_positions (symbol, side, qty, snapshot_at) "
        "VALUES (?, 'long', ?, ?)", (symbol, qty, snapshot_at))


def _add_order(c, oid, symbol="HPQ", order_type="stop", status="held",
               qty=222.0, stop_price=None, limit_price=None,
               snapshot_at=FRESH):
    c.execute(
        "INSERT INTO mm_alpaca_orders "
        "(id, symbol, side, order_type, qty, stop_price, limit_price, "
        " status, snapshot_at) VALUES (?, ?, 'sell', ?, ?, ?, ?, ?, ?)",
        (oid, symbol, order_type, qty, stop_price, limit_price, status,
         snapshot_at))


def _add_ledger_open(c, rec_id="aaaabbbbccccdddd", symbol="HPQ", qty=222,
                     stop_price=20.67, target_price=26.96,
                     exit_status="open"):
    c.execute(
        "INSERT INTO mm_recs_ledger "
        "(rec_id, as_of_date, entry_date, symbol, strategy_id, status, "
        " fill_status, fill_qty, stop_price, target_price, exit_status, "
        " created_at, updated_at) "
        "VALUES (?, '2026-07-06', '2026-07-07', ?, 17, 'approved', "
        "'filled', ?, ?, ?, ?, ?, ?)",
        (rec_id, symbol, qty, stop_price, target_price, exit_status,
         FRESH, FRESH))


def _kinds(result):
    return {(f["kind"], f["severity"]) for f in result["findings"]}


def _protected_position(c, symbol="HPQ", qty=222.0,
                        stop_price=20.67, limit_price=26.96):
    _add_position(c, symbol, qty)
    _add_order(c, f"stop-{symbol}", symbol, "stop", "held", qty,
               stop_price=stop_price)
    _add_order(c, f"tp-{symbol}", symbol, "limit", "new", qty,
               limit_price=limit_price)


class TestClean:
    def test_matching_state_is_clean(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c)
            _add_ledger_open(c)
        result = check(db)
        assert result["findings"] == []
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM mm_broker_divergences").fetchone()[0] == 0

    def test_exited_ledger_rows_are_ignored(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_order(c, "o1", "FCX", "stop", "filled", 81, stop_price=56.25)
            _add_ledger_open(c, rec_id="ffffbbbbccccdddd", symbol="FCX",
                             exit_status="stopped_out")
        result = check(db)
        assert ("ledger_open_no_position", "critical") not in _kinds(result)


class TestDesyncShapes:
    def test_ledger_open_broker_flat_is_critical(self, db):
        with sqlite3.connect(db) as c:
            _add_order(c, "o1", "ZZZ", "limit", "canceled", 1)
            _add_ledger_open(c)
        result = check(db)
        assert ("ledger_open_no_position", "critical") in _kinds(result)
        f = next(f for f in result["findings"]
                 if f["kind"] == "ledger_open_no_position")
        assert "never force-sell" in f["detail"]

    def test_approved_no_order_is_critical(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_order(c, "o1", "ZZZ", "limit", "canceled", 1)
            c.execute(
                "INSERT INTO mm_recs_ledger "
                "(rec_id, as_of_date, entry_date, symbol, strategy_id, status, "
                " decided_at, created_at, updated_at) "
                "VALUES ('fixture-avptshape1', '2026-07-09', '2026-07-10', 'AVPT', 17, "
                "'approved', ?, ?, ?)",
                ((NOW - timedelta(hours=3)).isoformat(), FRESH, FRESH))
        result = check(db)
        assert ("approved_no_order", "critical") in _kinds(result)

    def test_recently_approved_no_order_is_silent(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_order(c, "o1", "ZZZ", "limit", "canceled", 1)
            c.execute(
                "INSERT INTO mm_recs_ledger "
                "(rec_id, as_of_date, entry_date, symbol, strategy_id, status, "
                " decided_at, created_at, updated_at) "
                "VALUES ('fixture-avptshape2', '2026-07-09', '2026-07-10', 'AVPT', 17, "
                "'approved', ?, ?, ?)",
                ((NOW - timedelta(minutes=30)).isoformat(), FRESH, FRESH))
        result = check(db)
        assert ("approved_no_order", "critical") not in _kinds(result)

    def test_approved_with_order_id_is_silent(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_order(c, "o1", "ZZZ", "limit", "canceled", 1)
            c.execute(
                "INSERT INTO mm_recs_ledger "
                "(rec_id, as_of_date, entry_date, symbol, strategy_id, status, "
                " alpaca_order_id, decided_at, created_at, updated_at) "
                "VALUES ('fixture-avptshape3', '2026-07-09', '2026-07-10', 'AVPT', 17, "
                "'approved', 'ord-123', ?, ?, ?)",
                ((NOW - timedelta(hours=3)).isoformat(), FRESH, FRESH))
        result = check(db)
        assert ("approved_no_order", "critical") not in _kinds(result)

    def test_order_filled_but_ledger_pending_is_critical(self, db):
        ""
        with sqlite3.connect(db) as c:
            c.execute(
                "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, symbol, "
                "strategy_id, status, fill_status, alpaca_order_id, decided_at, "
                "created_at, updated_at) "
                "VALUES ('fixture-qarsa-meta', '2026-08-21', '2026-08-21', 'META', 7, "
                "'approved', 'pending', 'ord-meta-parent', '2026-08-21T09:23:29-04:00', "
                "'2026-08-21T09:23:29-04:00', '2026-08-21T09:23:29-04:00')")
            c.execute(
                "INSERT INTO mm_alpaca_orders (id, symbol, side, order_type, qty, "
                "filled_qty, filled_avg_price, filled_at, status, snapshot_at) "
                "VALUES ('ord-meta-parent', 'META', 'buy', 'limit', 9, 9, 546.58, "
                "'2026-08-21T13:34:14Z', 'filled', ?)", (FRESH,))
        res = check(db)
        assert ("order_filled_ledger_stale", "critical") in _kinds(res)
        assert res["critical"] >= 1

    def test_placed_order_still_resting_is_silent(self, db):
        ""
        with sqlite3.connect(db) as c:
            c.execute(
                "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, symbol, "
                "strategy_id, status, fill_status, alpaca_order_id, decided_at, "
                "created_at, updated_at) "
                "VALUES ('fixture-resting-asml', '2026-10-06', '2026-10-07', 'ASML', 7, "
                "'approved', 'placed', 'ord-asml-parent', '2026-10-06T09:26:54-04:00', "
                "'2026-10-06T09:26:54-04:00', '2026-10-06T09:26:54-04:00')")
            c.execute(
                "INSERT INTO mm_alpaca_orders (id, symbol, side, order_type, qty, "
                "filled_qty, status, snapshot_at) "
                "VALUES ('ord-asml-parent', 'ASML', 'buy', 'limit', 3, 0, 'new', ?)",
                (FRESH,))
        res = check(db)
        assert not any(k == "order_filled_ledger_stale" for k, _ in _kinds(res))

    def _add_buy_parent(self, c, oid, symbol, submitted_at, rec_id=None, status="new"):
        c.execute(
            "INSERT INTO mm_alpaca_orders (id, symbol, side, order_type, qty, "
            "limit_price, status, submitted_at, snapshot_at) "
            "VALUES (?, ?, 'buy', 'limit', 10, 50.0, ?, ?, ?)",
            (oid, symbol, status, submitted_at, FRESH))
        if rec_id:
            c.execute(
                "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, symbol, "
                "strategy_id, status, fill_status, alpaca_order_id, decided_at, "
                "created_at, updated_at) "
                "VALUES (?, '2026-10-06', '2026-10-07', ?, 7, 'approved', 'placed', ?, "
                "?, ?, ?)", (rec_id, symbol, oid, submitted_at, submitted_at, submitted_at))

    def test_resting_entry_from_yesterday_is_warn(self, db):
        ""
        yesterday = (NOW - timedelta(days=1, hours=2)).isoformat()
        with sqlite3.connect(db) as c:
            self._add_buy_parent(c, "ord-teck-old", "TECK", yesterday, rec_id="fixture-teck")
        res = check(db)
        assert ("stale_resting_entry", "warn") in _kinds(res)

    def test_resting_entry_from_today_is_silent(self, db):
        with sqlite3.connect(db) as c:
            self._add_buy_parent(c, "ord-adm-new", "ADM", FRESH, rec_id="fixture-adm")
        res = check(db)
        assert not any(k == "stale_resting_entry" for k, _ in _kinds(res))

    def test_operator_ladder_is_exempt_from_stale_check(self, db):
        old = (NOW - timedelta(days=8)).isoformat()
        with sqlite3.connect(db) as c:
            self._add_buy_parent(c, "ord-intu-t1", "INTU", old, rec_id="INTU-op-20260930-T1")
        res = check(db)
        assert not any(k == "stale_resting_entry" for k, _ in _kinds(res))

    def test_untracked_position_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c, symbol="MYST", qty=10)
        result = check(db)
        assert ("position_not_in_ledger", "warn") in _kinds(result)

    def test_operator_manual_position_is_info(self, db):
        ""
        with sqlite3.connect(db) as c:
            _protected_position(c, symbol="QQQI", qty=89)
            _add_ledger_open(c, rec_id="d44manual0000001", symbol="QQQI",
                             qty=89, exit_status="operator_manual")
        result = check(db)
        kinds = _kinds(result)
        assert ("operator_manual_position", "info") in kinds
        assert ("position_not_in_ledger", "warn") not in kinds
        assert result["warn"] == 0

    def test_operator_manual_position_without_legs_is_not_a_missing_stop(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_position(c, symbol="QQQI", qty=89.0)
            c.execute(
                "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, symbol, "
                "strategy_id, status, fill_status, fill_qty, exit_status, "
                "created_at, updated_at) "
                "VALUES ('fixture-qqqi', '2026-07-06', '2026-07-07', 'QQQI', 7, "
                "'approved', 'filled', 89, 'operator_manual', ?, ?)", (FRESH, FRESH))
        res = check(db)
        kinds = _kinds(res)
        assert ("operator_manual_position", "info") in kinds
        assert not any(k in ("missing_stop_leg", "missing_target_leg") for k, _ in kinds)

    def test_qty_mismatch_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c, qty=222.0)
            _add_ledger_open(c, qty=200)
        result = check(db)
        assert ("qty_mismatch", "warn") in _kinds(result)


class TestProtectionGaps:
    def test_missing_stop_leg_is_critical(self, db):
        with sqlite3.connect(db) as c:
            _add_position(c)
            _add_order(c, "tp1", order_type="limit", status="new",
                       limit_price=26.96)
            _add_ledger_open(c)
        result = check(db)
        assert ("missing_stop_leg", "critical") in _kinds(result)

    def test_missing_target_leg_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _add_position(c)
            _add_order(c, "st1", order_type="stop", status="held",
                       stop_price=20.67)
            _add_ledger_open(c)
        result = check(db)
        assert ("missing_target_leg", "warn") in _kinds(result)

    def test_filled_stop_leg_does_not_count_as_protection(self, db):
        ""
        with sqlite3.connect(db) as c:
            _add_position(c)
            _add_order(c, "st1", order_type="stop", status="filled",
                       stop_price=20.67)
            _add_ledger_open(c)
        result = check(db)
        assert ("missing_stop_leg", "critical") in _kinds(result)

    def test_leg_qty_mismatch_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _add_position(c, qty=222.0)
            _add_order(c, "st1", order_type="stop", status="held", qty=100.0,
                       stop_price=20.67)
            _add_order(c, "tp1", order_type="limit", status="new", qty=222.0,
                       limit_price=26.96)
            _add_ledger_open(c)
        result = check(db)
        assert ("leg_qty_mismatch", "warn") in _kinds(result)


class TestPriceDrift:
    def test_repeg_shows_as_info_only(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c, stop_price=21.50, limit_price=27.80)
            _add_ledger_open(c, stop_price=20.67, target_price=26.96)
        result = check(db)
        kinds = _kinds(result)
        assert ("leg_price_drift", "info") in kinds
        assert result["critical"] == 0 and result["warn"] == 0

    def test_matching_prices_no_drift(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c)
            _add_ledger_open(c)
        assert _kinds(check(db)) == set()


class TestMirrorFreshness:
    def test_empty_mirror_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _add_ledger_open(c)
        result = check(db)
        assert ("mirror_missing", "warn") in _kinds(result)

    def test_stale_mirror_is_warn(self, db):
        with sqlite3.connect(db) as c:
            _add_position(c, snapshot_at=STALE)
            _add_order(c, "st1", order_type="stop", status="held", qty=222.0,
                       stop_price=20.67, snapshot_at=STALE)
            _add_order(c, "tp1", order_type="limit", status="new", qty=222.0,
                       limit_price=26.96, snapshot_at=STALE)
            _add_ledger_open(c)
        result = check(db)
        assert ("mirror_stale", "warn") in _kinds(result)


class TestPersistenceAndSummary:
    def test_findings_replace_previous_run(self, db):
        with sqlite3.connect(db) as c:
            _add_ledger_open(c)
        check(db)
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM mm_broker_divergences").fetchone()[0] == 2
            _protected_position(c)
        check(db)
        with sqlite3.connect(db) as c:
            assert c.execute(
                "SELECT COUNT(*) FROM mm_broker_divergences").fetchone()[0] == 0

    def test_summary_text_skips_info(self, db):
        with sqlite3.connect(db) as c:
            _protected_position(c, stop_price=21.50)
            _add_ledger_open(c)
            _add_ledger_open(c, rec_id="eeeeffff00001111", symbol="GONE")
        result = check(db)
        text = summary_text(result)
        assert "GONE" in text and "drift" not in text
        assert "Triangulate" in text


class TestDividendReceivable:
    ""

    def _plant(self, db, ex_days_ago: int, credited: bool = False):
        with sqlite3.connect(db) as c:
            _protected_position(c, "QQQI", 89.0, stop_price=51.61,
                                limit_price=67.32)
            _add_ledger_open(c, "divrec0000000001", "QQQI", 89,
                             stop_price=51.61, target_price=67.32)
            ex = (NOW - timedelta(days=ex_days_ago)).strftime("%Y-%m-%d")
            c.execute(
                "INSERT INTO mm_dividends (symbol, ex_date, rate, special, "
                "is_foreign, source, fetched_at) VALUES "
                "('QQQI', ?, 0.6346, 0, 0, 'test', ?)", (ex, FRESH))
            if credited:
                c.execute(
                    "INSERT INTO mm_alpaca_activities "
                    "(id, activity_type, symbol, net_amount, snapshot_at) "
                    "VALUES (?, 'DIV', 'QQQI', 56.48, ?)",
                    (NOW.strftime("%Y%m%d") + "000000000::div-test", FRESH))
            c.commit()

    def test_uncredited_past_grace_is_warn(self, db):
        self._plant(db, ex_days_ago=10)
        assert ("dividend_not_credited", "warn") in _kinds(check(db))

    def test_credited_is_silent(self, db):
        self._plant(db, ex_days_ago=10, credited=True)
        assert "dividend_not_credited" not in {
            f["kind"] for f in check(db)["findings"]}

    def test_within_grace_is_silent(self, db):
        self._plant(db, ex_days_ago=2)
        assert "dividend_not_credited" not in {
            f["kind"] for f in check(db)["findings"]}
