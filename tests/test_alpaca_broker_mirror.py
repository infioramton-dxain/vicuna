""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from macmini.data import alpaca_broker_mirror as abm
from macmini.data.schema import init_schema


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch) -> Path:
    ""
    db_path = tmp_path / "macmini.db"
    sqlite3.connect(db_path).close()
    init_schema(db_path)
    monkeypatch.setattr(abm, "ALPACA_API_KEY", "test-key")
    monkeypatch.setattr(abm, "ALPACA_SECRET_KEY", "test-secret")
    return db_path


def _fake_get(routes: dict):
    ""
    def _get(path: str, params: dict | None = None):
        if path == "/v2/account/activities":
            key = ("activities", (params or {}).get("date"))
        elif path == "/v2/orders":
            key = ("orders", (params or {}).get("until"))
        else:
            key = (path,)
        if key not in routes:
            raise AssertionError(f"unexpected route: {key}")
        return routes[key]
    return _get


class TestPositions:

    def test_positions_truncate_and_insert(self, tmp_db: Path, monkeypatch) -> None:
        payload = [
            {"symbol": "AAA", "side": "long", "qty": "10", "avg_entry_price": "5.0",
             "market_value": "55", "cost_basis": "50", "current_price": "5.5",
             "unrealized_pl": "5", "unrealized_plpc": "0.1", "change_today": "0.01",
             "exchange": "NASDAQ", "asset_class": "us_equity"},
            {"symbol": "BBB", "side": "long", "qty": "2", "avg_entry_price": "100",
             "market_value": "180", "cost_basis": "200", "current_price": "90",
             "unrealized_pl": "-20", "unrealized_plpc": "-0.1", "change_today": "-0.02",
             "exchange": "NYSE", "asset_class": "us_equity"},
        ]
        monkeypatch.setattr(abm, "_get", _fake_get({("/v2/positions",): payload}))
        with sqlite3.connect(tmp_db) as c:
            c.execute(
                "INSERT INTO mm_alpaca_positions (symbol, side, qty, snapshot_at) "
                "VALUES ('OLD', 'long', 99, '2020-01-01T00:00:00Z')"
            )
            c.commit()
            n = abm.snapshot_positions(c, "2026-07-08T13:00:00Z")
        assert n == 2
        with sqlite3.connect(tmp_db) as c:
            rows = c.execute(
                "SELECT symbol, qty, unrealized_pl FROM mm_alpaca_positions "
                "ORDER BY symbol"
            ).fetchall()
        assert rows == [("AAA", 10.0, 5.0), ("BBB", 2.0, -20.0)]


class TestOrders:

    def test_orders_upsert_and_bracket_legs_flattened(
            self, tmp_db: Path, monkeypatch) -> None:
        bracket = {
            "id": "parent-1", "client_order_id": "cli-1", "symbol": "AAA",
            "side": "buy", "order_type": "market", "order_class": "bracket",
            "qty": "10", "filled_qty": "10", "filled_avg_price": "5.0",
            "limit_price": None, "stop_price": None,
            "time_in_force": "gtc", "status": "filled",
            "parent_id": None, "replaced_by": None, "replaces": None,
            "extended_hours": False,
            "submitted_at": "2026-07-06T13:30:00Z",
            "filled_at": "2026-07-06T13:30:05Z",
            "canceled_at": None, "expired_at": None,
            "replaced_at": None, "updated_at": "2026-07-06T13:30:05Z",
            "legs": [
                {"id": "leg-tp", "symbol": "AAA", "side": "sell",
                 "order_type": "limit", "qty": "10", "filled_qty": "0",
                 "limit_price": "6.0", "stop_price": None,
                 "time_in_force": "gtc", "status": "new",
                 "submitted_at": "2026-07-06T13:30:00Z"},
                {"id": "leg-sl", "symbol": "AAA", "side": "sell",
                 "order_type": "stop", "qty": "10", "filled_qty": "0",
                 "limit_price": None, "stop_price": "4.5",
                 "time_in_force": "gtc", "status": "held",
                 "submitted_at": "2026-07-06T13:30:00Z"},
            ],
        }
        monkeypatch.setattr(abm, "_get", _fake_get({
            ("orders", None): [bracket],
        }))
        with sqlite3.connect(tmp_db) as c:
            n = abm.snapshot_orders(c, "2026-07-08T13:00:00Z",
                                    "2026-07-01T00:00:00Z")
        assert n == 3
        with sqlite3.connect(tmp_db) as c:
            rows = c.execute(
                "SELECT id, order_class, parent_id, side, order_type, status "
                "FROM mm_alpaca_orders ORDER BY id"
            ).fetchall()
        by_id = {r[0]: r for r in rows}
        assert by_id["parent-1"][1] == "bracket"
        assert by_id["parent-1"][2] is None
        assert by_id["leg-sl"][2] == "parent-1"
        assert by_id["leg-sl"][4] == "stop"
        assert by_id["leg-sl"][5] == "held"
        assert by_id["leg-tp"][2] == "parent-1"
        assert by_id["leg-tp"][4] == "limit"
        assert by_id["leg-tp"][5] == "new"

    def test_orders_upsert_updates_mutable_fields(
            self, tmp_db: Path, monkeypatch) -> None:
        ""
        first = [{
            "id": "o-1", "client_order_id": "c-1", "symbol": "AAA",
            "side": "buy", "order_type": "market", "order_class": "simple",
            "qty": "10", "filled_qty": "0", "filled_avg_price": None,
            "limit_price": None, "stop_price": None,
            "time_in_force": "day", "status": "new",
            "submitted_at": "2026-07-06T13:30:00Z",
        }]
        second = [dict(first[0], status="filled", filled_qty="10",
                       filled_avg_price="5.0",
                       filled_at="2026-07-06T13:31:00Z")]
        monkeypatch.setattr(abm, "_get", _fake_get({("orders", None): first}))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-07-06T13:30:00Z",
                                "2026-07-01T00:00:00Z")
        monkeypatch.setattr(abm, "_get", _fake_get({("orders", None): second}))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-07-06T14:00:00Z",
                                "2026-07-01T00:00:00Z")
        with sqlite3.connect(tmp_db) as c:
            row = c.execute(
                "SELECT status, filled_qty, filled_avg_price "
                "FROM mm_alpaca_orders WHERE id='o-1'"
            ).fetchone()
        assert row == ("filled", 10.0, 5.0)


class TestStragglerRefresh:
    def test_live_row_outside_window_is_refetched(
            self, tmp_db: Path, monkeypatch) -> None:
        ""
        old = [{
            "id": "o-old", "client_order_id": "c-old", "symbol": "QQQI",
            "side": "buy", "order_type": "limit", "order_class": "simple",
            "qty": "89", "filled_qty": "0", "filled_avg_price": None,
            "limit_price": "49.11", "stop_price": None,
            "time_in_force": "gtc", "status": "new",
            "submitted_at": "2026-07-28T13:16:05Z",
        }]
        monkeypatch.setattr(abm, "_get", _fake_get({("orders", None): old}))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-08-11T02:35:00Z", "2026-07-20T00:00:00Z")
        monkeypatch.setattr(abm, "_get", _fake_get({
            ("orders", None): [],
            ("/v2/orders/o-old",): dict(old[0], status="canceled",
                                        canceled_at="2026-08-20T14:00:00Z"),
        }))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-10-07T22:21:00Z", "2026-09-23T00:00:00Z")
            row = c.execute(
                "SELECT status, canceled_at, snapshot_at FROM mm_alpaca_orders "
                "WHERE id='o-old'").fetchone()
        assert row == ("canceled", "2026-08-20T14:00:00Z", "2026-10-07T22:21:00Z")

    def test_refetch_keeps_parent_id(self, tmp_db: Path, monkeypatch) -> None:
        parent = [{
            "id": "p-1", "client_order_id": "c-p", "symbol": "KHC",
            "side": "buy", "order_type": "limit", "order_class": "bracket",
            "qty": "283", "filled_qty": "283", "filled_avg_price": "23.4",
            "limit_price": "23.4", "stop_price": None, "time_in_force": "gtc",
            "status": "filled", "submitted_at": "2026-09-29T14:00:00Z",
            "legs": [{
                "id": "leg-stop", "client_order_id": "c-s", "symbol": "KHC",
                "side": "sell", "order_type": "stop", "order_class": "bracket",
                "qty": "283", "filled_qty": "0", "stop_price": "21.61",
                "time_in_force": "gtc", "status": "held",
                "submitted_at": "2026-09-29T14:00:00Z",
            }],
        }]
        monkeypatch.setattr(abm, "_get", _fake_get({("orders", None): parent}))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-09-29T23:50:00Z", "2026-09-20T00:00:00Z")
        monkeypatch.setattr(abm, "_get", _fake_get({
            ("orders", None): [],
            ("/v2/orders/leg-stop",): dict(parent[0]["legs"][0], status="canceled"),
        }))
        with sqlite3.connect(tmp_db) as c:
            abm.snapshot_orders(c, "2026-10-20T23:50:00Z", "2026-10-06T00:00:00Z")
            row = c.execute(
                "SELECT status, parent_id FROM mm_alpaca_orders WHERE id='leg-stop'"
            ).fetchone()
        assert row == ("canceled", "p-1")


class TestActivities:

    def test_activities_insert_or_ignore(self, tmp_db: Path, monkeypatch) -> None:
        ""
        fill = {
            "id": "act-1", "activity_type": "FILL",
            "transaction_time": "2026-07-06T13:30:05Z",
            "symbol": "AAA", "side": "buy_short", "qty": "10", "price": "5.0",
            "cum_qty": "10", "leaves_qty": "0", "order_id": "o-1",
            "order_status": "filled",
        }
        div = {
            "id": "act-2", "activity_type": "DIV", "date": "2026-07-06",
            "symbol": "AAA", "net_amount": "0.50", "per_share_amount": "0.05",
            "description": "AAA CASH DIV",
        }
        monkeypatch.setattr(abm, "_get", _fake_get({
            ("activities", "2026-07-06"): [fill, div],
            ("activities", "2026-07-07"): [],
        }))
        with sqlite3.connect(tmp_db) as c:
            n1 = abm.snapshot_activities(
                c, "2026-07-08T13:00:00Z",
                date(2026, 7, 6), date(2026, 7, 7),
            )
        monkeypatch.setattr(abm, "_get", _fake_get({
            ("activities", "2026-07-06"): [fill, div],
            ("activities", "2026-07-07"): [],
        }))
        with sqlite3.connect(tmp_db) as c:
            n2 = abm.snapshot_activities(
                c, "2026-07-08T14:00:00Z",
                date(2026, 7, 6), date(2026, 7, 7),
            )
        assert n1 == 2 and n2 == 2
        with sqlite3.connect(tmp_db) as c:
            total = c.execute(
                "SELECT COUNT(*) FROM mm_alpaca_activities"
            ).fetchone()[0]
        assert total == 2


class TestHeadersGuard:

    def test_missing_creds_raise(self, monkeypatch) -> None:
        monkeypatch.setattr(abm, "ALPACA_API_KEY", None)
        monkeypatch.setattr(abm, "ALPACA_SECRET_KEY", "x")
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            abm._headers()
