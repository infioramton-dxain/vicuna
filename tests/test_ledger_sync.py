""
from __future__ import annotations

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from macmini.recs.ledger_sync import sync_ledger, _COLUMNS, _MINI_TABLE


@pytest.fixture
def laptop_db(tmp_path: Path) -> Path:
    ""
    from macmini.data.schema import init_schema
    db = tmp_path / "macmini.db"
    db.touch()
    init_schema(str(db))
    return db


def _make_mini_snapshot(path: Path, rows: list[dict]) -> None:
    ""
    with sqlite3.connect(path) as c:
        c.execute(f"""
            CREATE TABLE {_MINI_TABLE} (
                {', '.join(f'{col} TEXT' for col in _COLUMNS)}
            )
        """)
        if rows:
            placeholders = ", ".join("?" for _ in _COLUMNS)
            for row in rows:
                values = tuple(row.get(col) for col in _COLUMNS)
                c.execute(
                    f"INSERT INTO {_MINI_TABLE} ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                    values,
                )
        c.commit()


def _sample_row(rec_id: str = "abc123def456ff01", symbol: str = "QQQI") -> dict:
    return {
        "rec_id": rec_id, "as_of_date": "2026-07-04", "entry_date": "2026-07-06",
        "rank": 1, "symbol": symbol, "strategy_id": 7,
        "strategy_name": "Heikin-Ashi Color Streak", "sector": "Other",
        "suggested_size_pct": 0.05, "suggested_stop_loss_pct": 0.08,
        "suggested_target_pct": 0.20, "rationale": "3 green HA bars | DL=+1.12%",
        "status": "approved", "decided_at": "2026-07-04T13:15:00+00:00",
        "operator_reply": "y", "notes": "test row",
        "alpaca_order_id": "smoketest-QQQI-7", "order_qty": 87,
        "entry_est_price": 56.80, "stop_price": 52.256, "target_price": 68.16,
        "fill_status": "filled", "actual_fill_price": 56.87,
        "actual_fill_at": "2026-07-06T13:30:05+00:00", "fill_qty": 87,
        "last_polled_at": "2026-07-06T13:31:00+00:00",
        "created_at": "2026-07-04T13:15:00+00:00",
        "updated_at": "2026-07-06T13:31:00+00:00",
    }


class TestLedgerSync:

    def test_happy_path(self, laptop_db: Path, tmp_path: Path) -> None:
        ""
        snap = tmp_path / "trades_snapshot.db"
        _make_mini_snapshot(snap, [_sample_row("aaa", "QQQI"), _sample_row("bbb", "CVLT")])
        out = sync_ledger(snap, laptop_db)
        assert out["rows_synced"] == 2
        assert out["snapshot_rows"] == 2
        with sqlite3.connect(laptop_db) as c:
            rows = c.execute("SELECT rec_id, symbol, actual_fill_price FROM mm_recs_ledger ORDER BY rec_id").fetchall()
        assert rows == [("aaa", "QQQI", 56.87), ("bbb", "CVLT", 56.87)]

    def test_empty_snapshot_no_error(self, laptop_db: Path, tmp_path: Path) -> None:
        ""
        snap = tmp_path / "trades_snapshot.db"
        _make_mini_snapshot(snap, [])
        out = sync_ledger(snap, laptop_db)
        assert out["rows_synced"] == 0
        assert out.get("snapshot_rows") == 0
        with sqlite3.connect(laptop_db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_recs_ledger").fetchone()[0]
        assert n == 0

    def test_missing_snapshot_skips_gracefully(self, laptop_db: Path, tmp_path: Path) -> None:
        ""
        snap = tmp_path / "does_not_exist.db"
        out = sync_ledger(snap, laptop_db)
        assert out["rows_synced"] == 0
        assert out.get("skipped") == "snapshot_missing"

    def test_schema_mismatch_refuses_to_write(self, laptop_db: Path, tmp_path: Path) -> None:
        ""
        snap = tmp_path / "trades_snapshot.db"
        truncated_cols = _COLUMNS[:-1]
        with sqlite3.connect(snap) as c:
            c.execute(f"""
                CREATE TABLE {_MINI_TABLE} (
                    {', '.join(f'{col} TEXT' for col in truncated_cols)}
                )
            """)
            c.commit()
        out = sync_ledger(snap, laptop_db)
        assert out["rows_synced"] == 0
        assert "error" in out
        assert "missing_columns" in out["error"]

    def test_upsert_replaces_existing(self, laptop_db: Path, tmp_path: Path) -> None:
        ""
        snap = tmp_path / "trades_snapshot.db"
        row_v1 = _sample_row("dup", "QQQI")
        row_v1["fill_status"] = "pending"
        row_v1["actual_fill_price"] = None
        _make_mini_snapshot(snap, [row_v1])
        sync_ledger(snap, laptop_db)

        snap.unlink()
        row_v2 = _sample_row("dup", "QQQI")
        row_v2["fill_status"] = "filled"
        row_v2["actual_fill_price"] = 57.10
        _make_mini_snapshot(snap, [row_v2])
        sync_ledger(snap, laptop_db)

        with sqlite3.connect(laptop_db) as c:
            row = c.execute(
                "SELECT fill_status, actual_fill_price FROM mm_recs_ledger WHERE rec_id='dup'"
            ).fetchone()
        assert row == ("filled", 57.10)

        with sqlite3.connect(laptop_db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_recs_ledger").fetchone()[0]
        assert n == 1
