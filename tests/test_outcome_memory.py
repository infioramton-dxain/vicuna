""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.recs import outcome_memory as om

REC_DATE = date(2026, 7, 1)


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for name in ("MM_BARS", "MM_RECS_LEDGER", "MM_OUTCOME_MEMORY"):
            c.execute(getattr(schema, name))
        c.commit()
    return path


def _rec(c, sym, rec_id=None, day=REC_DATE, stop=0.08, status="expired",
         fill_price=None) -> None:
    c.execute(
        "INSERT INTO mm_recs_ledger (rec_id, as_of_date, entry_date, symbol, "
        "strategy_id, strategy_name, status, suggested_stop_loss_pct, "
        "fill_status, actual_fill_price, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 14, 'Test', ?, ?, ?, ?, 'x', 'x')",
        (rec_id or f"rec-{sym}", day.isoformat(),
         (day + timedelta(days=1)).isoformat(), sym, status, stop,
         "filled" if fill_price else None, fill_price))


def _bars(c, sym, closes, start=REC_DATE) -> None:
    ""
    prev = closes[0]
    for i, px in enumerate(closes, start=1):
        c.execute(
            "INSERT INTO mm_bars (symbol, date, open, high, low, close, source) "
            "VALUES (?, ?, ?, ?, ?, ?, 'test')",
            (sym, (start + timedelta(days=i)).isoformat(), prev, max(prev, px), px, px))
        prev = px


def _states(db: Path, as_of: date | None = None) -> tuple[dict, dict]:
    with sqlite3.connect(db) as c:
        report = om.rebuild(c, as_of or date(2026, 9, 1))
        states = dict(c.execute("SELECT symbol, state FROM mm_outcome_memory"))
    return states, report


def test_states_follow_the_outcome_standard(db: Path) -> None:
    with sqlite3.connect(db) as c:
        for sym, closes in (
            ("DOWN", [100, 95, 90, 88, 85]),
            ("BACK", [100, 90, 95, 101, 103]),
            ("GOOD", [100, 102, 105, 108, 110]),
            ("OPEN", [100, 98, 97, 96, 97]),
        ):
            _rec(c, sym)
            _bars(c, sym, closes)
        _rec(c, "NEW", day=date(2026, 8, 31))
        c.commit()
    states, report = _states(db)
    assert states == {"DOWN": "bad_down", "BACK": "recovered", "GOOD": "good",
                      "OPEN": "open", "NEW": "pending"}
    assert report["blocked"] == ["DOWN"]


def test_block_lifts_when_price_recovers(db: Path) -> None:
    with sqlite3.connect(db) as c:
        _rec(c, "LAR")
        _bars(c, "LAR", [100, 90, 85, 88, 92, 99, 101])
        c.commit()
    early, _ = _states(db, as_of=REC_DATE + timedelta(days=5))
    late, report = _states(db, as_of=REC_DATE + timedelta(days=30))
    assert early["LAR"] == "bad_down"
    assert late["LAR"] == "recovered" and report["blocked"] == []


def test_as_of_hides_the_future(db: Path) -> None:
    with sqlite3.connect(db) as c:
        _rec(c, "FUT")
        _bars(c, "FUT", [100, 101, 102, 80, 70])
        _rec(c, "LATE", day=date(2026, 8, 20))
        c.commit()
    states, _ = _states(db, as_of=REC_DATE + timedelta(days=3))
    assert states == {"FUT": "good"}, "the crash and the later rec are not visible yet"


def test_fill_price_is_the_entry_when_filled(db: Path) -> None:
    ""
    with sqlite3.connect(db) as c:
        _rec(c, "FILL", fill_price=110.0, status="approved")
        _bars(c, "FILL", [100, 100, 101, 105])
        c.commit()
        rows = om.evaluate(c, date(2026, 9, 1))
    assert rows[0]["entry_price"] == 110.0 and rows[0]["filled"] == 1
    assert rows[0]["state"] == "bad_down"


def test_fixture_rows_are_ignored(db: Path) -> None:
    with sqlite3.connect(db) as c:
        _rec(c, "FIX", rec_id="fixture-0001")
        _bars(c, "FIX", [100, 80, 70])
        c.commit()
    states, report = _states(db)
    assert states == {} and report["blocked"] == []


def test_one_bad_rec_blocks_despite_a_later_good_one(db: Path) -> None:
    with sqlite3.connect(db) as c:
        _rec(c, "MIX", rec_id="old")
        _rec(c, "MIX", rec_id="new", day=REC_DATE + timedelta(days=4))
        _bars(c, "MIX", [100, 90, 80, 78, 79, 82, 84])
        c.commit()
    _, report = _states(db)
    assert report["blocked"] == ["MIX"]
    with sqlite3.connect(db) as c:
        assert "stop level breached" in om.blocked_symbols(c)["MIX"]


def test_missing_table_blocks_nothing(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "empty.db") as c:
        assert om.blocked_symbols(c) == {}
