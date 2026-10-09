""
import sqlite3
from datetime import date

import pytest

from macmini.data import earnings_calendar as ec
from macmini.data.schema import MM_EARNINGS_EVENTS


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(MM_EARNINGS_EVENTS)
    conn.commit()
    return path, conn


def _run_ingest(monkeypatch, db_path, events, universe):
    monkeypatch.setattr(ec, "_client", lambda: object())
    monkeypatch.setattr(ec, "fetch_calendar", lambda client, start, end: events)
    return ec.ingest(
        db_path=db_path,
        symbols=universe,
        as_of=date(2026, 7, 12),
    )


def test_forward_rows_inserted_and_universe_filtered(monkeypatch, db):
    path, conn = db
    events = [
        {"symbol": "JPM", "date": "2026-07-14", "epsEstimate": 5.74,
         "epsActual": None, "revenueEstimate": 4.5e10, "revenueActual": None},
        {"symbol": "NOTUNIV", "date": "2026-07-14", "epsEstimate": 1.0},
        {"symbol": "", "date": "2026-07-15", "epsEstimate": 1.0},
        {"symbol": "GE", "date": None, "epsEstimate": 1.0},
    ]
    report = _run_ingest(monkeypatch, path, events, {"JPM", "GE"})
    assert report["universe_matches"] == 1
    assert report["symbols"] == ["JPM"]
    rows = conn.execute(
        "SELECT symbol, report_date, eps_estimate, eps_actual "
        "FROM mm_earnings_events"
    ).fetchall()
    assert rows == [("JPM", "2026-07-14", 5.74, None)]


def test_repull_never_clobbers_actuals(monkeypatch, db):
    path, conn = db
    conn.execute(
        "INSERT INTO mm_earnings_events "
        "(symbol, report_date, eps_actual, eps_estimate, eps_surprise_pct) "
        "VALUES ('HPQ', '2026-07-15', 0.82, 0.80, 2.5)"
    )
    conn.commit()
    events = [{"symbol": "HPQ", "date": "2026-07-15", "epsEstimate": 0.81,
               "epsActual": None}]
    _run_ingest(monkeypatch, path, events, {"HPQ"})
    row = conn.execute(
        "SELECT eps_actual, eps_estimate, eps_surprise_pct "
        "FROM mm_earnings_events WHERE symbol='HPQ'"
    ).fetchone()
    assert row == (0.82, 0.81, 2.5)


def test_estimate_not_nulled_by_sparse_repull(monkeypatch, db):
    path, conn = db
    events = [{"symbol": "JPM", "date": "2026-07-14", "epsEstimate": 5.74}]
    _run_ingest(monkeypatch, path, events, {"JPM"})
    events = [{"symbol": "JPM", "date": "2026-07-14", "epsEstimate": None}]
    _run_ingest(monkeypatch, path, events, {"JPM"})
    row = conn.execute(
        "SELECT eps_estimate FROM mm_earnings_events WHERE symbol='JPM'"
    ).fetchone()
    assert row == (5.74,)
