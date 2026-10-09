""
from __future__ import annotations

import sqlite3

import pytest

from macmini.data.schema import MM_SHADOW_SIZING
from macmini.ranking import shadow_sizing as sz


def test_multiplier_clamps_low():
    assert sz.conviction_multiplier(-0.5) == pytest.approx(sz.SHADOW_MULT_MIN)
    assert sz.conviction_multiplier(0.0) == pytest.approx(sz.SHADOW_MULT_MIN)


def test_multiplier_clamps_high():
    assert sz.conviction_multiplier(1.0) == pytest.approx(sz.SHADOW_MULT_MAX)
    assert sz.conviction_multiplier(3.0) == pytest.approx(sz.SHADOW_MULT_MAX)


def test_multiplier_linear_midpoint():
    assert sz.conviction_multiplier(0.5) == pytest.approx(1.25)


def test_multiplier_none_is_neutral():
    assert sz.conviction_multiplier(None) == pytest.approx(1.0)


def test_multiplier_monotone():
    vals = [sz.conviction_multiplier(x / 10) for x in range(11)]
    assert vals == sorted(vals)


def test_size_matches_flat_scheme_at_neutral():
    assert sz.shadow_size_pct(None, 0.10) == pytest.approx(0.05)


def test_size_cap_and_floor():
    assert sz.shadow_size_pct(1.0, 0.05) == pytest.approx(sz.SHADOW_SIZE_CAP)
    assert sz.shadow_size_pct(0.0, 0.15) == pytest.approx(sz.SHADOW_SIZE_FLOOR)


def test_size_scales_with_conviction():
    low = sz.shadow_size_pct(0.1, 0.10)
    high = sz.shadow_size_pct(0.9, 0.10)
    assert high > low


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.execute(MM_SHADOW_SIZING)
    yield c
    c.close()


def _write(c, rec_id="r1", dsr=0.4):
    sz.record_shadow_size(
        c, rec_id=rec_id, as_of_date="2026-08-06", symbol="CTSH",
        strategy_id=7, deflated_sharpe=dsr, real_size_pct=0.05,
        stop_frac=0.10)


def test_record_writes_row(conn):
    _write(conn)
    row = conn.execute(
        "SELECT symbol, conviction_mult, shadow_size_pct, real_size_pct "
        "FROM mm_shadow_sizing WHERE rec_id='r1'").fetchone()
    assert row[0] == "CTSH"
    assert row[1] == pytest.approx(1.1)
    assert row[2] == pytest.approx(0.055)
    assert row[3] == pytest.approx(0.05)


def test_record_is_idempotent_per_rec(conn):
    _write(conn, dsr=0.4)
    _write(conn, dsr=0.6)
    rows = conn.execute("SELECT COUNT(*) FROM mm_shadow_sizing").fetchone()
    assert rows[0] == 1
    mult = conn.execute(
        "SELECT conviction_mult FROM mm_shadow_sizing WHERE rec_id='r1'"
    ).fetchone()[0]
    assert mult == pytest.approx(1.4)
