""
from __future__ import annotations

import sqlite3

import pytest

from macmini.data.fundamentals_growth import (
    _cagr,
    _pct_change,
    compute_growth,
    refresh,
)
from macmini.data.schema import MM_FUNDAMENTALS_ANNUAL, MM_FUNDAMENTALS_QUARTERLY


def _ytd_year(year: int, q_rev: float, q_eps: float, n: int = 3) -> list[tuple]:
    ""
    ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30"]
    return [
        (ends[i], q_rev * (i + 1), round(q_eps * (i + 1), 6))
        for i in range(n)
    ]


def _annual(year: int, rev: float, eps: float, earnings: float | None = None) -> tuple:
    if earnings is None:
        earnings = eps * 1_000_000
    return (f"{year}-12-31", rev, eps, earnings)


def _fixture():
    qrows = []
    for year, qr in ((2021, 80.0), (2022, 90.0), (2023, 100.0), (2024, 110.0)):
        qrows += _ytd_year(year, qr, qr / 100)
    qrows += [("2025-03-31", 121.0, 1.21)]
    arows = [
        _annual(2019, 220.0, 2.20),
        _annual(2020, 260.0, 2.60),
        _annual(2021, 320.0, 3.20),
        _annual(2022, 360.0, 3.60),
        _annual(2023, 400.0, 4.00),
        _annual(2024, 440.0, 4.40),
    ]
    return qrows, arows


class TestHelpers:
    def test_pct_change(self):
        assert _pct_change(110.0, 100.0) == pytest.approx(10.0)
        assert _pct_change(90.0, 100.0) == pytest.approx(-10.0)
        assert _pct_change(110.0, 0.0) is None
        assert _pct_change(110.0, -5.0) is None
        assert _pct_change(None, 100.0) is None
        assert _pct_change(110.0, None) is None

    def test_cagr_actual_span(self):
        assert _cagr(440.0, 220.0, 1461) == pytest.approx(18.92, abs=0.05)
        assert _cagr(440.0, -1.0, 1461) is None
        assert _cagr(None, 220.0, 1461) is None


class TestComputeGrowth:
    def test_mrq_yoy_from_ytd_q1(self):
        qrows, arows = _fixture()
        g = compute_growth(qrows, arows)
        assert g["rev_chg_mrq_yoy"] == pytest.approx(10.0)
        assert g["eps_chg_mrq_yoy"] == pytest.approx(10.0, abs=0.01)

    def test_ttm_construction(self):
        qrows, arows = _fixture()
        g = compute_growth(qrows, arows)
        assert g["rev_chg_ttm"] == pytest.approx(10.0)
        assert g["eps_chg_ttm"] == pytest.approx(10.0, abs=0.01)

    def test_5y_avg_is_fy_cagr(self):
        qrows, arows = _fixture()
        g = compute_growth(qrows, arows)
        assert g["rev_growth_5y_avg"] == pytest.approx(14.86, abs=0.05)
        assert g["eps_growth_5y_avg"] == pytest.approx(14.86, abs=0.05)

    def test_mid_year_ytd_diff(self):
        qrows, arows = _fixture()
        qrows = [r for r in qrows if r[0] <= "2024-06-30"]
        g = compute_growth(qrows, arows[:-1])
        assert g["rev_chg_mrq_yoy"] == pytest.approx(10.0)

    def test_negative_eps_base_is_null_but_revenue_survives(self):
        qrows, arows = _fixture()
        qrows = [
            (pe, rev, -0.5 if pe == "2024-03-31" else eps) for (pe, rev, eps) in qrows
        ]
        g = compute_growth(qrows, arows)
        assert g["rev_chg_mrq_yoy"] == pytest.approx(10.0)
        assert g["eps_chg_mrq_yoy"] is None

    def test_split_nulls_eps_but_not_revenue(self):
        qrows, arows = _fixture()
        arows[0] = ("2019-12-31", 220.0, 8.80, 2.20 * 1_000_000)
        g = compute_growth(qrows, arows)
        assert g["rev_growth_5y_avg"] == pytest.approx(14.86, abs=0.05)
        assert g["eps_growth_5y_avg"] is None

    def test_filing_gap_nulls_ttm(self):
        qrows, arows = _fixture()
        qrows = [r for r in qrows if r[0] != "2024-03-31"]
        g = compute_growth(qrows, arows)
        assert g["rev_chg_ttm"] is None

    def test_annual_only_symbol(self):
        _, arows = _fixture()
        g = compute_growth([], arows)
        assert g["rev_chg_ttm"] == pytest.approx(10.0)
        assert g["rev_growth_5y_avg"] == pytest.approx(14.86, abs=0.05)
        assert g["rev_chg_mrq_yoy"] is None

    def test_empty(self):
        assert compute_growth([], []) is None


class TestRefresh:
    def test_end_to_end(self, tmp_path):
        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        conn.execute(MM_FUNDAMENTALS_QUARTERLY)
        conn.execute(MM_FUNDAMENTALS_ANNUAL)
        qrows, arows = _fixture()
        conn.executemany(
            "INSERT INTO mm_fundamentals_quarterly (symbol, period_end, revenue, eps, source, fetched_at) "
            "VALUES ('TEST', ?, ?, ?, 'test', 'now')",
            qrows,
        )
        conn.executemany(
            "INSERT INTO mm_fundamentals_annual (symbol, period_end, revenue, eps, earnings, source, fetched_at) "
            "VALUES ('TEST', ?, ?, ?, ?, 'test', 'now')",
            arows,
        )
        conn.commit()
        conn.close()

        report = refresh(db_path=db, as_of="2026-07-16")
        assert report["symbols_written"] == 1

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT rev_chg_mrq_yoy, rev_chg_ttm, rev_growth_5y_avg, latest_period_end "
            "FROM mm_fundamentals_growth WHERE symbol='TEST' AND as_of_date='2026-07-16'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(10.0)
        assert row[1] == pytest.approx(10.0)
        assert row[2] == pytest.approx(14.86, abs=0.05)
        assert row[3] == "2025-03-31"

        refresh(db_path=db, as_of="2026-07-16")
        n = conn.execute("SELECT COUNT(*) FROM mm_fundamentals_growth").fetchone()[0]
        conn.close()
        assert n == 1
