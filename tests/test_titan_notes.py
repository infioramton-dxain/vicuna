""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.recs import titan_notes as tn

AS_OF = "2026-09-28"
CFO = "NetCashProvidedByUsedInOperatingActivities"
CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment"
REV = "Revenues"


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for name in ("MM_BARS", "MM_SECTOR_MAP", "MM_SEC_FACTS",
                     "MM_FUNDAMENTALS_GROWTH", "MM_ANALYST_ESTIMATES",
                     "MM_EARNINGS_EVENTS", "MM_INSIDER_TRANSACTIONS",
                     "MM_DIVIDENDS", "MM_KRONOS_SIGNALS", "MM_ACTIVE_PICKS",
                     "MM_QUALITY_CANDIDATES", "MM_TITAN_NOTES"):
            c.execute(getattr(schema, name))
        c.commit()
    return path


def _fact(c, sym, tag, start, end, filed, value, unit="USD"):
    c.execute(
        "INSERT INTO mm_sec_facts (symbol, cik, tag, unit, period_end, "
        "period_start, fy, fp, form, filed, frame, value, fetched_at) "
        "VALUES (?, '0000000001', ?, ?, ?, ?, NULL, NULL, '10-Q', ?, NULL, ?, 'x')",
        (sym, tag, unit, end, start, filed, value))


def _seed(c, sym="TITN", close=50.0, cap=5_000.0) -> None:
    ""
    c.execute("INSERT INTO mm_bars (symbol, date, close, source) "
              "VALUES (?, ?, ?, 'test')", (sym, AS_OF, close))
    c.execute("INSERT INTO mm_sector_map (symbol, sector, finnhub_sector, "
              "market_cap_usd) VALUES (?, 'Software', 'Technology', ?)", (sym, cap))
    for tag, fy, ytd_now, ytd_prior in ((CFO, 400.0, 260.0, 200.0),
                                        (CAPEX, 100.0, 60.0, 50.0)):
        _fact(c, sym, tag, "2025-01-01", "2025-12-31", "2026-02-15", fy)
        _fact(c, sym, tag, "2026-01-01", "2026-06-30", "2026-08-01", ytd_now)
        _fact(c, sym, tag, "2025-01-01", "2025-06-30", "2025-08-01", ytd_prior)
    for end, filed, val in (("2022-12-31", "2023-02-15", 1000.0),
                            ("2023-12-31", "2024-02-15", 1100.0),
                            ("2024-12-31", "2025-02-15", 1210.0),
                            ("2025-12-31", "2026-02-15", 1331.0)):
        _fact(c, sym, REV, end[:4] + "-01-01", end, filed, val)
    _fact(c, sym, "CashAndCashEquivalentsAtCarryingValue", "", "2026-06-30", "2026-08-01", 300.0)
    _fact(c, sym, "LongTermDebt", "", "2026-06-30", "2026-08-01", 500.0)
    _fact(c, sym, "WeightedAverageNumberOfDilutedSharesOutstanding",
          "2026-01-01", "2026-06-30", "2026-08-01", 100.0, unit="shares")
    c.commit()


def _note(db: Path, sym="TITN", as_of=AS_OF) -> tn.Note:
    with sqlite3.connect(db) as c:
        return tn.build_note(c, sym, as_of)


class TestTrailingTwelveMonths:
    def test_ytd_flows_combine_into_ttm(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
        note = _note(db)
        assert note.get("sec", "cfo_ttm") == pytest.approx(400 + 260 - 200)
        assert note.get("sec", "capex_ttm") == pytest.approx(100 + 60 - 50)
        assert note.get("sec", "fcf_ttm") == pytest.approx(460 - 110)

    def test_filings_after_as_of_are_invisible(self, db: Path) -> None:
        ""
        with sqlite3.connect(db) as c:
            _seed(c)
            _fact(c, "TITN", CFO, "2025-01-01", "2025-12-31", "2026-10-15", 9999.0)
            c.commit()
        assert _note(db).get("sec", "cfo_ttm") == pytest.approx(460.0)
        before_q2 = _note(db, as_of="2026-07-15")
        assert before_q2.get("sec", "cfo_ttm") == pytest.approx(400.0)


class TestDcf:
    def test_matches_closed_form(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
        note = _note(db)
        assert note.get("sec", "revenue_cagr_3y") == pytest.approx(0.10)
        assert note.get("dcf", "base_growth") == pytest.approx(0.10)
        fcf, g, w, gt = 350.0, 0.10, 0.09, 0.025
        pv = sum(fcf * (1 + g) ** t / (1 + w) ** t for t in range(1, 6))
        pv += fcf * (1 + g) ** 5 * (1 + gt) / (w - gt) / (1 + w) ** 5
        expected = (pv - (500.0 - 300.0)) / 100.0
        assert note.get("dcf", "value_per_share_base") == pytest.approx(expected)
        assert (note.get("dcf", "value_per_share_bear")
                < note.get("dcf", "value_per_share_base")
                < note.get("dcf", "value_per_share_bull"))

    def _skip_reason(self, note: tn.Note) -> str | None:
        return next((r["narrative"] for r in note.rows
                     if r["section"] == "dcf" and r["metric"] == "skipped"), None)

    def test_skipped_without_debt(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
            c.execute("DELETE FROM mm_sec_facts WHERE tag = 'LongTermDebt'")
            c.commit()
        note = _note(db)
        assert note.get("dcf", "value_per_share_base") is None
        assert "net debt" in self._skip_reason(note)

    def test_skipped_on_negative_fcf(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
            c.execute("UPDATE mm_sec_facts SET value = value * 10 WHERE tag = ?", (CAPEX,))
            c.commit()
        assert "not positive" in self._skip_reason(_note(db))

    def test_skipped_when_share_count_disagrees_with_cap(self, db: Path) -> None:
        ""
        with sqlite3.connect(db) as c:
            _seed(c, cap=50_000.0)
        note = _note(db)
        assert "sanity check" in self._skip_reason(note)
        assert note.get("sec", "fcf_yield") is None

    def test_skipped_for_stale_and_missing_filers(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
            c.execute("INSERT INTO mm_bars (symbol, date, close, source) "
                      "VALUES ('ETF', ?, 10, 'test')", (AS_OF,))
            c.commit()
        assert "stale" in self._skip_reason(_note(db, as_of="2028-01-05"))
        assert "no SEC" in self._skip_reason(_note(db, sym="ETF"))

    def test_lease_bundled_debt_tags_are_read(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
            c.execute("DELETE FROM mm_sec_facts WHERE tag = 'LongTermDebt'")
            _fact(c, "TITN", "LongTermDebtAndCapitalLeaseObligations", "",
                  "2026-06-30", "2026-08-01", 450.0)
            _fact(c, "TITN", "LongTermDebtAndCapitalLeaseObligationsCurrent", "",
                  "2026-06-30", "2026-08-01", 50.0)
            c.commit()
        assert _note(db).get("sec", "total_debt") == pytest.approx(500.0)


class TestNarrative:
    FACTS = "TITN Last close: $50.00 Free cash flow: $350 EPS change: 150.0%"

    def test_clean_prose_passes(self) -> None:
        assert tn.validate_narrative(
            "TITN closed at $50.00; EPS rose 150%.", self.FACTS) is None

    def test_rounded_figure_rejected(self) -> None:
        assert tn.validate_narrative("TITN closed near $50.4.", self.FACTS) is not None

    def test_price_gap_direction_is_spelled_out(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c, close=80.0, cap=8_000.0)
        lines = tn.fact_lines(_note(db))
        assert any("above the DCF base case" in ln for ln in lines)
        assert not any(w in " ".join(lines) for w in ("Buy", "Sell", "Hold"))

    def test_invented_figure_rejected(self) -> None:
        reason = tn.validate_narrative("Fair value is $72.50.", self.FACTS)
        assert reason is not None and "72.50" in reason

    @pytest.mark.parametrize("text", [
        "This is a buy.", "We recommend adding.", "The stock looks undervalued.",
        "Investors should avoid it.", "Hold for now.",
    ])
    def test_verdict_words_rejected(self, text: str) -> None:
        assert tn.validate_narrative(text, self.FACTS) is not None

    def test_bad_llm_output_falls_back_to_template(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
        note = _note(db)
        bad = lambda *_: "SUMMARY: A strong buy at $999. DETAIL: x WRONG IF: y"
        assert tn.write_narrative(note, True, caller=bad)["model"] == "template"

        def down(*_):
            raise OSError("ollama unreachable")
        assert tn.write_narrative(note, True, caller=down)["model"] == "template"

    def test_good_llm_output_is_kept(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
        note = _note(db)
        good = lambda *_: ("<think>…</think>SUMMARY: TITN closed at $50.00.\n"
                           "DETAIL: Free cash flow was $350.\n"
                           "WRONG IF: Cash flow weakens.")
        out = tn.write_narrative(
            note, True, model="m",
            caller=lambda *a: good().split("</think>")[1])
        assert out["model"] == "m"
        assert out["detail"] == "TITN closed at $50.00. Free cash flow was $350."
        assert out["summary"] == tn.template_narrative(note)["summary"]

    def test_template_uses_only_note_figures(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
        note = _note(db)
        nar = tn.template_narrative(note)
        allowed = note.symbol + " " + " ".join(tn.fact_lines(note))
        assert len(nar["summary"]) <= tn.SUMMARY_MAX_CHARS
        assert tn.validate_narrative(
            " ".join((nar["summary"], nar["detail"])), allowed) is None


class TestRun:
    def test_writes_provenance_rows_for_picks(self, db: Path) -> None:
        with sqlite3.connect(db) as c:
            _seed(c)
            c.execute("INSERT INTO mm_active_picks (rank, strategy_id, "
                      "strategy_name, symbol, sector, as_of_date) "
                      "VALUES (1, 56, 'Test', 'TITN', 'Software', ?)", (AS_OF,))
            c.commit()
        report = tn.run(db, use_llm=False)
        assert report == {"as_of_date": AS_OF, "notes": ["TITN"], "llm": 0,
                          "template": 1, "dcf": 1}
        assert tn.run(db, use_llm=False)["notes"] == ["TITN"]
        with sqlite3.connect(db) as c:
            rows = c.execute(
                "SELECT section, metric, value, source_table, source_column, "
                "narrative FROM mm_titan_notes WHERE symbol = 'TITN'").fetchall()
        assert len(rows) == len({(r[0], r[1]) for r in rows})
        numeric = [r for r in rows if r[2] is not None]
        assert numeric and all(r[3] or r[4] for r in numeric), "every number needs a source"
        assert {r[1] for r in rows if r[0] == "narrative"} == {"summary", "detail", "wrong_if"}

    def test_no_picks_writes_nothing(self, db: Path) -> None:
        assert tn.run(db, use_llm=False)["notes"] == []
