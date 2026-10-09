""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from macmini.config import DB_PATH
from macmini.data.schema import SENTIMENT_NATIVE_TABLES, init_schema

EXPECTED_MM_TABLES = {
    "mm_active_picks",
    "mm_analyst_estimates",
    "mm_backtest_runs",
    "mm_bars",
    "mm_bars_ingest_log",
    "mm_cftc_cot_weekly",
    "mm_dl_runs",
    "mm_dl_signals",
    "mm_earnings_events",
    "mm_export_metadata",
    "mm_fundamentals_quarterly",
    "mm_indicators_daily",
    "mm_insider_transactions",
    "mm_leaderboard",
    "mm_recommendation_performance",
    "mm_recommendation_scoreboard",
    "mm_regime",
    "mm_rl_signals",
    "mm_sector_etf_flows_weekly",
    "mm_sector_map",
    "mm_strategy_meta",
    "mm_training_labels",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


class TestSchemaBootstrap:
    ""

    @pytest.fixture
    def tmp_db(self, tmp_path: Path) -> Path:
        dst = tmp_path / "macmini.db"
        shutil.copy2(DB_PATH, dst)
        return dst

    def test_all_mm_tables_created(self, tmp_db: Path) -> None:
        init_schema(tmp_db)
        with sqlite3.connect(tmp_db) as c:
            tables = _table_names(c)
        missing = EXPECTED_MM_TABLES - tables
        assert not missing, f"schema missing tables: {missing}"

    def test_idempotent(self, tmp_db: Path) -> None:
        init_schema(tmp_db)
        init_schema(tmp_db)
        with sqlite3.connect(tmp_db) as c:
            n = c.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'mm_%'"
            ).fetchone()[0]
        assert n >= len(EXPECTED_MM_TABLES)

    def test_sentsrc_tables_untouched(self, tmp_db: Path) -> None:
        with sqlite3.connect(tmp_db) as c:
            before = {
                t: c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                for t in SENTIMENT_NATIVE_TABLES
                if c.execute(
                    "SELECT 1 FROM sqlite_master WHERE name=?", (t,)
                ).fetchone()
            }
        init_schema(tmp_db)
        with sqlite3.connect(tmp_db) as c:
            after = {
                t: c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                for t in before
            }
        assert before == after


class TestCanonicalDb:
    ""

    def test_db_exists_and_has_sentsrc_tables(self) -> None:
        assert DB_PATH.exists(), f"missing {DB_PATH}"
        with sqlite3.connect(DB_PATH) as c:
            tables = _table_names(c)
        assert {"sentiment", "term", "runs", "checkpoint"}.issubset(tables)

    def test_mm_tables_present(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            tables = _table_names(c)
        missing = EXPECTED_MM_TABLES - tables
        assert not missing, f"schema not initialized — missing: {missing}"

    def test_sentiment_row_count_floor(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
        assert n >= 17_042

    def test_mm_bars_meets_phase1_gate(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(DISTINCT symbol) FROM mm_bars").fetchone()[0]
        assert n >= 390, f"only {n} symbols in mm_bars; expected ≥390"

    def test_ingest_log_has_at_least_one_run(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_bars_ingest_log").fetchone()[0]
        assert n >= 1


SPEC_SECTORS = {"Materials", "Mining", "Defense", "Software", "Computers", "Other",
                "Consumer Staples"}


class TestPhase2:
    ""

    def test_sector_map_uses_only_spec_labels(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            labels = {r[0] for r in c.execute("SELECT DISTINCT sector FROM mm_sector_map")}
        unknown = labels - SPEC_SECTORS
        assert not unknown, f"mm_sector_map has non-spec sector labels: {unknown}"

    def test_sector_map_covers_most_of_universe(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_sector_map").fetchone()[0]
        assert n >= 150, f"mm_sector_map only has {n} entries"

    def test_fundamentals_quarterly_meets_gate(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_fundamentals_quarterly").fetchone()[0]
        assert n >= 1000, (
            f"Phase 2 gate failed: only {n} quarterly fundamental rows "
            f"(expected ≥1000). Did finnhub_fund.py run?"
        )

    def test_analyst_estimates_populated(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_analyst_estimates").fetchone()[0]
        assert n >= 100, f"mm_analyst_estimates has only {n} rows"

    def test_earnings_events_populated(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_earnings_events").fetchone()[0]
        assert n >= 100, f"mm_earnings_events has only {n} rows"

    def test_cftc_cot_populated(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute(
                "SELECT commodity, COUNT(*) FROM mm_cftc_cot_weekly GROUP BY commodity"
            ).fetchall()
        commodities = {r[0]: r[1] for r in rows}
        for required in ("gold", "silver", "copper"):
            assert commodities.get(required, 0) >= 50, (
                f"mm_cftc_cot_weekly is light on {required}: {commodities.get(required, 0)}"
            )

    def test_sector_etf_flows_populated(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_sector_etf_flows_weekly").fetchone()[0]
            etfs = c.execute("SELECT COUNT(DISTINCT etf_ticker) FROM mm_sector_etf_flows_weekly").fetchone()[0]
        assert etfs >= 10, f"only {etfs} ETFs covered"
        assert n >= 500, f"mm_sector_etf_flows_weekly has only {n} rows"

    def test_insider_transactions_populated(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_insider_transactions").fetchone()[0]
            syms = c.execute(
                "SELECT COUNT(DISTINCT symbol) FROM mm_insider_transactions"
            ).fetchone()[0]
        assert n >= 100, f"mm_insider_transactions only has {n} rows"
        assert syms >= 30, f"only {syms} symbols have insider rows"

    def test_insider_transaction_dates_are_iso_10char(self) -> None:
        ""
        with sqlite3.connect(DB_PATH) as c:
            bad = c.execute(
                "SELECT COUNT(*) FROM mm_insider_transactions "
                "WHERE LENGTH(transaction_date) <> 10 "
                "OR transaction_date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'"
            ).fetchone()[0]
        assert bad == 0, f"{bad} rows have malformed transaction_date"

    def test_sec_form4_normalize_txn_date(self) -> None:
        ""
        from macmini.data.sec_form4 import _normalize_txn_date

        assert _normalize_txn_date("2026-05-22") == "2026-05-22"
        assert _normalize_txn_date("2025-11-12-05:00") == "2025-11-12"
        assert _normalize_txn_date("2025-11-12T14:30:00") == "2025-11-12"
        assert _normalize_txn_date("  2026-01-05  ") == "2026-01-05"
        assert _normalize_txn_date(None) is None
        assert _normalize_txn_date("") is None
        assert _normalize_txn_date("not-a-date") is None
        assert _normalize_txn_date("2026-13-01") is None
        assert _normalize_txn_date("2026-02-30") is None

    def test_sentiment_sync_idempotent(self, tmp_path: Path) -> None:
        ""
        from macmini.data.sentiment_sync import refresh

        dst = tmp_path / "macmini.db"
        src = sqlite3.connect(DB_PATH, timeout=30)
        dst_conn = sqlite3.connect(dst)
        with dst_conn:
            src.backup(dst_conn)
        src.close(); dst_conn.close()

        r1 = refresh(db_path=dst)
        r2 = refresh(db_path=dst)
        assert isinstance(r1["sentiment_inserted"], int)
        assert isinstance(r2["sentiment_inserted"], int)
        assert r2["watermark"] >= r1["watermark"]
