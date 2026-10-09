""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.recs.titan_ql import STRATEGY_ID, scan

AS_OF = "2026-07-31"


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for name in ("MM_BARS", "MM_SECTOR_MAP", "MM_FUNDAMENTALS_QUARTERLY",
                     "MM_FUNDAMENTALS_GROWTH", "MM_ANALYST_ESTIMATES",
                     "MM_EARNINGS_EVENTS", "MM_REGIME", "MM_ALPACA_POSITIONS",
                     "MM_ACTIVE_PICKS", "MM_INDICATORS_DAILY",
                     "MM_KRONOS_SIGNALS", "MM_QUALITY_CANDIDATES"):
            c.execute(getattr(schema, name))
        c.commit()
    return path


def _seed(path: Path, **over) -> None:
    ""
    dates = [d.strftime("%Y-%m-%d")
             for d in pd.bdate_range(end=AS_OF, periods=252)]
    close = over.get("close", 80.0)
    low_min = over.get("low_min", 75.0)
    drop_at = 252 - over.get("drop_last_n", 52)
    with sqlite3.connect(path) as c:
        c.execute("INSERT INTO mm_sector_map (symbol, sector, market_cap_usd) "
                  "VALUES ('TITN', 'Software', ?)",
                  (over.get("cap", 1.0e12),))
        for i in range(11):
            c.execute("INSERT INTO mm_sector_map (symbol, sector, market_cap_usd) "
                      "VALUES (?, 'Software', ?)", (f"F{i:02d}", 1.0e9 * (i + 1)))
        for i, d in enumerate(dates):
            px = 100.0 if i < drop_at else close
            lo = low_min if i == 100 else px - 1.0
            c.execute("INSERT INTO mm_bars (symbol, date, open, high, low, close, "
                      "volume, source) VALUES ('TITN', ?, ?, ?, ?, ?, 1000, 't')",
                      (d, px, px + 1, lo, px))
        c.execute("INSERT INTO mm_indicators_daily (symbol, date, rsi14, sma50, atr14) "
                  "VALUES ('TITN', ?, ?, ?, 2.0)",
                  (AS_OF, over.get("rsi", 35.0), over.get("sma50", 95.0)))
        c.execute("INSERT INTO mm_fundamentals_quarterly (symbol, period_end, revenue, "
                  "earnings, free_cash_flow, total_debt, total_equity, source, fetched_at) "
                  "VALUES ('TITN', '2026-03-31', 100e9, ?, ?, 20e9, 100e9, 't', 'now')",
                  (over.get("earnings", 20.0e9), over.get("fcf", 10.0e9)))
        c.execute("INSERT INTO mm_fundamentals_growth (symbol, as_of_date, "
                  "latest_period_end, n_quarters, rev_chg_ttm, computed_at) "
                  "VALUES ('TITN', ?, '2026-03-31', 8, ?, 'now')",
                  (AS_OF, over.get("rev_growth", 25.0)))
        c.execute("INSERT INTO mm_analyst_estimates (symbol, as_of_date, rating_buy, "
                  "rating_hold, rating_sell) VALUES ('TITN', ?, ?, ?, ?)",
                  (AS_OF, over.get("buy", 10), over.get("hold", 3),
                   over.get("sell", 0)))
        c.execute("INSERT INTO mm_kronos_signals (model_id, symbol, date, "
                  "predicted_5d_return, predicted_30d_return, predicted_5d_vol, "
                  "sample_count, quality_flag, created_at) "
                  "VALUES ('k', 'TITN', ?, ?, ?, 0.02, 3, ?, 'now')",
                  (over.get("kronos_date", AS_OF), over.get("k5", 0.05),
                   over.get("k30", 0.02), over.get("kflag", "ok")))
        c.execute("INSERT INTO mm_regime (as_of_date, earnings_blackout_symbols) "
                  "VALUES (?, ?)", (AS_OF, over.get("blackout", "")))
        if over.get("next_earnings"):
            c.execute("INSERT INTO mm_earnings_events (symbol, report_date) "
                      "VALUES ('TITN', ?)", (over["next_earnings"],))
        if over.get("held"):
            c.execute("INSERT INTO mm_alpaca_positions (symbol, qty, snapshot_at) "
                      "VALUES ('TITN', 10, 'now')")
        c.commit()


def _candidates(path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(path) as c:
        c.row_factory = sqlite3.Row
        return c.execute("SELECT * FROM mm_quality_candidates").fetchall()


class TestFullPass:
    def test_candidate_emitted_with_gate_values(self, db):
        _seed(db)
        result = scan(db)
        assert result["candidates"] == ["TITN"]
        (row,) = _candidates(db)
        assert row["strategy_id"] == STRATEGY_ID
        assert row["as_of_date"] == AS_OF
        assert row["net_margin"] == pytest.approx(0.2)
        assert row["debt_to_equity"] == pytest.approx(0.2)
        assert row["kronos_5d"] == pytest.approx(0.05)
        assert row["dip_triggers"] == "near_52w_low,sma50_rsi"
        assert row["suggested_stop"] == pytest.approx(80 * 0.92)
        assert (row["held"], row["in_picks"]) == (0, 0)

    def test_rescan_replaces_not_duplicates(self, db):
        _seed(db)
        scan(db)
        scan(db)
        assert len(_candidates(db)) == 1

    def test_held_symbol_is_flagged_not_dropped(self, db):
        _seed(db, held=True)
        assert scan(db)["candidates"] == ["TITN"]
        assert _candidates(db)[0]["held"] == 1


class TestGateFailures:
    def test_degenerate_kronos_blocks(self, db):
        _seed(db, kflag="degenerate")
        assert scan(db)["candidates"] == []

    def test_kronos_below_threshold_blocks(self, db):
        _seed(db, k5=0.01)
        assert scan(db)["candidates"] == []

    def test_kronos_negative_30d_blocks(self, db):
        _seed(db, k30=-0.01)
        assert scan(db)["candidates"] == []

    def test_stale_kronos_blocks(self, db):
        _seed(db, kronos_date="2026-07-20")
        assert scan(db)["candidates"] == []

    def test_any_sell_rating_blocks(self, db):
        _seed(db, sell=1)
        assert scan(db)["candidates"] == []

    def test_weak_buy_consensus_blocks(self, db):
        _seed(db, buy=5, hold=3)
        assert scan(db)["candidates"] == []

    def test_thin_margin_blocks(self, db):
        _seed(db, earnings=5.0e9)
        assert scan(db)["candidates"] == []

    def test_margin_alternate_needs_positive_fcf(self, db):
        _seed(db, earnings=12.0e9, fcf=-1.0e9)
        assert scan(db)["candidates"] == []

    def test_low_growth_blocks(self, db):
        _seed(db, rev_growth=8.0)
        assert scan(db)["candidates"] == []

    def test_no_dip_blocks(self, db):
        _seed(db, close=120.0, sma50=95.0, rsi=60.0, low_min=95.0)
        assert scan(db)["candidates"] == []

    def test_upcoming_earnings_blocks(self, db):
        _seed(db, next_earnings="2026-08-04")
        assert scan(db)["candidates"] == []

    def test_regime_blackout_blocks(self, db):
        _seed(db, blackout="AU,TITN,MRNA")
        assert scan(db)["candidates"] == []

    def test_small_cap_never_scanned(self, db):
        _seed(db, cap=5.0e8)
        result = scan(db)
        assert result["candidates"] == []


class TestDipTriggers:
    def test_post_earnings_gap_fires_alone(self, db):
        dates = [d.strftime("%Y-%m-%d")
                 for d in pd.bdate_range(end=AS_OF, periods=252)]
        _seed(db, close=90.0, sma50=85.0, rsi=50.0, drop_last_n=2)
        with sqlite3.connect(db) as c:
            c.execute("INSERT INTO mm_earnings_events (symbol, report_date, "
                      "eps_surprise_pct) VALUES ('TITN', ?, 5.6)", (dates[-3],))
            c.commit()
        assert scan(db)["candidates"] == ["TITN"]
        assert _candidates(db)[0]["dip_triggers"] == "post_earnings_gap"
