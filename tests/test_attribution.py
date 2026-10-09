""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from macmini.data.schema import init_schema
from macmini.recs.attribution import (
    Decision,
    Resolution,
    load_decisions_log,
    resolve_bracket,
    run_attribution,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    ""
    db_path = tmp_path / "macmini.db"
    sqlite3.connect(db_path).close()
    init_schema(db_path)
    return db_path


def _insert_bars(db_path: Path, symbol: str, bars: list[dict]) -> None:
    ""
    with sqlite3.connect(db_path) as c:
        for b in bars:
            c.execute(
                "INSERT INTO mm_bars (symbol, date, open, high, low, close, "
                "adj_close, volume, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, b["date"], b["open"], b["high"], b["low"], b["close"],
                 b["close"], 1_000_000, "test"),
            )


def _make_trading_days(start: str, n: int) -> list[str]:
    ""
    out: list[str] = []
    d = date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _flat_bars(dates: list[str], price: float, hi_delta: float = 0.5,
               lo_delta: float = 0.5) -> list[dict]:
    ""
    return [
        {"date": d, "open": price, "high": price + hi_delta,
         "low": price - lo_delta, "close": price}
        for d in dates
    ]


def _write_decisions_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _decision_record(rec_id: str, as_of: str, symbol: str, strategy_id: int,
                     decision: str, decided_at: str,
                     stop_pct: float = 0.10, target_pct: float = 0.20,
                     size_pct: float = 0.05,
                     alpaca_order_id: str | None = None,
                     operator_reply: str = "") -> dict:
    return {
        "rec_id": rec_id, "as_of_date": as_of, "symbol": symbol,
        "strategy_id": strategy_id, "strategy_name": f"strat_{strategy_id}",
        "decision": decision, "decided_at": decided_at,
        "operator_reply": operator_reply,
        "suggested_stop_loss_pct": stop_pct,
        "suggested_target_pct": target_pct,
        "suggested_size_pct": size_pct,
        "alpaca_order_id": alpaca_order_id,
        "notes": "",
    }


class TestResolveBracket:
    ""

    def _cal(self, dates: list[str]) -> list[dict]:
        ""
        return [{"date": d, "open": 100, "high": 100, "low": 100, "close": 100}
                for d in dates]

    def test_gap_target_via_negative_pct(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [{"date": dates[0], "open": 100.0, "high": 100, "low": 100, "close": 100}]
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=-0.05)
        assert r.status == "closed"
        assert r.exit_reason == "gap_target"
        assert r.days_held == 0
        assert r.exit_price == pytest.approx(100.0)

    def test_gap_stop(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [{"date": dates[0], "open": 100.0, "high": 105, "low": 99, "close": 101}]
        r = resolve_bracket(bars, cal, stop_pct=-0.05, target_pct=0.20)
        assert r.status == "closed"
        assert r.exit_reason == "gap_stop"
        assert r.days_held == 0

    def test_day0_intraday_target_ignored(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [{"date": dates[0], "open": 100, "high": 125, "low": 99, "close": 118}]
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.status == "open"
        assert r.exit_reason is None

    def test_day0_intraday_stop_ignored(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [{"date": dates[0], "open": 100, "high": 102, "low": 85, "close": 92}]
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.status == "open"
        assert r.exit_reason is None

    def test_same_day_stop_wins_on_day_1(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [
            {"date": dates[0], "open": 100, "high": 101, "low": 99, "close": 100},
            {"date": dates[1], "open": 100, "high": 130, "low": 85, "close": 100},
        ]
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.exit_reason == "stop"
        assert r.days_held == 1

    def test_target_midweek(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        bars = [{"date": dates[0], "open": 100, "high": 102, "low": 99, "close": 101}]
        bars += _flat_bars(dates[1:5], 101)
        bars.append({"date": dates[5], "open": 101, "high": 122, "low": 100, "close": 120})
        bars += _flat_bars(dates[6:21], 120)
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.exit_reason == "target"
        assert r.days_held == 5
        assert r.exit_date == dates[5]

    def test_time_20d(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 25)
        cal = self._cal(dates)
        bars = _flat_bars(dates[:21], 100)
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.exit_reason == "time_20d"
        assert r.days_held == 20
        assert r.exit_date == dates[20]
        assert r.exit_price == pytest.approx(100.0)

    def test_no_data_20d_halted(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 21)
        cal = self._cal(dates)
        r = resolve_bracket([], cal, stop_pct=0.10, target_pct=0.20)
        assert r.exit_reason == "no_data_20d"
        assert r.status == "closed"
        assert r.exit_price is None
        assert r.entry_price is None

    def test_open_position_short_window(self) -> None:
        ""
        dates = _make_trading_days("2026-07-02", 3)
        cal = self._cal(dates)
        bars = _flat_bars(dates, 100)
        r = resolve_bracket(bars, cal, stop_pct=0.10, target_pct=0.20)
        assert r.status == "open"
        assert r.exit_reason is None

    def test_no_calendar(self) -> None:
        ""
        r = resolve_bracket([], [], stop_pct=0.10, target_pct=0.20)
        assert r.status == "open"
        assert r.entry_date is None


class TestLoadDecisionsLog:

    def test_missing_file(self, tmp_path: Path) -> None:
        decisions, fills, exit_fills = load_decisions_log(tmp_path / "no.jsonl")
        assert decisions == {} and fills == {} and exit_fills == {}

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "d.jsonl"
        p.write_text('{"broken"\n')
        decisions, fills, exit_fills = load_decisions_log(p)
        assert decisions == {} and fills == {} and exit_fills == {}

    def test_latest_decision_wins(self, tmp_path: Path) -> None:
        ""
        p = tmp_path / "d.jsonl"
        _write_decisions_jsonl(p, [
            _decision_record("r1", "2026-07-02", "SYM", 1, "approve",
                             "2026-07-02T09:15:00Z", operator_reply="y"),
            _decision_record("r1", "2026-07-02", "SYM", 1, "reject",
                             "2026-07-02T09:20:00Z", operator_reply="wait no"),
        ])
        decisions, _, _ = load_decisions_log(p)
        assert decisions["r1"].decision == "reject"
        assert decisions["r1"].operator_reply == "wait no"

    def test_fill_update_separate(self, tmp_path: Path) -> None:
        p = tmp_path / "d.jsonl"
        _write_decisions_jsonl(p, [
            _decision_record("r1", "2026-07-02", "SYM", 1, "approve",
                             "2026-07-02T09:15:00Z",
                             alpaca_order_id="ord-1"),
            {"rec_id": "r1", "decision": "fill_update",
             "alpaca_order_id": "ord-1", "actual_fill_price": 123.45,
             "actual_fill_at": "2026-07-02T09:30:15Z", "fill_qty": 50,
             "notes": ""},
        ])
        decisions, fills, exit_fills = load_decisions_log(p)
        assert decisions["r1"].decision == "approve"
        assert fills["r1"].actual_fill_price == pytest.approx(123.45)
        assert exit_fills == {}

    def test_v2_exit_fill_routed_to_exit_dict(self, tmp_path: Path) -> None:
        ""
        p = tmp_path / "d.jsonl"
        _write_decisions_jsonl(p, [
            _decision_record("r1", "2026-07-02", "SYM", 1, "approve",
                             "2026-07-02T09:15:00Z",
                             alpaca_order_id="ord-1"),
            {"rec_id": "r1", "decision": "fill_update",
             "alpaca_order_id": "ord-1", "actual_fill_price": 100.0,
             "actual_fill_at": "2026-07-02T09:30:15Z", "fill_qty": 50},
            {"rec_id": "r1", "decision": "fill_update", "leg": "exit",
             "alpaca_order_id": "ord-target-1", "exit_price": 108.5,
             "exit_at": "2026-07-15T14:22:00Z", "exit_qty": 50,
             "exit_reason": "target"},
        ])
        decisions, fills, exit_fills = load_decisions_log(p)
        assert decisions["r1"].decision == "approve"
        assert fills["r1"].actual_fill_price == pytest.approx(100.0)
        assert fills["r1"].fill_qty == 50
        assert exit_fills["r1"].actual_exit_price == pytest.approx(108.5)
        assert exit_fills["r1"].exit_qty == 50
        assert exit_fills["r1"].exit_reason == "target"

    def test_v2_exit_fill_missing_required_dropped(self, tmp_path: Path) -> None:
        ""
        p = tmp_path / "d.jsonl"
        _write_decisions_jsonl(p, [
            _decision_record("r1", "2026-07-02", "SYM", 1, "approve",
                             "2026-07-02T09:15:00Z", alpaca_order_id="ord-1"),
            {"rec_id": "r1", "decision": "fill_update",
             "alpaca_order_id": "ord-1", "actual_fill_price": 100.0,
             "actual_fill_at": "2026-07-02T09:30:15Z", "fill_qty": 50},
            {"rec_id": "r1", "decision": "fill_update", "leg": "exit",
             "alpaca_order_id": "ord-target-1",
             "exit_at": "2026-07-15T14:22:00Z", "exit_qty": 50},
        ])
        decisions, fills, exit_fills = load_decisions_log(p)
        assert "r1" in fills
        assert fills["r1"].actual_fill_price == pytest.approx(100.0)
        assert "r1" not in exit_fills

    def test_v2_exit_fill_unknown_leg_dropped(self, tmp_path: Path) -> None:
        ""
        p = tmp_path / "d.jsonl"
        _write_decisions_jsonl(p, [
            _decision_record("r1", "2026-07-02", "SYM", 1, "approve",
                             "2026-07-02T09:15:00Z", alpaca_order_id="ord-1"),
            {"rec_id": "r1", "decision": "fill_update", "leg": "sideways",
             "alpaca_order_id": "ord-1", "actual_fill_price": 100.0,
             "actual_fill_at": "2026-07-02T09:30:15Z", "fill_qty": 50},
        ])
        _, fills, exit_fills = load_decisions_log(p)
        assert fills == {}
        assert exit_fills == {}


class TestRunAttribution:
    ""

    @pytest.fixture
    def seeded_env(self, tmp_db: Path, tmp_path: Path):
        ""
        as_of = "2026-07-02"
        dates = _make_trading_days("2026-07-03", 25)

        _insert_bars(tmp_db, "SPY", _flat_bars(dates, 500))

        win_bars = [
            {"date": dates[0], "open": 100, "high": 102, "low": 99, "close": 101},
            {"date": dates[1], "open": 101, "high": 105, "low": 100, "close": 104},
            {"date": dates[2], "open": 104, "high": 108, "low": 103, "close": 107},
            {"date": dates[3], "open": 107, "high": 125, "low": 106, "close": 122},
        ] + _flat_bars(dates[4:25], 122)
        _insert_bars(tmp_db, "WIN", win_bars)

        loss_bars = [
            {"date": dates[i], "open": 100 - i, "high": 100 - i + 1,
             "low": 100 - i - 1, "close": 100 - i}
            for i in range(4)
        ] + [{"date": dates[4], "open": 96, "high": 97, "low": 85, "close": 88}]
        loss_bars += _flat_bars(dates[5:25], 88)
        _insert_bars(tmp_db, "LOSS", loss_bars)

        _insert_bars(tmp_db, "FLAT", _flat_bars(dates, 100, hi_delta=0.5, lo_delta=0.5))


        decisions_path = tmp_path / "decisions.jsonl"
        _write_decisions_jsonl(decisions_path, [
            _decision_record("rec_win_ap",  as_of, "WIN", 1, "approve",
                             f"{as_of}T09:15:00Z", stop_pct=0.10, target_pct=0.20,
                             alpaca_order_id="ord-win"),
            {"rec_id": "rec_win_ap", "decision": "fill_update",
             "alpaca_order_id": "ord-win", "actual_fill_price": 100.5,
             "actual_fill_at": f"{as_of}T09:30:15Z", "fill_qty": 50, "notes": ""},
            _decision_record("rec_loss_rj", as_of, "LOSS", 1, "reject",
                             f"{as_of}T09:16:00Z", stop_pct=0.10, target_pct=0.20),
            _decision_record("rec_flat_ap", as_of, "FLAT", 2, "approve",
                             f"{as_of}T09:17:00Z", stop_pct=0.10, target_pct=0.20,
                             alpaca_order_id="ord-flat"),
            _decision_record("rec_halt_ex", as_of, "HALT", 3, "expired",
                             f"{as_of}T16:00:00Z", stop_pct=0.10, target_pct=0.20),
        ])

        return {"db": tmp_db, "decisions": decisions_path,
                "as_of": as_of, "dates": dates}

    def test_end_to_end(self, seeded_env: dict) -> None:
        scoreboard_as_of = seeded_env["dates"][24]
        result = run_attribution(
            db_path=seeded_env["db"],
            decisions_path=seeded_env["decisions"],
            as_of_date=scoreboard_as_of,
        )
        assert result["decisions_read"] == 4
        assert result["fills_read"] == 1
        assert result["perf_rows_written"] == 4

        with sqlite3.connect(seeded_env["db"]) as c:
            rows = {r[0]: dict(zip(
                ["decision", "exit_reason", "return_pct_counterfactual",
                 "return_pct_actual", "slippage_pct", "status"], r[1:]))
                for r in c.execute(
                    "SELECT rec_id, decision, exit_reason, "
                    "return_pct_counterfactual, return_pct_actual, "
                    "slippage_pct, status FROM mm_recommendation_performance")}
        assert rows["rec_win_ap"]["exit_reason"] == "target"
        assert rows["rec_win_ap"]["return_pct_counterfactual"] == pytest.approx(0.20)
        assert rows["rec_win_ap"]["return_pct_actual"] == pytest.approx(
            (100 * 1.2 - 100.5) / 100.5, rel=1e-4)
        assert rows["rec_win_ap"]["slippage_pct"] == pytest.approx(0.005)
        assert rows["rec_loss_rj"]["exit_reason"] == "stop"
        assert rows["rec_loss_rj"]["return_pct_counterfactual"] == pytest.approx(-0.10)
        assert rows["rec_loss_rj"]["return_pct_actual"] is None
        assert rows["rec_flat_ap"]["exit_reason"] == "time_20d"
        assert rows["rec_flat_ap"]["return_pct_counterfactual"] == pytest.approx(0.0, abs=1e-6)
        assert rows["rec_halt_ex"]["exit_reason"] == "no_data_20d"
        assert rows["rec_halt_ex"]["return_pct_counterfactual"] is None
        assert all(r["status"] == "closed" for r in rows.values())

    def test_scoreboard_execution_quality(self, seeded_env: dict) -> None:
        ""
        scoreboard_as_of = seeded_env["dates"][24]
        run_attribution(
            db_path=seeded_env["db"],
            decisions_path=seeded_env["decisions"],
            as_of_date=scoreboard_as_of,
        )
        with sqlite3.connect(seeded_env["db"]) as c:
            ap, rj = (c.execute(
                "SELECT fill_rate, avg_slippage_pct, avg_return_pct_actual, "
                "implementation_shortfall_pct FROM mm_recommendation_scoreboard "
                "WHERE window='all' AND decision_bucket=?", (b,)).fetchone()
                for b in ("approved", "rejected_or_expired"))
        assert ap[0] == pytest.approx(0.5)
        assert ap[1] == pytest.approx(0.005)
        actual = (100 * 1.2 - 100.5) / 100.5
        assert ap[2] == pytest.approx(actual, rel=1e-4)
        assert ap[3] == pytest.approx(actual - 0.20, rel=1e-3)
        assert rj == (None, None, None, None)

    def test_scoreboard_delta(self, seeded_env: dict) -> None:
        ""
        scoreboard_as_of = seeded_env["dates"][24]
        run_attribution(
            db_path=seeded_env["db"],
            decisions_path=seeded_env["decisions"],
            as_of_date=scoreboard_as_of,
        )
        with sqlite3.connect(seeded_env["db"]) as c:
            rows = c.execute(
                "SELECT window, decision_bucket, n_recs, n_closed, "
                "win_rate, avg_return_pct FROM mm_recommendation_scoreboard "
                "WHERE window = 'all' ORDER BY decision_bucket"
            ).fetchall()
        assert len(rows) == 2
        by_bucket = {r[1]: r for r in rows}
        win_actual = (100 * 1.2 - 100.5) / 100.5
        assert by_bucket["approved"][2] == 2
        assert by_bucket["approved"][3] == 2
        assert by_bucket["approved"][4] == pytest.approx(0.5)
        assert by_bucket["approved"][5] == pytest.approx((win_actual + 0.0) / 2, rel=1e-4)
        assert by_bucket["rejected_or_expired"][2] == 2
        assert by_bucket["rejected_or_expired"][3] == 1
        assert by_bucket["rejected_or_expired"][4] == pytest.approx(0.0)
        assert by_bucket["rejected_or_expired"][5] == pytest.approx(-0.10)

    def test_idempotent(self, seeded_env: dict) -> None:
        ""
        scoreboard_as_of = seeded_env["dates"][24]
        run_attribution(db_path=seeded_env["db"],
                        decisions_path=seeded_env["decisions"],
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(seeded_env["db"]) as c:
            snapshot_1 = c.execute(
                "SELECT rec_id, exit_date, exit_price, exit_reason, "
                "return_pct_counterfactual FROM mm_recommendation_performance "
                "ORDER BY rec_id").fetchall()
        run_attribution(db_path=seeded_env["db"],
                        decisions_path=seeded_env["decisions"],
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(seeded_env["db"]) as c:
            snapshot_2 = c.execute(
                "SELECT rec_id, exit_date, exit_price, exit_reason, "
                "return_pct_counterfactual FROM mm_recommendation_performance "
                "ORDER BY rec_id").fetchall()
        assert snapshot_1 == snapshot_2

    def test_late_fill_update_only_updates_actual(self, tmp_db: Path,
                                                   tmp_path: Path) -> None:
        ""
        as_of = "2026-07-02"
        dates = _make_trading_days("2026-07-03", 25)
        _insert_bars(tmp_db, "SPY", _flat_bars(dates, 500))
        win_bars = [
            {"date": dates[0], "open": 100, "high": 125, "low": 99, "close": 122},
        ] + _flat_bars(dates[1:25], 122)
        _insert_bars(tmp_db, "WIN", win_bars)

        dec = tmp_path / "d.jsonl"
        _write_decisions_jsonl(dec, [
            _decision_record("r1", as_of, "WIN", 1, "approve",
                             f"{as_of}T09:15:00Z", alpaca_order_id="ord-x"),
        ])
        scoreboard_as_of = dates[24]
        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(tmp_db) as c:
            before = c.execute(
                "SELECT exit_price, exit_reason, days_held, "
                "actual_fill_price, slippage_pct FROM mm_recommendation_performance "
                "WHERE rec_id = 'r1'").fetchone()
        assert before[0] == pytest.approx(120.0)
        assert before[3] is None

        with dec.open("a") as f:
            f.write(json.dumps({
                "rec_id": "r1", "decision": "fill_update",
                "alpaca_order_id": "ord-x", "actual_fill_price": 100.75,
                "actual_fill_at": f"{as_of}T09:30:00Z", "fill_qty": 50,
                "notes": ""}) + "\n")

        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(tmp_db) as c:
            after = c.execute(
                "SELECT exit_price, exit_reason, days_held, "
                "actual_fill_price, slippage_pct FROM mm_recommendation_performance "
                "WHERE rec_id = 'r1'").fetchone()
        assert after[0] == before[0]
        assert after[1] == before[1]
        assert after[2] == before[2]
        assert after[3] == pytest.approx(100.75)
        assert after[4] == pytest.approx((100.75 - 100.0) / 100.0)

    def test_v2_exit_fill_populates_real_real_return(self, tmp_db: Path,
                                                     tmp_path: Path) -> None:
        ""
        as_of = "2026-07-02"
        dates = _make_trading_days("2026-07-03", 25)
        _insert_bars(tmp_db, "SPY", _flat_bars(dates, 500))
        win_bars = [
            {"date": dates[0], "open": 100, "high": 101, "low": 99, "close": 101},
        ] + _flat_bars(dates[1:25], 122)
        _insert_bars(tmp_db, "WIN", win_bars)

        dec = tmp_path / "d.jsonl"
        _write_decisions_jsonl(dec, [
            _decision_record("r1", as_of, "WIN", 1, "approve",
                             f"{as_of}T09:15:00Z", alpaca_order_id="ord-x"),
            {"rec_id": "r1", "decision": "fill_update",
             "alpaca_order_id": "ord-x", "actual_fill_price": 101.0,
             "actual_fill_at": f"{as_of}T09:30:00Z", "fill_qty": 50},
            {"rec_id": "r1", "decision": "fill_update", "leg": "exit",
             "alpaca_order_id": "ord-target-1", "exit_price": 118.75,
             "exit_at": f"{dates[1]}T15:59:30Z", "exit_qty": 50,
             "exit_reason": "target"},
        ])
        scoreboard_as_of = dates[24]
        result = run_attribution(db_path=tmp_db, decisions_path=dec,
                                 as_of_date=scoreboard_as_of)
        assert result["exit_fills_read"] == 1
        with sqlite3.connect(tmp_db) as c:
            row = c.execute(
                "SELECT exit_price, exit_reason, actual_exit_price, "
                "actual_exit_at, actual_exit_qty, actual_exit_reason, "
                "actual_fill_price, return_pct_actual "
                "FROM mm_recommendation_performance WHERE rec_id='r1'"
            ).fetchone()
        assert row[0] == pytest.approx(120.0)
        assert row[1] == "target"
        assert row[2] == pytest.approx(118.75)
        assert row[3] == f"{dates[1]}T15:59:30Z"
        assert row[4] == 50
        assert row[5] == "target"
        assert row[6] == pytest.approx(101.0)
        assert row[7] == pytest.approx((118.75 - 101.0) / 101.0)

    def test_v2_exit_fill_only_updates_actual_on_closed_row(
            self, tmp_db: Path, tmp_path: Path) -> None:
        ""
        as_of = "2026-07-02"
        dates = _make_trading_days("2026-07-03", 25)
        _insert_bars(tmp_db, "SPY", _flat_bars(dates, 500))
        loss_bars = [
            {"date": dates[0], "open": 100, "high": 100, "low": 80, "close": 82},
        ] + _flat_bars(dates[1:25], 82)
        _insert_bars(tmp_db, "LOSS", loss_bars)

        dec = tmp_path / "d.jsonl"
        _write_decisions_jsonl(dec, [
            _decision_record("r1", as_of, "LOSS", 1, "approve",
                             f"{as_of}T09:15:00Z", alpaca_order_id="ord-x"),
            {"rec_id": "r1", "decision": "fill_update",
             "alpaca_order_id": "ord-x", "actual_fill_price": 100.0,
             "actual_fill_at": f"{as_of}T09:30:00Z", "fill_qty": 100},
        ])
        scoreboard_as_of = dates[24]
        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(tmp_db) as c:
            before = c.execute(
                "SELECT exit_price, exit_reason, days_held, status "
                "FROM mm_recommendation_performance WHERE rec_id='r1'"
            ).fetchone()
        assert before[0] == pytest.approx(90.0)
        assert before[1] == "stop"
        assert before[3] == "closed"

        with dec.open("a") as f:
            f.write(json.dumps({
                "rec_id": "r1", "decision": "fill_update", "leg": "exit",
                "alpaca_order_id": "ord-stop-1", "exit_price": 89.50,
                "exit_at": f"{dates[0]}T10:15:00Z", "exit_qty": 100,
                "exit_reason": "stop"}) + "\n")
        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=scoreboard_as_of)
        with sqlite3.connect(tmp_db) as c:
            after = c.execute(
                "SELECT exit_price, exit_reason, days_held, status, "
                "actual_exit_price, actual_exit_reason, return_pct_actual "
                "FROM mm_recommendation_performance WHERE rec_id='r1'"
            ).fetchone()
        assert after[0] == before[0]
        assert after[1] == before[1]
        assert after[2] == before[2]
        assert after[3] == before[3]
        assert after[4] == pytest.approx(89.50)
        assert after[5] == "stop"
        assert after[6] == pytest.approx((89.50 - 100.0) / 100.0)


class TestDividendCredit:
    ""

    def _seed(self, tmp_db: Path, tmp_path: Path, ex_offset: int):
        as_of = "2026-07-02"
        dates = _make_trading_days("2026-07-03", 25)
        _insert_bars(tmp_db, "SPY", _flat_bars(dates, 500))
        _insert_bars(tmp_db, "DIVI", _flat_bars(dates, 100, 0.5, 0.5))
        with sqlite3.connect(tmp_db) as c:
            c.execute(
                "INSERT INTO mm_dividends (symbol, ex_date, rate, special, "
                "is_foreign, source, fetched_at) VALUES (?, ?, ?, 0, 0, "
                "'test', '2026-07-01T00:00:00Z')",
                ("DIVI", dates[ex_offset], 2.0))
        dec = tmp_path / "d.jsonl"
        _write_decisions_jsonl(dec, [
            _decision_record("rec_div", as_of, "DIVI", 2, "approve",
                             f"{as_of}T09:15:00Z", stop_pct=0.10,
                             target_pct=0.20, alpaca_order_id="ord-div"),
            {"rec_id": "rec_div", "decision": "fill_update",
             "alpaca_order_id": "ord-div", "actual_fill_price": 100.0,
             "actual_fill_at": f"{as_of}T09:30:15Z", "fill_qty": 10,
             "notes": ""},
        ])
        return dates, dec

    def _row(self, tmp_db: Path):
        with sqlite3.connect(tmp_db) as c:
            return c.execute(
                "SELECT dividends_per_share, return_pct_counterfactual, "
                "return_pct_actual, exit_reason "
                "FROM mm_recommendation_performance WHERE rec_id='rec_div'"
            ).fetchone()

    def test_mid_hold_dividend_credited(self, tmp_db: Path, tmp_path: Path) -> None:
        dates, dec = self._seed(tmp_db, tmp_path, ex_offset=5)
        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=dates[24])
        div_ps, ret_cf, ret_act, reason = self._row(tmp_db)
        assert reason == "time_20d"
        assert div_ps == pytest.approx(2.0)
        assert ret_cf == pytest.approx(0.02)
        assert ret_act == pytest.approx(0.02)

    def test_entry_day_ex_date_not_credited(self, tmp_db: Path, tmp_path: Path) -> None:
        dates, dec = self._seed(tmp_db, tmp_path, ex_offset=0)
        run_attribution(db_path=tmp_db, decisions_path=dec,
                        as_of_date=dates[24])
        div_ps, ret_cf, ret_act, _ = self._row(tmp_db)
        assert div_ps == pytest.approx(0.0)
        assert ret_cf == pytest.approx(0.0)
        assert ret_act == pytest.approx(0.0)
