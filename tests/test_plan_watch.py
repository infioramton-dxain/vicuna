""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "hermes" / "plan_watch.py"

_spec = importlib.util.spec_from_file_location("plan_watch", SCRIPT)
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

TODAY = date(2026, 9, 29)


def _bars(closes: list[float], end: date = date(2026, 9, 28), spread: float = 2.0):
    ""
    days, d = [], end
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return [(dd.isoformat(), c, c + spread, c - spread, c)
            for dd, c in zip(reversed(days), closes)]


def _plan(**over):
    p = {"id": "fixture-plan", "symbol": "TEST", "side": "buy", "status": "active",
         "set_on": "2026-09-22", "price_when_set": 110.0, "valid_until": "2026-10-06",
         "anchor": {"kind": "fixed", "price": 100.0}, "overshoot_atr": 1.0}
    p.update(over)
    return p


def _tick(plan, state, price, bars=None, today=TODAY, orders=()):
    bars = bars if bars is not None else _bars([110.0] * 60)
    return pw.run_watch([plan], state, {"TEST": bars}, {"TEST": (price, "ts")},
                        list(orders), today)


def test_atr_of_flat_series_is_the_daily_range():
    assert pw.atr(_bars([110.0] * 60)) == pytest.approx(4.0)


def test_armed_plan_is_silent():
    msgs, state = _tick(_plan(), {}, 105.0)
    assert msgs == []
    assert state["plans"]["fixture-plan"]["state"] == "armed"


def test_crossing_alerts_once_then_silent():
    msgs, state = _tick(_plan(), {}, 99.5)
    assert len(msgs) == 1 and "first touch" in msgs[0]
    for price in (99.0, 98.0, 101.0, 99.9):
        msgs, state = _tick(_plan(), state, price)
        assert msgs == []
    assert state["plans"]["fixture-plan"]["state"] == "triggered"


def test_overshoot_invalidates_once_with_reanalysis():
    msgs, state = _tick(_plan(), {}, 95.0)
    assert len(msgs) == 1
    assert "INVALIDATED" in msgs[0] and "Re-analysis" in msgs[0]
    assert "not to buy" in msgs[0]
    for price in (94.0, 90.0, 99.0, 120.0):
        msgs, state = _tick(_plan(), state, price)
        assert msgs == []
    assert state["plans"]["fixture-plan"]["state"] == "invalidated"


def test_triggered_plan_can_still_die_on_overshoot():
    _, state = _tick(_plan(), {}, 99.0)
    msgs, state = _tick(_plan(), state, 90.0)
    assert len(msgs) == 1 and "INVALIDATED" in msgs[0]


def test_expiry_alerts_once():
    msgs, state = _tick(_plan(valid_until="2026-09-28"), {}, 105.0)
    assert len(msgs) == 1 and "EXPIRED" in msgs[0]
    msgs, _ = _tick(_plan(valid_until="2026-09-28"), state, 105.0)
    assert msgs == []


def test_broken_premise_invalidates():
    plan = _plan(premise=[{"kind": "above_sma", "n": 50}])
    bars = _bars([120.0] * 55 + [108.0] * 5)
    msgs, state = _tick(plan, {}, 108.0, bars=bars)
    assert len(msgs) == 1 and "premise broke" in msgs[0]
    assert state["plans"]["fixture-plan"]["state"] == "invalidated"


def test_floor_holds_premise():
    plan = _plan(premise=[{"kind": "floor_holds", "price": 109.0}])
    assert _tick(plan, {}, 108.0)[0] == []
    msgs, _ = _tick(plan, {}, 108.0, bars=_bars([110.0] * 59 + [108.5]))
    assert len(msgs) == 1 and "below the floor" in msgs[0]


def test_legacy_bare_number_is_rejected_once(tmp_path):
    f = tmp_path / "plan_levels.json"
    f.write_text(json.dumps({"TEST re-entry plan": {
        "symbol": "TEST", "level": 290.0, "set_on": "2026-08-27", "kind": "absolute"}}))
    plans = pw.load_plans(str(f))
    msgs, state = pw.run_watch(plans, {}, {"TEST": _bars([265.0] * 60)},
                               {"TEST": (265.0, "ts")}, [], TODAY)
    assert len(msgs) == 1 and "NOT ARMED" in msgs[0] and "bare number" in msgs[0]
    msgs, _ = pw.run_watch(plans, state, {"TEST": _bars([265.0] * 60)},
                           {"TEST": (265.0, "ts")}, [], TODAY)
    assert msgs == []


def test_inactive_plan_is_skipped():
    msgs, state = _tick(_plan(status="superseded"), {}, 50.0)
    assert msgs == [] and state["plans"] == {}


def test_stale_bars_pause_once_and_keep_the_state():
    _, state = _tick(_plan(), {}, 99.0)
    old = _bars([110.0] * 60, end=date(2026, 9, 18))
    msgs, state = _tick(_plan(), state, 99.0, bars=old)
    assert len(msgs) == 1 and "cannot be evaluated" in msgs[0]
    assert _tick(_plan(), state, 99.0, bars=old)[0] == []
    msgs, state = _tick(_plan(), state, 99.0)
    assert msgs == [] and state["plans"]["fixture-plan"]["state"] == "triggered"


def test_derived_anchor_moves_with_the_market():
    plan = _plan(anchor={"kind": "sma", "n": 20, "pct": -0.05})
    lvl_a, _ = pw.derive_level(plan["anchor"], _bars([110.0] * 60), 110.0)
    lvl_b, _ = pw.derive_level(plan["anchor"], _bars([110.0] * 40 + [90.0] * 20), 90.0)
    assert lvl_a == pytest.approx(104.5) and lvl_b == pytest.approx(85.5)


def _order(limit, submitted="2026-09-29T14:29:23Z", side="buy"):
    return {"id": "fixture-order", "symbol": "TEST", "side": side,
            "limit_price": str(limit), "submitted_at": submitted}


def test_resting_buy_below_market_is_not_stale():
    bars = _bars([265.0] * 60, spread=6.0)
    msgs, _ = pw.run_watch([], {}, {"TEST": bars}, {"TEST": (264.61, "ts")},
                           [_order(256.0)], TODAY)
    assert msgs == []


def test_runaway_resting_buy_is_flagged_once():
    bars = _bars([110.0] * 60)
    args = ([], {"TEST": bars}, {"TEST": (130.0, "ts")},
            [_order(100.0, submitted="2026-09-01T14:00:00Z")], TODAY)
    msgs, state = pw.run_watch(args[0], {}, *args[1:])
    assert len(msgs) == 1 and "abandoned by the market" in msgs[0]
    assert pw.run_watch(args[0], state, *args[1:])[0] == []


def test_young_order_far_below_market_is_not_flagged():
    bars = _bars([110.0] * 60)
    msgs, _ = pw.run_watch([], {}, {"TEST": bars}, {"TEST": (130.0, "ts")},
                           [_order(100.0)], TODAY)
    assert msgs == []


def _v_bars():
    ""
    closes = ([120.0 - i for i in range(40)] + [81.0, 80.0, 82.0, 86.0, 90.0, 94.0, 90.0,
              86.0, 82.0, 80.5, 81.5, 85.0] + [86.0 + i * 1.5 for i in range(17)])
    return _bars(closes, spread=1.0)


def test_reanalyse_finds_the_floor_and_sizes_the_stop():
    bars = _v_bars()
    r = pw.reanalyse(bars, 105.0, target_pct=0.25)
    assert r["floor"]["touches"] >= 2
    assert 78.0 <= r["entry"] <= 82.0
    assert r["stop"] == pytest.approx(r["entry"] - max(2 * r["atr"], 0.08 * r["entry"]), abs=0.01)
    assert r["target"] == pytest.approx(r["entry"] * 1.25, abs=0.01)
    assert "not an order" in pw.reanalysis_text("TEST", r)


def test_reanalyse_at_new_lows_stands_aside():
    bars = _bars([200.0 - i for i in range(80)], spread=1.0)
    r = pw.reanalyse(bars, 119.0)
    assert r["floor"] is None and "stand aside" in r["verdict"]


def test_validate_names_every_missing_field():
    errs = pw.validate({"id": "x", "symbol": "TEST", "anchor": {"kind": "fixed", "price": 1}})
    assert {"missing `set_on`", "missing `valid_until`", "missing `price_when_set`"} <= set(errs)


def test_new_plan_round_trips_through_the_file(tmp_path):
    plans = tmp_path / "plans.json"
    rc = pw.main(["--plans", str(plans), "--env", str(tmp_path / "none.env"), "new",
                  "--symbol", "TEST", "--anchor", "fixed:256", "--valid-days", "5",
                  "--premise", "floor_holds:251.72", "--price", "268.75",
                  "--target-pct", "0.25", "--id", "fixture-new", "--write"])
    assert rc == 0
    (plan,) = pw.load_plans(str(plans))
    assert pw.validate(plan) == []
    assert plan["anchor"] == {"kind": "fixed", "price": 256.0}
    assert plan["premise"] == [{"kind": "floor_holds", "price": 251.72}]


def test_load_bars_reads_the_shipped_schema(tmp_path):
    db = tmp_path / "fixture.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE mm_bars (symbol TEXT, date TEXT, open REAL, high REAL, "
                "low REAL, close REAL, adj_close REAL, volume INTEGER, source TEXT)")
    con.executemany("INSERT INTO mm_bars VALUES ('TEST',?,?,?,?,?,?,1,'fixture')",
                    [(b[0], b[1], b[2], b[3], b[4], b[4]) for b in _bars([110.0] * 30)])
    con.commit()
    con.close()
    bars = pw.load_bars(str(db), "TEST")
    assert len(bars) == 30 and bars[-1][0] == "2026-09-28" and bars[0][0] < bars[-1][0]
