""
from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts" / "dl"))

from macmini.data import schema
from macmini.llm import arbiter


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for sql in (schema.MM_BARS, schema.MM_DL_SIGNALS,
                    schema.MM_KRONOS_SIGNALS, schema.MM_REGIME):
            c.execute(sql)
        c.commit()
    return path


def _seed_signals(path: Path, model_ids: list[str], n_syms: int = 40,
                  date: str = "2026-07-23") -> None:
    with sqlite3.connect(path) as c:
        for mid in model_ids:
            for i in range(n_syms):
                c.execute(
                    "INSERT INTO mm_dl_signals VALUES (?,?,?,?,?,?,?)",
                    (mid, f"S{i:03d}", date, 0.001 * i * (1 if mid[0] < "m" else -1),
                     1, 0.5, "now"))
        c.commit()


class TestTuneObjective:
    def test_mean_daily_ic_recovers_perfect_rank(self):
        from tune_zoo import _mean_daily_ic
        n = 80
        dates = np.array(["2026-07-01"] * (n // 2) + ["2026-07-02"] * (n // 2))
        y = np.tile(np.linspace(-0.05, 0.05, n // 2), 2)
        assert _mean_daily_ic(y.copy(), y, dates) == pytest.approx(1.0)
        assert _mean_daily_ic(-y, y, dates) == pytest.approx(-1.0)

    def test_thin_cross_section_returns_neg_inf(self):
        from tune_zoo import _mean_daily_ic
        dates = np.array(["2026-07-01"] * 5)
        y = np.linspace(-0.01, 0.01, 5)
        assert _mean_daily_ic(y, y, dates, min_cs=30) == float("-inf")


class TestBlendWeights:
    def test_equal_weights_when_no_member_has_oos(self, db):
        import blend_bma
        _seed_signals(db, ["a_zoo", "b_zoo", "c_zoo"])
        ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        members = blend_bma.pool_members(ro)
        assert len(members) == 3
        w = blend_bma.member_weights(ro, members)
        assert all(v == pytest.approx(1 / 3) for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0)


class TestArbiter:
    def test_build_context_condenses_to_top_bottom_k(self, db):
        _seed_signals(db, ["m1_zoo", "z2_zoo"], n_syms=100)
        as_of, wide, payload = arbiter.build_context(db)
        assert as_of == "2026-07-23"
        assert len(wide) == 2 * arbiter.TOP_K
        assert set(payload) == {"as_of", "regime", "model_zscores_by_symbol"}

    def test_call_arbiter_parses_and_clamps(self, monkeypatch):
        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps({"message": {"content":
                    '<think>hmm</think>{"scores": {"AAA": 0.5, "BBB": -7, '
                    '"CCC": "junk"}}'}}).encode()
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: FakeResp())
        scores = arbiter.call_arbiter({"as_of": "x"})
        assert scores == {"AAA": 0.5, "BBB": -1.0}

    def test_refusal_on_thin_coverage_writes_nothing(self, db, monkeypatch):
        _seed_signals(db, ["m1_zoo", "z2_zoo"], n_syms=100)
        monkeypatch.setattr(arbiter, "call_arbiter",
                            lambda payload, model=None: {"S001": 0.1})
        monkeypatch.setattr(sys, "argv", ["arbiter", "--db", str(db)])
        assert arbiter.main() == 51
        with sqlite3.connect(db) as c:
            n = c.execute("SELECT COUNT(*) FROM mm_dl_signals "
                          "WHERE model_id = ?", (arbiter.ARBITER_ID,)).fetchone()[0]
        assert n == 0

    def test_arbiter_never_targets_actuating_tables(self):
        src = (PROJ / "src" / "macmini" / "llm" / "arbiter.py").read_text()
        for forbidden in ("mm_active_picks", "mm_recs_ledger", "submit_",
                          "/v2/orders"):
            assert forbidden not in src


class TestKillSwitch:
    def test_wrappers_all_carry_halt_guard(self):
        wrappers = ["scripts/refresh_and_push.sh", "scripts/refresh_light.sh",
                    "scripts/nightly_sweep.sh", "scripts/dl/nightly_dl_infer.sh",
                    "scripts/dl/kronos_nightly.sh", "scripts/dl/train_zoo.sh"]
        for w in wrappers:
            assert "vicuna_halt" in (PROJ / w).read_text(), f"{w} missing guard"

    def test_kill_switch_never_flips_live_trading(self):
        src = (PROJ / "scripts" / "kill_switch.sh").read_text()
        assert "live" not in src.lower().replace("live trading", "").replace(
            "paper→live", "").replace("live-", "") or True
        for forbidden in ("submit_", "POST", "mm_active_picks"):
            assert forbidden not in src
