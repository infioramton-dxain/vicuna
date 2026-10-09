""
from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.selector import evaluate as ev
from macmini.selector import fundamentals as fu
from macmini.selector import panel as pn


def _bars(n: int = 400, seed: int = 1, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1e6, 2e6, n).astype(float)}, index=idx)


class TestPanel:
    def test_features_ignore_the_future(self) -> None:
        b, spy = _bars(), _bars(seed=2)
        t = 300
        before = pn.features(b, spy).iloc[t]
        b2 = b.copy()
        b2.iloc[t + 1:] *= 3.0
        after = pn.features(b2, spy).iloc[t]
        pd.testing.assert_series_equal(before, after)

    def _flat(self, path: list[tuple[float, float, float, float]]) -> pd.DataFrame:
        ""
        rows = [(100.0, 101.0, 99.0, 100.0)] * 20 + path
        return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                            index=pd.bdate_range("2025-01-02", periods=len(rows)))

    def _label(self, path) -> pd.Series:
        b = self._flat(path)
        atr = np.full(len(b), 2.0)
        return pn.labels(b, atr, np.array([19])).iloc[0]

    def test_target_hit(self) -> None:
        lab = self._label([(100, 105, 99, 104), (104, 113, 103, 112)])
        assert lab["how"] == "target" and lab["br"] == pytest.approx(0.12)
        assert lab["stop_frac"] == pytest.approx(0.06) and lab["days"] == 1

    def test_stop_hit_and_gap_through(self) -> None:
        lab = self._label([(100, 101, 93, 94)])
        assert lab["how"] == "stop" and lab["br"] == pytest.approx(-0.06)
        gap = self._label([(100, 101, 99, 100), (90, 91, 89, 90)])
        assert gap["br"] == pytest.approx(-0.10), "a gap below the stop fills at the open"

    def test_bar_touching_both_is_a_stop(self) -> None:
        assert self._label([(100, 113, 93, 100)])["how"] == "stop"

    def test_label_never_uses_the_sample_bar(self) -> None:
        ""
        b = self._flat([(100, 105, 99, 104), (104, 113, 103, 112)])
        b.iloc[19] = (100.0, 101.0, 50.0, 100.0)
        lab = pn.labels(b, np.full(len(b), 2.0), np.array([19])).iloc[0]
        assert lab["how"] == "target"


class TestFundamentals:
    @pytest.fixture()
    def conn(self, tmp_path):
        c = sqlite3.connect(tmp_path / "f.db")
        c.execute(schema.MM_SEC_FACTS)
        def fact(tag, start, end, filed, value, unit="USD"):
            c.execute(
                "INSERT INTO mm_sec_facts (symbol, cik, tag, unit, period_end, "
                "period_start, form, filed, value, fetched_at) "
                "VALUES ('T', '1', ?, ?, ?, ?, '10-Q', ?, ?, 'x')",
                (tag, unit, end, start, filed, value))
        cfo = "NetCashProvidedByUsedInOperatingActivities"
        fact(cfo, "2024-01-01", "2024-12-31", "2025-02-15", 400.0)
        fact(cfo, "2024-01-01", "2024-06-30", "2024-08-01", 200.0)
        fact(cfo, "2025-01-01", "2025-06-30", "2025-08-01", 260.0)
        fact(cfo, "2024-01-01", "2024-12-31", "2025-11-01", 9999.0)
        c.commit()
        yield c
        c.close()

    def test_timeline_is_point_in_time(self, conn) -> None:
        t = fu.filing_timeline(conn, "T").set_index("filed")["cfo"]
        assert t[pd.Timestamp("2025-02-15")] == pytest.approx(400.0)
        assert t[pd.Timestamp("2025-08-01")] == pytest.approx(400 + 260 - 200)
        assert t[pd.Timestamp("2025-11-01")] == pytest.approx(9999 + 260 - 200)

    def test_no_facts_no_rows(self, conn) -> None:
        assert fu.filing_timeline(conn, "NONE").empty


class TestScoreCard:
    def _scored(self, signal: float) -> pd.DataFrame:
        ""
        rng = np.random.default_rng(7)
        rows = []
        for d in pd.bdate_range("2024-01-02", periods=30, freq="5B"):
            noise = rng.normal(0, 1, 40)
            score = rng.normal(0, 1, 40)
            r = signal * score + noise
            for i in range(40):
                rows.append({"date": d, "symbol": f"S{i}", "score": score[i],
                             "eligible": True, "stop_frac": 0.08,
                             "br": 0.08 * r[i], pn.OUTCOME: r[i],
                             "how": "stop" if r[i] < 0 else "target",
                             "f40": r[i] * 0.05})
        return pd.DataFrame(rows)

    def test_informative_score_beats_random(self) -> None:
        card = ev.score_card(self._scored(1.0))
        assert card["excess"] > 0.5 and card["rank_ic"] > 0.3
        assert card["top_stays_down"] < card["random_stays_down"]

    def test_uninformative_score_is_not_accepted(self) -> None:
        card = ev.score_card(self._scored(0.0))
        assert abs(card["excess"]) < 0.5
        assert not ev.verdict(card)["accepted"]

    def test_verdict_needs_every_check(self) -> None:
        card = {"excess": 0.1, "t_excess_quarterly": 3.0, "years_positive": 6,
                "top_stays_down": 0.30, "random_stays_down": 0.38,
                "rank_ic": 0.03, "t_ic": 2.0}
        v = ev.verdict(card)
        assert not v["accepted"] and not v["checks"]["4_rank_ic_t>=3"]
        assert ev.verdict({**card, "t_ic": 3.5})["accepted"]


class TestExporterHook:
    def test_ranks_load_and_go_stale(self, tmp_path) -> None:
        from macmini.ranking.exporter import _load_selector_ranks
        c = sqlite3.connect(tmp_path / "s.db")
        assert _load_selector_ranks(c, date(2026, 9, 29), 5) == {}
        c.execute(schema.MM_SELECTOR_SCORES)
        c.executemany(
            "INSERT INTO mm_selector_scores VALUES (?, ?, ?, ?, ?, 'm', 'x')",
            [("2026-09-28", "KHC", 0.4, 0.25, 1), ("2026-09-28", "LAR", 0.1, None, 0),
             ("2026-10-05", "KHC", 0.9, 0.99, 1)])
        c.commit()
        assert _load_selector_ranks(c, date(2026, 9, 29), 5) == {"KHC": 0.25}
        assert _load_selector_ranks(c, date(2026, 10, 4), 5) == {}, "too old"
        assert _load_selector_ranks(c, date(2026, 10, 5), 5) == {"KHC": 0.99}
