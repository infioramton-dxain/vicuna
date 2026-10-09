""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from macmini.rl.exit_policy import dataset as ds
from macmini.rl.exit_policy import infer
from macmini.rl.exit_policy.obs_spec import (
    ARCHETYPES, MAX_EPISODE_BARS, OBS_FIELDS, SECTORS,
)


@pytest.fixture()
def tiny_db(tmp_path):
    db = tmp_path / "tiny.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE mm_bars (symbol TEXT, date TEXT, open REAL,"
                " high REAL, low REAL, close REAL, adj_close REAL, volume REAL)")
    con.execute("CREATE TABLE mm_sector_map (symbol TEXT, sector TEXT)")
    con.execute("INSERT INTO mm_sector_map VALUES ('TST','Defense')")
    dates = pd.bdate_range("2026-01-02", periods=80)
    px = 100 + np.linspace(0, 10, 80)
    for d, p in zip(dates, px):
        con.execute("INSERT INTO mm_bars VALUES ('TST',?,?,?,?,?,?,1e6)",
                    (d.isoformat(), p, p + 1, p - 1, p, p))
        con.execute("INSERT INTO mm_bars VALUES ('^VIX',?,?,?,?,?,?,0)",
                    (d.isoformat(), 30, 30, 30, 30, 30))
    con.commit()
    con.close()
    return db


def test_build_observation_matches_spec(tiny_db):
    dates = pd.bdate_range("2026-01-02", periods=80)
    entry_i = 60
    as_of_i = 70
    rec = {
        "rec_id": "r1", "symbol": "TST", "strategy_id": 1,
        "entry_date": dates[entry_i].strftime("%Y-%m-%d"),
        "actual_fill_price": float(100 + 10 * entry_i / 79),
        "stop_price": None, "target_price": None,
        "exit_status": None, "broker_position": True,
    }
    rec["stop_price"] = rec["actual_fill_price"] * 0.92
    rec["target_price"] = rec["actual_fill_price"] * 1.16
    out = infer.build_observation(tiny_db, rec, dates[as_of_i].strftime("%Y-%m-%d"))
    assert out is not None
    obs, ctx = out
    assert obs.shape == (len(OBS_FIELDS),)

    px = float(100 + 10 * as_of_i / 79)
    frac0 = 0.08
    assert obs[0] == pytest.approx(
        (px / rec["actual_fill_price"] - 1) / frac0, rel=1e-5)
    assert obs[1] == pytest.approx((as_of_i - entry_i + 1) / MAX_EPISODE_BARS)
    assert obs[8] == 0.0 and obs[9] == 0.0 and obs[10] == 1.0
    sec = obs[11:11 + len(SECTORS)]
    assert sec[SECTORS.index("Defense")] == 1.0 and sec.sum() == 1.0
    arch = obs[11 + len(SECTORS):]
    assert arch[ARCHETYPES.index("trend")] == 1.0 and arch.sum() == 1.0
    assert obs[4] > 0
    assert ctx["vix_regime"] == "high"


def test_build_observation_none_without_bars(tiny_db):
    rec = {"rec_id": "r2", "symbol": "NOPE", "strategy_id": 1,
           "entry_date": "2026-03-02", "actual_fill_price": 10.0,
           "stop_price": 9.0, "target_price": 12.0,
           "exit_status": None, "broker_position": False}
    assert infer.build_observation(tiny_db, rec, "2026-03-31") is None


def _write_cache(tmp_path, entry_dates, variant="bracket"):
    rows = []
    for ed in entry_dates:
        for step in range(3):
            row = {"episode_id": f"{ed}_{variant}", "variant": variant,
                   "entry_date": pd.Timestamp(ed),
                   "date": pd.Timestamp(ed) + pd.Timedelta(days=step + 1),
                   "action": 0 if step < 2 else 1,
                   "reward": np.float32(0.01),
                   "terminal": step == 2,
                   "strategy_id": 7, "symbol": "TST"}
            for f in OBS_FIELDS:
                row[f] = np.float32(0.0)
            rows.append(row)
    pd.DataFrame(rows).to_parquet(tmp_path / "sid7_TST.parquet", index=False)


def test_load_arrays_temporal_split(tmp_path):
    _write_cache(tmp_path, ["2025-06-02", "2026-02-02"])
    full = ds.load_arrays(cache_dir=tmp_path)
    assert full["n_episodes"] == 2 and len(full["actions"]) == 6

    train = ds.load_arrays(cache_dir=tmp_path, entry_before="2026-01-01")
    assert train["n_episodes"] == 1
    ev = ds.load_arrays(cache_dir=tmp_path, entry_from="2026-01-01")
    assert ev["n_episodes"] == 1
    assert full["terminals"].sum() == 2
    assert full["observations"].shape == (6, len(OBS_FIELDS))
    with pytest.raises(ValueError):
        ds.load_arrays(cache_dir=tmp_path, entry_before="2020-01-01")
