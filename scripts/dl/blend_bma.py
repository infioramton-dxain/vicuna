""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macmini.config import DB_PATH
from eval_dl_gate import daily_rank_ic, evaluate

BLEND_ID = "blend_bma"
IC_LOOKBACK_DAYS = 30


def pool_members(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    members = [
        ("mm_dl_signals", mid) for (mid,) in conn.execute(
            "SELECT DISTINCT model_id FROM mm_dl_signals WHERE model_id NOT LIKE 'blend_%'")
    ]
    members += [("mm_kronos_signals", mid) for (mid,) in conn.execute(
        "SELECT DISTINCT model_id FROM mm_kronos_signals")]
    return members


def member_weights(conn: sqlite3.Connection,
                   members: list[tuple[str, str]]) -> dict[str, float]:
    ""
    raw: dict[str, float] = {}
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=IC_LOOKBACK_DAYS * 2)
              ).strftime("%Y-%m-%d")
    for table, mid in members:
        e = evaluate(daily_rank_ic(conn, mid, cutoff, table=table))
        t = e["t_stat"]
        raw[mid] = float(np.logaddexp(0.0, t)) if t is not None else 0.0
    if all(v == 0.0 for v in raw.values()):
        raw = {mid: 1.0 for mid in raw}
    total = sum(raw.values())
    return {mid: v / total for mid, v in raw.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--backfill-days", type=int, default=3)
    p.add_argument("--json-out", default="exports/blend_bma_weights.json")
    args = p.parse_args()

    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    members = pool_members(ro)
    if not members:
        print("blend_bma: empty pool — nothing to do", flush=True)
        return 0
    weights = member_weights(ro, members)

    frames = []
    for table, mid in members:
        f = pd.read_sql_query(
            f"SELECT symbol, date, predicted_5d_return FROM {table} "
            "WHERE model_id = ? AND predicted_5d_return IS NOT NULL",
            ro, params=(mid,))
        f["model_id"] = mid
        frames.append(f)
    sig = pd.concat(frames, ignore_index=True)
    dates = sorted(sig["date"].unique())[-args.backfill_days:]
    sig = sig[sig["date"].isin(dates)]

    def _z(g: pd.Series) -> pd.Series:
        s = g.std(ddof=0)
        return (g - g.mean()) / (s if s > 1e-12 else 1.0)

    sig["z"] = sig.groupby(["model_id", "date"])["predicted_5d_return"].transform(_z)
    sig["wz"] = sig["z"] * sig["model_id"].map(weights)
    blend = (sig.groupby(["symbol", "date"])
             .agg(score=("wz", "sum"), n_models=("wz", "size")).reset_index())
    blend = blend[blend["n_models"] >= max(2, len(members) // 3)]

    now_iso = datetime.now(timezone.utc).isoformat()
    written = 0
    for sym, grp in blend.groupby("symbol"):
        rows = [(BLEND_ID, sym, r.date, float(r.score),
                 1 if r.score > 0 else 0, min(1.0, abs(float(r.score)) / 2.0),
                 now_iso) for r in grp.itertuples()]
        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.executemany(
                "INSERT OR REPLACE INTO mm_dl_signals (model_id, symbol, date, "
                "predicted_5d_return, predicted_class, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?)", rows)
            c.commit()
        written += len(rows)

    Path(args.json_out).parent.mkdir(exist_ok=True)
    Path(args.json_out).write_text(json.dumps({
        "generated_at": now_iso, "dates": dates, "rows_written": written,
        "weights": weights,
        "note": ("equal weights — no member has scoreable OOS yet"
                 if len(set(weights.values())) == 1 else
                 f"softplus(t-stat) over last ~{IC_LOOKBACK_DAYS} OOS days"),
    }, indent=2) + "\n")
    print(f"blend_bma: {written} rows over {dates} "
          f"({len(members)} members)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
