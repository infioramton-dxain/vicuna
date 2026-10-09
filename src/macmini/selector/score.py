""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.data.schema import MM_SELECTOR_SCORES
from macmini.selector import panel as panel_mod
from macmini.selector.evaluate import (PARAMS, SEEDS, feature_columns,
                                       fit_predict, prepare)
from macmini.selector.panel import MAX_HOLD

log = logging.getLogger(__name__)

MODEL_ID = "selector_xgb_rank_v1"
OBJECTIVE = "rank"


def score_latest(panel: pd.DataFrame) -> pd.DataFrame:
    ""
    feats = feature_columns(panel)
    df = prepare(panel, feats, OBJECTIVE)
    latest = df["date"].max()
    cutoff = latest - pd.Timedelta(days=int((MAX_HOLD + 2) * 7 / 5) + 5)
    train = df[(df["date"] < cutoff) & df["y"].notna()]
    today = df[df["date"] == latest].copy()
    today["score"] = fit_predict(train, today, feats, OBJECTIVE)
    pool = today[today["eligible"]]
    today["rank_pct"] = pool["score"].rank(pct=True).reindex(today.index)
    log.info("trained on %d rows to %s; scored %d symbols for %s (%d eligible)",
             len(train), train["date"].max().date(), len(today),
             latest.date(), len(pool))
    return today[["date", "symbol", "score", "rank_pct", "eligible"]]


def run(db_path: Path | str | None = None) -> dict:
    db = resolve_db(db_path)
    panel = panel_mod.build(db, step=1)
    scores = score_latest(panel)
    as_of = scores["date"].iloc[0].date().isoformat()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(db) as conn:
        conn.execute(MM_SELECTOR_SCORES)
        conn.execute("DELETE FROM mm_selector_scores WHERE as_of_date = ? "
                     "AND model_id = ?", (as_of, MODEL_ID))
        conn.executemany(
            "INSERT INTO mm_selector_scores (as_of_date, symbol, score, rank_pct, "
            "eligible, model_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(as_of, r.symbol, float(r.score),
              None if pd.isna(r.rank_pct) else float(r.rank_pct),
              int(r.eligible), MODEL_ID, now) for r in scores.itertuples()])
        conn.commit()
    top = scores[scores["eligible"]].nlargest(10, "score")["symbol"].tolist()
    return {"as_of_date": as_of, "model_id": MODEL_ID, "scored": int(len(scores)),
            "eligible": int(scores["eligible"].sum()), "top_10": top,
            "params": PARAMS, "seeds": list(SEEDS), "mode": "shadow"}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Nightly selector scores (shadow).")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    print(json.dumps(run(args.db)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
