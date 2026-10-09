""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.ranking.exporter import (
    K_PER_SECTOR,
    MAX_PICKS_TOTAL,
    RESEARCH_MIN_DEFLATED_SHARPE,
    MIN_PROFIT_FACTOR,
    MAX_DRAWDOWN_FLOOR,
    SURVIVAL_WINDOWS,
)


def build(db_path: Path, run_id: str | None, out_path: Path) -> None:
    t0 = time.time()
    with sqlite3.connect(db_path) as c:
        if run_id is None:
            row = c.execute(
                "SELECT run_id FROM mm_backtest_runs "
                "GROUP BY run_id ORDER BY MAX(created_at) DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise SystemExit("mm_backtest_runs is empty")
            run_id = row[0]
        print(f"loading mm_backtest_runs for run_id={run_id}", flush=True)
        df = pd.read_sql_query(
            "SELECT strategy_id, symbol, sector, window_end, "
            "       deflated_sharpe, profit_factor, max_drawdown "
            "FROM mm_backtest_runs WHERE run_id = ? "
            "ORDER BY strategy_id, symbol, window_end",
            c, params=[run_id],
        )
    print(f"  rows={len(df):,}  window_ends={df['window_end'].nunique()}  "
          f"strategies={df['strategy_id'].nunique()}  symbols={df['symbol'].nunique()}",
          flush=True)

    df["dsr_avg"] = (
        df.groupby(["strategy_id", "symbol"])["deflated_sharpe"]
        .rolling(SURVIVAL_WINDOWS, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    df["pf_avg"] = (
        df.groupby(["strategy_id", "symbol"])["profit_factor"]
        .rolling(SURVIVAL_WINDOWS, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
    df["dd_worst"] = (
        df.groupby(["strategy_id", "symbol"])["max_drawdown"]
        .rolling(SURVIVAL_WINDOWS, min_periods=1).min().reset_index(level=[0, 1], drop=True)
    )

    qual = (
        (df["dsr_avg"] >= RESEARCH_MIN_DEFLATED_SHARPE)
        & (df["pf_avg"] >= MIN_PROFIT_FACTOR)
        & (df["dd_worst"] >= MAX_DRAWDOWN_FLOOR)
    )
    df = df.loc[qual, ["window_end", "strategy_id", "symbol", "sector",
                       "dsr_avg", "pf_avg", "dd_worst"]].copy()
    print(f"  qualifying (strategy,symbol,date) triples: {len(df):,}", flush=True)

    df = df.sort_values(["window_end", "dsr_avg"], ascending=[True, False])

    rows_out = []
    dates = df["window_end"].unique()
    for i, d in enumerate(dates):
        sub = df[df["window_end"] == d]
        sub = (
            sub.sort_values("dsr_avg", ascending=False)
            .drop_duplicates(subset=["symbol"], keep="first")
        )
        sector_count: dict[str, int] = {}
        picked = 0
        rank = 0
        for _, row in sub.iterrows():
            if picked >= MAX_PICKS_TOTAL:
                break
            sec = row["sector"] or "Other"
            if sector_count.get(sec, 0) >= K_PER_SECTOR:
                continue
            rank += 1
            picked += 1
            sector_count[sec] = sector_count.get(sec, 0) + 1
            rows_out.append({
                "date": d, "rank": rank,
                "strategy_id": int(row["strategy_id"]),
                "symbol": row["symbol"], "sector": sec,
                "dsr": float(row["dsr_avg"]), "pf": float(row["pf_avg"]),
                "max_dd": float(row["dd_worst"]),
            })
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(dates)} dates  out={len(rows_out):,}", flush=True)

    out_df = pd.DataFrame(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    print(f"done: wrote {len(out_df):,} rows → {out_path}  "
          f"({(time.time() - t0):.1f}s)", flush=True)
    print(f"  date range: {out_df['date'].min()} → {out_df['date'].max()}", flush=True)
    print(f"  unique symbols in cache: {out_df['symbol'].nunique()}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(REPO_ROOT / "macmini.db"), type=Path)
    p.add_argument("--run-id", default=None,
                   help="mm_backtest_runs run_id; default = latest")
    p.add_argument("--out", default=str(REPO_ROOT / "data/rl_cache/universe_topk.parquet"),
                   type=Path)
    args = p.parse_args()
    build(Path(args.db), args.run_id, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
