""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from macmini.config import DB_PATH

log = logging.getLogger(__name__)

GATE_MIN_DAYS = 60
GATE_MIN_T = 3.0
MIN_CROSS_SECTION = 30
FWD_ENTRY_SHIFT = 1
FWD_EXIT_SHIFT = 6


def _latest_model_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT model_id FROM mm_dl_signals GROUP BY model_id "
        "ORDER BY MAX(created_at) DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def daily_rank_ic(
    conn: sqlite3.Connection,
    model_id: str,
    min_date: str | None = None,
    table: str = "mm_dl_signals",
) -> pd.Series:
    ""
    if table not in ("mm_dl_signals", "mm_kronos_signals"):
        raise ValueError(f"unsupported signals table: {table}")
    quality = (" AND COALESCE(quality_flag, 'ok') = 'ok'"
               if table == "mm_kronos_signals" else "")
    sig = pd.read_sql_query(
        f"SELECT symbol, date, predicted_5d_return FROM {table} "
        f"WHERE model_id = ? AND predicted_5d_return IS NOT NULL{quality}",
        conn, params=(model_id,),
    )
    if min_date:
        sig = sig[sig["date"] >= min_date]
    if sig.empty:
        return pd.Series(dtype=float)

    symbols = sorted(sig["symbol"].unique())
    placeholders = ",".join("?" * len(symbols))
    bars = pd.read_sql_query(
        f"SELECT symbol, date, open FROM mm_bars "
        f"WHERE symbol IN ({placeholders}) AND date >= ?",
        conn, params=(*symbols, min(sig["date"])),
    )
    opens = bars.pivot(index="date", columns="symbol", values="open").sort_index()
    fwd = opens.shift(-FWD_EXIT_SHIFT) / opens.shift(-FWD_ENTRY_SHIFT) - 1.0

    fwd_long = fwd.stack().rename("realized_5d").reset_index()
    merged = sig.merge(fwd_long, on=["date", "symbol"], how="inner").dropna()

    ics = {}
    for day, grp in merged.groupby("date"):
        if len(grp) < MIN_CROSS_SECTION:
            continue
        ics[day] = grp["predicted_5d_return"].corr(
            grp["realized_5d"], method="spearman"
        )
    return pd.Series(ics, dtype=float).dropna().sort_index()


def evaluate(ic: pd.Series) -> dict:
    n = len(ic)
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
    return {
        "n_days": n,
        "mean_ic": round(mean, 5) if n else None,
        "ic_std": round(std, 5) if n > 1 else None,
        "t_stat": round(t, 3) if not math.isnan(t) else None,
        "gate_min_days": GATE_MIN_DAYS,
        "gate_min_t": GATE_MIN_T,
        "gate_pass": bool(n >= GATE_MIN_DAYS and not math.isnan(t) and t >= GATE_MIN_T),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DL rank-IC promotion gate.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--model-id", default=None,
                        help="default: model_id with the newest created_at")
    parser.add_argument("--min-date", default=None,
                        help="restrict to signals on/after this date "
                             "(e.g. the checkpoint's training cutoff)")
    parser.add_argument("--json-out", default="exports/dl_gate_latest.json")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    model_id = args.model_id or _latest_model_id(conn)
    if not model_id:
        print("eval_dl_gate: mm_dl_signals is empty — nothing to evaluate")
        return 2

    ic = daily_rank_ic(conn, model_id, args.min_date)
    result = {
        "model_id": model_id,
        "min_date": args.min_date,
        "window": (f"{ic.index.min()}..{ic.index.max()}" if len(ic) else None),
        **evaluate(ic),
        "convention": "open(t+1)->open(t+6), Spearman per day, "
                      f"cross-section >= {MIN_CROSS_SECTION}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out = Path(args.json_out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"model_id      : {result['model_id']}")
    print(f"window        : {result['window']}  ({result['n_days']} days)")
    print(f"mean rank IC  : {result['mean_ic']}")
    print(f"t-stat        : {result['t_stat']}")
    print(f"GATE ({GATE_MIN_DAYS}d & t>={GATE_MIN_T}) : "
          f"{'PASS' if result['gate_pass'] else 'FAIL'}")
    print(f"written       : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
