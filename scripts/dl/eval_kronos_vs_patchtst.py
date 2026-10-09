""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macmini.config import DB_PATH
from eval_dl_gate import daily_rank_ic, evaluate


def _latest_model_id(conn: sqlite3.Connection, table: str) -> str | None:
    row = conn.execute(
        f"SELECT model_id FROM {table} GROUP BY model_id "
        "ORDER BY MAX(created_at) DESC LIMIT 1").fetchone()
    return row[0] if row else None


def compare(db_path: str, kronos_id: str | None = None,
            patchtst_id: str | None = None,
            min_date: str | None = None) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    kronos_id = kronos_id or _latest_model_id(conn, "mm_kronos_signals")
    patchtst_id = patchtst_id or _latest_model_id(conn, "mm_dl_signals")

    result: dict = {
        "kronos_model_id": kronos_id,
        "patchtst_model_id": patchtst_id,
        "min_date": min_date,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not kronos_id or not patchtst_id:
        result["error"] = "missing signals for one or both models"
        return result

    ic_k = daily_rank_ic(conn, kronos_id, min_date, table="mm_kronos_signals")
    ic_p = daily_rank_ic(conn, patchtst_id, min_date, table="mm_dl_signals")
    result["kronos"] = evaluate(ic_k)
    result["patchtst"] = evaluate(ic_p)

    common = ic_k.index.intersection(ic_p.index)
    diff = (ic_k.loc[common] - ic_p.loc[common]).dropna()
    n = len(diff)
    if n > 1:
        mean = float(diff.mean())
        std = float(diff.std(ddof=1))
        t = mean / (std / math.sqrt(n)) if std > 0 else float("nan")
        result["paired_diff_kronos_minus_patchtst"] = {
            "n_common_days": n,
            "mean_ic_diff": round(mean, 5),
            "t_stat": round(t, 3) if not math.isnan(t) else None,
        }
    else:
        result["paired_diff_kronos_minus_patchtst"] = {"n_common_days": n}
    result["daily_ic"] = {
        "kronos": {d: round(float(v), 5) for d, v in ic_k.items()},
        "patchtst": {d: round(float(v), 5) for d, v in ic_p.items()},
    }
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--kronos-model-id", default=None)
    p.add_argument("--patchtst-model-id", default=None)
    p.add_argument("--min-date", default=None)
    p.add_argument("--json-out", default="exports/kronos_vs_patchtst_latest.json")
    args = p.parse_args()

    result = compare(args.db, args.kronos_model_id, args.patchtst_model_id,
                     args.min_date)
    out = Path(args.json_out)
    out.parent.mkdir(exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "daily_ic"}
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(slim, indent=2))
    if "error" in result:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
