""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macmini.config import DB_PATH
from eval_dl_gate import GATE_MIN_DAYS, GATE_MIN_T, daily_rank_ic, evaluate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--min-date", default=None)
    p.add_argument("--json-out", default="exports/model_scoreboard.json")
    args = p.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    pool: list[tuple[str, str]] = []
    for table in ("mm_dl_signals", "mm_kronos_signals"):
        for (mid,) in conn.execute(
                f"SELECT DISTINCT model_id FROM {table}").fetchall():
            pool.append((table, mid))

    m = len(pool) or 1
    adjusted_t = float(norm.ppf(1 - 0.05 / m / 2))
    effective_bar = max(GATE_MIN_T, adjusted_t)

    rows = []
    for table, mid in pool:
        ic = daily_rank_ic(conn, mid, args.min_date, table=table)
        e = evaluate(ic)
        e["model_id"] = mid
        e["table"] = table
        e["clears_adjusted_bar"] = bool(
            e["n_days"] >= GATE_MIN_DAYS and e["t_stat"] is not None
            and e["t_stat"] >= effective_bar)
        rows.append(e)
    rows.sort(key=lambda r: (r["t_stat"] is None, -(r["t_stat"] or -99)))

    exit_rows = []
    for algo in ("cql", "bcq"):
        entry = {"algo": algo, "instrument": "exit_uplift"}
        rpt = Path(f"exports/rl_exit_uplift_{algo}.json")
        if rpt.exists():
            r = json.loads(rpt.read_text())
            entry.update({
                "model_id": r.get("model_id"),
                "mean_uplift_logret": r.get("overall_mean_uplift"),
                "t_stat": r.get("overall_t"),
                "n_eval_episodes": r.get("n_episodes"),
                "oos_start": r.get("oos_start"),
            })
        days = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM mm_rl_signals "
            "WHERE policy_id LIKE ? AND policy_id NOT LIKE '%smoke%'",
            (f"rl_exit_{algo}_%",)).fetchone()[0]
        entry["oos_annotation_days"] = days
        entry["clears_gate"] = bool(
            days >= GATE_MIN_DAYS
            and (entry.get("t_stat") or 0) >= GATE_MIN_T)
        exit_rows.append(entry)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_models_in_pool": m,
        "gate_min_days": GATE_MIN_DAYS,
        "single_model_t_bar": GATE_MIN_T,
        "bonferroni_adjusted_t_bar": round(adjusted_t, 3),
        "effective_t_bar": round(effective_bar, 3),
        "models": rows,
        "exit_policies": exit_rows,
    }
    path = Path(args.json_out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"scoreboard: {m} model_ids, effective t bar {effective_bar:.2f} "
          f"→ {path}", flush=True)
    for r in rows[:8]:
        print(f"  {r['model_id']:36s} days={r['n_days']:<4d} "
              f"ic={r['mean_ic']} t={r['t_stat']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
