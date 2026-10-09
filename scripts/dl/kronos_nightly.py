""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.kronos_infer import (
    DEFAULT_MODEL_ID, N_SAMPLES, PRED_LEN, forecast_and_store,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=REPO_ROOT / "macmini.db")
    p.add_argument("--symbols-csv", type=Path, default=REPO_ROOT / "symbols.csv")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--pred-len", type=int, default=PRED_LEN)
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--limit", type=int, default=0, help="first N symbols only (smoke)")
    args = p.parse_args()

    symbols = [
        line.strip() for line in args.symbols_csv.read_text().splitlines()[1:]
        if line.strip()
    ]
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"kronos_nightly: {len(symbols)} symbols, model_id={args.model_id}",
          flush=True)

    started_at = datetime.now(timezone.utc).isoformat()

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total} symbols", flush=True)

    summary = forecast_and_store(
        args.db, symbols, model_id=args.model_id,
        pred_len=args.pred_len, n_samples=args.samples, progress_cb=progress)
    print(f"done: {json.dumps(summary)}", flush=True)

    run_id = (f"kronos_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
              f"_{uuid.uuid4().hex[:6]}")
    with sqlite3.connect(args.db, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute(
            "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
            "epochs_or_timesteps, best_val_metric, best_metric_name, config_json, "
            "checkpoint_path, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, "kronos_inference", started_at,
             datetime.now(timezone.utc).isoformat(), None, None, None,
             json.dumps(summary), "hf:NeoQuasar/Kronos-base",
             f"kronos_nightly: {summary['symbols_written']} written, "
             f"{summary['symbols_skipped']} skipped, {summary['elapsed_s']}s"))
        c.commit()
    print(f"logged mm_dl_runs row: run_id={run_id}", flush=True)

    if summary["symbols_written"] == 0:
        print("ERROR: zero symbols written", flush=True)
        return 25
    return 0


if __name__ == "__main__":
    sys.exit(main())
