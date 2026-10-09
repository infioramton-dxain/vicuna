""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from macmini.dl.datasets import _numeric_feature_cols, load_and_concat
from macmini.dl.zoo import gbdt as zoo_gbdt
from macmini.dl.zoo.registry import SPECS
from populate_dl_signals import (
    _date_str, _iter_symbol_windows, _staleness_guard, _train_stats,
)

ZOO_CKPT_ROOT = REPO_ROOT / "data" / "checkpoints" / "zoo"


def _write_rows(db: Path, model_id: str, sym: str,
                rows: list[tuple[str, float]], now_iso: str) -> None:
    payload = [
        (model_id, sym, d, pred, 1 if pred > 0 else 0,
         min(1.0, abs(pred) / 0.05), now_iso)
        for d, pred in rows
    ]
    with sqlite3.connect(db, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute("PRAGMA journal_mode = WAL")
        c.executemany(
            "INSERT OR REPLACE INTO mm_dl_signals (model_id, symbol, date, "
            "predicted_5d_return, predicted_class, confidence, created_at) "
            "VALUES (?,?,?,?,?,?,?)", payload)
        c.commit()


def infer_torch(deployed: dict, df: pl.DataFrame, feat_cols: list[str],
                target_dates: set[str], db: Path, batch_size: int = 512) -> int:
    import torch

    from macmini.dl.device import require_cuda
    from macmini.dl.zoo.wrapper import ZooRegressor

    device = require_cuda("zoo inference")
    model = ZooRegressor.load_from_checkpoint(deployed["checkpoint"],
                                              map_location=device)
    model.eval().to(device)
    stats = _train_stats(df, feat_cols, deployed["train_end"])
    seq_len = int(deployed["seq_len"])
    now_iso = datetime.now(timezone.utc).isoformat()

    total = 0
    for sym in sorted(df["symbol"].unique().to_list()):
        df_sym = df.filter(pl.col("symbol") == sym).sort("date")
        pairs = list(_iter_symbol_windows(df_sym, feat_cols, stats, seq_len,
                                          target_dates))
        if not pairs:
            continue
        preds: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                x = torch.from_numpy(
                    np.stack([w for _, w in pairs[i:i + batch_size]])).to(device)
                preds.extend(model(x).float().cpu().numpy().reshape(-1).tolist())
        _write_rows(db, deployed["model_id"], sym,
                    [(d, float(p)) for (d, _), p in zip(pairs, preds)], now_iso)
        total += len(pairs)
    return total


def infer_gbdt(deployed: dict, df: pl.DataFrame,
               target_dates: set[str], db: Path) -> int:
    dates = _date_str(df)
    rows = df.filter(dates.is_in(sorted(target_dates)))
    if rows.height == 0:
        return 0
    preds = zoo_gbdt.predict_gbdt(deployed, rows)
    rows_dates = _date_str(rows).to_list()
    symbols = rows["symbol"].to_list()
    now_iso = datetime.now(timezone.utc).isoformat()
    by_sym: dict[str, list[tuple[str, float]]] = {}
    for sym, d, p in zip(symbols, rows_dates, preds):
        by_sym.setdefault(sym, []).append((d, float(p)))
    total = 0
    for sym, srows in sorted(by_sym.items()):
        _write_rows(db, deployed["model_id"], sym, srows, now_iso)
        total += len(srows)
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "dl_cache")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "macmini.db")
    p.add_argument("--backfill-days", type=int, default=3)
    p.add_argument("--allow-smoke", action="store_true",
                   help="score smoke-trained checkpoints (testing only)")
    args = p.parse_args()

    names = args.model or [n for n, s in SPECS.items() if s.family != "deferred"]
    deployed_list = []
    for name in names:
        dj = ZOO_CKPT_ROOT / name / "deployed.json"
        if not dj.exists():
            print(f"skip {name}: no deployed.json (not trained yet)", flush=True)
            continue
        deployed = json.loads(dj.read_text())
        if deployed.get("smoke") and not args.allow_smoke:
            print(f"skip {name}: smoke checkpoint (train for real first)",
                  flush=True)
            continue
        deployed_list.append((name, deployed))
    if not deployed_list:
        print("zoo_infer: nothing deployed yet — exiting cleanly", flush=True)
        return 0

    print(f"loading cache from {args.cache}", flush=True)
    df = load_and_concat(args.cache)
    feat_cols = _numeric_feature_cols(df)
    unique_dates = sorted(_date_str(df).unique().to_list())
    target_dates = set(unique_dates[-args.backfill_days:])
    print(f"target_dates: {min(target_dates)} .. {max(target_dates)}", flush=True)
    _staleness_guard(args.db, max(target_dates))

    for name, deployed in deployed_list:
        t0 = time.time()
        if deployed["family"] == "gbdt":
            n = infer_gbdt(deployed, df, target_dates, args.db)
        else:
            n = infer_torch(deployed, df, feat_cols, target_dates, args.db)
        print(f"{name}: {n} rows as {deployed['model_id']} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
