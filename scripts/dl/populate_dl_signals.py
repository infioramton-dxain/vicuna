""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.datasets import _numeric_feature_cols, load_and_concat
from macmini.dl.models.patchtst import PatchTSTRegressor

TARGET = "forward_5d_return"


def _date_str(df: pl.DataFrame) -> pl.Series:
    ""
    col = df["date"]
    if col.dtype in (pl.Date, pl.Datetime):
        return col.dt.strftime("%Y-%m-%d")
    return col.cast(pl.Utf8).str.slice(0, 10)


def _train_stats(df: pl.DataFrame, feat_cols: list[str], train_end: str) -> dict:
    ""
    train_df = df.filter(_date_str(df) <= train_end)
    X = train_df.select(feat_cols).to_pandas().to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def _iter_symbol_windows(
    df_sym: pl.DataFrame,
    feat_cols: list[str],
    stats: dict,
    seq_len: int,
    target_dates: set[str],
):
    ""
    if df_sym.height < seq_len:
        return
    dates = _date_str(df_sym).to_numpy()
    X_all = df_sym.select(feat_cols).to_pandas().to_numpy(dtype=np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = (X_all - stats["mean"]) / stats["std"]
    for i in range(seq_len - 1, len(dates)):
        d = dates[i]
        if d not in target_dates:
            continue
        yield d, X_all[i - seq_len + 1 : i + 1]


def _batch_infer(model: PatchTSTRegressor, batch: list[np.ndarray], device: str) -> np.ndarray:
    ""
    x = torch.from_numpy(np.stack(batch, axis=0)).to(device)
    with torch.no_grad():
        out = model.model(past_values=x)
    pred = out.regression_outputs if hasattr(out, "regression_outputs") else out
    return pred.detach().float().cpu().numpy().reshape(-1)


def _staleness_guard(db_path: Path, cache_max_date: str) -> None:
    ""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
            lag_days = c.execute(
                "SELECT COUNT(DISTINCT date) FROM mm_bars WHERE date > ?",
                (cache_max_date,),
            ).fetchone()[0]
            db_max = c.execute("SELECT MAX(date) FROM mm_bars").fetchone()[0]
    except sqlite3.Error as exc:
        print(f"WARNING: staleness guard could not read {db_path}: {exc}", flush=True)
        return
    if lag_days <= 3:
        return
    msg = (f"DL feature cache is stale: cache ends {cache_max_date}, mm_bars ends "
           f"{db_max} ({lag_days} trading days behind). Nightly DL inference is "
           f"re-scoring old dates. Check scripts/dl/nightly_dl_infer.sh cache rebuild.")
    print(f"WARNING: {msg}", flush=True)
    import subprocess
    subprocess.run(
        [str(REPO_ROOT / "scripts" / "notify_operator.sh"), "DL cache stale", msg],
        check=False, timeout=60,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-id", default=None)
    p.add_argument("--deployed-config", type=Path, default=None,
                   help="JSON with checkpoint/model_id/train_end/seq_len; keeps the "
                        "cron line stable across retrains (see retrain_pipeline.sh)")
    p.add_argument("--cache", default=REPO_ROOT / "data" / "dl_cache", type=Path)
    p.add_argument("--db", default=REPO_ROOT / "macmini.db", type=Path)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-end", default="2024-12-31",
                   help="upper date used for feature-normalization stats (match training)")
    p.add_argument("--backfill-days", type=int, default=30,
                   help="score the last N distinct dates in the cache")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    best_val_metric = None
    if args.deployed_config:
        cfg_deployed = json.loads(args.deployed_config.read_text())
        args.checkpoint = Path(cfg_deployed["checkpoint"])
        args.model_id = cfg_deployed["model_id"]
        args.train_end = cfg_deployed.get("train_end", args.train_end)
        args.seq_len = int(cfg_deployed.get("seq_len", args.seq_len))
        best_val_metric = cfg_deployed.get("best_val_metric")
    if not args.checkpoint or not args.model_id:
        raise SystemExit("need --deployed-config, or both --checkpoint and --model-id")
    if not args.checkpoint.is_absolute():
        args.checkpoint = REPO_ROOT / args.checkpoint

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    started_at = datetime.now(timezone.utc).isoformat()
    print(f"loading cache from {args.cache}", flush=True)
    df = load_and_concat(args.cache)
    feat_cols = _numeric_feature_cols(df)
    print(f"features={len(feat_cols)}  rows={df.height:,}  symbols={df['symbol'].n_unique()}",
          flush=True)

    stats = _train_stats(df, feat_cols, args.train_end)
    print(f"stats: train_end={args.train_end}  mean.shape={stats['mean'].shape}", flush=True)

    unique_dates = sorted(_date_str(df).unique().to_list())
    target_dates = set(unique_dates[-args.backfill_days:])
    print(f"target_dates: {min(target_dates)} .. {max(target_dates)}  count={len(target_dates)}",
          flush=True)

    _staleness_guard(args.db, max(target_dates))

    from macmini.dl.device import require_cuda
    device = require_cuda("PatchTST inference")
    print(f"loading checkpoint on {device}: {args.checkpoint.name}", flush=True)
    model = PatchTSTRegressor.load_from_checkpoint(str(args.checkpoint), map_location=device)
    model.eval()
    model.to(device)

    now_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    symbols = sorted(df["symbol"].unique().to_list())
    total_rows = 0
    total_symbols_written = 0
    total_symbols_skipped = 0

    for si, sym in enumerate(symbols, 1):
        df_sym = df.filter(pl.col("symbol") == sym).sort("date")
        pairs = list(_iter_symbol_windows(df_sym, feat_cols, stats, args.seq_len, target_dates))
        if not pairs:
            total_symbols_skipped += 1
            continue

        preds: list[float] = []
        for i in range(0, len(pairs), args.batch_size):
            batch = [x for _, x in pairs[i : i + args.batch_size]]
            preds.extend(_batch_infer(model, batch, device).tolist())

        if args.dry_run:
            total_rows += len(pairs)
            total_symbols_written += 1
            if si % 25 == 0:
                print(f"  [dry] {si}/{len(symbols)} symbols, rows so far: {total_rows}",
                      flush=True)
            continue

        rows_to_write = []
        for (d, _), pred in zip(pairs, preds):
            rows_to_write.append((
                args.model_id, sym, d, float(pred),
                1 if pred > 0 else 0,
                float(min(1.0, abs(pred) / 0.05)),
                now_iso,
            ))
        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute("PRAGMA journal_mode = WAL")
            c.executemany(
                "INSERT OR REPLACE INTO mm_dl_signals "
                "(model_id, symbol, date, predicted_5d_return, predicted_class, "
                " confidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows_to_write,
            )
            c.commit()
        total_rows += len(rows_to_write)
        total_symbols_written += 1
        if si % 25 == 0:
            print(f"  {si}/{len(symbols)} symbols, rows so far: {total_rows}", flush=True)

    elapsed = time.time() - t0
    ended_at = datetime.now(timezone.utc).isoformat()
    print(f"done: symbols_written={total_symbols_written}  skipped={total_symbols_skipped}  "
          f"rows={total_rows}  elapsed={elapsed:.1f}s", flush=True)

    if not args.dry_run:
        run_id = f"infer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        cfg = {
            "checkpoint": str(args.checkpoint),
            "model_id": args.model_id,
            "seq_len": args.seq_len,
            "train_end": args.train_end,
            "backfill_days": args.backfill_days,
            "batch_size": args.batch_size,
            "target_dates_min": min(target_dates),
            "target_dates_max": max(target_dates),
        }
        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute(
                "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
                "epochs_or_timesteps, best_val_metric, best_metric_name, config_json, "
                "checkpoint_path, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, "patchtst_inference", started_at, ended_at,
                    None, best_val_metric, "val_dir_acc",
                    json.dumps(cfg), str(args.checkpoint),
                    f"populate_dl_signals: {total_rows} rows, "
                    f"{total_symbols_written} symbols, {elapsed:.1f}s",
                ),
            )
            c.commit()
        print(f"logged mm_dl_runs row: run_id={run_id}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
