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

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.datasets import (
    SequenceDataset, _numeric_feature_cols, load_and_concat,
)
from macmini.dl.zoo import gbdt as zoo_gbdt
from macmini.dl.zoo.registry import trainable

ZOO_CKPT_ROOT = REPO_ROOT / "data" / "checkpoints" / "zoo"


def _date_str(df: pl.DataFrame) -> pl.Series:
    col = df["date"]
    if col.dtype in (pl.Date, pl.Datetime):
        return col.dt.strftime("%Y-%m-%d")
    return col.cast(pl.Utf8).str.slice(0, 10)


def _log_run(db: Path, name: str, started: str, cfg: dict,
             best_metric: float | None, metric_name: str,
             ckpt: str, note: str) -> str:
    run_id = (f"zoo_{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
              f"_{uuid.uuid4().hex[:6]}")
    with sqlite3.connect(db, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute(
            "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
            "epochs_or_timesteps, best_val_metric, best_metric_name, config_json, "
            "checkpoint_path, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, f"zoo_{name}", started,
             datetime.now(timezone.utc).isoformat(), cfg.get("epochs"),
             best_metric, metric_name, json.dumps(cfg), ckpt, note))
        c.commit()
    return run_id


def train_torch_model(spec, df, feat_cols, args) -> dict:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    from macmini.dl.zoo.wrapper import ZooRegressor

    dates = _date_str(df)
    train_df = df.filter(dates <= args.train_end)
    val_df = df.filter((dates > args.train_end) & (dates <= args.val_end))

    tuned: dict = {}
    tp = ZOO_CKPT_ROOT / spec.name / "tuned_params.json"
    if tp.exists():
        tuned = dict(json.loads(tp.read_text()).get("params", {}))
        print(f"  using tuned params ({tp.name}): {tuned}", flush=True)
    batch_size = tuned.pop("batch_size", spec.batch_size)
    lr = tuned.pop("learning_rate", 3e-4)
    wd = tuned.pop("weight_decay", 1e-4)
    import inspect
    from macmini.dl.zoo.wrapper import BACKBONES
    accepted = set(inspect.signature(BACKBONES[spec.name].__init__).parameters)
    stray = {k: tuned.pop(k) for k in list(tuned) if k not in accepted}
    if stray:
        print(f"  WARN: ignoring tuned params {spec.name} does not accept: "
              f"{stray}", flush=True)

    ds_train = SequenceDataset(train_df, feat_cols, seq_len=spec.seq_len)
    ds_val = SequenceDataset(val_df, feat_cols, seq_len=spec.seq_len,
                             stats=ds_train.stats)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=2)

    model = ZooRegressor(backbone=spec.name, n_feat=len(feat_cols),
                         seq_len=spec.seq_len, max_epochs=spec.max_epochs,
                         learning_rate=lr, weight_decay=wd,
                         **{**spec.backbone_kwargs, **tuned})
    out_dir = ZOO_CKPT_ROOT / spec.name
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, monitor="val_loss", mode="min",
                              filename="best-{epoch:02d}-{val_loss:.5f}",
                              save_top_k=1)
    from macmini.dl.device import require_cuda
    require_cuda(f"zoo training ({spec.name})")
    trainer = L.Trainer(
        max_epochs=1 if args.smoke else spec.max_epochs,
        accelerator="cuda", devices=1,
        callbacks=[ckpt_cb, EarlyStopping("val_loss", patience=3, mode="min")],
        default_root_dir=out_dir, enable_progress_bar=False,
        log_every_n_steps=50,
        limit_train_batches=40 if args.smoke else 1.0,
        limit_val_batches=20 if args.smoke else 1.0)
    trainer.fit(model, dl_train, dl_val)

    best_val = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None
    model_id = f"{spec.name}_zoo_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    deployed = {
        "model_id": model_id, "family": "torch", "backbone": spec.name,
        "checkpoint": str(ckpt_cb.best_model_path), "seq_len": spec.seq_len,
        "n_feat": len(feat_cols), "train_end": args.train_end,
        "val_end": args.val_end, "best_val_loss": best_val,
        "smoke": bool(args.smoke),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "deployed.json").write_text(json.dumps(deployed, indent=2))
    return deployed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=None,
                   help="zoo model name; repeatable")
    p.add_argument("--all", action="store_true")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "dl_cache")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "macmini.db")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-end", default="2026-06-30")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 40 train batches — plumbing check only")
    args = p.parse_args()

    if not args.model and not args.all:
        raise SystemExit("pass --model <name> (repeatable) or --all")
    specs = trainable(args.model)
    from macmini.dl.zoo.registry import SPECS
    for name in (args.model or SPECS):
        s = SPECS.get(name)
        if s and s.family == "deferred":
            print(f"SKIP {name}: deferred — {s.deferred}", flush=True)

    print(f"loading cache from {args.cache}", flush=True)
    df = load_and_concat(args.cache)
    feat_cols = _numeric_feature_cols(df)
    print(f"features={len(feat_cols)}  rows={df.height:,}", flush=True)

    for spec in specs:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        print(f"=== training {spec.name} ({spec.family}) ===", flush=True)
        if spec.family == "gbdt":
            dates = _date_str(df)
            gbdt_tuned: dict = {}
            gtp = ZOO_CKPT_ROOT / spec.name / "tuned_params.json"
            if gtp.exists():
                gbdt_tuned = dict(json.loads(gtp.read_text()).get("params", {}))
                print(f"  using tuned params: {gbdt_tuned}", flush=True)
            deployed = zoo_gbdt.train_gbdt(
                df.filter(dates <= args.train_end),
                df.filter((dates > args.train_end) & (dates <= args.val_end)),
                feat_cols, ZOO_CKPT_ROOT / spec.name,
                model_id=f"gbdt_zoo_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                params=gbdt_tuned)
            best, metric = deployed["best_val_rmse"], "val_rmse"
        else:
            deployed = train_torch_model(spec, df, feat_cols, args)
            best, metric = deployed["best_val_loss"], "val_loss"
        elapsed = time.time() - t0
        run_id = _log_run(
            args.db, spec.name, started,
            {"epochs": spec.max_epochs, "seq_len": spec.seq_len,
             "smoke": args.smoke, "train_end": args.train_end,
             "val_end": args.val_end},
            best, metric, deployed["checkpoint"],
            f"train_zoo: {spec.name} {elapsed:.0f}s"
            + (" [SMOKE — not for evaluation]" if args.smoke else ""))
        print(f"{spec.name}: {metric}={best} elapsed={elapsed:.0f}s "
              f"run_id={run_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
