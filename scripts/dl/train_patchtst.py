""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.datasets import SequenceDataset, _numeric_feature_cols, load_and_concat
from macmini.dl.models.patchtst import PatchTSTRegressor

TARGET = "forward_5d_return"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/dl_cache", type=Path)
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-end", default="2025-12-31")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--patch-length", type=int, default=16)
    p.add_argument("--patch-stride", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--out", default="data/checkpoints/patchtst", type=Path)
    p.add_argument("--limit-symbols", type=int, default=0, help="0 = all, else use N most-recent symbols")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA not available.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    df = load_and_concat(args.cache)
    df = df.filter(pl.col(TARGET).is_not_null())
    feat_cols = _numeric_feature_cols(df)
    print(f"features={len(feat_cols)}  total_rows={df.height:,}", flush=True)

    d = df.with_columns(pl.col("date").cast(pl.Utf8))
    train_df = d.filter(pl.col("date") <= args.train_end)
    val_df = d.filter((pl.col("date") > args.train_end) & (pl.col("date") <= args.val_end))
    test_df = d.filter(pl.col("date") > args.val_end)
    print(f"split: train={train_df.height:,}  val={val_df.height:,}  test={test_df.height:,}", flush=True)

    if args.limit_symbols > 0:
        top_syms = train_df.group_by("symbol").agg(pl.len().alias("n")).sort("n", descending=True).head(args.limit_symbols)["symbol"].to_list()
        train_df = train_df.filter(pl.col("symbol").is_in(top_syms))
        val_df = val_df.filter(pl.col("symbol").is_in(top_syms))
        test_df = test_df.filter(pl.col("symbol").is_in(top_syms))
        print(f"limited to top {args.limit_symbols} syms: train={train_df.height:,}  val={val_df.height:,}  test={test_df.height:,}", flush=True)

    train_ds = SequenceDataset(train_df, feat_cols, TARGET, seq_len=args.seq_len, stats=None)
    val_ds = SequenceDataset(val_df, feat_cols, TARGET, seq_len=args.seq_len, stats=train_ds.stats)
    test_ds = SequenceDataset(test_df, feat_cols, TARGET, seq_len=args.seq_len, stats=train_ds.stats) if test_df.height else None

    print(f"sequences: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds) if test_ds else 0:,}", flush=True)
    print(f"data prep in {time.time()-t0:.1f}s", flush=True)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("FAIL: empty train or val — reduce seq_len or shift split dates")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    model = PatchTSTRegressor(
        n_channels=len(feat_cols),
        context_length=args.seq_len,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        max_epochs=args.epochs,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out, monitor="val_dir_acc", mode="max", save_top_k=1,
                           filename="best-{epoch:02d}-{val_dir_acc:.3f}")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=[ckpt, early],
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    t1 = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"train wall-clock: {time.time()-t1:.1f}s", flush=True)

    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        results = trainer.validate(model, test_loader, verbose=False)
        print(f"test-fold metrics: {results}", flush=True)

    print(f"best ckpt: {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
