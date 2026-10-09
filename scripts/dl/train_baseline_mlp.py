""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.models.baseline_mlp import BaselineMLP

DROP_ALWAYS = {
    "symbol", "date", "sector", "archetype",
    "forward_1d_return", "forward_5d_return", "forward_10d_return",
    "forward_5d_class", "forward_5d_sharpe", "max_dd_5d",
}
LABEL_COL = "forward_5d_class"


def load_all_shards(cache_dir: Path) -> pl.DataFrame:
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no Parquet shards in {cache_dir}")
    print(f"loading {len(files)} shards from {cache_dir}", flush=True)
    frames = []
    for p in files:
        try:
            frames.append(pl.read_parquet(p))
        except Exception as e:
            print(f"  WARN {p.name}: {e}", flush=True)
    df = pl.concat(frames, how="diagonal_relaxed")
    print(f"  → {df.height:,} total rows across {len(frames)} shards", flush=True)
    return df


def _numeric_feature_cols(df: pl.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in DROP_ALWAYS and df.schema[c] in (
            pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
        )
    ]


def _split_by_date(df: pl.DataFrame, train_end: str, val_end: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ""
    d = df.with_columns(pl.col("date").cast(pl.Utf8))
    train = d.filter(pl.col("date") <= train_end)
    val = d.filter((pl.col("date") > train_end) & (pl.col("date") <= val_end))
    test = d.filter(pl.col("date") > val_end)
    return train, val, test


def _to_tensors(df: pl.DataFrame, feat_cols: list[str], stats: dict | None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ""
    X_pd = df.select(feat_cols).to_pandas()
    y = df.select(LABEL_COL).to_pandas()[LABEL_COL].to_numpy().astype(np.int64)

    X = X_pd.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        stats = {"mean": mean, "std": std}

    X = (X - stats["mean"]) / stats["std"]
    X = np.clip(X, -10, 10)

    return torch.from_numpy(X), torch.from_numpy(y), stats


def _class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float()
    total = counts.sum()
    w = total / (n_classes * counts.clamp(min=1))
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/dl_cache", type=Path)
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-end", default="2025-12-31")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed", help="fp32, 16-mixed, bf16-mixed")
    p.add_argument("--out", default="data/checkpoints/baseline_mlp", type=Path)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA not available.")
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}", flush=True)

    t0 = time.time()
    df = load_all_shards(args.cache)
    df = df.filter(pl.col(LABEL_COL).is_not_null())
    feat_cols = _numeric_feature_cols(df)
    print(f"feature count: {len(feat_cols)}", flush=True)
    print(f"label class counts: {df.group_by(LABEL_COL).len().sort(LABEL_COL)}", flush=True)

    train_df, val_df, test_df = _split_by_date(df, args.train_end, args.val_end)
    print(f"split rows: train={train_df.height:,}  val={val_df.height:,}  test={test_df.height:,}", flush=True)

    if train_df.height == 0 or val_df.height == 0:
        raise SystemExit("FAIL: empty train or val fold — check --train-end / --val-end vs data range")

    X_train, y_train, stats = _to_tensors(train_df, feat_cols, stats=None)
    X_val, y_val, _ = _to_tensors(val_df, feat_cols, stats=stats)
    X_test, y_test, _ = _to_tensors(test_df, feat_cols, stats=stats) if test_df.height else (None, None, stats)

    print(f"tensors ready in {time.time()-t0:.1f}s  X_train={X_train.shape}  X_val={X_val.shape}", flush=True)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    cw = _class_weights(y_train, n_classes=3)
    print(f"class weights (train): {cw.tolist()}", flush=True)

    model = BaselineMLP(
        n_features=X_train.shape[1],
        n_classes=3,
        hidden_dims=(256, 128, 64),
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        class_weights=cw,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=args.out, monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch:02d}-{val_acc:.3f}")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=[ckpt, early],
        log_every_n_steps=25,
        enable_progress_bar=True,
        deterministic=False,
    )

    t1 = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"train wall-clock: {time.time()-t1:.1f}s", flush=True)

    if X_test is not None and X_test.shape[0] > 0:
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=args.batch_size, shuffle=False, pin_memory=True,
        )
        results = trainer.validate(model, test_loader, verbose=False)
        print(f"test-fold metrics: {results}", flush=True)

    print(f"best ckpt: {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
