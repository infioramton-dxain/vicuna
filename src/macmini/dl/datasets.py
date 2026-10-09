""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

DROP_ALWAYS = {
    "symbol", "date", "sector", "archetype",
    "forward_1d_return", "forward_5d_return", "forward_10d_return",
    "forward_5d_class", "forward_5d_sharpe", "max_dd_5d",
}


def _numeric_feature_cols(df: pl.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in DROP_ALWAYS and df.schema[c] in (
            pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
        )
    ]


def load_and_concat(cache_dir: Path) -> pl.DataFrame:
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no Parquet shards in {cache_dir}")
    frames = [pl.read_parquet(p) for p in files]
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.sort(["symbol", "date"])
    return df


class SequenceDataset(Dataset):
    ""

    def __init__(
        self,
        df: pl.DataFrame,
        feat_cols: list[str],
        target_col: str = "forward_5d_return",
        seq_len: int = 64,
        stats: dict | None = None,
    ):
        self.seq_len = seq_len

        symbols = df["symbol"].to_numpy()
        y = df[target_col].to_numpy().astype(np.float32)

        X = df.select(feat_cols).to_pandas().to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if stats is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std < 1e-8] = 1.0
            stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

        X = (X - stats["mean"]) / stats["std"]
        X = np.clip(X, -10, 10).astype(np.float32)

        self.X = X
        self.y = y
        self.stats = stats

        boundaries: list[tuple[int, int]] = []
        curr_start = 0
        for i in range(1, len(symbols)):
            if symbols[i] != symbols[i - 1]:
                boundaries.append((curr_start, i))
                curr_start = i
        boundaries.append((curr_start, len(symbols)))

        valid = []
        for start, end in boundaries:
            if end - start < seq_len:
                continue
            for gi in range(start + seq_len - 1, end):
                if np.isfinite(y[gi]):
                    valid.append(gi)
        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i):
        gi = int(self.valid_indices[i])
        X_seq = self.X[gi - self.seq_len + 1 : gi + 1]
        y = self.y[gi]
        y = float(np.clip(y, -0.5, 0.5))
        return torch.from_numpy(X_seq), torch.tensor(y, dtype=torch.float32)
