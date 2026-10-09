""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


def _xy(df: pl.DataFrame, feat_cols: list[str],
        target: str = "forward_5d_return") -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feat_cols).to_pandas().to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(df[target].to_numpy().astype(np.float32), -0.5, 0.5)
    mask = np.isfinite(y)
    return X[mask], y[mask]


def train_gbdt(train_df: pl.DataFrame, val_df: pl.DataFrame,
               feat_cols: list[str], out_dir: Path, model_id: str,
               device: str = "cuda", params: dict | None = None) -> dict:
    import xgboost as xgb

    from macmini.dl.device import require_cuda
    require_cuda("gbdt training")

    Xtr, ytr = _xy(train_df, feat_cols)
    Xva, yva = _xy(val_df, feat_cols)
    defaults = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20)
    model = xgb.XGBRegressor(
        **{**defaults, **(params or {})},
        reg_lambda=1.0, device=device, tree_method="hist",
        early_stopping_rounds=30, eval_metric="rmse")
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.ubj"
    model.save_model(ckpt)
    best_score = float(model.best_score) if model.best_score is not None else None
    deployed = {
        "model_id": model_id, "checkpoint": str(ckpt), "family": "gbdt",
        "feat_cols": feat_cols, "best_val_rmse": best_score,
        "best_iteration": int(model.best_iteration or 0),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "deployed.json").write_text(json.dumps(deployed, indent=2))
    return deployed


def predict_gbdt(deployed: dict, rows: pl.DataFrame,
                 device: str = "cuda") -> np.ndarray:
    import xgboost as xgb

    from macmini.dl.device import require_cuda
    require_cuda("gbdt inference")

    model = xgb.XGBRegressor(device=device)
    model.load_model(deployed["checkpoint"])
    X = rows.select(deployed["feat_cols"]).to_pandas().to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return model.predict(X).astype(float)
