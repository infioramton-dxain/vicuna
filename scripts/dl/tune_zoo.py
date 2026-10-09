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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.dl.zoo.registry import trainable

ZOO_CKPT_ROOT = REPO_ROOT / "data" / "checkpoints" / "zoo"
OPTUNA_DB = REPO_ROOT / "data" / "optuna_zoo.db"

TRIAL_EPOCHS = 6
TRIAL_BATCH_FRACTION = 0.5
LOSS_GUARD_FRAC = 0.5


def _date_str(df):
    import polars as pl
    col = df["date"]
    if col.dtype in (pl.Date, pl.Datetime):
        return col.dt.strftime("%Y-%m-%d")
    return col.cast(pl.Utf8).str.slice(0, 10)


def _mean_daily_ic(preds: np.ndarray, targets: np.ndarray,
                  dates: np.ndarray, min_cs: int = 30) -> float:
    import pandas as pd
    df = pd.DataFrame({"pred": preds, "y": targets, "date": dates})
    ics = [g["pred"].corr(g["y"], method="spearman")
           for _, g in df.groupby("date") if len(g) >= min_cs]
    ics = [i for i in ics if i == i]
    return float(np.mean(ics)) if ics else float("-inf")


def _common_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }


def _arch_space(name: str, trial) -> dict:
    d: dict = {}
    if name in ("gru", "deepar"):
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        d["d_model"] = trial.suggest_categorical("d_model", [64, 128, 256])
        d["n_layers"] = trial.suggest_int("n_layers", 1, 3)
    elif name in ("attention_lstm", "tft"):
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        d["d_model"] = trial.suggest_categorical("d_model", [64, 96, 128, 192])
    elif name in ("tcn", "wavenet"):
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        d["d_model"] = trial.suggest_categorical("d_model", [64, 96, 128])
        d["n_blocks"] = trial.suggest_int("n_blocks", 3, 6)
    elif name == "transformer":
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        d["d_model"] = trial.suggest_categorical("d_model", [64, 128, 192])
        d["n_layers"] = trial.suggest_int("n_layers", 2, 4)
    elif name == "nhits":
        d["d_hidden"] = trial.suggest_categorical("d_hidden", [128, 256, 384])
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    elif name in ("autoencoder", "vae"):
        d["d_latent"] = trial.suggest_categorical("d_latent", [16, 32, 64])
        d["recon_weight"] = trial.suggest_float("recon_weight", 0.05, 1.0)
    else:
        d["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    return d


def _gbdt_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 60),
    }


def _torch_trial(spec, params: dict, data: dict) -> tuple[float, float]:
    ""
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping
    from torch.utils.data import DataLoader

    from macmini.dl.zoo.wrapper import ZooRegressor

    batch = params.pop("batch_size")
    lr = params.pop("learning_rate")
    wd = params.pop("weight_decay")
    model = ZooRegressor(backbone=spec.name, n_feat=data["n_feat"],
                         seq_len=spec.seq_len, learning_rate=lr,
                         weight_decay=wd, max_epochs=TRIAL_EPOCHS, **params)
    dl_train = DataLoader(data["ds_train"], batch_size=batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    dl_val = DataLoader(data["ds_val"], batch_size=1024, num_workers=2)
    trainer = L.Trainer(
        max_epochs=TRIAL_EPOCHS, accelerator="cuda", devices=1,
        callbacks=[EarlyStopping("val_loss", patience=2, mode="min")],
        limit_train_batches=TRIAL_BATCH_FRACTION,
        enable_progress_bar=False, enable_checkpointing=False, logger=False,
        enable_model_summary=False)
    trainer.fit(model, dl_train, dl_val)
    val_loss = float(trainer.callback_metrics["val_loss"])

    model.eval().cuda()
    preds = []
    with torch.no_grad():
        for x, _ in dl_val:
            preds.append(model(x.cuda()).float().cpu().numpy())
    preds = np.concatenate(preds)
    ic = _mean_daily_ic(preds, data["val_y"], data["val_dates"])
    return val_loss, ic


def _gbdt_trial(params: dict, data: dict) -> tuple[float, float]:
    import xgboost as xgb
    model = xgb.XGBRegressor(device="cuda", tree_method="hist",
                             early_stopping_rounds=30, eval_metric="rmse",
                             reg_lambda=1.0, **params)
    model.fit(data["Xtr"], data["ytr"], eval_set=[(data["Xva"], data["yva"])],
              verbose=False)
    preds = model.predict(data["Xva"]).astype(float)
    ic = _mean_daily_ic(preds, data["yva"], data["flat_val_dates"])
    return float(model.best_score), ic


def _prep(df, feat_cols, spec, train_end, val_end) -> dict:
    from macmini.dl.datasets import SequenceDataset
    dates = _date_str(df)
    train_df = df.filter(dates <= train_end)
    val_df = df.filter((dates > train_end) & (dates <= val_end))
    if spec.family == "gbdt":
        from macmini.dl.zoo.gbdt import _xy
        Xtr, ytr = _xy(train_df, feat_cols)
        Xva, yva = _xy(val_df, feat_cols)
        vd = _date_str(val_df).to_numpy()
        mask = np.isfinite(np.clip(
            val_df["forward_5d_return"].to_numpy().astype(np.float32), -0.5, 0.5))
        return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
                "flat_val_dates": vd[mask]}
    ds_train = SequenceDataset(train_df, feat_cols, seq_len=spec.seq_len)
    ds_val = SequenceDataset(val_df, feat_cols, seq_len=spec.seq_len,
                             stats=ds_train.stats)
    val_dates = _date_str(val_df).to_numpy()[ds_val.valid_indices]
    val_y = ds_val.y[ds_val.valid_indices]
    return {"ds_train": ds_train, "ds_val": ds_val, "n_feat": len(feat_cols),
            "val_dates": val_dates, "val_y": np.clip(val_y, -0.5, 0.5)}


def tune_model(spec, df, feat_cols, args) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = ZOO_CKPT_ROOT / spec.name
    incumbent: dict = {}
    tp = out_dir / "tuned_params.json"
    if tp.exists():
        incumbent = json.loads(tp.read_text()).get("params", {})

    data = _prep(df, feat_cols, spec, args.train_end, args.val_end)
    incumbent_loss = {"v": None}

    def objective(trial):
        if spec.family == "gbdt":
            params = _gbdt_space(trial)
            loss, ic = _gbdt_trial(params, data)
        else:
            params = {**_common_space(trial), **_arch_space(spec.name, trial)}
            loss, ic = _torch_trial(spec, dict(params), data)
        if trial.number == 0:
            incumbent_loss["v"] = loss
        elif incumbent_loss["v"] is not None:
            guard = incumbent_loss["v"] + abs(incumbent_loss["v"]) * LOSS_GUARD_FRAC
            if loss > guard:
                return float("-inf")
        trial.set_user_attr("val_loss", loss)
        return ic

    study = optuna.create_study(
        study_name=f"{spec.name}_{args.train_end}",
        storage=f"sqlite:///{OPTUNA_DB}", direction="maximize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=20260724))
    if len(study.trials) == 0:
        defaults = {"learning_rate": 3e-4, "weight_decay": 1e-4,
                    "batch_size": spec.batch_size}
        if spec.family == "gbdt":
            defaults = {"n_estimators": 600, "max_depth": 6,
                        "learning_rate": 0.05, "subsample": 0.8,
                        "colsample_bytree": 0.8, "min_child_weight": 20}
        study.enqueue_trial({**defaults, **incumbent})
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout_s)

    baseline_ic = study.trials[0].value if study.trials else float("-inf")
    best = study.best_trial
    swapped = bool(best.value is not None and baseline_ic is not None
                   and best.number != 0
                   and best.value >= baseline_ic + args.margin)
    record = {
        "model": spec.name, "n_trials": len(study.trials),
        "baseline_ic": baseline_ic, "best_ic": best.value,
        "best_trial": best.number, "margin": args.margin, "swapped": swapped,
        "params": dict(best.params) if swapped else incumbent,
        "candidate_params": dict(best.params),
        "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if swapped:
        out_dir.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--margin", type=float, default=0.002,
                   help="min absolute mean-IC improvement before dial swap")
    p.add_argument("--timeout-s", type=int, default=3000,
                   help="per-model wall clock cap (keeps 11 models inside "
                        "the Sunday window)")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "dl_cache")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "macmini.db")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--val-end", default="2026-06-30")
    args = p.parse_args()

    if not args.model and not args.all:
        raise SystemExit("pass --model <name> (repeatable) or --all")
    from macmini.dl.datasets import _numeric_feature_cols, load_and_concat
    from macmini.dl.device import require_cuda
    require_cuda("zoo tuning")
    specs = trainable(args.model)

    print(f"loading cache from {args.cache}", flush=True)
    df = load_and_concat(args.cache)
    feat_cols = _numeric_feature_cols(df)

    for spec in specs:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        print(f"=== tuning {spec.name}: {args.trials} trials ===", flush=True)
        rec = tune_model(spec, df, feat_cols, args)
        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute(
                "INSERT INTO mm_dl_runs (run_id, model_type, started_at, "
                "ended_at, epochs_or_timesteps, best_val_metric, "
                "best_metric_name, config_json, checkpoint_path, notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"tune_{spec.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                 f"tune_{spec.name}", started,
                 datetime.now(timezone.utc).isoformat(), rec["n_trials"],
                 rec["best_ic"], "val_mean_ic", json.dumps(rec),
                 str(ZOO_CKPT_ROOT / spec.name / "tuned_params.json"),
                 f"optuna: swapped={rec['swapped']} baseline_ic="
                 f"{rec['baseline_ic']} best_ic={rec['best_ic']}"))
            c.commit()
        print(f"{spec.name}: baseline_ic={rec['baseline_ic']} "
              f"best_ic={rec['best_ic']} swapped={rec['swapped']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
