""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.dl.device import require_cuda
from macmini.rl.exit_policy import dataset as ds
from macmini.rl.exit_policy.obs_spec import CKPT_ROOT, OBS_FIELDS, OBS_VERSION

log = logging.getLogger(__name__)

ALGOS = ("cql", "bcq")
DEFAULT_TRAIN_END = "2026-01-01"
DEFAULT_STEPS = 200_000
SMOKE_STEPS = 2_000


def _make_algo(name: str, device: str):
    import d3rlpy

    if name == "cql":
        cfg = d3rlpy.algos.DiscreteCQLConfig()
    elif name == "bcq":
        cfg = d3rlpy.algos.DiscreteBCQConfig()
    else:
        raise ValueError(f"unknown algo {name!r}")
    return cfg.create(device=device)


def _write_run_row(db_path, model_id: str, algo: str, meta: dict) -> None:
    with connect(db_path) as c:
        c.execute(
            """INSERT OR REPLACE INTO mm_dl_runs
               (run_id, model_type, started_at, ended_at,
                epochs_or_timesteps, best_val_metric, best_metric_name,
                config_json, checkpoint_path, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, f"rl_exit_{algo}", meta["started_at"], meta["ended_at"],
             meta["n_steps"], meta.get("final_loss"), "td_loss",
             json.dumps(meta["config"]), meta["checkpoint"],
             "Phase C2 (D42/D47/D48); annotation-only"),
        )
        c.commit()


def train_one(algo_name: str, arrays: dict, out_root: Path, db_path,
              n_steps: int, smoke: bool) -> dict:
    import d3rlpy

    require_cuda("rl exit-policy training")
    dataset = ds.to_mdp_dataset(arrays)
    algo = _make_algo(algo_name, "cuda:0")

    started = datetime.now(timezone.utc).isoformat()
    n_steps_per_epoch = max(1000, min(10_000, n_steps // 10))
    history = algo.fit(dataset, n_steps=n_steps,
                       n_steps_per_epoch=n_steps_per_epoch,
                       logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
                       show_progress=False)
    ended = datetime.now(timezone.utc).isoformat()

    final_metrics = history[-1][1] if history else {}
    final_loss = float(final_metrics.get("loss", float("nan")))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    model_id = f"rl_exit_{algo_name}_{stamp}" + ("_smoke" if smoke else "")
    out_dir = out_root / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_id}.d3"
    algo.save(str(ckpt))

    deployed = {
        "model_id": model_id,
        "family": "d3rlpy",
        "algo": algo_name,
        "checkpoint": str(ckpt),
        "obs_version": OBS_VERSION,
        "obs_dim": len(OBS_FIELDS),
        "n_actions": 3,
        "train_end": None,
        "n_episodes": arrays["n_episodes"],
        "n_steps_trained": n_steps,
        "final_loss": final_loss,
        "smoke": smoke,
        "trained_at": ended,
    }
    meta = {"started_at": started, "ended_at": ended, "n_steps": n_steps,
            "final_loss": final_loss, "checkpoint": str(ckpt),
            "config": {"algo": algo_name, "n_episodes": arrays["n_episodes"],
                       "n_transitions": int(len(arrays["actions"])),
                       "obs_version": OBS_VERSION, "smoke": smoke}}
    return {"deployed": deployed, "meta": meta, "model_id": model_id}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=ALGOS + ("all",), default="all")
    p.add_argument("--db", default=None)
    p.add_argument("--cache", default=None, help="episode cache dir override")
    p.add_argument("--out", default=None, help="checkpoint root override")
    p.add_argument("--train-end", default=DEFAULT_TRAIN_END,
                   help="episodes entered on/after this date are excluded")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--smoke", action="store_true",
                   help="few steps, limited files, checkpoint marked smoke")
    args = p.parse_args(argv)

    require_cuda("rl exit-policy training")
    db_path = resolve_db(args.db)
    out_root = Path(args.out) if args.out else CKPT_ROOT
    n_steps = SMOKE_STEPS if args.smoke else args.steps

    arrays = ds.load_arrays(
        cache_dir=args.cache,
        entry_before=args.train_end,
        limit_files=200 if args.smoke else None,
    )
    log.info("training set: %d episodes / %d transitions (entry < %s)",
             arrays["n_episodes"], len(arrays["actions"]), args.train_end)

    names = ALGOS if args.algo == "all" else (args.algo,)
    results = {}
    for name in names:
        log.info("=== training %s ===", name)
        res = train_one(name, arrays, out_root, db_path, n_steps, args.smoke)
        res["deployed"]["train_end"] = args.train_end
        (out_root / name / "deployed.json").write_text(
            json.dumps(res["deployed"], indent=2))
        _write_run_row(db_path, res["model_id"], name, res["meta"])
        results[name] = res["model_id"]
        log.info("%s done -> %s", name, res["deployed"]["checkpoint"])

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
