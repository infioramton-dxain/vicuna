""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.rl.exit_policy.obs_spec import (
    ACTIONS, CACHE_DIR, CKPT_ROOT, OBS_FIELDS, OBS_VERSION,
)

log = logging.getLogger(__name__)


def _episodes(cache_dir: Path, oos_start: str):
    ""
    for fp in sorted(Path(cache_dir).glob("sid*_*.parquet")):
        df = pd.read_parquet(fp)
        df = df[df["entry_date"] >= pd.Timestamp(oos_start)]
        if df.empty:
            continue
        starts = [0] + list(np.flatnonzero(df["terminal"].to_numpy()) + 1)
        for s, e in zip(starts[:-1], starts[1:]):
            ep = df.iloc[s:e]
            yield (ep["variant"].iloc[0],
                   ep[OBS_FIELDS].to_numpy(np.float32),
                   ep["reward"].to_numpy(np.float64))


def evaluate(algo, cache_dir: Path, oos_start: str) -> dict:
    per_variant: dict[str, list[float]] = {}
    n_ep = 0
    exit_step_fracs = []
    for variant, obs, rewards in _episodes(cache_dir, oos_start):
        n_ep += 1
        actions = algo.predict(obs)
        exit_idx = np.flatnonzero(actions == ACTIONS["EXIT"])
        behavior_ret = float(rewards.sum())
        if len(exit_idx) == 0 or exit_idx[0] == len(rewards) - 1:
            policy_ret = behavior_ret
        else:
            k = int(exit_idx[0])
            policy_ret = float(rewards[:k].sum())
            exit_step_fracs.append(k / len(rewards))
        per_variant.setdefault(variant, []).append(policy_ret - behavior_ret)

    out = {"n_episodes": n_ep, "oos_start": oos_start,
           "obs_version": OBS_VERSION, "variants": {}}
    all_uplifts: list[float] = []
    for variant, ups in sorted(per_variant.items()):
        arr = np.asarray(ups)
        t = float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))) \
            if len(arr) > 2 and arr.std(ddof=1) > 0 else 0.0
        out["variants"][variant] = {
            "n": len(arr),
            "mean_uplift_logret": float(arr.mean()),
            "t_stat": t,
            "pct_changed": float((arr != 0).mean()),
        }
        all_uplifts.extend(ups)
    if all_uplifts:
        arr = np.asarray(all_uplifts)
        out["overall_mean_uplift"] = float(arr.mean())
        out["overall_t"] = float(
            arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))) \
            if arr.std(ddof=1) > 0 else 0.0
    out["mean_exit_step_frac"] = (
        float(np.mean(exit_step_fracs)) if exit_step_fracs else None)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=None)
    p.add_argument("--ckpt-root", default=None)
    p.add_argument("--algo", choices=("cql", "bcq"), required=True)
    p.add_argument("--oos-start", default="2026-01-01")
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--out", default=None, help="write JSON report here")
    args = p.parse_args(argv)

    import d3rlpy

    from macmini.dl.device import require_cuda
    require_cuda("rl exit-policy eval")

    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else CKPT_ROOT
    meta = json.loads((ckpt_root / args.algo / "deployed.json").read_text())
    if meta.get("smoke") and not args.allow_smoke:
        raise SystemExit(f"{args.algo} checkpoint is smoke=true; pass --allow-smoke")
    if meta.get("obs_version") != OBS_VERSION:
        raise SystemExit("obs_version mismatch — retrain first")
    algo = d3rlpy.load_learnable(meta["checkpoint"], device="cuda:0")

    report = evaluate(algo, Path(args.cache) if args.cache else CACHE_DIR,
                      args.oos_start)
    report["model_id"] = meta["model_id"]
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
