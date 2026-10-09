""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from macmini.rl.exit_policy.obs_spec import (
    ACTIONS, BEHAVIOR_VARIANTS, CACHE_DIR, OBS_VERSION,
)

COLS = ["variant", "action", "reward", "terminal", "entry_date", "strategy_id"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(CACHE_DIR))
    p.add_argument("--json-out", default="exports/rl_exit_cache_stats.json")
    args = p.parse_args()

    cache = Path(args.cache)
    files = sorted(cache.glob("sid*_*.parquet"))
    if not files:
        print(f"no parquet files under {cache}", file=sys.stderr)
        return 1

    dset = pads.dataset([str(f) for f in files])
    tbl = dset.to_table(columns=COLS)
    df = tbl.to_pandas()

    ep_key = df["terminal"].cumsum().shift(1, fill_value=0)
    ep_returns = df.groupby([df["strategy_id"], df["variant"], ep_key])["reward"].sum()

    per_variant = {}
    for v in BEHAVIOR_VARIANTS:
        sub = df[df["variant"] == v]
        eps = int(sub["terminal"].sum())
        r = ep_returns.xs(v, level=1)
        per_variant[v] = {
            "episodes": eps,
            "steps": int(len(sub)),
            "mean_episode_logret": float(r.mean()),
            "std_episode_logret": float(r.std()),
            "p05": float(r.quantile(0.05)), "p50": float(r.quantile(0.50)),
            "p95": float(r.quantile(0.95)),
        }

    action_counts = {name: int((df["action"] == code).sum())
                     for name, code in ACTIONS.items()}
    per_strategy = (df[df["terminal"]].groupby("strategy_id").size()
                    .sort_index().to_dict())
    ep_len = df.groupby(ep_key).size()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "obs_version": OBS_VERSION,
        "n_pair_files": len(files),
        "n_strategies": int(df["strategy_id"].nunique()),
        "episodes_total": int(df["terminal"].sum()),
        "steps_total": int(len(df)),
        "entry_date_min": str(df["entry_date"].min())[:10],
        "entry_date_max": str(df["entry_date"].max())[:10],
        "mean_episode_len_bars": float(ep_len.mean()),
        "action_counts": action_counts,
        "action_share": {k: round(v / len(df), 4) for k, v in action_counts.items()},
        "per_variant": per_variant,
        "episodes_per_strategy": {str(k): int(v) for k, v in per_strategy.items()},
        "cache_bytes": int(sum(f.stat().st_size for f in files)),
    }
    path = Path(args.json_out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"cache: {len(files)} pair files, {out['episodes_total']:,} episodes, "
          f"{out['steps_total']:,} steps, entries {out['entry_date_min']} → "
          f"{out['entry_date_max']} → {path}")
    for v, s in per_variant.items():
        print(f"  {v:10s} eps={s['episodes']:>7,} mean_ret={s['mean_episode_logret']:+.4f} "
              f"p05={s['p05']:+.3f} p95={s['p95']:+.3f}")
    print(f"  actions: {out['action_share']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
