""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.rl.env import EPISODE_LEN, INIT_CASH, MacminiSwingEnv
from macmini.backtest.stop_counterfactual import regime_series


def _run_episode(env: MacminiSwingEnv, start_idx: int, actor) -> dict:
    env._reset_state()
    env.episode_start_idx = start_idx
    env.step_idx = 0
    obs = env._observation()
    total_reward, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        action = actor(obs, env.action_masks())
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
    pv = float(env.state.port_value_series[-1])
    peak = float(env.state.running_max)
    return {
        "reward": total_reward,
        "final_return": pv / INIT_CASH - 1.0,
        "max_dd": (min(env.state.port_value_series) / peak - 1.0) if peak else 0.0,
        "dd_terminated": bool(terminated),
    }


def _summarize(episodes: list[dict]) -> dict:
    if not episodes:
        return {"n": 0}
    return {
        "n": len(episodes),
        "mean_reward": round(float(np.mean([e["reward"] for e in episodes])), 4),
        "mean_return": round(float(np.mean([e["final_return"] for e in episodes])), 4),
        "worst_dd": round(float(min(e["max_dd"] for e in episodes)), 4),
        "dd_terminate_share": round(
            sum(e["dd_terminated"] for e in episodes) / len(episodes), 3),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--masked", action="store_true")
    p.add_argument("--db", default=REPO_ROOT / "macmini.db", type=Path)
    p.add_argument("--cache", default=REPO_ROOT / "data/rl_cache/universe_topk.parquet",
                   type=Path)
    p.add_argument("--episodes-per-regime", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json-out", default=REPO_ROOT / "exports/rl_3regime_latest.json",
                   type=Path)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    ModelCls = MaskablePPO if args.masked else PPO
    from macmini.dl.device import require_cuda
    model = ModelCls.load(args.checkpoint, device=require_cuda("RL 3-regime eval"))

    env = MacminiSwingEnv(db_path=args.db, universe_topk_parquet=args.cache,
                          seed=args.seed)
    labels = regime_series(args.db)
    max_start = len(env.trading_dates) - env.episode_len - 1
    by_regime: dict[str, list[int]] = {"bull": [], "bear": [], "chop": []}
    for idx, d in enumerate(env.trading_dates[: max_start + 1]):
        lab = labels.get(d)
        if lab in by_regime:
            by_regime[lab].append(idx)

    rng = np.random.default_rng(args.seed)

    def policy_actor(obs, masks):
        act, _ = (model.predict(obs, deterministic=True, action_masks=masks)
                  if args.masked else model.predict(obs, deterministic=True))
        return act

    def random_actor(_obs, masks):
        legal = np.flatnonzero(masks)
        return rng.choice(legal)

    result: dict = {
        "checkpoint": str(args.checkpoint),
        "episode_len": EPISODE_LEN,
        "episodes_per_regime": args.episodes_per_regime,
        "regimes": {},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    gate_pass = True
    for regime, starts in by_regime.items():
        if not starts:
            result["regimes"][regime] = {"n_start_dates": 0, "note": "no data in window"}
            gate_pass = False
            continue
        chosen = rng.choice(starts, size=min(args.episodes_per_regime, len(starts)),
                            replace=len(starts) < args.episodes_per_regime)
        pol = [_run_episode(env, int(s), policy_actor) for s in chosen]
        ran = [_run_episode(env, int(s), random_actor) for s in chosen]
        pol_s, ran_s = _summarize(pol), _summarize(ran)
        regime_pass = (
            pol_s["mean_reward"] >= ran_s["mean_reward"]
            and pol_s["mean_return"] >= ran_s["mean_return"]
            and pol_s["dd_terminate_share"] == 0.0
        )
        gate_pass = gate_pass and regime_pass
        result["regimes"][regime] = {
            "n_start_dates": len(starts), "policy": pol_s, "random": ran_s,
            "regime_pass": regime_pass,
        }
        print(f"{regime:>5}: policy {pol_s} | random {ran_s} | "
              f"{'PASS' if regime_pass else 'FAIL'}", flush=True)

    result["gate_pass"] = gate_pass
    args.json_out.parent.mkdir(exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"3-REGIME GATE: {'PASS' if gate_pass else 'FAIL'}  → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
