""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.rl.env import ACTION_DIM, MacminiSwingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def _mask_fn(env):
    ""
    while hasattr(env, "env"):
        env = env.env
    return env.action_masks()


def make_env(
    db_path: Path, start_date: str, end_date: str, seed: int, masked: bool,
) -> callable:
    def _thunk():
        env = MacminiSwingEnv(
            db_path=db_path,
            start_date=start_date, end_date=end_date, seed=seed,
        )
        if masked:
            env = ActionMasker(env, _mask_fn)
        return Monitor(env)
    return _thunk


def evaluate_random(
    db_path: Path, start_date: str, end_date: str, n_episodes: int,
) -> dict:
    env = MacminiSwingEnv(
        db_path=db_path, start_date=start_date, end_date=end_date, seed=0,
    )
    rng = np.random.default_rng(0)
    rewards, finals = [], []
    for ep in range(n_episodes):
        env.reset(seed=ep)
        ep_r = 0.0; terminated = truncated = False
        last = 100_000.0
        while not (terminated or truncated):
            a = int(rng.integers(0, ACTION_DIM))
            _, r, terminated, truncated, info = env.step(a)
            ep_r += r; last = info["port_value"]
        rewards.append(ep_r); finals.append(last)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "final_min": float(np.min(finals)),
        "final_max": float(np.max(finals)),
        "n_episodes": n_episodes,
    }


def evaluate_policy(
    model, db_path: Path, start_date: str, end_date: str,
    n_episodes: int, masked: bool,
) -> dict:
    env = MacminiSwingEnv(
        db_path=db_path, start_date=start_date, end_date=end_date, seed=1000,
    )
    rewards, finals = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_r = 0.0; terminated = truncated = False
        last = 100_000.0
        while not (terminated or truncated):
            if masked:
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            ep_r += r; last = info["port_value"]
        rewards.append(ep_r); finals.append(last)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "final_min": float(np.min(finals)),
        "final_max": float(np.max(finals)),
        "n_episodes": n_episodes,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(REPO_ROOT / "macmini.db"))
    p.add_argument("--tag", default="baseline_v1")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--eval-start", default="2025-01-01")
    p.add_argument("--eval-end", default="2025-06-30")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", action="store_true",
                   help="Ignored — GPU is mandatory; kept so old invocations don't break")
    p.add_argument("--masked", action="store_true",
                   help="Use MaskablePPO (sb3-contrib) with env.action_masks()")
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    ckpt_dir = REPO_ROOT / "data/checkpoints/ppo" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = ckpt_dir / "tb"

    from macmini.dl.device import require_cuda
    device = require_cuda("PPO training")
    print(f"device={device}  n_envs={args.n_envs}  total_timesteps={args.total_timesteps:,}",
          flush=True)

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env_fns = [make_env(db_path, args.train_start, args.train_end, args.seed + i, args.masked)
               for i in range(args.n_envs)]
    vec_env = VecCls(env_fns)

    print(f"training window: {args.train_start} .. {args.train_end}", flush=True)
    print(f"eval window:     {args.eval_start} .. {args.eval_end}", flush=True)

    print("random-policy baseline (eval window)…", flush=True)
    rand_stats = evaluate_random(db_path, args.eval_start, args.eval_end,
                                 args.eval_episodes)
    print(f"  random: mean_reward={rand_stats['reward_mean']:+.4f}±{rand_stats['reward_std']:.4f}  "
          f"final=${rand_stats['final_mean']:,.0f}±${rand_stats['final_std']:,.0f}  "
          f"[{rand_stats['final_min']:,.0f} .. {rand_stats['final_max']:,.0f}]",
          flush=True)

    ModelCls = MaskablePPO if args.masked else PPO
    print(f"initializing {ModelCls.__name__}…", flush=True)
    model = ModelCls(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(tb_dir),
        learning_rate=args.lr,
        device=device,
        seed=args.seed,
    )
    print(f"  policy params: {sum(p.numel() for p in model.policy.parameters()):,}", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s "
          f"({args.total_timesteps / elapsed:,.0f} steps/s)", flush=True)

    ckpt_path = ckpt_dir / "final.zip"
    model.save(ckpt_path)
    print(f"saved: {ckpt_path}", flush=True)

    print("trained-policy eval (deterministic)…", flush=True)
    ppo_stats = evaluate_policy(model, db_path, args.eval_start,
                                args.eval_end, args.eval_episodes, args.masked)
    print(f"  PPO:    mean_reward={ppo_stats['reward_mean']:+.4f}±{ppo_stats['reward_std']:.4f}  "
          f"final=${ppo_stats['final_mean']:,.0f}±${ppo_stats['final_std']:,.0f}  "
          f"[{ppo_stats['final_min']:,.0f} .. {ppo_stats['final_max']:,.0f}]",
          flush=True)

    delta_reward = ppo_stats["reward_mean"] - rand_stats["reward_mean"]
    delta_final = ppo_stats["final_mean"] - rand_stats["final_mean"]
    print()
    print(f"delta vs random: reward={delta_reward:+.4f}   final=${delta_final:+,.0f}", flush=True)
    vec_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
