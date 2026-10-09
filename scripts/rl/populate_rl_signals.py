""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from macmini.rl.env import (
    ACTION_DIM, EPISODE_LEN, K_SLOTS, MacminiSwingEnv,
)
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO


def _decode_action(a: int) -> tuple[str, int | None]:
    ""
    if a == 0:
        return "HOLD", None
    if 1 <= a <= K_SLOTS:
        return "BUY", a - 1
    return "SELL", a - 1 - K_SLOTS


def _get_action_and_probs(model, obs: np.ndarray, masks: np.ndarray | None,
                          masked: bool) -> tuple[int, np.ndarray]:
    ""
    if masked:
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
    else:
        action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
        if masked:
            m = torch.as_tensor(masks.reshape(1, -1), device=obs_t.device)
            dist = model.policy.get_distribution(obs_t, action_masks=m)
        else:
            dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs[0].cpu().numpy().astype(np.float64)
    return action, probs


def _prepare_env(db_path: Path, cache_path: Path, universe_dates: list[str]) -> MacminiSwingEnv:
    ""
    start = min(universe_dates)
    end = max(universe_dates)
    env = MacminiSwingEnv(
        db_path=db_path,
        universe_topk_parquet=cache_path,
        start_date=start,
        end_date=end,
        seed=0,
    )
    return env


def _force_date(env: MacminiSwingEnv, date_str: str) -> bool:
    ""
    env._reset_state()
    if date_str not in env.trading_dates:
        return False
    idx = env.trading_dates.index(date_str)
    max_start = len(env.trading_dates) - env.episode_len - 1
    if idx > max_start:
        idx = min(idx, len(env.trading_dates) - 1)
    env.episode_start_idx = idx
    env.step_idx = 0
    return True


def _ledger_portfolio(
    db_path: Path, d: str, trading_dates: list[str],
) -> tuple[float, dict[str, tuple[float, float, int]]]:
    ""
    from macmini.rl.env import INIT_CASH
    didx = {dt: i for i, dt in enumerate(trading_dates)}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT symbol, fill_qty, actual_fill_price, "
            "       substr(actual_fill_at, 1, 10), "
            "       exit_qty, exit_price, substr(exit_at, 1, 10) "
            "FROM mm_recs_ledger "
            "WHERE actual_fill_price IS NOT NULL AND fill_qty > 0 "
            "  AND substr(actual_fill_at, 1, 10) <= ?", (d,),
        ).fetchall()
    finally:
        conn.close()

    cash = INIT_CASH
    open_lots: dict[str, list[tuple[float, float, int]]] = {}
    for sym, qty, px, fill_d, xqty, xpx, exit_d in rows:
        cash -= qty * px
        if exit_d is not None and exit_d <= d and xpx is not None:
            cash += (xqty if xqty else qty) * xpx
            continue
        dh = max(didx.get(d, 0) - didx.get(fill_d, didx.get(d, 0)), 0)
        open_lots.setdefault(sym, []).append((float(qty), float(px), dh))

    positions: dict[str, tuple[float, float, int]] = {}
    for sym, lots in open_lots.items():
        shares = sum(q for q, _, _ in lots)
        avg_cost = sum(q * p for q, p, _ in lots) / shares
        days_held = max(dh for _, _, dh in lots)
        positions[sym] = (shares, avg_cost, days_held)
    return cash, positions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--policy-id", required=True,
                   help="policy_id column value, e.g. maskable_v2_novol_20260704")
    p.add_argument("--cache", default=REPO_ROOT / "data/rl_cache/universe_topk.parquet", type=Path)
    p.add_argument("--db", default=REPO_ROOT / "macmini.db", type=Path)
    p.add_argument("--backfill-days", type=int, default=60,
                   help="score the last N distinct dates in the universe cache")
    p.add_argument("--masked", action="store_true",
                   help="Load with MaskablePPO (must match training)")
    p.add_argument("--position-size-pct", type=float, default=0.05,
                   help="value written for suggested_size_pct on BUY rows")
    p.add_argument("--portfolio-from-ledger", action="store_true",
                   help="seed each date's observation with the real portfolio "
                        "reconstructed from mm_recs_ledger (audit H6); "
                        "default remains cold-start for reproducibility")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.cache.exists():
        raise SystemExit(f"universe cache not found: {args.cache}. "
                         f"Build it: scripts/rl/build_universe_cache.py")

    started_at = datetime.now(timezone.utc).isoformat()
    ModelCls = MaskablePPO if args.masked else PPO
    print(f"loading {ModelCls.__name__} from {args.checkpoint}", flush=True)
    from macmini.dl.device import require_cuda
    model = ModelCls.load(args.checkpoint, device=require_cuda("RL inference"))

    import pandas as pd
    uc = pd.read_parquet(args.cache)
    uc["date_str"] = uc["date"].astype(str).str.slice(0, 10)
    all_dates = sorted(uc["date_str"].unique().tolist())
    target_dates = all_dates[-args.backfill_days:]
    print(f"target dates: {target_dates[0]} .. {target_dates[-1]}  "
          f"(count={len(target_dates)})", flush=True)

    print(f"initializing env over full cache window …", flush=True)
    env = _prepare_env(args.db, args.cache, all_dates)
    print(f"env ready: {len(env.trading_dates)} trading dates in-window", flush=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    dates_done = 0
    dates_skipped = 0
    rows_total = 0
    action_summary = {"HOLD": 0, "BUY": 0, "SELL": 0}

    for d in target_dates:
        if not _force_date(env, d):
            dates_skipped += 1
            continue

        if args.portfolio_from_ledger:
            cash, positions = _ledger_portfolio(args.db, d, env.trading_dates)
            env.seed_portfolio(cash, positions)

        obs = env._observation()
        if args.masked:
            masks = env.action_masks()
        else:
            masks = None

        action, probs = _get_action_and_probs(model, obs, masks, args.masked)
        kind, slot = _decode_action(action)
        action_summary[kind] += 1

        buy_probs = probs[1 : K_SLOTS + 1].tolist()
        hold_prob = float(probs[0])
        picked_symbol = None
        if kind == "BUY" and slot is not None:
            uni = env._universe_for_date(d)
            if slot < len(uni):
                picked_symbol = uni[slot]

        logits_snapshot = {
            "action": int(action),
            "kind": kind,
            "slot": slot,
            "hold_prob": round(hold_prob, 6),
            "buy_probs": [round(x, 6) for x in buy_probs],
            "picked_symbol": picked_symbol,
        }

        uni = [s for s in env._universe_for_date(d) if s != "__PAD__"]
        rows = []
        for i, sym in enumerate(uni):
            per_symbol_action = 0
            per_symbol_size = 0.0
            if kind == "BUY" and slot is not None and i == slot:
                per_symbol_action = 1
                per_symbol_size = float(args.position_size_pct)
            elif kind == "SELL" and slot is not None and i == slot:
                per_symbol_action = 2
                per_symbol_size = 0.0
            rows.append((
                args.policy_id, sym, d, per_symbol_action, per_symbol_size,
                json.dumps(logits_snapshot), now_iso,
            ))

        if args.dry_run:
            rows_total += len(rows)
            dates_done += 1
            if dates_done % 10 == 0:
                print(f"  [dry] {dates_done}/{len(target_dates)} dates, "
                      f"rows so far: {rows_total}", flush=True)
            continue

        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute("PRAGMA journal_mode = WAL")
            c.executemany(
                "INSERT OR REPLACE INTO mm_rl_signals "
                "(policy_id, symbol, date, action, suggested_size_pct, "
                " policy_logits_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            c.commit()
        rows_total += len(rows)
        dates_done += 1
        if dates_done % 10 == 0:
            print(f"  {dates_done}/{len(target_dates)} dates, "
                  f"rows so far: {rows_total}", flush=True)

    elapsed = time.time() - t0
    ended_at = datetime.now(timezone.utc).isoformat()
    print(f"done: dates_done={dates_done}  dates_skipped={dates_skipped}  "
          f"rows={rows_total}  elapsed={elapsed:.1f}s", flush=True)
    print(f"action mix: HOLD={action_summary['HOLD']}  BUY={action_summary['BUY']}  "
          f"SELL={action_summary['SELL']}", flush=True)

    if not args.dry_run:
        run_id = f"infer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        model_type = "maskable_ppo_inference" if args.masked else "ppo_inference"
        cfg = {
            "checkpoint": str(args.checkpoint),
            "policy_id": args.policy_id,
            "backfill_days": args.backfill_days,
            "position_size_pct": args.position_size_pct,
            "masked": args.masked,
            "target_dates_min": target_dates[0],
            "target_dates_max": target_dates[-1],
            "action_mix": action_summary,
        }
        with sqlite3.connect(args.db, timeout=30) as c:
            c.execute("PRAGMA busy_timeout = 30000")
            c.execute(
                "INSERT INTO mm_dl_runs (run_id, model_type, started_at, ended_at, "
                "epochs_or_timesteps, best_val_metric, best_metric_name, config_json, "
                "checkpoint_path, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, model_type, started_at, ended_at,
                    None, None, None,
                    json.dumps(cfg), str(args.checkpoint),
                    f"populate_rl_signals: {rows_total} rows over "
                    f"{dates_done} dates, {elapsed:.1f}s",
                ),
            )
            c.commit()
        print(f"logged mm_dl_runs row: run_id={run_id}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
