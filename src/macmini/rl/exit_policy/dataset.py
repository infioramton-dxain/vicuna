""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.rl.exit_policy.obs_spec import CACHE_DIR, OBS_FIELDS, OBS_VERSION

log = logging.getLogger(__name__)


def load_arrays(cache_dir: Path | None = None,
                entry_before: str | None = None,
                entry_from: str | None = None,
                variants: tuple[str, ...] | None = None,
                limit_files: int | None = None) -> dict:
    ""
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    files = sorted(cache_dir.glob("sid*_*.parquet"))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"no episode parquets under {cache_dir}")

    obs_list, act_list, rew_list, term_list = [], [], [], []
    n_episodes = 0
    for fp in files:
        df = pd.read_parquet(fp)
        if entry_before is not None:
            df = df[df["entry_date"] < pd.Timestamp(entry_before)]
        if entry_from is not None:
            df = df[df["entry_date"] >= pd.Timestamp(entry_from)]
        if variants is not None:
            df = df[df["variant"].isin(variants)]
        if df.empty:
            continue
        obs_list.append(df[OBS_FIELDS].to_numpy(np.float32))
        act_list.append(df["action"].to_numpy(np.int64))
        rew_list.append(df["reward"].to_numpy(np.float32))
        term_list.append(df["terminal"].to_numpy(bool))
        n_episodes += int(df["terminal"].sum())

    if not obs_list:
        raise ValueError("episode filter left nothing to train on")
    out = {
        "observations": np.concatenate(obs_list),
        "actions": np.concatenate(act_list),
        "rewards": np.concatenate(rew_list),
        "terminals": np.concatenate(term_list),
        "n_episodes": n_episodes,
        "n_files": len(files),
        "obs_version": OBS_VERSION,
    }
    log.info("loaded %d episodes / %d steps from %d files",
             n_episodes, len(out["actions"]), len(files))
    return out


def to_mdp_dataset(arrays: dict):
    ""
    import d3rlpy

    return d3rlpy.dataset.MDPDataset(
        observations=arrays["observations"],
        actions=arrays["actions"],
        rewards=arrays["rewards"],
        terminals=arrays["terminals"],
        action_space=d3rlpy.constants.ActionSpace.DISCRETE,
    )
