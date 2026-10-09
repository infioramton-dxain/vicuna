""
from __future__ import annotations

from pathlib import Path

from macmini.config import DB_PATH

CACHE_DIR = Path(DB_PATH).parent / "data" / "rl_exit_cache"
CKPT_ROOT = Path(DB_PATH).parent / "data" / "checkpoints" / "rl_exit"

ACTIONS = {"HOLD": 0, "EXIT": 1, "TIGHTEN_STOP": 2}
MAX_EPISODE_BARS = 40
BEHAVIOR_VARIANTS = ("bracket", "trail", "trail_tp", "time_stop")

SECTORS = ("Computers", "Consumer Staples", "Defense", "Materials",
           "Mining", "Other", "Software")
ARCHETYPES = ("breakout", "composite", "fundamental", "mean_reversion",
              "mtf", "pattern", "sentiment", "trend", "volatility")

OBS_FIELDS = [
    "unrealized_r",
    "days_held_frac",
    "dist_to_stop_atr",
    "dist_to_target_atr",
    "ret_5d", "ret_10d", "ret_20d",
    "atr_percentile",
    "vix_low", "vix_normal", "vix_high",
] + [f"sector_{s.lower().replace(' ', '_')}" for s in SECTORS] \
  + [f"arch_{a}" for a in ARCHETYPES]
OBS_VERSION = 1
