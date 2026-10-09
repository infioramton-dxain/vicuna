""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.rl.exit_policy.obs_spec import (
    ACTIONS, ARCHETYPES, CKPT_ROOT, MAX_EPISODE_BARS, OBS_FIELDS,
    OBS_VERSION, SECTORS,
)
from macmini.strategies._indicators import atr

log = logging.getLogger(__name__)

ACTION_NAMES = {v: k for k, v in ACTIONS.items()}
STOP_ATR_MULT, STOP_FRAC_MIN, STOP_FRAC_MAX, TARGET_R_MULT = 3.0, 0.05, 0.15, 2.0


def _open_recs(db_path) -> list[dict]:
    with connect(db_path) as c:
        cur = c.execute(
            """SELECT rec_id, symbol, strategy_id, entry_date,
                      actual_fill_price, stop_price, target_price, exit_status
               FROM mm_recs_ledger
               WHERE fill_status='filled'
                 AND (exit_status IS NULL
                      OR exit_status IN ('open', 'manual_override'))""")
        cols = [d[0] for d in cur.description]
        recs = [dict(zip(cols, r)) for r in cur]
        held = {r[0] for r in c.execute("SELECT DISTINCT symbol FROM mm_alpaca_positions")}
    for r in recs:
        r["broker_position"] = r["symbol"] in held
    return recs


def _bars(db_path, symbol: str, upto: str) -> pd.DataFrame:
    with connect(db_path) as c:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close FROM mm_bars "
            "WHERE symbol=? AND date<=? ORDER BY date",
            c, params=[symbol, upto + "T23:59:59"], parse_dates=["date"])
    return df.set_index("date")


def _vix_regime(db_path, upto: str) -> str:
    with connect(db_path) as c:
        row = c.execute(
            "SELECT close FROM mm_bars WHERE symbol='^VIX' AND date<=? "
            "ORDER BY date DESC LIMIT 1", (upto + "T23:59:59",)).fetchone()
    if not row or row[0] is None:
        return "normal"
    return "low" if row[0] < 15 else ("high" if row[0] > 25 else "normal")


def _sector(db_path, symbol: str) -> str:
    with connect(db_path) as c:
        row = c.execute("SELECT sector FROM mm_sector_map WHERE symbol=?",
                        (symbol,)).fetchone()
    return (row[0] if row else "Other") or "Other"


def _archetype(strategy_id: int) -> str:
    from macmini.strategies import STRATEGY_REGISTRY
    cls = STRATEGY_REGISTRY.get(strategy_id)
    return getattr(cls, "category", "trend") if cls else "trend"


def build_observation(db_path, rec: dict, as_of: str) -> tuple[np.ndarray, dict] | None:
    bars = _bars(db_path, rec["symbol"], as_of)
    if bars.empty or len(bars) < 30:
        return None
    close = bars["close"]
    px = float(close.iloc[-1])
    entry = rec["actual_fill_price"]
    if not entry:
        return None

    a14 = atr(bars["high"], bars["low"], close, 14)
    a_now = float(a14.iloc[-1]) if np.isfinite(a14.iloc[-1]) else None
    atr_frac = (a14 / close)
    atr_pct = atr_frac.rolling(252, min_periods=60).rank(pct=True).iloc[-1]

    stop = rec["stop_price"]
    target = rec["target_price"]
    if not stop or stop >= entry:
        frac0 = float(np.clip(STOP_ATR_MULT * (a_now or 0) / px,
                              STOP_FRAC_MIN, STOP_FRAC_MAX)) or STOP_FRAC_MIN
        stop = entry * (1 - frac0)
    frac0 = (entry - stop) / entry
    if not target:
        target = entry * (1 + TARGET_R_MULT * frac0)

    entry_ts = pd.Timestamp(str(rec["entry_date"])[:10])
    days_held = int((bars.index >= entry_ts).sum())

    obs = np.zeros(len(OBS_FIELDS), dtype=np.float32)
    obs[0] = (px / entry - 1.0) / frac0
    obs[1] = min(days_held, MAX_EPISODE_BARS) / MAX_EPISODE_BARS
    obs[2] = (px - stop) / a_now if a_now else 0.0
    obs[3] = (target - px) / a_now if a_now else 0.0
    for i, n in ((4, 5), (5, 10), (6, 20)):
        r = close.pct_change(n).iloc[-1]
        obs[i] = r if np.isfinite(r) else 0.0
    obs[7] = atr_pct if np.isfinite(atr_pct) else 0.5
    regime = _vix_regime(db_path, as_of)
    obs[8 + ["low", "normal", "high"].index(regime)] = 1.0
    sector = _sector(db_path, rec["symbol"])
    sec_i = SECTORS.index(sector) if sector in SECTORS else SECTORS.index("Other")
    obs[11 + sec_i] = 1.0
    arch = _archetype(rec["strategy_id"])
    if arch in ARCHETYPES:
        obs[11 + len(SECTORS) + ARCHETYPES.index(arch)] = 1.0
    ctx = {"price": px, "entry": entry, "stop": stop, "target": target,
           "days_held": days_held, "vix_regime": regime,
           "broker_position": rec["broker_position"]}
    return obs, ctx


def _load_policies(ckpt_root: Path, algos: tuple[str, ...], allow_smoke: bool):
    import d3rlpy

    out = []
    for name in algos:
        dj = ckpt_root / name / "deployed.json"
        if not dj.exists():
            log.warning("no deployed.json for %s — skipping", name)
            continue
        meta = json.loads(dj.read_text())
        if meta.get("smoke") and not allow_smoke:
            log.warning("%s checkpoint is smoke=true — skipping", name)
            continue
        if meta.get("obs_version") != OBS_VERSION:
            raise RuntimeError(
                f"{name}: checkpoint obs_version {meta.get('obs_version')} "
                f"!= code {OBS_VERSION} — retrain before inferring")
        algo = d3rlpy.load_learnable(meta["checkpoint"], device="cuda:0")
        out.append((meta["model_id"], algo))
    return out


def run(db_path=None, ckpt_root: Path | None = None, algos=("cql", "bcq"),
        as_of: str | None = None, allow_smoke: bool = False) -> dict:
    from macmini.dl.device import require_cuda

    require_cuda("rl exit-policy inference")
    db_path = resolve_db(db_path)
    ckpt_root = Path(ckpt_root) if ckpt_root else CKPT_ROOT
    as_of = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    policies = _load_policies(ckpt_root, tuple(algos), allow_smoke)
    if not policies:
        log.warning("no deployable policies found under %s", ckpt_root)
        return {"annotated": 0, "policies": 0}

    recs = _open_recs(db_path)
    log.info("annotating %d open recs as of %s with %d policies",
             len(recs), as_of, len(policies))
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    for rec in recs:
        built = build_observation(db_path, rec, as_of)
        if built is None:
            log.warning("%s: no observation (missing bars/fill) — skipped",
                        rec["symbol"])
            continue
        obs, ctx = built
        batch = obs[None, :]
        for model_id, algo in policies:
            action = int(algo.predict(batch)[0])
            q = [float(algo.predict_value(batch, np.array([a]))[0])
                 for a in range(3)]
            payload = {"action": action, "kind": ACTION_NAMES[action],
                       "q_values": q, "obs_version": OBS_VERSION,
                       "rec_id": rec["rec_id"], **ctx}
            with connect(db_path) as c:
                c.execute(
                    """INSERT OR REPLACE INTO mm_rl_signals
                       (policy_id, symbol, date, action,
                        suggested_size_pct, policy_logits_json, created_at)
                       VALUES (?, ?, ?, ?, NULL, ?, ?)""",
                    (model_id, rec["symbol"], as_of, action,
                     json.dumps(payload), now))
                c.commit()
            written += 1
            log.info("  %s %s -> %s (q=%s)", model_id, rec["symbol"],
                     ACTION_NAMES[action], [round(x, 4) for x in q])
    return {"annotated": written, "policies": len(policies),
            "open_recs": len(recs), "as_of": as_of}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None)
    p.add_argument("--ckpt-root", default=None)
    p.add_argument("--algo", choices=("cql", "bcq", "all"), default="all")
    p.add_argument("--date", default=None, help="as-of date (default today)")
    p.add_argument("--allow-smoke", action="store_true")
    args = p.parse_args(argv)

    algos = ("cql", "bcq") if args.algo == "all" else (args.algo,)
    out = run(db_path=args.db, ckpt_root=args.ckpt_root, algos=algos,
              as_of=args.date, allow_smoke=args.allow_smoke)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
