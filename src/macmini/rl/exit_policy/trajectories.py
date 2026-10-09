""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.backtest import exec_variants
from macmini.backtest.engine import WalkForwardConfig, signals_to_arrays
from macmini.backtest.loader import load_aux, load_bars, slice_aux
from macmini.config import DB_PATH, SYMBOLS_CSV, resolve_db
from macmini.data._db import connect
from macmini.strategies import STRATEGY_REGISTRY
from macmini.strategies._indicators import atr

log = logging.getLogger(__name__)

from macmini.rl.exit_policy.obs_spec import (
    ACTIONS, ARCHETYPES, BEHAVIOR_VARIANTS, CACHE_DIR, MAX_EPISODE_BARS,
    OBS_FIELDS, OBS_VERSION, SECTORS,
)


@dataclass
class EpisodeBatch:
    ""

    strategy_id: int
    symbol: str
    n_episodes: int
    observations: "object"
    actions: "object"
    rewards: "object"
    terminals: "object"


def _collect_signals(strategy, symbol: str, bars: pd.DataFrame, aux: dict,
                     config: WalkForwardConfig) -> tuple[pd.Series, pd.Series]:
    ""
    n = len(bars)
    entries = pd.Series(False, index=bars.index)
    exits = pd.Series(False, index=bars.index)
    train, test, step = (config.train_window_days, config.test_window_days,
                         config.step_days)
    start = config.min_history_days
    while start + test <= n:
        test_end = start + test
        slice_bars = bars.iloc[:test_end]
        window_aux = slice_aux(aux, slice_bars.index[-1])
        try:
            signals = strategy.generate_signals(symbol, slice_bars, **window_aux)
        except Exception:
            signals = []
        test_window = bars.iloc[start:test_end]
        t0, t1 = test_window.index[0], test_window.index[-1]
        w_entries, w_exits = signals_to_arrays(
            [s for s in signals
             if t0 <= pd.Timestamp(s.date).normalize() <= t1],
            test_window,
        )
        entries.loc[w_entries.index[w_entries]] = True
        exits.loc[w_exits.index[w_exits]] = True
        start += step
    return entries, exits


def _vix_onehot_series(bars_index: pd.DatetimeIndex, db_path: Path) -> pd.DataFrame:
    ""
    with connect(db_path) as c:
        vix = pd.read_sql_query(
            "SELECT date, close FROM mm_bars WHERE symbol='^VIX' ORDER BY date",
            c, parse_dates=["date"],
        )
    out = pd.DataFrame(0.0, index=bars_index,
                       columns=["vix_low", "vix_normal", "vix_high"])
    if vix.empty:
        out["vix_normal"] = 1.0
        return out
    aligned = pd.merge_asof(
        pd.DataFrame({"date": bars_index}), vix, on="date",
    )["close"]
    lo = (aligned < 15).to_numpy()
    hi = (aligned > 25).to_numpy()
    out["vix_low"] = lo.astype(float)
    out["vix_high"] = hi.astype(float)
    out["vix_normal"] = (~lo & ~hi | aligned.isna().to_numpy()).astype(float)
    return out


def _dividend_rate_series(bars_index: pd.DatetimeIndex, symbol: str,
                          db_path: Path) -> pd.Series:
    ""
    with connect(db_path) as c:
        divs = pd.read_sql_query(
            "SELECT ex_date, rate FROM mm_dividends WHERE symbol = ?",
            c, params=[symbol], parse_dates=["ex_date"])
    s = pd.Series(0.0, index=bars_index)
    if not divs.empty:
        hits = divs[divs["ex_date"].isin(bars_index)]
        for _, r in hits.iterrows():
            s.loc[r["ex_date"]] += float(r["rate"] or 0.0)
    return s


def _feature_panel(bars: pd.DataFrame, db_path: Path,
                   symbol: str | None = None) -> pd.DataFrame:
    close = bars["close"]
    a14 = atr(bars["high"], bars["low"], close, 14)
    feats = pd.DataFrame(index=bars.index)
    feats["atr14"] = a14
    feats["div_rate"] = (_dividend_rate_series(bars.index, symbol, db_path)
                         if symbol else 0.0)
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_10d"] = close.pct_change(10)
    feats["ret_20d"] = close.pct_change(20)
    atr_frac = a14 / close
    feats["atr_percentile"] = (
        atr_frac.rolling(252, min_periods=60)
        .rank(pct=True)
    )
    vix = _vix_onehot_series(bars.index, db_path)
    return pd.concat([feats, vix], axis=1)


def _static_onehots(sector: str, archetype: str) -> np.ndarray:
    sec = np.zeros(len(SECTORS))
    if sector in SECTORS:
        sec[SECTORS.index(sector)] = 1.0
    else:
        sec[SECTORS.index("Other")] = 1.0
    arch = np.zeros(len(ARCHETYPES))
    if archetype in ARCHETYPES:
        arch[ARCHETYPES.index(archetype)] = 1.0
    return np.concatenate([sec, arch])


def _roll_episodes(bars: pd.DataFrame, entries: pd.Series, exits: pd.Series,
                   variant: str, feats: pd.DataFrame, static_oh: np.ndarray,
                   cost_per_side: float, stats: dict) -> list[dict]:
    ""
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    dates = bars.index
    frac_series = exec_variants.stop_frac_series(bars).to_numpy(float)
    ent = entries.to_numpy(bool)
    sig_exit = exits.to_numpy(bool)
    atr14 = feats["atr14"].to_numpy(float)
    n = len(bars)

    rows: list[dict] = []
    entry_cost = float(np.log1p(-cost_per_side))
    exit_cost = float(np.log1p(-cost_per_side))

    k = 0
    while k < n - 1:
        if not ent[k]:
            k += 1
            continue
        f = k + 1
        entry_px = opens[f]
        frac0 = frac_series[f]
        stop0 = entry_px * (1.0 - frac0)
        target = entry_px * (1.0 + exec_variants.TARGET_R_MULT * frac0)
        has_stop = variant in ("bracket", "trail", "trail_tp")
        has_target = variant in ("bracket", "trail_tp")
        trailing = variant in ("trail", "trail_tp")

        ep_rows: list[dict] = []
        ep_id = f"{dates[f].date()}_{variant}"
        stop_level = stop0
        hwm = entry_px
        terminal_bar = None
        exit_px = None

        j = f
        while j < n:
            if has_stop:
                lvl = stop_level
                if opens[j] <= lvl:
                    terminal_bar, exit_px = j, opens[j]
                    break
                if lows[j] <= lvl:
                    terminal_bar, exit_px = j, lvl
                    break
            if has_target:
                if opens[j] >= target:
                    terminal_bar, exit_px = j, opens[j]
                    break
                if highs[j] >= target:
                    terminal_bar, exit_px = j, target
                    break
            ratcheted = False
            if trailing and highs[j] > hwm:
                hwm = highs[j]
                new_stop = hwm * (1.0 - frac0)
                if new_stop > stop_level:
                    stop_level = new_stop
                    ratcheted = True

            held = j - f + 1
            if sig_exit[j]:
                action = ACTIONS["EXIT"]
            elif variant == "time_stop" and held >= exec_variants.TIME_STOP_HORIZON:
                action = ACTIONS["EXIT"]
            elif held >= MAX_EPISODE_BARS:
                action = ACTIONS["EXIT"]
            elif ratcheted:
                action = ACTIONS["TIGHTEN_STOP"]
            else:
                action = ACTIONS["HOLD"]

            obs = np.empty(len(OBS_FIELDS))
            obs[0] = (closes[j] / entry_px - 1.0) / frac0
            obs[1] = held / MAX_EPISODE_BARS
            a = atr14[j] if np.isfinite(atr14[j]) and atr14[j] > 0 else np.nan
            obs[2] = (closes[j] - stop_level) / a if np.isfinite(a) else 0.0
            obs[3] = (target - closes[j]) / a if np.isfinite(a) else 0.0
            fr = feats.iloc[j]
            obs[4] = fr["ret_5d"] if np.isfinite(fr["ret_5d"]) else 0.0
            obs[5] = fr["ret_10d"] if np.isfinite(fr["ret_10d"]) else 0.0
            obs[6] = fr["ret_20d"] if np.isfinite(fr["ret_20d"]) else 0.0
            obs[7] = fr["atr_percentile"] if np.isfinite(fr["atr_percentile"]) else 0.5
            obs[8:11] = fr[["vix_low", "vix_normal", "vix_high"]].to_numpy(float)
            obs[11:] = static_oh

            ep_rows.append({
                "episode_id": ep_id, "variant": variant,
                "entry_date": dates[f], "date": dates[j],
                "obs": obs, "action": action,
                "reward": 0.0, "terminal": False,
            })

            if action == ACTIONS["EXIT"]:
                if j + 1 < n:
                    terminal_bar, exit_px = j + 1, opens[j + 1]
                else:
                    terminal_bar, exit_px = j, closes[j]
                break
            j += 1

        if terminal_bar is None:
            terminal_bar, exit_px = n - 1, closes[n - 1]

        if not ep_rows:
            stats["same_bar_exits"] += 1
            k = terminal_bar + 1
            continue

        bar_ix = [dates.get_loc(r["date"]) for r in ep_rows]
        pxs = [closes[i] for i in bar_ix]
        div_by_bar = feats["div_rate"].to_numpy(float)
        for i, r in enumerate(ep_rows[:-1]):
            div_next = div_by_bar[bar_ix[i + 1]]
            r["reward"] = float(np.log((pxs[i + 1] + div_next) / pxs[i]))
        last = ep_rows[-1]
        div_term = div_by_bar[terminal_bar] if terminal_bar > bar_ix[-1] else 0.0
        last["reward"] = float(np.log((exit_px + div_term) / pxs[-1]))
        ep_rows[0]["reward"] += float(np.log(pxs[0] / entry_px)) + entry_cost
        last["reward"] += exit_cost
        last["terminal"] = True
        rows.extend(ep_rows)
        stats["episodes"] += 1
        stats[f"episodes_{variant}"] = stats.get(f"episodes_{variant}", 0) + 1
        stats["episode_return_sum"] += float(np.log(exit_px / entry_px))
        k = terminal_bar + 1

    return rows


def _sector_of(symbol: str, db_path: Path) -> str:
    with connect(db_path) as c:
        row = c.execute("SELECT sector FROM mm_sector_map WHERE symbol = ?",
                        (symbol,)).fetchone()
    return (row[0] if row else "Other") or "Other"


def _empty_agg(start: str, end: str) -> dict:
    return {"pairs": 0, "pairs_skipped": 0, "pairs_existing": 0,
            "episodes": 0, "steps": 0, "same_bar_exits": 0,
            "episode_return_sum": 0.0, "obs_version": OBS_VERSION,
            "start": start, "end": end,
            "action_counts": {"HOLD": 0, "EXIT": 0, "TIGHTEN_STOP": 0},
            "episodes_per_strategy": {}}


def _merge_agg(agg: dict, part: dict) -> None:
    for k in ("pairs", "pairs_skipped", "pairs_existing", "episodes",
              "steps", "same_bar_exits", "episode_return_sum"):
        agg[k] += part[k]
    for name, n in part["action_counts"].items():
        agg["action_counts"][name] += n
    for variant in BEHAVIOR_VARIANTS:
        key = f"episodes_{variant}"
        agg[key] = agg.get(key, 0) + part.get(key, 0)
    for sid, n in part["episodes_per_strategy"].items():
        agg["episodes_per_strategy"][sid] = (
            agg["episodes_per_strategy"].get(sid, 0) + n)


def _process_symbol(symbol: str, sids: list[int], db_path_str: str,
                    out_dir_str: str, start: str, end: str,
                    skip_existing: bool) -> dict:
    ""
    db_path, out_dir = Path(db_path_str), Path(out_dir_str)
    config = WalkForwardConfig()
    cost_per_side = (config.transaction_cost_bps + config.slippage_bps) / 1e4
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    agg = _empty_agg(start, end)

    bars = load_bars(symbol, db_path=db_path)
    if bars.empty or len(bars) < config.min_history_days:
        agg["pairs_skipped"] += len(sids)
        return agg
    feats = _feature_panel(bars, db_path, symbol=symbol)
    sector = _sector_of(symbol, db_path)
    for sid in sids:
        cls = STRATEGY_REGISTRY.get(sid)
        if cls is None:
            continue
        if skip_existing and (out_dir / f"sid{sid}_{symbol}.parquet").exists():
            agg["pairs_existing"] += 1
            continue
        strategy = cls()
        aux = load_aux(symbol, cls, db_path=db_path)
        entries, exits = _collect_signals(strategy, symbol, bars, aux, config)
        in_range = entries.copy()
        fill_dates = bars.index[1:].append(bars.index[-1:])
        mask = (fill_dates >= lo) & (fill_dates <= hi)
        in_range &= pd.Series(mask, index=bars.index)
        if not in_range.any():
            agg["pairs_skipped"] += 1
            continue
        static_oh = _static_onehots(sector, cls.category)
        stats = {"episodes": 0, "same_bar_exits": 0,
                 "episode_return_sum": 0.0}
        rows: list[dict] = []
        for variant in BEHAVIOR_VARIANTS:
            rows.extend(_roll_episodes(
                bars, in_range, exits, variant, feats, static_oh,
                cost_per_side, stats))
        if not rows:
            agg["pairs_skipped"] += 1
            agg["same_bar_exits"] += stats["same_bar_exits"]
            continue
        obs_mat = np.stack([r.pop("obs") for r in rows])
        df = pd.DataFrame(rows)
        for i, fname in enumerate(OBS_FIELDS):
            df[fname] = obs_mat[:, i].astype(np.float32)
        df["strategy_id"] = sid
        df["symbol"] = symbol
        df["action"] = df["action"].astype(np.int8)
        df["reward"] = df["reward"].astype(np.float32)
        df.to_parquet(out_dir / f"sid{sid}_{symbol}.parquet", index=False)

        agg["pairs"] += 1
        agg["episodes"] += stats["episodes"]
        agg["steps"] += len(df)
        agg["same_bar_exits"] += stats["same_bar_exits"]
        agg["episode_return_sum"] += stats["episode_return_sum"]
        for name, code in ACTIONS.items():
            agg["action_counts"][name] += int((df["action"] == code).sum())
        for variant in BEHAVIOR_VARIANTS:
            key = f"episodes_{variant}"
            agg[key] = agg.get(key, 0) + stats.get(key, 0)
        agg["episodes_per_strategy"][str(sid)] = (
            agg["episodes_per_strategy"].get(str(sid), 0) + stats["episodes"])
    return agg


def extract_trajectories(start: str, end: str, db_path: Path | None = None,
                         strategy_ids: list[int] | None = None,
                         symbols: list[str] | None = None,
                         out_dir: Path | None = None,
                         workers: int = 1,
                         skip_existing: bool = False) -> dict:
    ""
    db_path = resolve_db(db_path)
    out_dir = Path(out_dir) if out_dir else CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = [s.strip() for s in
                   Path(SYMBOLS_CSV).read_text().splitlines()
                   if s.strip() and not s.startswith("symbol")]
    sids = sorted(strategy_ids or STRATEGY_REGISTRY)

    agg = _empty_agg(start, end)
    t0 = time.time()
    done = 0

    if workers <= 1:
        for symbol in symbols:
            part = _process_symbol(symbol, sids, str(db_path), str(out_dir),
                                   start, end, skip_existing)
            _merge_agg(agg, part)
            done += 1
            log.info("%s done (%d/%d) — cumulative episodes=%d steps=%d",
                     symbol, done, len(symbols), agg["episodes"], agg["steps"])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process_symbol, symbol, sids, str(db_path),
                                str(out_dir), start, end, skip_existing): symbol
                    for symbol in symbols}
            for fut in as_completed(futs):
                symbol = futs[fut]
                try:
                    _merge_agg(agg, fut.result())
                except Exception:
                    log.exception("%s: worker failed — pairs lost", symbol)
                done += 1
                if done % 25 == 0 or done == len(symbols):
                    log.info("%d/%d symbols — cumulative episodes=%d steps=%d",
                             done, len(symbols), agg["episodes"], agg["steps"])

    agg["episodes_per_strategy"] = {
        k: v for k, v in sorted(agg["episodes_per_strategy"].items(),
                                key=lambda kv: int(kv[0]))}
    agg["mean_episode_logret"] = (
        agg["episode_return_sum"] / agg["episodes"] if agg["episodes"] else 0.0)
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
    agg["workers"] = workers
    agg["skip_existing"] = skip_existing
    (out_dir / "_extract_stats.json").write_text(json.dumps(agg, indent=2))
    return agg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Phase C1 trajectory extractor")
    p.add_argument("--db", default=None)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2099-01-01")
    p.add_argument("--strategies", default=None,
                   help="comma-separated strategy ids (default: all)")
    p.add_argument("--symbols", default=None,
                   help="comma-separated symbols (default: symbols.csv)")
    p.add_argument("--out", default=None, help="cache dir override")
    p.add_argument("--workers", type=int,
                   default=max(1, min(8, (os.cpu_count() or 2) - 2)),
                   help="symbol-level process fan-out (default: cores-2, cap 8)")
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip pairs whose parquet already exists "
                        "(stats undercount; see extract_trajectories doc)")
    args = p.parse_args(argv)

    sids = [int(x) for x in args.strategies.split(",")] if args.strategies else None
    syms = [x.strip().upper() for x in args.symbols.split(",")] if args.symbols else None
    out = extract_trajectories(args.start, args.end, db_path=args.db,
                               strategy_ids=sids, symbols=syms,
                               out_dir=args.out, workers=args.workers,
                               skip_existing=args.skip_existing)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "episodes_per_strategy"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
