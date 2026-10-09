""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.backtest.engine import WalkForwardConfig, signals_to_arrays
from macmini.backtest.loader import load_aux, load_bars
from macmini.backtest.metrics import max_drawdown, profit_factor
from macmini.data._db import connect
from macmini.strategies import STRATEGY_REGISTRY
from macmini.strategies._indicators import atr

log = logging.getLogger(__name__)

STOP_FRAC_MIN, STOP_FRAC_MAX = 0.05, 0.15
TARGET_R_MULT = 2.0
RULES: dict[str, dict] = {
    "none": {"kind": "none"},
    "flat8": {"kind": "flat", "sl": 0.08, "tp": 0.20},
    "atr15": {"kind": "atr", "mult": 1.5},
    "atr20": {"kind": "atr", "mult": 2.0},
    "atr25": {"kind": "atr", "mult": 2.5},
    "atr30": {"kind": "atr", "mult": 3.0},
    "atr25_trail": {"kind": "atr", "mult": 2.5, "trail": True},
}

PAIR_FIELDS = [
    "rule", "regime", "strategy_id", "strategy_name", "category", "symbol",
    "sector", "n_trades", "hit_rate", "profit_factor", "total_return",
    "max_drawdown", "avg_holding_days", "mean_trade_ret", "median_trade_ret",
    "pct_stop_exits", "pct_target_exits", "pct_signal_exits",
]

REGIME_LABELS = ("bull", "bear", "chop")


def regime_series(db_path: Path) -> dict[str, str]:
    ""
    with connect(db_path) as c:
        spy = pd.read_sql_query(
            "SELECT date, close FROM mm_bars WHERE symbol='SPY' ORDER BY date", c)
        vix = pd.read_sql_query(
            "SELECT date, close FROM mm_bars WHERE symbol='^VIX' ORDER BY date", c)
    if spy.empty:
        return {}
    s = spy.set_index("date")["close"]
    sma50, sma200 = s.rolling(50).mean(), s.rolling(200).mean()
    v = vix.set_index("date")["close"].reindex(s.index).ffill()
    out: dict[str, str] = {}
    for d in s.index:
        if pd.isna(sma200[d]):
            continue
        if sma50[d] <= sma200[d]:
            out[d] = "bear"
        elif not pd.isna(v[d]) and v[d] > 25:
            out[d] = "chop"
        else:
            out[d] = "bull"
    return out


def _sl_tp_arrays(rule: dict, bars: pd.DataFrame) -> tuple:
    ""
    if rule["kind"] == "none":
        return np.nan, np.nan, False
    if rule["kind"] == "flat":
        return rule["sl"], rule["tp"], False
    atr_frac = (atr(bars["high"], bars["low"], bars["close"], 14)
                / bars["close"])
    sl = (rule["mult"] * atr_frac).clip(STOP_FRAC_MIN, STOP_FRAC_MAX)
    sl = sl.fillna(0.08).to_numpy()
    return sl, TARGET_R_MULT * sl, bool(rule.get("trail"))


def _exit_causes(trades: pd.DataFrame, exits_shifted: pd.Series,
                 sl, tp, bars: pd.DataFrame) -> pd.Series:
    ""
    closed = trades[trades["Status"] == "Closed"] if "Status" in trades.columns else trades
    if len(closed) == 0:
        return pd.Series(dtype=object)
    causes: list[str] = []
    idx = bars.index
    tp_is_arr = isinstance(tp, np.ndarray)
    entry_pos = idx.searchsorted(pd.to_datetime(closed["Entry Timestamp"]))
    for k, (_, t) in enumerate(closed.iterrows()):
        exit_ts = pd.Timestamp(t["Exit Timestamp"])
        if bool(exits_shifted.get(exit_ts, False)):
            causes.append("signal")
            continue
        if isinstance(sl, float) and np.isnan(sl):
            causes.append("signal")
            continue
        ep = entry_pos[k]
        tpf = float(tp[ep]) if tp_is_arr else float(tp)
        target_level = float(t["Avg Entry Price"]) * (1 + tpf)
        causes.append("target"
                      if float(t["Avg Exit Price"]) >= target_level * 0.9925
                      else "stop")
    return pd.Series(causes, index=closed.index)


def _trade_stats(trades: pd.DataFrame, causes: pd.Series) -> dict:
    ""
    tr = trades["Return"].to_numpy()
    pf_val = profit_factor(tr)
    sub_causes = causes[causes.index.isin(trades.index)]
    n_closed = len(sub_causes)
    share = (sub_causes.value_counts() / n_closed) if n_closed else {}
    return {
        "n_trades": int(len(trades)),
        "hit_rate": round(float((tr > 0).mean()), 4),
        "profit_factor": round(min(float(pf_val), 20.0)
                               if np.isfinite(pf_val) else 20.0, 4),
        "avg_holding_days": round(float(
            (pd.to_datetime(trades["Exit Timestamp"])
             - pd.to_datetime(trades["Entry Timestamp"])
             ).dt.days.mean()), 2),
        "mean_trade_ret": round(float(tr.mean()), 5),
        "median_trade_ret": round(float(np.median(tr)), 5),
        "pct_stop_exits": round(float(share.get("stop", 0.0)), 4),
        "pct_target_exits": round(float(share.get("target", 0.0)), 4),
        "pct_signal_exits": round(float(share.get("signal", 0.0)), 4),
    }


def evaluate_symbol(symbol: str, strategy_ids: list[int], db_path_str: str,
                    min_bars: int, regimes: dict[str, str] | None = None) -> list[dict]:
    ""
    import vectorbt as vbt

    db_path = Path(db_path_str)
    bars = load_bars(symbol, db_path=db_path)
    if bars.empty or len(bars) < min_bars:
        return []
    with connect(db_path) as c:
        row = c.execute("SELECT sector FROM mm_sector_map WHERE symbol = ?",
                        (symbol,)).fetchone()
    sector = (row[0] if row else "Other") or "Other"

    cfg = WalkForwardConfig()
    close, open_ = bars["close"], bars["open"]
    high, low = bars["high"], bars["low"]

    rows: list[dict] = []
    for sid in strategy_ids:
        cls = STRATEGY_REGISTRY.get(sid)
        if cls is None:
            continue
        aux = load_aux(symbol, cls, db_path=db_path)
        try:
            signals = cls().generate_signals(symbol, bars, **aux)
        except Exception:
            continue
        if not signals:
            continue
        entries, exits = signals_to_arrays(signals, bars)
        entries = entries.shift(1, fill_value=False)
        exits = exits.shift(1, fill_value=False)
        if not entries.any():
            continue

        for rule_name, rule in RULES.items():
            sl, tp, trail = _sl_tp_arrays(rule, bars)
            try:
                pf = vbt.Portfolio.from_signals(
                    close=close, entries=entries, exits=exits,
                    price=open_, high=high, low=low,
                    sl_stop=sl, tp_stop=tp, sl_trail=trail,
                    stop_entry_price="fillprice",
                    stop_exit_price="stopmarket",
                    fees=cfg.transaction_cost_bps / 10_000,
                    slippage=cfg.slippage_bps / 10_000,
                    init_cash=10_000, freq="1D",
                )
            except Exception:
                log.exception("%s × #%d × %s: simulation failed",
                              symbol, sid, rule_name)
                continue
            trades = pf.trades.records_readable
            if len(trades) == 0:
                continue
            causes = _exit_causes(trades, exits, sl, tp, bars)
            base = {"rule": rule_name, "strategy_id": sid,
                    "strategy_name": cls.name, "category": cls.category,
                    "symbol": symbol, "sector": sector}
            rows.append({
                **base, "regime": "ALL", **_trade_stats(trades, causes),
                "total_return": round(float(pf.total_return()), 4),
                "max_drawdown": round(float(max_drawdown(pf.value().to_numpy())), 4),
            })
            if regimes:
                entry_day = pd.to_datetime(
                    trades["Entry Timestamp"]).dt.strftime("%Y-%m-%d")
                trade_regime = entry_day.map(regimes)
                for label in REGIME_LABELS:
                    sub = trades[trade_regime == label]
                    if len(sub) == 0:
                        continue
                    rows.append({
                        **base, "regime": label, **_trade_stats(sub, causes),
                        "total_return": None, "max_drawdown": None,
                    })
    return rows


def _universe(db_path: Path, sectors: list[str] | None) -> list[str]:
    ""
    root = Path(__file__).resolve().parents[3]
    syms = [s.strip().upper() for s in
            (root / "symbols.csv").read_text().splitlines()
            if s.strip() and not s.startswith("symbol")]
    if not sectors:
        return syms
    with connect(db_path) as c:
        rows = c.execute(
            f"SELECT symbol FROM mm_sector_map WHERE sector IN "
            f"({','.join('?' * len(sectors))})", sectors).fetchall()
    keep = {r[0] for r in rows}
    return [s for s in syms if s in keep]


def run(db_path: Path, out_csv: Path, *, sectors: list[str] | None = None,
        strategy_ids: list[int] | None = None, workers: int | None = None,
        min_bars: int = 504, limit: int | None = None) -> dict:
    import macmini.strategies

    strategy_ids = strategy_ids or sorted(STRATEGY_REGISTRY)
    symbols = _universe(db_path, sectors)
    if limit:
        symbols = symbols[:limit]
    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 2))
    regimes = regime_series(db_path)
    if regimes:
        counts = pd.Series(list(regimes.values())).value_counts().to_dict()
        days = sorted(regimes)
        log.info("regime series: %d labeled days %s → %s, %s",
                 len(regimes), days[0], days[-1], counts)
    else:
        log.warning("no SPY history in %s — regime slicing disabled, "
                    "only ALL rows will be emitted", db_path)
    log.info("stop_counterfactual: %d symbols × %d strategies × %d rules, "
             "workers=%d, db=%s", len(symbols), len(strategy_ids), len(RULES),
             workers, db_path)

    started = time.time()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()

        if workers == 1:
            for sym in symbols:
                rows = evaluate_symbol(sym, strategy_ids, str(db_path),
                                       min_bars, regimes)
                writer.writerows(rows)
                n_rows += len(rows)
                log.info("%s: %d rows", sym, len(rows))
        else:
            pending = deque(symbols)
            in_flight: dict = {}
            with ProcessPoolExecutor(max_workers=workers,
                                     max_tasks_per_child=64) as pool:
                def _submit(cap: int) -> None:
                    while pending and len(in_flight) < cap:
                        sym = pending.popleft()
                        in_flight[pool.submit(
                            evaluate_symbol, sym, strategy_ids,
                            str(db_path), min_bars, regimes)] = sym

                _submit(2 * workers)
                while in_flight:
                    done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                    for fut in done:
                        sym = in_flight.pop(fut)
                        try:
                            rows = fut.result()
                        except Exception:
                            log.exception("worker %s raised", sym)
                            continue
                        writer.writerows(rows)
                        fh.flush()
                        n_rows += len(rows)
                        log.info("%s: %d rows (%d symbols left)",
                                 sym, len(rows), len(pending) + len(in_flight))
                    _submit(2 * workers)

    elapsed = round(time.time() - started, 1)
    log.info("done: %d rows → %s in %.1fs", n_rows, out_csv, elapsed)
    return {"rows": n_rows, "out": str(out_csv), "elapsed_seconds": elapsed}


def summarize(csv_path: Path) -> pd.DataFrame:
    ""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    if "regime" not in df.columns:
        df["regime"] = "ALL"

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "pairs": len(g),
            "trades": int(g["n_trades"].sum()),
            "median_pf": round(float(g["profit_factor"].median()), 3),
            "mean_trade_ret": round(float(g["mean_trade_ret"].mean()), 5),
            "mean_total_ret": round(float(g["total_return"].mean()), 4),
            "mean_hit_rate": round(float(g["hit_rate"].mean()), 4),
            "mean_max_dd": round(float(g["max_drawdown"].mean()), 4),
            "mean_hold_days": round(float(g["avg_holding_days"].mean()), 2),
            "stop_exit_share": round(float(g["pct_stop_exits"].mean()), 4),
            "target_exit_share": round(float(g["pct_target_exits"].mean()), 4),
        })

    overall = df.groupby(["regime", "rule"]).apply(agg, include_groups=False)
    overall = overall.reset_index()
    overall.insert(0, "scope", "ALL")
    all_rows = df[df["regime"] == "ALL"]
    by_sector = all_rows.groupby(["sector", "rule"]).apply(agg, include_groups=False)
    by_sector = by_sector.reset_index().rename(columns={"sector": "scope"})
    by_sector.insert(1, "regime", "ALL")
    return pd.concat([overall, by_sector], ignore_index=True)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Stop-width counterfactual replay (D20)")
    p.add_argument("--db", required=True, help="SCRATCH DB copy (never the live DB)")
    p.add_argument("--out", default=None, help="pair-level CSV output path")
    p.add_argument("--sectors", default=None,
                   help="comma-separated sector filter (e.g. Mining)")
    p.add_argument("--strategies", default=None,
                   help="comma-separated strategy ids (default: all)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--min-bars", type=int, default=504)
    p.add_argument("--limit", type=int, default=None,
                   help="cap symbol count (smoke tests)")
    p.add_argument("--summary", action="store_true",
                   help="print per-rule aggregates when the run finishes")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    out_csv = Path(args.out) if args.out else Path("logs") / (
        f"stop_counterfactual_{time.strftime('%Y%m%dT%H%M%S')}.csv")
    result = run(
        db_path, out_csv,
        sectors=[s.strip() for s in args.sectors.split(",")] if args.sectors else None,
        strategy_ids=[int(x) for x in args.strategies.split(",")] if args.strategies else None,
        workers=args.workers, min_bars=args.min_bars, limit=args.limit,
    )
    print(result)
    if args.summary and result["rows"]:
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(summarize(out_csv).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
