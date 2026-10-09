""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from macmini.backtest.engine import WalkForwardConfig, walk_forward_backtest
from macmini.backtest.loader import load_aux, load_bars, load_dividends
from macmini.backtest.metrics import bonferroni_pvalue, deflated_sharpe_ratio
from macmini.config import SYMBOLS_CSV, resolve_db
from macmini.data._db import connect
from macmini.strategies import STRATEGY_REGISTRY

log = logging.getLogger(__name__)


def _read_symbols(path: Path) -> list[str]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["symbol"] for r in rows if r.get("symbol")]


def _ensure_sector(symbol: str, db_path: Path) -> str:
    with connect(db_path) as c:
        row = c.execute("SELECT sector FROM mm_sector_map WHERE symbol = ?", (symbol,)).fetchone()
    return (row[0] if row else "Other") or "Other"


def _write_rows(
    db_path: Path,
    run_id: str,
    strategy_id: int,
    symbol: str,
    sector: str,
    window_stats: list[dict],
    n_trials: int,
    entry_type: str | None = None,
    exit_type: str | None = None,
) -> None:
    ""
    if not window_stats:
        return
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as c:
        for w in window_stats:
            ds = deflated_sharpe_ratio(
                sharpe=w["sharpe"],
                n_returns=max(2, int(w.get("n_returns", 63))),
                n_trials=n_trials,
                skew=w["skew"],
                kurt=w["kurt"],
            )
            pb = bonferroni_pvalue(w["p_value"], n_trials)
            c.execute(
                """INSERT INTO mm_backtest_runs
                   (run_id, strategy_id, symbol, sector, window_start, window_end,
                    n_trades, sharpe, deflated_sharpe, profit_factor, max_drawdown,
                    hit_rate, total_return, avg_holding_days, skew, kurt,
                    p_value, p_value_bonferroni, created_at, entry_type, exit_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, strategy_id, symbol, sector,
                 w["window_start"], w["window_end"],
                 w["n_trades"], w["sharpe"], ds, w["profit_factor"], w["max_drawdown"],
                 w["hit_rate"], w["total_return"], w["avg_holding_days"], w["skew"], w["kurt"],
                 w["p_value"], pb, now, entry_type, exit_type),
            )
        c.commit()


def run(
    symbols: list[str],
    strategy_ids: list[int],
    db_path: Path,
    config: WalkForwardConfig | None = None,
) -> dict:
    if config is None:
        config = WalkForwardConfig()
    run_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    log.info("backtest run_id=%s symbols=%d strategies=%d", run_id, len(symbols), len(strategy_ids))
    n_trials = max(1, len(symbols) * len(strategy_ids))

    started = time.time()
    counters = {"pairs": 0, "rows_written": 0, "skipped": 0}

    for symbol in symbols:
        bars = load_bars(symbol, db_path=db_path)
        if bars.empty or len(bars) < config.min_history_days:
            log.warning("%s: insufficient history (%d bars); skipping", symbol, len(bars))
            counters["skipped"] += 1
            continue
        sector = _ensure_sector(symbol, db_path)
        dividends = load_dividends(symbol, db_path=db_path)
        for sid in strategy_ids:
            cls = STRATEGY_REGISTRY.get(sid)
            if cls is None:
                log.warning("unknown strategy id %d; skipping", sid)
                continue
            aux = load_aux(symbol, cls, db_path=db_path)
            strategy = cls()
            df = walk_forward_backtest(strategy, symbol, bars, config=config, aux=aux,
                                       dividends=dividends)
            if df.empty:
                counters["skipped"] += 1
                continue
            rows = df.to_dict(orient="records")
            _write_rows(db_path, run_id, sid, symbol, sector, rows, n_trials)
            counters["pairs"] += 1
            counters["rows_written"] += len(rows)
            log.info("  %s × #%d: %d windows", symbol, sid, len(rows))

    counters["run_id"] = run_id
    counters["elapsed_seconds"] = round(time.time() - started, 1)
    return counters


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="macmini walk-forward backtest runner")
    p.add_argument("--db", default=None, help="path to macmini.db (defaults to project root)")
    p.add_argument("--symbols", help="comma-separated tickers (overrides --all)")
    p.add_argument("--symbols-file", default=str(SYMBOLS_CSV))
    p.add_argument("--strategies", help="comma-separated strategy ids (default: all registered)")
    p.add_argument("--all", action="store_true", help="run full universe")
    p.add_argument(
        "--parallel", action="store_true",
        help="run symbol-level ProcessPoolExecutor (see parallel_runner.py)",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="worker cap for --parallel; default = min(4, cpu_count-2)",
    )
    args = p.parse_args(argv)

    db_path = resolve_db(args.db)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.all:
        symbols = _read_symbols(Path(args.symbols_file))
    else:
        p.error("either --symbols or --all is required")
        return 2

    if args.strategies:
        ids = [int(s) for s in args.strategies.split(",") if s.strip()]
    else:
        ids = sorted(STRATEGY_REGISTRY)

    if args.parallel:
        from macmini.backtest.parallel_runner import run_parallel
        out = run_parallel(symbols, ids, db_path, max_workers=args.workers)
    else:
        out = run(symbols, ids, db_path)
    print(
        f"Run {out['run_id']} done in {out['elapsed_seconds']}s — "
        f"{out['pairs']} (symbol,strategy) pairs, "
        f"{out['rows_written']} rows written, {out['skipped']} skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
