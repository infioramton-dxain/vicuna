""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from macmini.backtest.engine import WalkForwardConfig, walk_forward_backtest
from macmini.backtest.loader import load_aux, load_bars, load_dividends
from macmini.backtest.runner import _write_rows
from macmini.data._db import connect
from macmini.strategies import STRATEGY_REGISTRY

log = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS_CAP = 4
MAX_TASKS_PER_CHILD = 64
PARALLEL_MIN_SYMBOLS = 8


def _resolve_max_workers(requested: int | None) -> int:
    ""
    cpu = os.cpu_count() or 2
    safe = max(1, min(_DEFAULT_MAX_WORKERS_CAP, cpu - 2))
    if requested is None:
        return safe
    return max(1, min(requested, cpu))


def _evaluate_symbol_task(
    symbol: str,
    strategy_ids: list[int],
    db_path_str: str,
    config: WalkForwardConfig,
) -> dict:
    ""
    db_path = Path(db_path_str)
    bars = load_bars(symbol, db_path=db_path)
    if bars.empty or len(bars) < config.min_history_days:
        return {
            "symbol": symbol,
            "sector": None,
            "skipped_no_history": True,
            "n_bars": len(bars),
            "per_strategy": [],
            "unknown_strategies": [],
        }

    with connect(db_path) as c:
        row = c.execute(
            "SELECT sector FROM mm_sector_map WHERE symbol = ?", (symbol,)
        ).fetchone()
    sector = (row[0] if row else "Other") or "Other"

    per_strategy: list[tuple[int, list[dict]]] = []
    unknown: list[int] = []
    dividends = load_dividends(symbol, db_path=db_path)
    for sid in strategy_ids:
        cls = STRATEGY_REGISTRY.get(sid)
        if cls is None:
            unknown.append(sid)
            continue
        aux = load_aux(symbol, cls, db_path=db_path)
        strategy = cls()
        df = walk_forward_backtest(strategy, symbol, bars, config=config, aux=aux,
                                   dividends=dividends)
        rows = [] if df.empty else df.to_dict(orient="records")
        per_strategy.append((sid, rows))

    return {
        "symbol": symbol,
        "sector": sector,
        "skipped_no_history": False,
        "n_bars": len(bars),
        "per_strategy": per_strategy,
        "unknown_strategies": unknown,
    }


def run_parallel(
    symbols: list[str],
    strategy_ids: list[int],
    db_path: Path,
    config: WalkForwardConfig | None = None,
    *,
    max_workers: int | None = None,
    run_id: str | None = None,
) -> dict:
    ""
    if config is None:
        config = WalkForwardConfig()

    workers = _resolve_max_workers(max_workers)

    if workers == 1 or len(symbols) < PARALLEL_MIN_SYMBOLS:
        from macmini.backtest.runner import run as sequential_run
        log.info(
            "parallel_runner: sequential fall-through (workers=%d, symbols=%d)",
            workers, len(symbols),
        )
        return sequential_run(symbols, strategy_ids, db_path, config=config)

    if run_id is None:
        run_id = (
            f"bt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f"_{uuid.uuid4().hex[:6]}"
        )
    log.info(
        "backtest run_id=%s symbols=%d strategies=%d workers=%d (parallel)",
        run_id, len(symbols), len(strategy_ids), workers,
    )

    started = time.time()
    counters = {"pairs": 0, "rows_written": 0, "skipped": 0}
    n_trials = max(1, len(symbols) * len(strategy_ids))

    pending = deque(symbols)
    in_flight: dict = {}
    db_path_str = str(db_path)

    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=MAX_TASKS_PER_CHILD,
    ) as pool:
        def _submit_up_to(cap: int) -> None:
            while pending and len(in_flight) < cap:
                sym = pending.popleft()
                fut = pool.submit(
                    _evaluate_symbol_task,
                    sym, strategy_ids, db_path_str, config,
                )
                in_flight[fut] = sym

        _submit_up_to(2 * workers)
        while in_flight:
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                sym = in_flight.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    log.exception("worker %s raised: %s", sym, exc)
                    counters["skipped"] += 1
                    continue

                if result["skipped_no_history"]:
                    log.warning(
                        "%s: insufficient history (%d bars); skipping",
                        sym, result["n_bars"],
                    )
                    counters["skipped"] += 1
                    continue

                for sid in result["unknown_strategies"]:
                    log.warning("unknown strategy id %d; skipping", sid)

                sector = result["sector"]
                for sid, rows in result["per_strategy"]:
                    if not rows:
                        counters["skipped"] += 1
                        continue
                    _write_rows(
                        db_path, run_id, sid, sym, sector,
                        rows, n_trials,
                    )
                    counters["pairs"] += 1
                    counters["rows_written"] += len(rows)
                    log.info("  %s × #%d: %d windows", sym, sid, len(rows))
            _submit_up_to(2 * workers)

    counters["run_id"] = run_id
    counters["elapsed_seconds"] = round(time.time() - started, 1)
    return counters
