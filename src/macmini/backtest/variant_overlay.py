""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from macmini.backtest import exec_variants
from macmini.backtest.engine import WalkForwardConfig, walk_forward_backtest
from macmini.backtest.loader import load_aux, load_bars, load_dividends
from macmini.backtest.runner import _ensure_sector, _write_rows
from macmini.config import SYMBOLS_CSV, resolve_db
from macmini.data._db import connect
from macmini.ranking.exporter import (
    MAX_DRAWDOWN_FLOOR,
    RESEARCH_MIN_DEFLATED_SHARPE,
    MIN_PROFIT_FACTOR,
    MIN_TOTAL_TRADES,
    SURVIVAL_WINDOWS,
    _winsor_median,
)
from macmini.strategies import STRATEGY_REGISTRY

log = logging.getLogger(__name__)

DEFAULT_MARGIN = 0.05


def _universe_size(symbols_file: Path = SYMBOLS_CSV) -> int:
    with open(symbols_file, newline="") as f:
        return sum(1 for r in csv.DictReader(f) if r.get("symbol"))


def _aggregate(rows: list[dict]) -> dict:
    ""
    recent = rows[-SURVIVAL_WINDOWS:]
    if not recent:
        return {"dsr_avg": 0.0, "qualifies": False, "total_trades": 0}
    dsr_avg = sum(r["deflated_sharpe"] for r in recent) / len(recent)
    dd_worst = min(r["max_drawdown"] for r in recent)
    traded = [r for r in recent if r["n_trades"] >= 1]
    total_trades = sum(r["n_trades"] for r in traded)
    pf_median = _winsor_median([r["profit_factor"] for r in traded])
    qualifies = (
        total_trades >= MIN_TOTAL_TRADES
        and dsr_avg >= RESEARCH_MIN_DEFLATED_SHARPE
        and pf_median >= MIN_PROFIT_FACTOR
        and dd_worst >= MAX_DRAWDOWN_FLOOR
    )
    return {
        "dsr_avg": float(dsr_avg),
        "pf_median": float(pf_median),
        "dd_worst": float(dd_worst),
        "total_trades": int(total_trades),
        "qualifies": bool(qualifies),
    }


def _read_back(db_path: Path, run_id: str, sid: int, symbol: str,
               entry_type: str, exit_type: str) -> list[dict]:
    with connect(db_path) as c:
        c.row_factory = None
        cur = c.execute(
            """SELECT deflated_sharpe, profit_factor, max_drawdown, n_trades
               FROM mm_backtest_runs
               WHERE run_id = ? AND strategy_id = ? AND symbol = ?
                 AND entry_type = ? AND exit_type = ?
               ORDER BY id""",
            (run_id, sid, symbol, entry_type, exit_type),
        )
        return [
            {"deflated_sharpe": r[0], "profit_factor": r[1],
             "max_drawdown": r[2], "n_trades": r[3]}
            for r in cur.fetchall()
        ]


def run_overlay(
    db_path: Path,
    margin: float = DEFAULT_MARGIN,
    cells: list[tuple[int, str]] | None = None,
    config: WalkForwardConfig | None = None,
) -> dict:
    if config is None:
        config = WalkForwardConfig()
    grid = exec_variants.variant_grid()
    ref_entry, ref_exit = exec_variants.REFERENCE_VARIANT

    if cells is None:
        with connect(db_path) as c:
            cells = [
                (int(r[0]), r[1])
                for r in c.execute(
                    "SELECT strategy_id, symbol FROM mm_leaderboard "
                    "WHERE qualifies = 1 ORDER BY deflated_sharpe_avg DESC"
                )
            ]
    if not cells:
        log.warning("no qualifying cells; nothing to overlay")
        return {"cells": 0, "run_id": None}

    n_trials = max(1, _universe_size() * len(STRATEGY_REGISTRY) * len(grid))
    run_id = (
        f"vt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"_{uuid.uuid4().hex[:6]}"
    )
    log.info("variant overlay run_id=%s cells=%d grid=%d n_trials=%d",
             run_id, len(cells), len(grid), n_trials)

    started = time.time()
    now = datetime.now(timezone.utc).isoformat()
    counters = {"cells": 0, "rows_written": 0, "winners_non_reference": 0}

    for sid, symbol in cells:
        cls = STRATEGY_REGISTRY.get(sid)
        if cls is None:
            log.warning("unknown strategy id %d; skipping", sid)
            continue
        bars = load_bars(symbol, db_path=db_path)
        if bars.empty or len(bars) < config.min_history_days:
            log.warning("%s: insufficient history; skipping", symbol)
            continue
        sector = _ensure_sector(symbol, db_path)
        aux = load_aux(symbol, cls, db_path=db_path)
        dividends = load_dividends(symbol, db_path=db_path)
        strategy = cls()

        agg_by_variant: dict[tuple[str, str], dict] = {}
        for entry_type, exit_type in grid:
            df = walk_forward_backtest(
                strategy, symbol, bars, config=config, aux=aux,
                entry_type=entry_type, exit_type=exit_type,
                dividends=dividends,
            )
            if df.empty:
                continue
            rows = df.to_dict(orient="records")
            _write_rows(db_path, run_id, sid, symbol, sector, rows, n_trials,
                        entry_type=entry_type, exit_type=exit_type)
            counters["rows_written"] += len(rows)
            agg_by_variant[(entry_type, exit_type)] = _aggregate(
                _read_back(db_path, run_id, sid, symbol, entry_type, exit_type)
            )

        ref = agg_by_variant.get((ref_entry, ref_exit))
        dsr_ref = ref["dsr_avg"] if ref else 0.0
        winner = (ref_entry, ref_exit)
        dsr_best = dsr_ref
        for variant, agg in agg_by_variant.items():
            if variant == winner:
                continue
            if agg["qualifies"] and agg["dsr_avg"] >= dsr_ref + margin \
                    and agg["dsr_avg"] > dsr_best:
                winner, dsr_best = variant, agg["dsr_avg"]
        if winner != (ref_entry, ref_exit):
            counters["winners_non_reference"] += 1
        with connect(db_path) as c:
            c.execute(
                """INSERT OR REPLACE INTO mm_variant_selection
                   (strategy_id, symbol, entry_type, exit_type,
                    dsr_ref, dsr_best, n_variants, run_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, symbol, winner[0], winner[1],
                 dsr_ref, dsr_best, len(grid), run_id, now),
            )
            c.commit()
        counters["cells"] += 1
        log.info("  %s × #%d → %s/%s (dsr %0.3f vs ref %0.3f)",
                 symbol, sid, winner[0], winner[1], dsr_best, dsr_ref)

    counters["run_id"] = run_id
    counters["elapsed_seconds"] = round(time.time() - started, 1)
    return counters


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="order-type variant overlay")
    p.add_argument("--db", default=None,
                   help="path to DB (use a scratch copy for validation runs)")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help="DSR margin a challenger needs over the reference")
    p.add_argument("--cells", default=None,
                   help="explicit cells 'sid:SYM,sid:SYM' (default: leaderboard qualifiers)")
    args = p.parse_args(argv)

    cells = None
    if args.cells:
        cells = []
        for part in args.cells.split(","):
            sid, sym = part.split(":")
            cells.append((int(sid), sym.strip().upper()))

    out = run_overlay(resolve_db(args.db), margin=args.margin, cells=cells)
    print(
        f"Overlay {out['run_id']} done in {out.get('elapsed_seconds', 0)}s — "
        f"{out['cells']} cells, {out['rows_written']} rows, "
        f"{out['winners_non_reference']} non-reference winners"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
