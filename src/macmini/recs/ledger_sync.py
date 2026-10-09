""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from macmini.config import DB_PATH, resolve_db

log = logging.getLogger(__name__)


_MINI_TABLE = "recs"
_LAPTOP_TABLE = "mm_recs_ledger"

_COLUMNS = (
    "rec_id", "as_of_date", "entry_date", "rank",
    "symbol", "strategy_id", "strategy_name", "sector",
    "suggested_size_pct", "suggested_stop_loss_pct", "suggested_target_pct",
    "rationale", "status", "decided_at", "operator_reply", "notes",
    "alpaca_order_id", "order_qty", "entry_est_price", "stop_price",
    "target_price", "fill_status", "actual_fill_price", "actual_fill_at",
    "fill_qty", "last_polled_at", "created_at", "updated_at",
    "stop_order_id", "target_order_id", "exit_status", "exit_price",
    "exit_at", "exit_qty", "superseded_order_ids",
)


def sync_ledger(
    snapshot_path: Path | str,
    db_path: Path | str | None = None,
) -> dict:
    ""
    snap = Path(snapshot_path)
    dest = resolve_db(db_path)

    if not snap.exists():
        log.warning("ledger_sync: snapshot missing at %s — skipping", snap)
        return {"rows_synced": 0, "skipped": "snapshot_missing"}

    src = sqlite3.connect(f"file:{snap}?mode=ro", uri=True, timeout=30)
    try:
        src_cols = {r[1] for r in src.execute(f"PRAGMA table_info({_MINI_TABLE})")}
        missing = set(_COLUMNS) - src_cols
        if missing:
            log.error("ledger_sync: mini snapshot missing columns %s — refusing to sync",
                      sorted(missing))
            return {"rows_synced": 0, "error": f"missing_columns={sorted(missing)}"}

        rows = src.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM {_MINI_TABLE}"
        ).fetchall()
    finally:
        src.close()

    if not rows:
        return {"rows_synced": 0, "snapshot_rows": 0}

    placeholders = ", ".join("?" for _ in _COLUMNS)
    sql = (f"INSERT OR REPLACE INTO {_LAPTOP_TABLE} "
           f"({', '.join(_COLUMNS)}) VALUES ({placeholders})")

    written = 0
    with sqlite3.connect(dest, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute("PRAGMA journal_mode = WAL")
        for row in rows:
            c.execute(sql, row)
            c.commit()
            written += 1

    log.info("ledger_sync: %d rows synced from %s → %s (table %s)",
             written, snap.name, dest.name, _LAPTOP_TABLE)
    return {"rows_synced": written, "snapshot_rows": len(rows),
            "snapshot_path": str(snap)}


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Sync mini's trades.db::recs into macmini.db::mm_recs_ledger")
    p.add_argument("--snapshot", required=True,
                   help="path to rsync'd trades.db snapshot")
    p.add_argument("--db", default=None, help=f"target laptop DB (default {DB_PATH})")
    args = p.parse_args(argv)
    out = sync_ledger(args.snapshot, args.db)
    print(f"ledger_sync done: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
