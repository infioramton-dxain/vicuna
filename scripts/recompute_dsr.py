""
from __future__ import annotations

import logging
import sys

from macmini.backtest.metrics import deflated_sharpe_ratio
from macmini.config import resolve_db
from macmini.data._db import connect

log = logging.getLogger(__name__)

N_TRIALS = 55
N_RETURNS_PER_WINDOW = 63


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    db = resolve_db()
    with connect(db) as c:
        rows = c.execute(
            "SELECT id, sharpe, skew, kurt FROM mm_backtest_runs"
        ).fetchall()
        log.info("recomputing DSR for %d rows…", len(rows))

        updates = []
        for row_id, sharpe, skew, kurt in rows:
            ds = deflated_sharpe_ratio(
                sharpe=sharpe or 0.0,
                n_returns=N_RETURNS_PER_WINDOW,
                n_trials=N_TRIALS,
                skew=skew or 0.0,
                kurt=kurt or 0.0,
            )
            updates.append((ds, row_id))

        c.executemany(
            "UPDATE mm_backtest_runs SET deflated_sharpe = ? WHERE id = ?",
            updates,
        )
        c.commit()

    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
