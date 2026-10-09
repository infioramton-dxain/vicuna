""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from macmini.config import DB_PATH, SENTIMENT_DB_PATH, resolve_db

log = logging.getLogger(__name__)


def refresh(
    db_path: Path | str | None = None,
    sentsrc_path: str | None = None,
) -> dict:
    ""
    local = resolve_db(db_path)
    sentsrc = Path(sentsrc_path or SENTIMENT_DB_PATH)
    if not sentsrc.exists():
        raise FileNotFoundError(f"SENTIMENT db not found at {sentsrc}")

    from macmini.data._db import connect as _db_connect
    with _db_connect(local) as c:

        row = c.execute("SELECT MAX(scored_at) FROM sentiment").fetchone()
        watermark = row[0] or "1970-01-01T00:00:00Z"

        c.execute(f"ATTACH DATABASE 'file:{sentsrc}?mode=ro' AS sentsrc")
        try:
            cur = c.execute(
                "INSERT OR IGNORE INTO sentiment "
                "SELECT * FROM sentsrc.sentiment WHERE scored_at > ?",
                (watermark,),
            )
            sentiment_inserted = cur.rowcount

            cur = c.execute(
                "INSERT OR IGNORE INTO term "
                "SELECT * FROM sentsrc.term WHERE window_start > ?",
                (watermark,),
            )
            term_inserted = cur.rowcount
            c.commit()
        finally:
            c.execute("DETACH DATABASE sentsrc")

    return {
        "watermark": watermark,
        "sentiment_inserted": sentiment_inserted,
        "term_inserted": term_inserted,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Sync sentiment + term → macmini.db.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--sentsrc", default=SENTIMENT_DB_PATH)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = refresh(args.db, args.sentsrc)
    print("sentiment_refresh OK")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
