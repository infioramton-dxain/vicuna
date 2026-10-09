""
from __future__ import annotations

import sqlite3
from pathlib import Path

from macmini.config import resolve_db

BUSY_TIMEOUT_MS = 30_000


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    ""
    path = resolve_db(db_path)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
