""
from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.config import SHIPPED_DB, TRADES_DB


def open_shipped_ro() -> sqlite3.Connection:
    ""
    uri = f"file:{SHIPPED_DB}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def open_trades() -> sqlite3.Connection:
    ""
    TRADES_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRADES_DB)
    conn.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy_id INTEGER NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            broker_order_id TEXT,
            status TEXT,
            status_reason TEXT,
            llm_rationale TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of_date TEXT NOT NULL,
            snapshot_path TEXT NOT NULL,
            picks_considered INTEGER,
            picks_taken INTEGER,
            llm_response_json TEXT,
            regime_read TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
