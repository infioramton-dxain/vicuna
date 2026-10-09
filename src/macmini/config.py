""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env", override=False)

DB_PATH: Path = PROJECT_ROOT / "macmini.db"
SYMBOLS_CSV: Path = PROJECT_ROOT / "symbols.csv"
EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val if val not in ("", None) else default


ALPACA_API_KEY = _env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _env("ALPACA_SECRET_KEY") or _env("ALPACA_API_SECRET")
ALPACA_BASE_URL = _env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = _env("ALPACA_DATA_URL", "https://data.alpaca.markets")

MASSIVE_API_KEY = _env("MASSIVE_API_KEY")
FINNHUB_API_KEY = _env("FINNHUB_API_KEY")

GEMINI_API_KEY = (
    _env("GEMINI_API_KEY")
    or _env("GOOGLE_API_KEY")
    or _env("GOOGLE_API_KEYU")
)
GOOGLE_PROJECT_NUMBER = _env("GOOGLE_PROJECT_NUMBER")

DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

SENTIMENT_DB_PATH = _env(
    "SENTIMENT_DB_PATH",
    str(Path.home() / "Projects" / "SENTIMENT" / "data" / "sentsrc.db"),
)

MACMINI_HOST = _env("MACMINI_HOST")
MACMINI_USER = _env("MACMINI_USER")
MACMINI_DEST_DIR = _env("MACMINI_DEST_DIR", "/var/agent/db/")


def resolve_db(override: str | os.PathLike | None = None) -> Path:
    ""
    return Path(override) if override else DB_PATH
