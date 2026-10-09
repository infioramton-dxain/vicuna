""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

AGENT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(AGENT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val if val not in ("", None) else default


SHIPPED_DB = Path(_env("AGENT_SHIPPED_DB",
                       "/home/agent/agent/db/macmini.db"))
TRADES_DB = Path(_env("AGENT_TRADES_DB",
                      "/home/agent/agent/trades.db"))
DECISIONS_LOG = Path(_env("AGENT_DECISIONS_LOG",
                          "/home/agent/agent/decisions.jsonl"))
LOG_DIR = Path(_env("AGENT_LOG_DIR", "/home/agent/agent/logs"))

ALPACA_API_KEY = _env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _env("ALPACA_SECRET_KEY") or _env("ALPACA_API_SECRET")
ALPACA_BASE_URL = _env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

LLM_PROVIDER = _env("AGENT_LLM_PROVIDER", "deepseek")
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-chat")
GEMINI_API_KEY = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.0-flash-exp")

MAX_RISK_PER_TRADE_PCT = float(_env("AGENT_MAX_RISK_PER_TRADE", "0.01"))
MAX_DAILY_DD_PCT = float(_env("AGENT_MAX_DAILY_DD", "0.05"))
MAX_MONTHLY_DD_PCT = float(_env("AGENT_MAX_MONTHLY_DD", "0.10"))
MAX_CONCURRENT_POSITIONS = int(_env("AGENT_MAX_POSITIONS", "5"))
MAX_SECTOR_CONCENTRATION = float(_env("AGENT_MAX_SECTOR", "0.25"))
MAX_POSITION_NOTIONAL_PCT = float(_env("AGENT_MAX_POSITION_PCT", "0.10"))
MAX_POSITION_NOTIONAL_USD = float(_env("AGENT_MAX_POSITION_USD", "10000"))
MIN_NOTIONAL = float(_env("AGENT_MIN_NOTIONAL", "20.0"))
EARNINGS_BLACKOUT_DAYS = 2
FOMC_HALVE_SIZE_HOURS = 24

DRY_RUN = _env("AGENT_DRY_RUN", "0") == "1"
MAX_PICKS_TO_SUBMIT = int(_env("AGENT_MAX_PICKS", "5"))
