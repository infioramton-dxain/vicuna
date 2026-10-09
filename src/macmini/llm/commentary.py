""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB = REPO_ROOT / "macmini.db"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-r1:14b"
TIMEOUT_S = 540
TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are the daily briefing writer for Vicuna, a fully \
autonomous long-only swing-trading system trading a paper account. \
Write a \
plain-English morning briefing from the JSON context you are given.

Hard rules:
- Use ONLY numbers present in the context. Never invent, extrapolate, or \
estimate a number. If something is missing, say it is missing.
- Kronos forecasts are experimental and unproven on this universe — when you \
mention them, say they are unproven. Never present them as expectations.
- You advise nobody. The system is autonomous; do not suggest manual trades \
or overrides. You may flag things worth watching.
- Structure: ## Account · ## Positions · ## Today's picks · ## Model views \
(Kronos + sentiment/regime) · ## Watch-outs. Keep it under 400 words. \
Markdown, no preamble."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _fetch_account() -> dict | None:
    ""
    env = _load_env(REPO_ROOT / ".env")
    key, sec = env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        return None
    base = env.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    req = urllib.request.Request(
        base + "/v2/account",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            acct = json.load(resp)
        return {"equity": float(acct["equity"]), "cash": float(acct["cash"]),
                "fetched_at": _now_iso()}
    except Exception:
        return None


def assemble_context(db_path: Path | str = DEFAULT_DB) -> dict:
    ""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row

        def rows(sql: str, params: tuple = ()) -> list[dict]:
            try:
                return [dict(r) for r in c.execute(sql, params).fetchall()]
            except sqlite3.Error:
                return []

        latest_bar = c.execute("SELECT MAX(date) FROM mm_bars").fetchone()[0]
        positions = rows(
            "SELECT symbol, qty, avg_entry_price, current_price, unrealized_pl, "
            "unrealized_plpc, snapshot_at FROM mm_alpaca_positions")
        picks = rows(
            "SELECT rank, symbol, sector, strategy_name, deflated_sharpe, "
            "suggested_size_pct, as_of_date FROM mm_active_picks ORDER BY rank")
        watch_syms = sorted({p["symbol"] for p in positions}
                            | {p["symbol"] for p in picks})
        kronos = [
            r for s in watch_syms for r in rows(
                "SELECT symbol, date, predicted_5d_return, predicted_30d_return, "
                "predicted_5d_vol FROM mm_kronos_signals WHERE symbol = ? "
                "AND COALESCE(quality_flag, 'ok') = 'ok' "
                "ORDER BY date DESC LIMIT 1", (s,))
        ]
        sentiment = [
            r for s in watch_syms for r in rows(
                "SELECT symbol, window_end, sentiment, article_count FROM sentiment "
                "WHERE symbol = ? AND status = 'ok' "
                "ORDER BY window_end DESC LIMIT 1", (s,))
        ]
        regime = rows("SELECT * FROM mm_regime ORDER BY as_of_date DESC LIMIT 1")
        divergences = rows(
            "SELECT snapshot_at, severity, kind, symbol FROM mm_broker_divergences "
            "ORDER BY snapshot_at DESC LIMIT 5")
    return {
        "generated_at_utc": _now_iso(),
        "data_as_of": latest_bar,
        "account": _fetch_account(),
        "positions": positions,
        "active_picks": picks,
        "kronos_forecasts_unproven": kronos,
        "latest_sentiment": sentiment,
        "market_regime": regime[0] if regime else None,
        "recent_divergence_alerts": divergences,
        "notes": {
            "kronos": "zero-shot foundation-model forecasts; evidence ledger "
                      "started 2026-07-24; no measured skill yet",
            "divergences": "most recent broker-vs-ledger check results; a "
                           "repeat of the same kind+symbol is one ongoing "
                           "condition, not a new event",
            "divergence_kinds": {
                "ledger_open_no_position": "the trade ledger still marks this "
                    "symbol open but the broker shows no position (usually a "
                    "stop/target already filled and the ledger hasn't synced)",
                "approved_no_order": "a recommendation was approved but no "
                    "broker order exists for it",
            },
        },
    }


def strip_think(text: str) -> str:
    ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def call_ollama(context: dict, model: str = OLLAMA_MODEL) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Context JSON:\n" + json.dumps(context, indent=1)
                + "\n\nWrite today's briefing."},
        ],
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = json.load(resp)
    return strip_think(body["message"]["content"])


def generate_briefing(db_path: Path | str = DEFAULT_DB,
                      model: str = OLLAMA_MODEL) -> dict:
    context = assemble_context(db_path)
    text = call_ollama(context, model=model)
    if not text:
        raise RuntimeError("model returned empty briefing")
    row = {
        "as_of_date": context["data_as_of"],
        "kind": "daily_briefing",
        "generated_at": _now_iso(),
        "model": model,
        "commentary_md": text,
        "context_json": json.dumps(context),
    }
    with sqlite3.connect(db_path, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute(
            "INSERT INTO mm_llm_commentary "
            "(as_of_date, kind, generated_at, model, commentary_md, context_json) "
            "VALUES (:as_of_date, :kind, :generated_at, :model, :commentary_md, "
            ":context_json)", row)
        c.commit()
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--model", default=OLLAMA_MODEL)
    args = p.parse_args()
    if not args.generate:
        p.print_help()
        return 2
    row = generate_briefing(args.db, model=args.model)
    print(f"briefing stored: as_of={row['as_of_date']} model={row['model']} "
          f"chars={len(row['commentary_md'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
