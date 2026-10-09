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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB = REPO_ROOT / "macmini.db"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-r1:14b"
ARBITER_ID = "llm_arbiter_deepseek"
TOP_K = 30
TIMEOUT_S = 540

SYSTEM_PROMPT = """You are a quantitative signal arbiter. You receive, for
each stock symbol, the standardized (z-scored) 5-day return predictions of
several models plus market-regime context. Output STRICT JSON only:
{"scores": {"SYM": s, ...}} where s is your relative-attractiveness score
in [-1, 1] for the next 5 trading days (1 = most attractive long).
Score every symbol given. Consider model agreement/disagreement and
regime. No prose, no markdown, JSON only."""


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


def build_context(db_path: Path | str) -> tuple[str, pd.DataFrame, dict]:
    ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sig = pd.read_sql_query(
        "SELECT model_id, symbol, date, predicted_5d_return FROM mm_dl_signals "
        "WHERE model_id NOT IN (?, 'blend_bma') AND predicted_5d_return IS NOT NULL",
        ro, params=(ARBITER_ID,))
    kr = pd.read_sql_query(
        "SELECT model_id, symbol, date, predicted_5d_return FROM mm_kronos_signals "
        "WHERE COALESCE(quality_flag, 'ok') = 'ok'",
        ro)
    sig = pd.concat([sig, kr], ignore_index=True)
    if sig.empty:
        raise SystemExit("arbiter: no pool signals")
    as_of = sig["date"].max()
    sig = sig[sig["date"] == as_of]

    def _z(g: pd.Series) -> pd.Series:
        s = g.std(ddof=0)
        return (g - g.mean()) / (s if s > 1e-12 else 1.0)

    sig["z"] = sig.groupby("model_id")["predicted_5d_return"].transform(_z)
    wide = sig.pivot_table(index="symbol", columns="model_id", values="z",
                           aggfunc="mean")
    consensus = wide.mean(axis=1).sort_values()
    chosen = pd.concat([consensus.tail(TOP_K), consensus.head(TOP_K)]).index
    wide = wide.loc[wide.index.isin(chosen)].round(2)

    regime = pd.read_sql_query(
        "SELECT * FROM mm_regime ORDER BY as_of_date DESC LIMIT 1", ro)
    regime_d = regime.iloc[0].to_dict() if not regime.empty else {}
    payload = {
        "as_of": as_of,
        "regime": regime_d,
        "model_zscores_by_symbol": {
            sym: {m: (None if pd.isna(v) else v) for m, v in row.items()}
            for sym, row in wide.iterrows()},
    }
    return as_of, wide, payload


def call_arbiter(payload: dict, model: str = OLLAMA_MODEL) -> dict[str, float]:
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ],
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        out = json.load(resp)
    text = strip_think(out["message"]["content"])
    scores = json.loads(text).get("scores", {})
    clean = {}
    for sym, v in scores.items():
        try:
            clean[str(sym)] = max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            continue
    return clean


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--model", default=OLLAMA_MODEL)
    args = p.parse_args()

    as_of, wide, payload = build_context(args.db)
    scores = call_arbiter(payload, model=args.model)
    known = {s: v for s, v in scores.items() if s in wide.index}
    if len(known) < max(10, len(wide) // 2):
        print(f"arbiter: only {len(known)}/{len(wide)} symbols scored — "
              "refusing to write a degenerate cross-section", flush=True)
        return 51

    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(args.db, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        for sym, v in sorted(known.items()):
            c.execute(
                "INSERT OR REPLACE INTO mm_dl_signals (model_id, symbol, date, "
                "predicted_5d_return, predicted_class, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (ARBITER_ID, sym, as_of, v, 1 if v > 0 else 0, abs(v), now_iso))
            c.commit()
    print(f"arbiter: {len(known)} symbols scored as {ARBITER_ID} "
          f"(as_of {as_of})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
