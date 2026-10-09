""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

SYSTEM_PROMPT = """You are a swing-trading assistant ranking pre-screened
long picks. The picks have already passed walk-forward backtests with
deflated Sharpe + Bonferroni correction; your job is to apply same-day
context (regime, news, open positions, sector concentration) and decide
how to rank them.

Return ONLY a JSON object matching this schema — no prose, no markdown:

{
  "schema_version": "1.0",
  "as_of": "<ISO timestamp>",
  "ranked_picks": [
    {
      "rank": <int>, "symbol": "<TICKER>", "strategy_id": <int>,
      "conviction": <0..1 float>,
      "rationale": "<one sentence>"
    }
  ],
  "overrides": {
    "skip_today": ["<TICKER>"],
    "halve_size": ["<TICKER>"],
    "reason": "<one sentence>"
  },
  "regime_read": "<one short sentence>"
}

Be conservative: when in doubt, skip rather than override the model's pick."""


@dataclass
class LlmRanking:
    ranked_picks: list[dict[str, Any]]
    overrides: dict[str, Any]
    regime_read: str
    raw_response: str


def _build_user_prompt(picks: list[dict], quotes: dict, regime: dict | None,
                       positions: list, context: dict | None) -> str:
    return json.dumps({
        "picks": picks,
        "live_quotes": {k: {"bid": v.bid, "ask": v.ask, "mid": v.mid}
                        for k, v in (quotes or {}).items()},
        "regime": regime or {},
        "open_positions": [
            {"symbol": p.symbol, "qty": p.qty, "unrealized_pl": p.unrealized_pl}
            for p in (positions or [])
        ],
        "extra_context": context or {},
    }, indent=2, default=str)


def rank_picks(
    picks: list[dict],
    quotes: dict,
    regime: dict | None,
    positions: list,
    context: dict | None = None,
) -> LlmRanking:
    ""
    if not picks:
        return LlmRanking(ranked_picks=[], overrides={"skip_today": [], "halve_size": []},
                          regime_read="no picks today", raw_response="")
    user = _build_user_prompt(picks, quotes, regime, positions, context)
    try:
        if LLM_PROVIDER == "gemini":
            raw = _call_gemini(user)
        else:
            raw = _call_deepseek(user)
        parsed = json.loads(_strip_codefence(raw))
        return LlmRanking(
            ranked_picks=parsed.get("ranked_picks", []),
            overrides=parsed.get("overrides", {"skip_today": [], "halve_size": []}),
            regime_read=parsed.get("regime_read", ""),
            raw_response=raw,
        )
    except Exception as exc:
        log.warning("LLM ranking failed (%s); falling back to raw pick order", exc)
        fallback = [
            {"rank": i + 1, "symbol": p["symbol"], "strategy_id": p["strategy_id"],
             "conviction": 0.5, "rationale": "raw mm_active_picks order (LLM fallback)"}
            for i, p in enumerate(picks)
        ]
        return LlmRanking(
            ranked_picks=fallback,
            overrides={"skip_today": [], "halve_size": [], "reason": "LLM fallback"},
            regime_read="LLM unavailable; using deterministic order",
            raw_response="",
        )


def _strip_codefence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _call_deepseek(user_prompt: str) -> str:
    from openai import OpenAI
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def _call_gemini(user_prompt: str) -> str:
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    resp = model.generate_content(
        user_prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    )
    return resp.text or ""
