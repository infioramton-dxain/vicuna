""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.llm import commentary


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "scratch.db"
    with sqlite3.connect(path) as c:
        for sql in (schema.MM_BARS, schema.MM_ACTIVE_PICKS,
                    schema.MM_ALPACA_POSITIONS, schema.MM_KRONOS_SIGNALS,
                    schema.MM_REGIME, schema.MM_BROKER_DIVERGENCES,
                    schema.MM_LLM_COMMENTARY):
            c.execute(sql)
        c.execute("CREATE TABLE sentiment (symbol TEXT, window_end TEXT, "
                  "sentiment REAL, article_count INTEGER, status TEXT)")
        c.execute("INSERT INTO mm_bars (symbol, date, open, high, low, close, "
                  "adj_close, volume, source) VALUES "
                  "('HPQ','2026-07-23',1,1,1,1,1,100,'t')")
        c.execute("INSERT INTO mm_alpaca_positions (symbol, side, qty, "
                  "avg_entry_price, market_value, cost_basis, current_price, "
                  "unrealized_pl, unrealized_plpc, change_today, exchange, "
                  "asset_class, snapshot_at) VALUES ('HPQ','long',222,22.85,"
                  "5473,5072,24.65,400.6,0.079,0,'NYSE','us_equity','now')")
        c.execute("INSERT INTO sentiment VALUES ('HPQ','2026-07-23',0.8,4,'ok')")
        c.commit()
    return path


def test_strip_think_closed_unclosed_and_absent():
    assert commentary.strip_think("<think>reasoning</think>Hello") == "Hello"
    assert commentary.strip_think("Hello\n<think>trailing never closed") == "Hello"
    assert commentary.strip_think("plain text") == "plain text"
    assert commentary.strip_think(
        "<think>a</think>keep<think>b</think> this") == "keep this"


def test_assemble_context_uses_only_db_numbers(db, monkeypatch):
    monkeypatch.setattr(commentary, "_fetch_account", lambda: None)
    ctx = commentary.assemble_context(db)
    assert ctx["data_as_of"] == "2026-07-23"
    assert ctx["account"] is None
    assert [p["symbol"] for p in ctx["positions"]] == ["HPQ"]
    assert ctx["latest_sentiment"][0]["sentiment"] == pytest.approx(0.8)
    assert ctx["kronos_forecasts_unproven"] == []
    assert "divergence_kinds" in ctx["notes"]


def test_generate_briefing_persists_with_audit_trail(db, monkeypatch):
    monkeypatch.setattr(commentary, "_fetch_account", lambda: None)
    monkeypatch.setattr(commentary, "call_ollama",
                        lambda ctx, model=None: "## Account\nequity missing")
    row = commentary.generate_briefing(db)
    with sqlite3.connect(db) as c:
        stored = c.execute(
            "SELECT as_of_date, model, commentary_md, context_json "
            "FROM mm_llm_commentary").fetchall()
    assert len(stored) == 1
    as_of, model, md, ctx_json = stored[0]
    assert as_of == "2026-07-23"
    assert md.startswith("## Account")
    assert json.loads(ctx_json)["positions"][0]["symbol"] == "HPQ"
    assert row["kind"] == "daily_briefing"


def test_generate_briefing_rejects_empty_output(db, monkeypatch):
    monkeypatch.setattr(commentary, "_fetch_account", lambda: None)
    monkeypatch.setattr(commentary, "call_ollama",
                        lambda ctx, model=None: "")
    with pytest.raises(RuntimeError):
        commentary.generate_briefing(db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM mm_llm_commentary").fetchone()[0] == 0
