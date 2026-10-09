""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

os.environ["AGENT_SHIPPED_DB"] = str(PROJ / "macmini.db")
os.environ.setdefault("AGENT_LLM_PROVIDER", "deepseek")


def test_stub_client_no_creds_required() -> None:
    ""
    from agent.alpaca_client import StubAlpacaClient

    cli = StubAlpacaClient()
    assert cli.get_account().cash > 0
    quotes = cli.get_quotes(["AAPL", "MSFT"])
    assert set(quotes.keys()) == {"AAPL", "MSFT"}
    assert quotes["AAPL"].mid > 0


def test_risk_overlay_caps_per_trade_risk() -> None:
    from agent.alpaca_client import AccountSnapshot, Position, Quote
    from agent.config import MAX_RISK_PER_TRADE_PCT
    from agent.risk_overlay import filter_and_size

    quote = Quote(symbol="AAPL", bid=100.0, ask=100.5, mid=100.25)
    pick_meta = {"AAPL": {"strategy_id": 1, "sector": "Software",
                          "suggested_stop_loss_pct": 0.08, "suggested_target_pct": 0.20}}
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": 1, "conviction": 0.8,
               "rationale": "test"}]
    account = AccountSnapshot(cash=100_000.0, equity=100_000.0,
                              buying_power=100_000.0, portfolio_value=100_000.0)
    sized = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": quote},
        regime={"fomc_within_3d": 0},
    )
    assert sized, "expected one accepted order"
    order = sized[0]
    risk_per_share = order.entry_price - order.stop_price
    total_risk = order.qty * risk_per_share
    assert total_risk <= account.equity * MAX_RISK_PER_TRADE_PCT + 1e-6


def test_position_notional_capped_d56() -> None:
    ""
    from agent.alpaca_client import AccountSnapshot, Quote
    from agent.config import MAX_POSITION_NOTIONAL_PCT, MAX_POSITION_NOTIONAL_USD
    from agent.risk_overlay import filter_and_size

    quote = Quote(symbol="AAPL", bid=100.0, ask=100.5, mid=100.25)
    pick_meta = {"AAPL": {"strategy_id": 1, "sector": "Software",
                          "suggested_stop_loss_pct": 0.05, "suggested_target_pct": 0.10}}
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": 1, "conviction": 0.9,
               "rationale": "x"}]
    account = AccountSnapshot(cash=1_000_000.0, equity=1_000_000.0,
                              buying_power=1_000_000.0, portfolio_value=1_000_000.0)
    sized = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": quote},
        regime={"fomc_within_3d": 0},
    )
    assert sized, "expected one accepted order"
    order = sized[0]
    cap = min(MAX_POSITION_NOTIONAL_USD, MAX_POSITION_NOTIONAL_PCT * account.equity)
    assert order.qty * order.entry_price <= cap + order.entry_price


def test_sector_cap_25pct_d56() -> None:
    ""
    from agent.alpaca_client import AccountSnapshot, Position, Quote
    from agent.risk_overlay import filter_and_size

    positions = [Position(symbol="MSFT", qty=100, market_value=24_000.0,
                          unrealized_pl=0.0, avg_entry_price=240.0)]
    pick_meta = {
        "MSFT": {"strategy_id": 2, "sector": "Software"},
        "AAPL": {"strategy_id": 1, "sector": "Software",
                 "suggested_stop_loss_pct": 0.08, "suggested_target_pct": 0.20},
    }
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": 1, "conviction": 0.8,
               "rationale": "x"}]
    account = AccountSnapshot(cash=76_000.0, equity=100_000.0,
                              buying_power=76_000.0, portfolio_value=100_000.0)
    sized = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=positions,
        quotes={"AAPL": Quote("AAPL", 100.0, 100.5, 100.25)},
        regime={"fomc_within_3d": 0},
    )
    assert sized == [], "Software pick should be vetoed by the 25% sector cap"


def test_risk_overlay_blackout_skip(tmp_path: Path) -> None:
    from agent.alpaca_client import AccountSnapshot, Quote
    from agent.risk_overlay import filter_and_size

    pick_meta = {"AAPL": {"strategy_id": 1, "sector": "Software"}}
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": 1, "conviction": 0.8, "rationale": "x"}]
    account = AccountSnapshot(cash=1000.0, equity=1000.0, buying_power=1000.0, portfolio_value=1000.0)
    sized = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": Quote("AAPL", 100, 100.5, 100.25)},
        regime={}, earnings_blackout={"AAPL"},
    )
    assert sized == [], "AAPL should be vetoed by earnings blackout"


def test_vix_high_regime_halves_size() -> None:
    ""
    from agent.alpaca_client import AccountSnapshot, Quote
    from agent.risk_overlay import filter_and_size

    pick_meta = {"AAPL": {"strategy_id": 1, "sector": "Software",
                          "suggested_stop_loss_pct": 0.08, "suggested_target_pct": 0.20}}
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": 1, "conviction": 0.8, "rationale": "x"}]
    account = AccountSnapshot(cash=10_000, equity=10_000, buying_power=10_000, portfolio_value=10_000)
    quote = Quote("AAPL", 100.0, 100.5, 100.25)

    normal = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": quote},
        regime={"vix_regime": "normal", "fomc_within_3d": 0},
    )
    high_vix = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": quote},
        regime={"vix_regime": "high", "fomc_within_3d": 0},
    )
    assert normal and high_vix
    assert high_vix[0].qty == normal[0].qty // 2


def test_pead_strategy_exempt_from_blackout() -> None:
    from agent.alpaca_client import AccountSnapshot, Quote
    from agent.risk_overlay import PEAD_STRATEGY_ID, filter_and_size

    pick_meta = {"AAPL": {"strategy_id": PEAD_STRATEGY_ID, "sector": "Software"}}
    ranked = [{"rank": 1, "symbol": "AAPL", "strategy_id": PEAD_STRATEGY_ID,
               "conviction": 0.8, "rationale": "PEAD"}]
    account = AccountSnapshot(cash=100_000.0, equity=100_000.0,
                              buying_power=100_000.0, portfolio_value=100_000.0)
    sized = filter_and_size(
        ranked_picks=ranked, pick_meta=pick_meta, overrides=None,
        account=account, positions=[], quotes={"AAPL": Quote("AAPL", 100, 100.5, 100.25)},
        regime={}, earnings_blackout={"AAPL"},
    )
    assert sized, "PEAD (#35) is exempt from earnings blackout"


def test_llm_router_fallback_without_keys(monkeypatch) -> None:
    ""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    import importlib
    import agent.llm_router as router
    importlib.reload(router)

    picks = [{"symbol": "AAPL", "strategy_id": 1, "rationale": "x"}]
    out = router.rank_picks(picks=picks, quotes={}, regime=None, positions=[])
    assert out.ranked_picks and out.ranked_picks[0]["symbol"] == "AAPL"
    assert "fallback" in out.overrides.get("reason", "").lower() or out.ranked_picks


def test_run_once_dry_run_against_live_db(tmp_path: Path, monkeypatch) -> None:
    ""
    monkeypatch.setenv("AGENT_DRY_RUN", "1")
    monkeypatch.setenv("AGENT_TRADES_DB", str(tmp_path / "trades.db"))
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path / "logs"))
    import importlib
    import agent.config
    importlib.reload(agent.config)
    import agent.db
    importlib.reload(agent.db)
    import agent.main as agent_main
    importlib.reload(agent_main)

    out = agent_main.run_once(dry_run=True, as_of="2026-06-26")
    assert isinstance(out, dict)
    assert "submitted" in out
    if out["picks"] > 0:
        with sqlite3.connect(tmp_path / "trades.db") as c:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"trades", "agent_decisions"}.issubset(tables)
    else:
        from agent.db import open_trades
        with open_trades() as c:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"trades", "agent_decisions"}.issubset(tables)
