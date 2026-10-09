""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from macmini.config import DB_PATH
from macmini.strategies import STRATEGY_REGISTRY
from macmini.strategies.base import Signal, Strategy

KNOWN_NO_DATA: frozenset[int] = frozenset({
    49,
})


@pytest.fixture(scope="module")
def aapl_bars() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM mm_bars WHERE symbol='AAPL' ORDER BY date",
            c,
            parse_dates=["date"],
        )
    return df.set_index("date")


def _make_estimates_with_positive_revision(date_str: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"as_of_date": date_str, "fy0_eps_consensus": 6.0, "fy1_eps_consensus": 7.0,
         "eps_revision_30d": 0.10, "rating_buy": 25, "rating_hold": 5, "rating_sell": 0},
    ])


def _make_estimates_with_upgrade_cascade(date_pair: tuple[str, str]) -> pd.DataFrame:
    ""
    d0, d1 = date_pair
    return pd.DataFrame([
        {"as_of_date": d0, "fy0_eps_consensus": 6.0, "fy1_eps_consensus": 7.0,
         "eps_revision_30d": 0.0, "rating_buy": 20, "rating_hold": 10, "rating_sell": 2},
        {"as_of_date": d1, "fy0_eps_consensus": 6.0, "fy1_eps_consensus": 7.0,
         "eps_revision_30d": 0.0, "rating_buy": 25, "rating_hold": 10, "rating_sell": 2},
    ])


def _first_uptrend_bar_index(bars: pd.DataFrame, sma_window: int = 50, after: int = 60) -> int:
    ""
    sma = bars["close"].rolling(sma_window).mean()
    candidates = bars.index[bars["close"] > sma]
    candidates = candidates[candidates >= bars.index[after]]
    return int(bars.index.get_indexer([candidates[0]])[0])


def _bars_ending_at_uptrend_peak(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    half = len(bars) // 2
    peak_pos = int(bars["close"].iloc[half:].idxmax().value)
    peak_idx = bars.index.get_indexer([pd.Timestamp(peak_pos)])[0]
    return bars.iloc[: peak_idx + 1]


def _synthetic_ascending_triangle_bars() -> pd.DataFrame:
    ""
    n_warmup = 55
    n_triangle = 40
    rng = np.random.default_rng(11)
    warmup = 100 + rng.normal(0, 0.3, n_warmup)
    triangle_close = np.linspace(102, 109, n_triangle) + rng.normal(0, 0.2, n_triangle)
    closes = np.concatenate([warmup, triangle_close, [115.0]])
    n = len(closes)
    highs = closes.copy()
    lows = closes.copy()
    for i in range(n_warmup, n_warmup + n_triangle):
        highs[i] = 110 - rng.uniform(0, 0.5)
        lows[i] = closes[i] - 1.5
    highs[-1] = 116
    lows[-1] = 113
    opens = closes - rng.uniform(0, 0.5, n)
    vol = np.full(n, 1_000_000.0)
    vol[-1] = 3_000_000.0
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "adj_close": closes, "volume": vol},
        index=idx,
    )


def _synthetic_panic_setup() -> tuple[pd.DataFrame, pd.DataFrame]:
    ""
    rng = np.random.default_rng(45)
    n_up = 250
    up = np.linspace(100, 200, n_up) + rng.normal(0, 0.4, n_up)
    crash = [up[-1] * 0.97, up[-1] * 0.93, up[-1] * 0.88]
    closes = np.concatenate([up, crash])
    n = len(closes)
    vol = np.full(n, 1_000_000.0)
    vol[-3:] = 2_500_000.0
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    bars = pd.DataFrame(
        {"open": closes + 0.2, "high": closes + 1.0, "low": closes - 1.0,
         "close": closes, "adj_close": closes, "volume": vol},
        index=idx,
    )
    vix = np.full(n, 15.0)
    vix[-3:] = 32.0
    market = pd.DataFrame(
        {"date": idx.strftime("%Y-%m-%d"), "vix_close": vix})
    return bars, market


def _accelerating_revenue_fundamentals(period_dates: list[str]) -> pd.DataFrame:
    ""
    revs = [10e9, 10e9, 10e9, 10e9, 11e9, 13e9, 16e9, 20e9]
    rows = []
    for d, rev in zip(period_dates[-8:], revs):
        rows.append({"period_end": d, "revenue": rev, "earnings": 2e9, "eps": 5.0,
                     "operating_cash_flow": 3e9, "free_cash_flow": 2.5e9,
                     "total_debt": 5e8, "total_equity": 1e10, "ebitda": 4e9,
                     "market_cap": 1e11, "pe_ratio": 20, "pb_ratio": 5, "ev_ebitda": 12,
                     "roic": 0.22, "shares_outstanding": 2e9, "dividend_yield": 0.02,
                     "buyback_yield": 0.05, "capex": 5e8, "f_score": 8,
                     "source": "synthetic", "fetched_at": "2026-06-30T00:00:00Z"})
    return pd.DataFrame(rows)


def _deleveraging_fundamentals(period_dates: list[str]) -> pd.DataFrame:
    ""
    rows = []
    for i, d in enumerate(period_dates[-8:]):
        debt = 3e9 - (i / 7) * 2e9
        rows.append({"period_end": d, "revenue": 1e10, "earnings": 1e9, "eps": 5.0,
                     "operating_cash_flow": 1.5e9, "free_cash_flow": 1e9,
                     "total_debt": debt, "total_equity": 1e10, "ebitda": 1e9,
                     "market_cap": 1e11, "pe_ratio": 20, "pb_ratio": 5, "ev_ebitda": 12,
                     "roic": 0.10, "shares_outstanding": 2e9, "dividend_yield": 0.02,
                     "buyback_yield": 0.05, "capex": 5e8, "f_score": 8,
                     "source": "synthetic", "fetched_at": "2026-06-30T00:00:00Z"})
    return pd.DataFrame(rows)


def _make_squeeze_setup(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ""
    high = bars["high"]
    vol = bars["volume"]
    vol_avg = vol.rolling(50).mean()
    twenty_high = high.rolling(20).max()
    cond = (bars["close"] >= twenty_high) & (vol >= 2 * vol_avg)
    matches = bars.index[cond.fillna(False)]
    if matches.empty:
        return pd.DataFrame(), -1
    d = matches[len(matches) // 2]
    si_date = (d - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    df = pd.DataFrame([{"date": si_date, "short_interest_pct": 0.25, "days_to_cover": 8.0}])
    return df, int(bars.index.get_indexer([d])[0])


def _make_insider_cluster(anchor: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for d, name in enumerate(["A", "B", "C", "D"]):
        rows.append({
            "transaction_date": (anchor - pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
            "insider_name": name, "insider_title": "CEO" if d == 0 else None,
            "transaction_type": "P", "shares": 1000, "value_usd": 150_000.0,
        })
    return pd.DataFrame(rows)


def _make_insider_long_history(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    rows = []
    dates = bars.index
    n = min(len(dates), 252)
    for i in range(0, n, 30):
        rows.append({
            "transaction_date": dates[i].strftime("%Y-%m-%d"),
            "insider_name": f"bg_{i}", "insider_title": None,
            "transaction_type": "P", "shares": 100, "value_usd": 5_000.0,
        })
    anchor = dates[-1]
    rows.extend(_make_insider_cluster(anchor).to_dict(orient="records"))
    return pd.DataFrame(rows)


def _make_fundamentals_quality(period_dates: list[str]) -> pd.DataFrame:
    ""
    rows = []
    for i, d in enumerate(period_dates):
        rows.append({
            "period_end": d, "revenue": 1e10 + i * 1e8, "earnings": 2e9 + i * 1e7,
            "eps": 5.0 + 0.01 * i, "operating_cash_flow": 3e9, "free_cash_flow": 2.5e9,
            "total_debt": 5e8, "total_equity": 1e10, "ebitda": 4e9, "market_cap": 1e11,
            "pe_ratio": 20, "pb_ratio": 5, "ev_ebitda": 12, "roic": 0.22,
            "shares_outstanding": 2e9, "dividend_yield": 0.02, "buyback_yield": 0.05,
            "capex": 5e8 - i * 1e7,
            "f_score": 8, "source": "synthetic", "fetched_at": "2026-06-30T00:00:00Z",
        })
    return pd.DataFrame(rows)


def _make_earnings_with_big_beat(report_date: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "report_date": report_date,
        "eps_actual": 2.5, "eps_estimate": 2.0, "eps_surprise_pct": 25.0,
        "revenue_actual": 1.1e10, "revenue_estimate": 1.0e10, "revenue_surprise_pct": 10.0,
        "guidance_raised": 1,
    }])


def _make_earnings_pair_for_tone(d1: str, d2: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"report_date": d1, "eps_actual": 1.0, "eps_estimate": 1.0, "eps_surprise_pct": 0.0,
         "revenue_actual": 1e10, "revenue_estimate": 1e10, "revenue_surprise_pct": 0.0,
         "guidance_raised": 0},
        {"report_date": d2, "eps_actual": 2.5, "eps_estimate": 2.0, "eps_surprise_pct": 25.0,
         "revenue_actual": 1.1e10, "revenue_estimate": 1e10, "revenue_surprise_pct": 10.0,
         "guidance_raised": 1},
    ])


def _make_sentiment_shock(date_str: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": date_str, "sentiment_mean": 0.75, "sentiment_confidence_mean": 0.8,
         "materiality_max": 0.8, "article_count_sum": 50},
    ])


def _make_sentiment_for_tone(d1: str, d2: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d1, "sentiment_mean": 0.10, "sentiment_confidence_mean": 0.5,
         "materiality_max": 0.3, "article_count_sum": 10},
        {"date": d2, "sentiment_mean": 0.80, "sentiment_confidence_mean": 0.8,
         "materiality_max": 0.8, "article_count_sum": 60},
    ])


def _make_sentiment_volume_spike(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    dates = bars.index
    rows = []
    for d in dates[-40:]:
        rows.append({"date": d.strftime("%Y-%m-%d"), "sentiment_mean": 0.2,
                     "sentiment_confidence_mean": 0.5, "materiality_max": 0.3,
                     "article_count_sum": 5})
    rows[-1].update({"article_count_sum": 200, "sentiment_mean": 0.7})
    return pd.DataFrame(rows)


def _make_short_interest_squeeze(bars: pd.DataFrame) -> pd.DataFrame:
    d = bars.index[-5]
    return pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"), "short_interest_pct": 0.25, "days_to_cover": 8.0,
    }])


def _make_cot_cycle() -> pd.DataFrame:
    ""
    dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    net = list(np.linspace(-50_000, -20_000, 59)) + [50_000]
    return pd.DataFrame({
        "commodity": "gold",
        "report_date": [d.strftime("%Y-%m-%d") for d in dates],
        "commercial_long": [100_000] * 60,
        "commercial_short": [100_000 - n for n in net],
        "commercial_net": net,
        "noncommercial_long": [50_000] * 60,
        "noncommercial_short": [40_000] * 60,
    })


def _make_etf_flow_spike() -> pd.DataFrame:
    dates = pd.date_range("2025-01-03", periods=20, freq="W-FRI")
    flows = [1e6] * 19 + [50e6]
    return pd.DataFrame({
        "etf_ticker": "XLK",
        "week_end": [d.strftime("%Y-%m-%d") for d in dates],
        "net_flow_usd": flows,
        "aum_usd": [1e10] * 20,
    })


def _make_staples_dividends(dts: pd.DatetimeIndex) -> pd.DataFrame:
    ""
    ex_dates = pd.date_range(dts[0], dts[-1], freq="QS-JAN")
    return pd.DataFrame({
        "symbol": "KO",
        "ex_date": [d.strftime("%Y-%m-%d") for d in ex_dates],
        "rate": 2.5,
        "special": 0,
        "is_foreign": 0,
        "source": "synthetic",
        "fetched_at": "2026-07-29T00:00:00Z",
    })


def _make_staples_pricing_power(period_dates: list[str]) -> pd.DataFrame:
    ""
    rows = []
    for i, d in enumerate(period_dates):
        rows.append({
            "period_end": d, "revenue": 1e10 * (1.015 ** i), "earnings": 2e9,
            "eps": 5.0 * (1.03 ** i), "operating_cash_flow": 3e9,
            "free_cash_flow": 2.5e9, "total_debt": 5e8, "total_equity": 1e10,
            "ebitda": None, "market_cap": None, "pe_ratio": None,
            "pb_ratio": None, "ev_ebitda": None, "roic": None,
            "shares_outstanding": None, "dividend_yield": None,
            "buyback_yield": None, "capex": 5e8, "f_score": None,
            "source": "synthetic", "fetched_at": "2026-07-29T00:00:00Z",
        })
    return pd.DataFrame(rows)


def _make_staples_annual(dts: pd.DatetimeIndex, eps: float = 20.0) -> pd.DataFrame:
    ""
    year_ends = pd.date_range(dts[0], dts[-1], freq="YE")
    return pd.DataFrame([{
        "period_end": d.strftime("%Y-%m-%d"), "revenue": 8e10, "earnings": 1.6e10,
        "eps": eps, "operating_cash_flow": 1.8e10, "free_cash_flow": 1.5e10,
        "total_debt": 2e10, "total_equity": 5e10, "shares_outstanding": 8e8,
        "capex": 3e9, "source": "synthetic", "fetched_at": "2026-07-29T00:00:00Z",
    } for d in year_ends])


def _make_staples_flow_weeks(dts: pd.DatetimeIndex) -> pd.DataFrame:
    ""
    weeks = pd.date_range(dts[0], dts[-1], freq="W-FRI")
    flows = [60e6 if i % 8 == 7 else 1e6 for i in range(len(weeks))]
    return pd.DataFrame({
        "etf_ticker": "XLP",
        "week_end": [d.strftime("%Y-%m-%d") for d in weeks],
        "net_flow_usd": flows,
        "aum_usd": [1.5e10] * len(weeks),
    })


def _make_peer_close(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    rng = np.random.default_rng(7)
    base = bars["close"].copy()
    out = {"AAPL": base}
    for sym in ("PEER1", "PEER2", "PEER3", "PEER4"):
        out[sym] = base * (1 + rng.normal(0, 0.001, len(base))) * 0.95
    return pd.DataFrame(out, index=bars.index)


def _build_kwargs(strategy_id: int, bars: pd.DataFrame) -> dict:
    dts = bars.index
    last = dts[-1]
    quarter_ends = pd.date_range(end=last, periods=21, freq="QE").strftime("%Y-%m-%d").tolist()

    if strategy_id == 20:
        return {"peers_close": _make_peer_close(bars)}
    if strategy_id == 31:
        return {"fundamentals": _make_fundamentals_quality(quarter_ends)}
    if strategy_id == 32:
        return {"fundamentals": _make_fundamentals_quality(quarter_ends), "sector_z": -1.5}
    if strategy_id == 33:
        bars_peak = _bars_ending_at_uptrend_peak(bars)
        peak_q = pd.date_range(end=bars_peak.index[-1], periods=20, freq="QE").strftime("%Y-%m-%d").tolist()
        return {"fundamentals": _make_fundamentals_quality(peak_q), "_bars_override": bars_peak}
    if strategy_id == 34:
        return {"estimates": _make_estimates_with_positive_revision(dts[100].strftime("%Y-%m-%d"))}
    if strategy_id == 35:
        return {"earnings": _make_earnings_with_big_beat(dts[200].strftime("%Y-%m-%d"))}
    if strategy_id == 36:
        bars_peak = _bars_ending_at_uptrend_peak(bars)
        peak_q = pd.date_range(end=bars_peak.index[-1], periods=20, freq="QE").strftime("%Y-%m-%d").tolist()
        return {"fundamentals": _accelerating_revenue_fundamentals(peak_q), "_bars_override": bars_peak}
    if strategy_id == 37:
        return {"fundamentals": _make_fundamentals_quality(quarter_ends)}
    if strategy_id == 38:
        return {"fundamentals": _deleveraging_fundamentals(quarter_ends)}
    if strategy_id == 39:
        return {"fundamentals": _make_fundamentals_quality(quarter_ends), "market_cap": 2.5e10}
    if strategy_id == 40:
        return {"fundamentals": _make_fundamentals_quality(quarter_ends)}
    if strategy_id == 41:
        return {"insider_tx": _make_insider_cluster(dts[210])}
    if strategy_id == 42:
        return {"valuation_pct": 0.05, "sector_rs_positive": True}
    if strategy_id == 43:
        prior_close = bars["close"].shift(1)
        up_days = bars[bars["close"] >= prior_close * 1.02]
        d = up_days.index[0]
        return {"sentiment_daily": _make_sentiment_shock(d.strftime("%Y-%m-%d"))}
    if strategy_id == 44:
        d1 = dts[200].strftime("%Y-%m-%d")
        d2 = dts[400 if len(dts) > 410 else -10].strftime("%Y-%m-%d")
        return {
            "earnings": _make_earnings_pair_for_tone(d1, d2),
            "sentiment_daily": _make_sentiment_for_tone(d1, d2),
        }
    if strategy_id == 45:
        df, _ = _make_squeeze_setup(bars)
        if df.empty:
            return {"short_interest": _make_short_interest_squeeze(bars)}
        return {"short_interest": df}
    if strategy_id == 46:
        d1_idx = 200
        sm50 = bars["close"].rolling(50).mean()
        candidates = bars.index[(bars["close"] > sm50)]
        candidates = candidates[candidates > bars.index[d1_idx + 5]]
        d2 = candidates[0] if not candidates.empty else bars.index[d1_idx + 50]
        return {"estimates": _make_estimates_with_upgrade_cascade(
            (bars.index[d1_idx].strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")))}
    if strategy_id == 47:
        return {"insider_tx": _make_insider_long_history(bars)}
    if strategy_id == 48:
        up_idx = _first_uptrend_bar_index(bars, sma_window=20, after=40)
        window_start = max(0, up_idx - 39)
        window = bars.iloc[window_start: up_idx + 1].index
        rows = [
            {"date": d.strftime("%Y-%m-%d"), "sentiment_mean": 0.2,
             "sentiment_confidence_mean": 0.5, "materiality_max": 0.3,
             "article_count_sum": 5}
            for d in window
        ]
        if rows:
            rows[-1].update({"article_count_sum": 200, "sentiment_mean": 0.7})
        return {"sentiment_daily": pd.DataFrame(rows)}
    if strategy_id == 49:
        return {}
    if strategy_id == 50:
        return {"etf_flows": _make_etf_flow_spike(), "rs_pctile": 0.95}
    if strategy_id == 51:
        return {"cot": _make_cot_cycle()}
    if strategy_id == 53:
        anchor_idx = min(300, len(dts) - 1)
        return {"estimates": _make_estimates_with_positive_revision(dts[anchor_idx].strftime("%Y-%m-%d"))}
    if strategy_id == 54:
        return {"momentum_pct": 0.9, "vol_pct": 0.3}

    if strategy_id in range(56, 66):
        kw: dict = {"_symbol_override": "KO"}
        if strategy_id == 56:
            kw["dividends"] = _make_staples_dividends(dts)
        elif strategy_id == 57:
            kw["fundamentals"] = _make_fundamentals_quality(quarter_ends)
        elif strategy_id == 59:
            kw.update(fundamentals=_make_fundamentals_quality(quarter_ends),
                      fundamentals_annual=_make_staples_annual(dts),
                      dividends=_make_staples_dividends(dts))
        elif strategy_id == 60:
            kw["fundamentals"] = _make_fundamentals_quality(quarter_ends)
        elif strategy_id == 61:
            kw["fundamentals"] = _make_staples_pricing_power(quarter_ends)
        elif strategy_id == 62:
            kw["etf_flows"] = _make_staples_flow_weeks(dts)
        elif strategy_id == 63:
            kw.update(fundamentals=_make_fundamentals_quality(quarter_ends),
                      dividends=_make_staples_dividends(dts))
        elif strategy_id == 64:
            kw["earnings"] = _make_earnings_with_big_beat(dts[200].strftime("%Y-%m-%d"))
        return kw

    if strategy_id == 66:
        crash_bars, market = _synthetic_panic_setup()
        return {"market": market, "_bars_override": crash_bars}
    return {}


class TestRegistry:
    def test_all_66_strategies_registered(self) -> None:
        missing = set(range(1, 67)) - STRATEGY_REGISTRY.keys()
        assert not missing, f"missing strategy ids: {sorted(missing)}"

    def test_every_class_subclasses_Strategy(self) -> None:
        for sid, cls in STRATEGY_REGISTRY.items():
            assert issubclass(cls, Strategy), f"#{sid} {cls.__name__} not a Strategy"

    def test_every_class_declares_required_attrs(self) -> None:
        for sid, cls in STRATEGY_REGISTRY.items():
            for attr in ("id", "name", "category", "data_required"):
                assert hasattr(cls, attr), f"#{sid} missing {attr}"

    def test_categories_match_spec_counts(self) -> None:
        ""
        from collections import Counter
        counts = Counter(c.category for c in STRATEGY_REGISTRY.values())
        expected = {"trend": 8, "mean_reversion": 9, "breakout": 7, "pattern": 5,
                    "volatility": 3, "mtf": 3, "fundamental": 16, "sentiment": 11,
                    "composite": 4}
        for cat, n in expected.items():
            assert counts.get(cat) == n, f"{cat}: got {counts.get(cat)}, expected {n}"


@pytest.mark.parametrize("strategy_id", sorted(STRATEGY_REGISTRY))
def test_strategy_emits_signal(strategy_id: int, aapl_bars: pd.DataFrame) -> None:
    cls = STRATEGY_REGISTRY[strategy_id]
    kwargs = _build_kwargs(strategy_id, aapl_bars)
    bars_to_use = aapl_bars
    if strategy_id == 24:
        bars_to_use = _synthetic_ascending_triangle_bars()
    if "_bars_override" in kwargs:
        bars_to_use = kwargs.pop("_bars_override")
    symbol = kwargs.pop("_symbol_override", "AAPL")
    sigs = cls().generate_signals(symbol, bars_to_use, **kwargs)
    if strategy_id in KNOWN_NO_DATA:
        return
    assert sigs, f"#{strategy_id} {cls.name} emitted no signals on synthetic-trigger input"
    for s in sigs:
        assert isinstance(s, Signal)
        assert s.action in {"enter", "exit"}
        assert 0 <= s.conviction <= 1
