""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.data.schema import MM_TITAN_NOTES

log = logging.getLogger(__name__)


DCF_PARAMS = {
    "horizon_years": 5,
    "terminal_growth": 0.025,
    "growth_floor": -0.05,
    "growth_cap": 0.15,
    "scenarios": {
        "bear": (0.10, -0.03),
        "base": (0.09, 0.00),
        "bull": (0.08, +0.03),
    },
}
SEC_MAX_AGE_DAYS = 460
CAP_CHECK_MAX_RATIO = 2.0
INSIDER_WINDOW_DAYS = 90
INSIDER_LAG_DAYS = 2
KRONOS_MAX_AGE_DAYS = 7
NO_DCF_FINNHUB_SECTORS = frozenset({"Banking"})

REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet")
LONG_DEBT_TAG_SETS = (
    ("LongTermDebt",),
    ("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
    ("LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",),
    ("LongTermDebtAndCapitalLeaseObligations",
     "LongTermDebtAndCapitalLeaseObligationsCurrent"),
)
SHORT_DEBT_TAGS = ("ShortTermBorrowings", "CommercialPaper")
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets")
CASH_TAGS = ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
SHARE_TAGS = ("WeightedAverageNumberOfDilutedSharesOutstanding",
              "EntityCommonStockSharesOutstanding",
              "CommonStockSharesOutstanding")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_TIMEOUT_S = 300
MAX_LLM_NOTES = 5
SUMMARY_MAX_CHARS = 300

SYSTEM_PROMPT = """You write a short note about one listed company, named by \
its ticker symbol. The reader is smart but not finance-deep.

Hard rules:
- Use ONLY the figures in the FACTS list, copied exactly as written. Never \
compute, round, estimate or invent a number.
- Give no verdict and no advice: never say buy, sell, hold, avoid, \
undervalued or overvalued, and never recommend anything. Describe; do not judge.
- The DCF range comes from fixed assumptions, not a forecast. Say so.
- Kronos forecasts are experimental and unproven. Say so if you mention them.
- If the facts say something is missing or skipped, say it is missing.

Format, exactly:
SUMMARY: one paragraph, at most 280 characters.
DETAIL: 80-140 words in plain English.
WRONG IF: one or two sentences on what would make this picture wrong."""

_VERDICT_RE = re.compile(
    r"\b(buy|sell|hold|avoid|overweight|underweight|undervalued?|overvalued?"
    r"|recommend\w*|vicuna)\b", re.IGNORECASE)
_RATING_PHRASE_RE = re.compile(r"\b(buy|hold|sell)\s+ratings?\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Fact:
    tag: str
    unit: str
    start: str
    end: str
    filed: str
    value: float

    @property
    def days(self) -> int:
        if not self.start:
            return 0
        return (date.fromisoformat(self.end) - date.fromisoformat(self.start)).days


def load_facts(c, symbol: str, as_of: str) -> dict[str, list[Fact]]:
    ""
    latest: dict[tuple, Fact] = {}
    for tag, unit, start, end, filed, value in c.execute(
        "SELECT tag, unit, period_start, period_end, filed, value "
        "FROM mm_sec_facts WHERE symbol = ? AND filed <= ? ORDER BY filed",
        (symbol, as_of),
    ):
        latest[(tag, unit, start or "", end)] = Fact(
            tag, unit, start or "", end, filed, float(value))
    out: dict[str, list[Fact]] = {}
    for f in latest.values():
        out.setdefault(f.tag, []).append(f)
    return out


def ttm(facts: list[Fact]) -> tuple[float, str, str] | None:
    ""
    annual = [f for f in facts if 340 <= f.days <= 380]
    if not annual:
        return None
    fy = max(annual, key=lambda f: (f.end, f.filed))
    fy_end = date.fromisoformat(fy.end)
    ytd = [f for f in facts
           if f.start and f.end > fy.end and f.days < 340
           and abs((date.fromisoformat(f.start) - fy_end).days - 1) <= 10]
    if ytd:
        cur = max(ytd, key=lambda f: (f.end, f.filed))
        cur_end = date.fromisoformat(cur.end)
        prior = [f for f in facts
                 if f.start and abs(f.days - cur.days) <= 10
                 and abs((cur_end - date.fromisoformat(f.end)).days - 365) <= 10]
        if prior:
            p = max(prior, key=lambda f: f.filed)
            return (fy.value + cur.value - p.value, cur.end,
                    max(fy.filed, cur.filed, p.filed))
    return fy.value, fy.end, fy.filed


def annual_series(facts: list[Fact]) -> list[Fact]:
    ""
    by_end: dict[str, Fact] = {}
    for f in sorted(facts, key=lambda f: f.filed):
        if 340 <= f.days <= 380:
            by_end[f.end] = f
    return [by_end[k] for k in sorted(by_end)]


def latest_instant(facts: list[Fact]) -> Fact | None:
    inst = [f for f in facts if not f.start]
    return max(inst, key=lambda f: (f.end, f.filed)) if inst else None


def _first(facts: dict[str, list[Fact]], tags: tuple[str, ...], fn):
    ""
    best = None
    for tag in tags:
        res = fn(facts.get(tag, []))
        if res is None:
            continue
        end = res.end if isinstance(res, Fact) else res[1]
        if best is None or end > best[2]:
            best = (tag, res, end)
    return (best[0], best[1]) if best else None


def dcf_per_share(fcf0: float, growth: float, wacc: float, net_debt: float,
                  shares: float, params: dict = DCF_PARAMS) -> float:
    ""
    g_t = params["terminal_growth"]
    n = params["horizon_years"]
    pv, fcf = 0.0, fcf0
    for t in range(1, n + 1):
        fcf *= 1.0 + growth
        pv += fcf / (1.0 + wacc) ** t
    pv += fcf * (1.0 + g_t) / (wacc - g_t) / (1.0 + wacc) ** n
    return (pv - net_debt) / shares


class Note:
    ""

    def __init__(self, as_of: str, symbol: str) -> None:
        self.as_of, self.symbol = as_of, symbol
        self.rows: list[dict] = []

    def add(self, section: str, metric: str, value, unit: str | None,
            table: str | None, column: str | None, asof: str | None,
            text: str | None = None) -> None:
        self.rows.append({
            "section": section, "metric": metric,
            "value": None if value is None else float(value), "unit": unit,
            "source_table": table, "source_column": column,
            "source_asof": asof, "narrative": text,
        })

    def skip(self, section: str, reason: str) -> None:
        self.add(section, "skipped", None, None, None, None, None, reason)

    def get(self, section: str, metric: str) -> float | None:
        for r in self.rows:
            if r["section"] == section and r["metric"] == metric:
                return r["value"]
        return None


def _sec_sections(c, note: Note, close: float | None) -> None:
    sym, as_of = note.symbol, note.as_of
    facts = load_facts(c, sym, as_of)
    T = "mm_sec_facts"
    if not facts:
        note.skip("sec", "no SEC company facts on file (ETF or foreign filer)")
        note.skip("dcf", "no SEC company facts on file")
        return
    units = {f.unit for fs in facts.values() for f in fs
             if f.unit not in ("shares",) and "/" not in f.unit}
    if units - {"USD"}:
        cur = sorted(units - {"USD"})[0]
        note.skip("sec", f"filer reports in {cur}; figures not converted")
        note.skip("dcf", f"filer reports in {cur}")
        return
    newest = max(f.end for fs in facts.values() for f in fs)
    if (date.fromisoformat(as_of) - date.fromisoformat(newest)).days > SEC_MAX_AGE_DAYS:
        note.skip("sec", f"latest SEC period is {newest} — too old to use")
        note.skip("dcf", "SEC data stale")
        return

    def flow(metric: str, tags: tuple[str, ...]):
        hit = _first(facts, tags, ttm)
        if hit is None:
            return None
        tag, (val, _end, filed) = hit
        note.add("sec", metric, val, "USD", T, tag, filed)
        return val

    rev = flow("revenue_ttm", REVENUE_TAGS)
    op = flow("operating_income_ttm", ("OperatingIncomeLoss",))
    ni = flow("net_income_ttm", ("NetIncomeLoss",))
    cfo = flow("cfo_ttm", ("NetCashProvidedByUsedInOperatingActivities",))
    capex = flow("capex_ttm", CAPEX_TAGS)
    flow("stock_comp_ttm", ("ShareBasedCompensation",))
    fcf = None
    if cfo is not None and capex is not None:
        fcf = cfo - capex
        note.add("sec", "fcf_ttm", fcf, "USD", T,
                 "derived: cfo_ttm − capex_ttm", None)
    if rev:
        for metric, num in (("operating_margin", op), ("net_margin", ni),
                            ("fcf_margin", fcf)):
            if num is not None:
                note.add("sec", metric, num / rev, "ratio", T,
                         f"derived: {metric.replace('_margin', '')} / revenue_ttm", None)

    cash_hit = _first(facts, CASH_TAGS, latest_instant)
    cash = cash_hit[1].value if cash_hit else None
    if cash_hit:
        note.add("sec", "cash", cash, "USD", T, cash_hit[0], cash_hit[1].filed)

    def inst(tag: str) -> Fact | None:
        f = latest_instant(facts.get(tag, []))
        if f and (date.fromisoformat(newest) - date.fromisoformat(f.end)).days > 400:
            return None
        return f

    parts: list[Fact] = []
    for tags in LONG_DEBT_TAG_SETS:
        parts = [f for f in map(inst, tags) if f]
        if parts:
            break
    if parts:
        short = next((f for f in map(inst, SHORT_DEBT_TAGS) if f), None)
        if short:
            parts.append(short)
    debt = sum(f.value for f in parts) if parts else None
    if debt is not None:
        note.add("sec", "total_debt", debt, "USD", T,
                 "derived: " + " + ".join(f.tag for f in parts),
                 min(f.filed for f in parts))
    net_debt = debt - cash if debt is not None and cash is not None else None
    if net_debt is not None:
        note.add("sec", "net_debt", net_debt, "USD", T,
                 "derived: total_debt − cash", None)

    share_hit = _first(facts, SHARE_TAGS, lambda fs: max(
        fs, key=lambda f: (f.end, f.filed)) if fs else None)
    shares = share_hit[1].value if share_hit else None
    if share_hit:
        note.add("sec", "shares", shares, "shares", T, share_hit[0], share_hit[1].filed)

    cap_ok = False
    if shares and close:
        mcap = shares * close
        note.add("sec", "market_cap", mcap, "USD", T,
                 "derived: shares × close", None)
        snap = c.execute("SELECT market_cap_usd FROM mm_sector_map WHERE symbol = ?",
                         (sym,)).fetchone()
        if snap and snap[0]:
            ratio = mcap / float(snap[0])
            cap_ok = 1.0 / CAP_CHECK_MAX_RATIO <= ratio <= CAP_CHECK_MAX_RATIO
            if not cap_ok:
                note.add("sec", "market_cap_check", ratio, "ratio", "mm_sector_map",
                         "market_cap_usd", None,
                         "share count disagrees with the market-cap snapshot "
                         "(split, ADR ratio or stale filing)")
        else:
            cap_ok = True
        if fcf is not None and cap_ok:
            note.add("sec", "fcf_yield", fcf / mcap, "ratio", T,
                     "derived: fcf_ttm / market_cap", None)
        if net_debt is not None and cap_ok:
            note.add("sec", "enterprise_value", mcap + net_debt, "USD", T,
                     "derived: market_cap + net_debt", None)

    rev_cagr = None
    rev_hit = _first(facts, REVENUE_TAGS,
                     lambda fs: (annual_series(fs)[-1] if annual_series(fs) else None))
    if rev_hit:
        series = annual_series(facts[rev_hit[0]])
        if len(series) >= 4 and series[-4].value > 0 and series[-1].value > 0:
            rev_cagr = (series[-1].value / series[-4].value) ** (1.0 / 3.0) - 1.0
            note.add("sec", "revenue_cagr_3y", rev_cagr, "ratio", T,
                     f"derived: {rev_hit[0]} FY {series[-4].end} → {series[-1].end}",
                     series[-1].filed)

    _dcf_section(c, note, fcf, net_debt, shares, close, cap_ok, rev_cagr)


def _dcf_section(c, note: Note, fcf, net_debt, shares, close, cap_ok: bool,
                 rev_cagr) -> None:
    fs_row = c.execute("SELECT finnhub_sector FROM mm_sector_map WHERE symbol = ?",
                       (note.symbol,)).fetchone()
    if fs_row and fs_row[0] in NO_DCF_FINNHUB_SECTORS:
        return note.skip("dcf", "free-cash-flow DCF is not meaningful for a bank")
    missing = [n for n, v in (("free cash flow", fcf), ("net debt", net_debt),
                              ("share count", shares), ("price", close)) if v is None]
    if missing:
        return note.skip("dcf", "missing " + ", ".join(missing))
    if not cap_ok:
        return note.skip("dcf", "share count failed the market-cap sanity check")
    if fcf <= 0:
        return note.skip("dcf", "trailing free cash flow is not positive")

    p = DCF_PARAMS
    if rev_cagr is not None:
        growth, g_src, g_tab = rev_cagr, "sec.revenue_cagr_3y", "mm_sec_facts"
    else:
        row = c.execute(
            "SELECT rev_chg_ttm, as_of_date FROM mm_fundamentals_growth "
            "WHERE symbol = ? AND as_of_date <= ? AND rev_chg_ttm IS NOT NULL "
            "ORDER BY as_of_date DESC LIMIT 1", (note.symbol, note.as_of)).fetchone()
        if row:
            growth, g_src, g_tab = float(row[0]) / 100.0, "rev_chg_ttm", "mm_fundamentals_growth"
        else:
            growth, g_src, g_tab = p["terminal_growth"], "DCF_PARAMS.terminal_growth", None
    growth = min(max(growth, 0.0), p["growth_cap"])

    note.add("dcf", "horizon_years", p["horizon_years"], "years", None, "DCF_PARAMS", None)
    note.add("dcf", "terminal_growth", p["terminal_growth"], "ratio", None, "DCF_PARAMS", None)
    note.add("dcf", "base_growth", growth, "ratio", g_tab,
             f"{g_src}, clamped to [0, {p['growth_cap']:g}]", None)
    for name, (wacc, shift) in p["scenarios"].items():
        g = min(max(growth + shift, p["growth_floor"]), p["growth_cap"])
        note.add("dcf", f"discount_rate_{name}", wacc, "ratio", None, "DCF_PARAMS", None)
        note.add("dcf", f"growth_{name}", g, "ratio", None,
                 f"derived: base_growth {shift:+.2f}, clamped to "
                 f"[{p['growth_floor']:g}, {p['growth_cap']:g}]", None)
        note.add("dcf", f"value_per_share_{name}",
                 dcf_per_share(fcf, g, wacc, net_debt, shares), "USD/share", "mm_sec_facts",
                 "derived: DCF on fcf_ttm, net_debt, shares", None)
    base = note.get("dcf", "value_per_share_base")
    if base and base > 0:
        note.add("dcf", "price_vs_base", close / base - 1.0, "ratio", "mm_bars",
                 "derived: close / value_per_share_base − 1", None)


def _context_sections(c, note: Note, close: float | None) -> None:
    sym, as_of = note.symbol, note.as_of
    d_as_of = date.fromisoformat(as_of)

    g = c.execute(
        "SELECT as_of_date, rev_chg_ttm, eps_chg_ttm, rev_growth_5y_avg "
        "FROM mm_fundamentals_growth WHERE symbol = ? AND as_of_date <= ? "
        "ORDER BY as_of_date DESC LIMIT 1", (sym, as_of)).fetchone()
    if g:
        for metric, val in zip(("rev_chg_ttm", "eps_chg_ttm", "rev_growth_5y_avg"), g[1:]):
            if val is not None:
                note.add("growth", metric, val, "pct", "mm_fundamentals_growth", metric, g[0])

    a = c.execute(
        "SELECT as_of_date, fy0_eps_consensus, fy1_eps_consensus, rating_buy, "
        "rating_hold, rating_sell FROM mm_analyst_estimates "
        "WHERE symbol = ? AND as_of_date <= ? ORDER BY as_of_date DESC LIMIT 1",
        (sym, as_of)).fetchone()
    if a:
        for metric, val, unit in zip(
            ("fy0_eps_consensus", "fy1_eps_consensus", "rating_buy",
             "rating_hold", "rating_sell"), a[1:],
            ("USD/share", "USD/share", "count", "count", "count")):
            if val is not None:
                note.add("estimates", metric, val, unit, "mm_analyst_estimates", metric, a[0])

    past = c.execute(
        "SELECT report_date, eps_surprise_pct FROM mm_earnings_events "
        "WHERE symbol = ? AND report_date <= ? AND eps_surprise_pct IS NOT NULL "
        "ORDER BY report_date DESC LIMIT 4", (sym, as_of)).fetchall()
    if past:
        note.add("earnings", "last_eps_surprise", past[0][1], "pct",
                 "mm_earnings_events", "eps_surprise_pct", past[0][0])
        note.add("earnings", "beats_in_last_reports", sum(1 for _, s in past if s > 0),
                 "count", "mm_earnings_events",
                 f"derived: eps_surprise_pct > 0 over last {len(past)} reports", past[0][0])
    nxt = c.execute(
        "SELECT MIN(report_date) FROM mm_earnings_events "
        "WHERE symbol = ? AND report_date > ?", (sym, as_of)).fetchone()
    if nxt and nxt[0]:
        note.add("earnings", "days_to_next_report",
                 (date.fromisoformat(nxt[0]) - d_as_of).days, "days",
                 "mm_earnings_events", "report_date", nxt[0])

    lo = (d_as_of - timedelta(days=INSIDER_WINDOW_DAYS)).isoformat()
    hi = (d_as_of - timedelta(days=INSIDER_LAG_DAYS)).isoformat()
    ins = c.execute(
        "SELECT transaction_type, COUNT(*), COALESCE(SUM(value_usd), 0) "
        "FROM mm_insider_transactions WHERE symbol = ? "
        "AND transaction_date BETWEEN ? AND ? AND transaction_type IN ('P', 'S') "
        "GROUP BY transaction_type", (sym, lo, hi)).fetchall()
    if ins:
        by = {t: (n, v) for t, n, v in ins}
        buys, sells = by.get("P", (0, 0.0)), by.get("S", (0, 0.0))
        col = f"transaction_type P/S, last {INSIDER_WINDOW_DAYS}d"
        note.add("insider", "purchases", buys[0], "count", "mm_insider_transactions", col, hi)
        note.add("insider", "sales", sells[0], "count", "mm_insider_transactions", col, hi)
        note.add("insider", "net_purchases_usd", buys[1] - sells[1], "USD",
                 "mm_insider_transactions", "derived: value_usd P − S", hi)

    dv = c.execute(
        "SELECT COALESCE(SUM(rate), 0), MAX(ex_date) FROM mm_dividends "
        "WHERE symbol = ? AND special = 0 AND ex_date <= ? AND ex_date > ?",
        (sym, as_of, (d_as_of - timedelta(days=365)).isoformat())).fetchone()
    if dv and dv[0]:
        note.add("dividends", "dividends_ttm", dv[0], "USD/share",
                 "mm_dividends", "rate", dv[1])
        if close:
            note.add("dividends", "dividend_yield", dv[0] / close, "ratio",
                     "mm_dividends", "derived: dividends_ttm / close", dv[1])

    k = c.execute(
        "SELECT date, predicted_5d_return, predicted_30d_return "
        "FROM mm_kronos_signals WHERE symbol = ? AND quality_flag = 'ok' "
        "AND date <= ? AND date >= ? ORDER BY date DESC, created_at DESC LIMIT 1",
        (sym, as_of, (d_as_of - timedelta(days=KRONOS_MAX_AGE_DAYS)).isoformat()),
    ).fetchone()
    if k:
        for metric, val in (("kronos_5d", k[1]), ("kronos_30d", k[2])):
            if val is not None:
                note.add("kronos", metric, val, "ratio", "mm_kronos_signals",
                         metric.replace("kronos", "predicted") + "_return", k[0],
                         "experimental, unvalidated")


def build_note(c, symbol: str, as_of: str) -> Note:
    note = Note(as_of, symbol)
    bar = c.execute(
        "SELECT date, close FROM mm_bars WHERE symbol = ? AND date <= ? "
        "ORDER BY date DESC LIMIT 1", (symbol, as_of)).fetchone()
    close = float(bar[1]) if bar and bar[1] else None
    if close:
        note.add("price", "close", close, "USD/share", "mm_bars", "close", bar[0])
    _sec_sections(c, note, close)
    _context_sections(c, note, close)
    return note


_LABELS = {
    ("price", "close"): "Last close",
    ("sec", "revenue_ttm"): "Revenue, last 12 months",
    ("sec", "operating_income_ttm"): "Operating income, last 12 months",
    ("sec", "net_income_ttm"): "Net income, last 12 months",
    ("sec", "cfo_ttm"): "Cash from operations, last 12 months",
    ("sec", "capex_ttm"): "Capital spending, last 12 months",
    ("sec", "fcf_ttm"): "Free cash flow (cash from operations minus capital spending)",
    ("sec", "operating_margin"): "Operating margin",
    ("sec", "net_margin"): "Net margin",
    ("sec", "fcf_margin"): "Free-cash-flow margin",
    ("sec", "cash"): "Cash",
    ("sec", "total_debt"): "Debt",
    ("sec", "net_debt"): "Net debt (debt minus cash)",
    ("sec", "market_cap"): "Market value",
    ("sec", "fcf_yield"): "Free-cash-flow yield (free cash flow / market value)",
    ("sec", "revenue_cagr_3y"): "Revenue growth per year, last 3 fiscal years",
    ("dcf", "base_growth"): "DCF assumed growth, base case",
    ("dcf", "value_per_share_bear"): "DCF value per share, cautious case",
    ("dcf", "value_per_share_base"): "DCF value per share, base case",
    ("dcf", "value_per_share_bull"): "DCF value per share, optimistic case",
    ("growth", "rev_chg_ttm"): "Revenue change, last 12 months vs prior",
    ("growth", "eps_chg_ttm"): "EPS change, last 12 months vs prior",
    ("estimates", "rating_buy"): "Analysts with a positive rating",
    ("estimates", "rating_hold"): "Analysts with a neutral rating",
    ("estimates", "rating_sell"): "Analysts with a negative rating",
    ("earnings", "last_eps_surprise"): "Last earnings surprise",
    ("earnings", "beats_in_last_reports"): "Earnings beats in recent reports",
    ("earnings", "days_to_next_report"): "Days to next earnings report",
    ("insider", "purchases"): "Insider open-market purchases, last 90 days",
    ("insider", "sales"): "Insider open-market sales, last 90 days",
    ("insider", "net_purchases_usd"): "Insider net purchases, last 90 days",
    ("dividends", "dividend_yield"): "Dividend yield, last 12 months",
    ("kronos", "kronos_5d"): "Kronos 5-day forecast (experimental, unproven)",
    ("kronos", "kronos_30d"): "Kronos 30-day forecast (experimental, unproven)",
}


def fmt(value: float, unit: str | None) -> str:
    if unit == "USD":
        a, sign = abs(value), "-" if value < 0 else ""
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"{sign}${a / div:.2f}{suf}"
        return f"{sign}${a:.0f}"
    if unit == "USD/share":
        return f"${value:.2f}"
    if unit == "ratio":
        return f"{value * 100:+.1f}%" if value < 0 else f"{value * 100:.1f}%"
    if unit == "pct":
        return f"{value:+.1f}%" if value < 0 else f"{value:.1f}%"
    return f"{value:.0f}"


def fact_lines(note: Note) -> list[str]:
    ""
    lines = []
    for r in note.rows:
        if r["metric"] == "skipped":
            lines.append(f"{r['section'].upper()} section skipped: {r['narrative']}")
            continue
        label = _LABELS.get((r["section"], r["metric"]))
        if label is None or r["value"] is None:
            continue
        line = f"{label}: {fmt(r['value'], r['unit'])}"
        if r["section"] == "estimates" and r["source_asof"]:
            line += f" (as of {r['source_asof']})"
        lines.append(line)
    gap = note.get("dcf", "price_vs_base")
    if gap is not None:
        lines.append(f"The last close is {fmt(abs(gap), 'ratio')} "
                     f"{'above' if gap > 0 else 'below'} the DCF base case")
    p = DCF_PARAMS
    if note.get("dcf", "value_per_share_base") is not None:
        rates = ", ".join(f"{fmt(w, 'ratio')}" for w, _ in p["scenarios"].values())
        lines.append(
            f"DCF fixed assumptions: {p['horizon_years']} years, discount rates "
            f"{rates}, long-run growth {fmt(p['terminal_growth'], 'ratio')}")
    return lines


def template_narrative(note: Note) -> dict[str, str]:
    ""
    def f(section, metric, unit):
        v = note.get(section, metric)
        return None if v is None else fmt(v, unit)

    s = note.symbol
    close, fcf = f("price", "close", "USD/share"), f("sec", "fcf_ttm", "USD")
    fy = f("sec", "fcf_yield", "ratio")
    lo, base, hi = (f("dcf", f"value_per_share_{k}", "USD/share")
                    for k in ("bear", "base", "bull"))
    head = f"{s} last closed at {close}." if close else f"{s}: no recent price."
    body = ""
    if fcf:
        body += (f" It produced {fcf} of free cash flow over the last 12 months"
                 + (f" ({fy} of its market value)" if fy else "") + ".")
    if base:
        body += (f" A DCF on fixed assumptions gives {lo} to {hi} per share, "
                 f"base case {base}.")
        wrong = ("Free cash flow falling below its trailing level, or growth "
                 "and discount rates outside the fixed assumptions.")
    else:
        why = next((r["narrative"] for r in note.rows
                    if r["section"] == "dcf" and r["metric"] == "skipped"), "inputs missing")
        body += f" No DCF was computed: {why}."
        wrong = ("Without a valuation there is little here to be wrong about; "
                 "the pick rests mainly on its price signal.")
    summary = (head + body)[:SUMMARY_MAX_CHARS]
    extra = []
    for sec, met, unit, text in (
        ("growth", "rev_chg_ttm", "pct", "Revenue changed {} over the last 12 months."),
        ("earnings", "last_eps_surprise", "pct", "The last earnings report surprised by {}."),
        ("dividends", "dividend_yield", "ratio", "Dividends over the last 12 months yield {}."),
        ("insider", "net_purchases_usd", "USD", "Insiders' net open-market purchases over 90 days: {}."),
        ("kronos", "kronos_5d", "ratio", "Kronos (experimental, unproven) forecasts {} over 5 days."),
    ):
        v = f(sec, met, unit)
        if v:
            extra.append(text.format(v))
    return {"summary": summary, "detail": " ".join([head + body] + extra),
            "wrong_if": wrong, "model": "template"}


def validate_narrative(text: str, allowed: str) -> str | None:
    ""
    m = _VERDICT_RE.search(_RATING_PHRASE_RE.sub("", text))
    if m:
        return f"barred word '{m.group(0)}'"
    ok = {float(n.replace(",", "")) for n in _NUM_RE.findall(allowed)}
    for n in _NUM_RE.findall(text):
        if float(n.replace(",", "")) not in ok:
            return f"figure '{n}' is not in the facts"
    return None


def _call_ollama(symbol: str, lines: list[str], model: str) -> str:
    from macmini.llm.commentary import strip_think
    payload = json.dumps({
        "model": model, "stream": False, "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Symbol: {symbol}\nFACTS:\n- " + "\n- ".join(lines)
                + "\n\nWrite the note."},
        ],
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
        return strip_think(json.load(resp)["message"]["content"])


def parse_sections(text: str) -> dict[str, str] | None:
    m = re.search(r"SUMMARY:\s*(.+?)\s*DETAIL:\s*(.+?)\s*WRONG IF:\s*(.+)",
                  text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    summary, detail, wrong = (re.sub(r"\s+", " ", g).strip(" *") for g in m.groups())
    if not summary or len(summary) > SUMMARY_MAX_CHARS:
        return None
    return {"summary": summary, "detail": detail, "wrong_if": wrong}


def write_narrative(note: Note, use_llm: bool, model: str = OLLAMA_MODEL,
                    caller=_call_ollama) -> dict[str, str]:
    ""
    lines = fact_lines(note)
    if use_llm and lines:
        try:
            raw = caller(note.symbol, lines, model)
            parsed = parse_sections(raw)
            reason = ("unparseable or over-long output" if parsed is None else
                      validate_narrative(" ".join(parsed.values()),
                                         note.symbol + " " + " ".join(lines)))
            if reason is None:
                return {"summary": template_narrative(note)["summary"],
                        "detail": f"{parsed['summary']} {parsed['detail']}",
                        "wrong_if": parsed["wrong_if"], "model": model}
            log.warning("%s: narrative rejected (%s) — using template", note.symbol, reason)
        except Exception as exc:
            log.warning("%s: narrative call failed (%s) — using template", note.symbol, exc)
    return template_narrative(note)


def _targets(c) -> tuple[str | None, list[str]]:
    picks = c.execute(
        "SELECT symbol, as_of_date FROM mm_active_picks ORDER BY rank").fetchall()
    as_of = max((r[1] for r in picks), default=None)
    cands = c.execute(
        "SELECT symbol, as_of_date FROM mm_quality_candidates WHERE as_of_date = "
        "(SELECT MAX(as_of_date) FROM mm_quality_candidates)").fetchall()
    if as_of is None:
        as_of = max((r[1] for r in cands), default=None)
    symbols: list[str] = []
    for sym, d in list(picks) + [r for r in cands if r[1] == as_of]:
        if sym not in symbols:
            symbols.append(sym)
    return as_of, symbols


def run(db_path: Path | str | None = None, as_of: str | None = None,
        symbols: list[str] | None = None, use_llm: bool = True,
        model: str = OLLAMA_MODEL) -> dict:
    ""
    out = {"as_of_date": None, "notes": [], "llm": 0, "template": 0, "dcf": 0}
    with connect(resolve_db(db_path)) as c:
        c.execute(MM_TITAN_NOTES)
        c.commit()
        auto_as_of, auto_syms = _targets(c)
        as_of = as_of or auto_as_of
        symbols = symbols or auto_syms
        if not as_of or not symbols:
            log.info("no picks or candidates — no notes written")
            return out
        out["as_of_date"] = as_of
        for i, sym in enumerate(symbols):
            note = build_note(c, sym, as_of)
            nar = write_narrative(note, use_llm and i < MAX_LLM_NOTES, model)
            src = f"model={nar['model']}"
            note.add("narrative", "summary", None, None, None, src, None, nar["summary"])
            note.add("narrative", "detail", None, None, None, src, None, nar["detail"])
            note.add("narrative", "wrong_if", None, None, None, src, None, nar["wrong_if"])
            now = _now_iso()
            c.execute("DELETE FROM mm_titan_notes WHERE as_of_date = ? AND symbol = ?",
                      (as_of, sym))
            c.executemany(
                "INSERT INTO mm_titan_notes (as_of_date, symbol, section, metric, "
                "value, unit, source_table, source_column, source_asof, narrative, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(as_of, sym, r["section"], r["metric"], r["value"], r["unit"],
                  r["source_table"], r["source_column"], r["source_asof"],
                  r["narrative"], now) for r in note.rows])
            c.commit()
            out["notes"].append(sym)
            out["template" if nar["model"] == "template" else "llm"] += 1
            out["dcf"] += int(note.get("dcf", "value_per_share_base") is not None)
            log.info("%s: %d rows, narrative=%s", sym, len(note.rows), nar["model"])
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Titan analyst notes (context only).")
    p.add_argument("--db", default=None)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; default = picks' as_of_date")
    p.add_argument("--symbols", default=None, help="comma list; default = picks + candidates")
    p.add_argument("--no-llm", action="store_true", help="template narrative only")
    p.add_argument("--model", default=OLLAMA_MODEL)
    args = p.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    print(json.dumps(run(args.db, args.as_of, syms, not args.no_llm, args.model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
