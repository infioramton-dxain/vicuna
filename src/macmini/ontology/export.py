""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import numpy as np

from macmini.config import DATA_DIR, DB_PATH, PROJECT_ROOT, SYMBOLS_CSV

log = logging.getLogger(__name__)


GENERATOR = "macmini.ontology.export"
MARKER_KEY = "generated_by"
MARKER_LINE = f"{MARKER_KEY}: {GENERATOR}"
INDEX_STEM = "Vicuna Ontology"
README_STEM = "README"

DEFAULT_VAULT = Path.home() / "ObsidianVault" / "10_Projects" / "Vicuna"
DEFAULT_DATA_DIR = DATA_DIR / "ontology"
DEFAULT_POLICY = PROJECT_ROOT / "POLICY.md"

TRADING_DAYS = 252
VOL_LOOKBACK = 60
HORIZONS = (5, 10, 20, 40)
EXCURSION_BARS = 20
FLOAT_DECIMALS = 6
FIXTURE_PREFIX = "fixture"

LABEL_DEFINITIONS: dict[str, str] = {
    "good": "r20 > 0 — twenty bars after entry the stock closed above the "
            "entry price.",
    "bad_stays_down": "r20 <= 0, AND the low of the first 20 bars reached the "
                      "suggested stop (mae20 <= -suggested_stop_loss_pct), AND "
                      "the stock is still below entry 40 bars on (r40 < 0; "
                      "r_now is used while r40 is not yet available).",
    "bad": "r20 <= 0 but it does not meet bad_stays_down (it never reached "
           "the stop, or it recovered above entry, or the rec has no "
           "suggested stop to test against).",
    "pending": "r20 is not known yet — fewer than 21 bars exist after as_of.",
}

MEMORY_DEFINITIONS: dict[str, str] = {
    "good": "Latest close is at or above the entry price and the stop level "
            "was never breached.",
    "open": "Latest close is below entry, but the stop level was never "
            "breached — undecided.",
    "recovered": "The stop level was breached at some point, but price has "
                 "since closed back at or above the entry price.",
    "bad_down": "The stop level was breached at some point after the rec AND "
                "the latest close is still below the entry price. This is the "
                "only state that blocks a symbol.",
    "pending": "No bar after the rec yet.",
}
MEMORY_BLOCKING = "bad_down"

LABEL_SYSTEMS: tuple[tuple[str, str, str], ...] = (
    ("Purpose",
     "Training label: was this recommendation a good decision, judged the "
     "same way for every rec.",
     "Live state: is this name a standing bad decision right now. The "
     "ranking exporter uses it to hold a name back."),
    ("Horizon",
     "Fixed: 20 bars (r20, mae20), confirmed at 40 bars (r40).",
     "Open-ended: every bar from the rec to the newest bar."),
    ("Changes over time",
     "Settles. Pending until 21 bars exist; bad_stays_down can still move "
     "while r_now stands in for r40; final once 41 bars exist.",
     "Re-evaluated on every run. bad_down becomes recovered the day price "
     "closes back at or above the entry, which also unblocks the symbol."),
    ("Entry price",
     "Always the open of the first bar after as_of_date (close if missing), "
     "filled or not.",
     "The actual fill price when the rec was filled, else the first open "
     "after as_of_date."),
    ("Stop test",
     "Lowest low of the first 20 bars vs suggested_stop_loss_pct. A rec "
     "without a suggested stop can never be bad_stays_down.",
     "Any low since the rec vs suggested_stop_loss_pct; recs without one "
     "use an 8% fallback stop."),
    ("Values",
     "good | bad_stays_down | bad | pending",
     "good | open | recovered | bad_down | pending"),
    ("Use it for",
     "Model training and evaluation targets.",
     "Describing today's state. Never a training target for a historical "
     "rec: it depends on bars that arrived long after it."),
)

DATASET_SPECS: tuple[tuple[str, str, str], ...] = (
    ("data/selector/panel.parquet", "parquet",
     "Selector weekly feature + label panel."),
    ("data/selector/panel_daily.parquet", "parquet",
     "Selector daily feature + label panel."),
    ("exports/selector_eval_*.json", "json",
     "Selector walk-forward test report."),
    ("scripts/analysis/bad_picks", "directory",
     "Analysis harness: event study of bad picks."),
    ("scripts/analysis/funnel", "directory",
     "Analysis harness: recommendation funnel diagnosis."),
    ("docs/bad_picks_root_cause_2026-09-29.md", "markdown",
     "Write-up: why the models keep recommending names that fall."),
    ("docs/funnel_diagnosis_2026-09-29.md", "markdown",
     "Write-up: recommendation funnel diagnosis."),
)

_UNSAFE_CHARS = re.compile(r'[\\:*?"<>|#^\[\]%\x00-\x1f]')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_POLICY_BULLET = re.compile(
    r"^- \*\*(\d{4}-\d{2}-\d{2}) — v(\d+(?:\.\d+)*)\.?\*\*\s*(.*)$")
_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


@dataclass(frozen=True)
class Prop:
    ""

    name: str
    type: str
    unit: str | None
    description: str
    role: str = "attribute"


@dataclass(frozen=True)
class Rel:
    name: str
    to: str
    cardinality: str
    description: str


@dataclass(frozen=True)
class ClassDef:
    name: str
    folder: str
    jsonl: str
    description: str
    props: tuple[Prop, ...]
    rels: tuple[Rel, ...]
    extras: tuple[Prop, ...] = ()


_FEATURE_NOTE = "Computed from mm_bars rows dated <= as_of_date only."
_OUTCOME_NOTE = ("Measured from the first bar AFTER as_of_date; entry price is "
                 "that bar's open (close if the open is missing).")

SYMBOL = ClassDef(
    name="Symbol", folder="Symbols", jsonl="symbols.jsonl",
    description="A tradable instrument. One per line of symbols.csv, plus any "
                "symbol a recommendation, analyst note or qualifying "
                "leaderboard row still points at (in_universe = false).",
    props=(
        Prop("ticker", "string", None, "Exchange ticker, exactly as in symbols.csv.", "key"),
        Prop("in_universe", "boolean", None, "True when the ticker is in symbols.csv today."),
        Prop("sector", "string", None, "mm_sector_map.sector (project sector bucket)."),
        Prop("finnhub_sector", "string", None, "mm_sector_map.finnhub_sector."),
        Prop("archetype", "string", None, "mm_sector_map.archetype."),
        Prop("market_cap_usd", "number", "USD", "mm_sector_map.market_cap_usd."),
        Prop("n_bars", "integer", "bars", "Daily bars held in mm_bars."),
        Prop("first_bar_date", "date", None, "Oldest bar date."),
        Prop("last_close", "number", "USD/share", "Close of the newest bar."),
        Prop("last_close_date", "date", None, "Date of the newest bar."),
        Prop("vol60", "number", "fraction",
             "Annualized realized volatility: stdev (ddof=1) of the last 60 "
             "daily close-to-close returns x sqrt(252)."),
        Prop("off_252d_high", "number", "fraction",
             "Last close / highest close of the last 252 bars - 1 (<= 0)."),
        Prop("mom21", "number", "fraction", "Close / close 21 bars earlier - 1."),
        Prop("mom63", "number", "fraction", "Close / close 63 bars earlier - 1."),
        Prop("mom252", "number", "fraction", "Close / close 252 bars earlier - 1."),
        Prop("memory_blocked", "boolean", None,
             "True while any rec on this symbol has memory_state = bad_down. "
             "Live state, not a blacklist: it lifts by itself when price "
             "closes back at or above the entry. Null when outcome memory "
             "is unavailable.", "label"),
        Prop("memory_blocked_reason", "string", None,
             "Plain-English reason for memory_blocked (oldest bad_down rec).",
             "label"),
        Prop("selector_score", "number", None,
             "Latest mm_selector_scores.score. Only its order within one "
             "as-of date matters. Shadow model output — context only."),
        Prop("selector_rank_pct", "number", "fraction",
             "Percentile among eligible names that day (1.0 = best); null "
             "when the symbol was not eligible."),
        Prop("selector_eligible", "boolean", None,
             "Passed the selector's live price / volatility / liquidity / "
             "scope screens that day."),
        Prop("selector_as_of", "date", None, "Date of that selector score."),
        Prop("selector_model_id", "string", None, "Model that produced it."),
    ),
    rels=(
        Rel("recommendations", "Recommendation", "many", "Every ledger rec on this symbol."),
        Rel("trades", "Trade", "many", "Every filled rec on this symbol."),
        Rel("qualifying_strategies", "Strategy", "many",
            "mm_leaderboard rows with qualifies = 1 for this symbol."),
        Rel("analyst_notes", "AnalystNote", "many", "mm_titan_notes for this symbol."),
    ),
)

STRATEGY = ClassDef(
    name="Strategy", folder="Strategies", jsonl="strategies.jsonl",
    description="A signal generator. One per entry of "
                "macmini.strategies.STRATEGY_REGISTRY, plus a stub for any "
                "strategy_id the ledger uses that is not in the registry "
                "(in_registry = false).",
    props=(
        Prop("strategy_id", "integer", None, "Registry id.", "key"),
        Prop("name", "string", None, "Strategy.name."),
        Prop("category", "string", None, "Strategy.category (archetype)."),
        Prop("timeframe", "string", None, "Strategy.timeframe."),
        Prop("data_required", "list[string]", None, "Tables the engine loads for it."),
        Prop("in_registry", "boolean", None, "False for ledger-only origination lanes."),
        Prop("module", "string", None, "Python module that defines it."),
        Prop("description", "string", None, "Class docstring, else module docstring."),
        Prop("ledger_names", "list[string]", None,
             "Distinct strategy_name values the ledger recorded for this id "
             "(verbatim, historical)."),
    ),
    rels=(
        Rel("recommendations", "Recommendation", "many", "Every ledger rec from this strategy."),
        Rel("qualifying_symbols", "Symbol", "many",
            "mm_leaderboard rows with qualifies = 1 for this strategy."),
    ),
)

RECOMMENDATION = ClassDef(
    name="Recommendation", folder="Recommendations", jsonl="recommendations.jsonl",
    description="One row of mm_recs_ledger (fixture-* rows excluded): what "
                "the system proposed, what the stock looked like that day, "
                "and what it did afterwards.",
    props=(
        Prop("rec_id", "string", None, "mm_recs_ledger.rec_id.", "key"),
        Prop("as_of_date", "date", None, "Signal date; the last bar the rec could see.", "key"),
        Prop("entry_date", "date", None, "Planned entry date recorded by the ledger."),
        Prop("rank", "integer", None, "Rank within that night's list (1 = top)."),
        Prop("ticker", "string", None, "Symbol recommended."),
        Prop("strategy_id", "integer", None, "Strategy that produced it."),
        Prop("strategy_name", "string", None, "Ledger strategy_name, verbatim."),
        Prop("sector", "string", None, "Ledger sector, verbatim."),
        Prop("status", "string", None, "pending | approved | rejected | expired."),
        Prop("fill_status", "string", None, "Ledger fill_status (filled | pending | shadow | null)."),
        Prop("rationale", "string", None,
             "Ledger rationale, verbatim. Historical text: it can embed legacy "
             "model annotations (DL=, RL=) and voided decision numbers — never "
             "use it as a model feature."),
        Prop("suggested_size_pct", "number", "fraction", "Suggested position size / equity."),
        Prop("suggested_stop_loss_pct", "number", "fraction", "Suggested stop distance below entry."),
        Prop("suggested_target_pct", "number", "fraction", "Suggested target distance above entry."),
        Prop("feature_bar_date", "date", None,
             "Newest bar dated <= as_of_date; every feature is measured on it.", "feature"),
        Prop("close", "number", "USD/share", f"Close on feature_bar_date. {_FEATURE_NOTE}", "feature"),
        Prop("vol60", "number", "fraction",
             f"Annualized 60-day realized volatility. {_FEATURE_NOTE}", "feature"),
        Prop("close_vs_sma50", "number", "fraction",
             f"Close / 50-bar average close - 1. {_FEATURE_NOTE}", "feature"),
        Prop("close_vs_sma200", "number", "fraction",
             f"Close / 200-bar average close - 1. {_FEATURE_NOTE}", "feature"),
        Prop("mom21", "number", "fraction", f"21-bar return. {_FEATURE_NOTE}", "feature"),
        Prop("mom63", "number", "fraction", f"63-bar return. {_FEATURE_NOTE}", "feature"),
        Prop("mom252", "number", "fraction", f"252-bar return. {_FEATURE_NOTE}", "feature"),
        Prop("off_252d_high", "number", "fraction",
             f"Close / highest close of the last 252 bars - 1. {_FEATURE_NOTE}", "feature"),
        Prop("outcome_entry_date", "date", None, f"Date of the entry bar. {_OUTCOME_NOTE}", "outcome"),
        Prop("outcome_entry_price", "number", "USD/share", f"Entry price. {_OUTCOME_NOTE}", "outcome"),
        Prop("bars_forward", "integer", "bars",
             "Bars available from the entry bar to the newest bar, inclusive.", "outcome"),
        Prop("r5", "number", "fraction", "Close 5 bars after the entry bar / entry - 1.", "outcome"),
        Prop("r10", "number", "fraction", "Close 10 bars after the entry bar / entry - 1.", "outcome"),
        Prop("r20", "number", "fraction", "Close 20 bars after the entry bar / entry - 1.", "outcome"),
        Prop("r40", "number", "fraction", "Close 40 bars after the entry bar / entry - 1.", "outcome"),
        Prop("r_now", "number", "fraction", "Newest close / entry - 1 (moves every run).", "outcome"),
        Prop("r_now_date", "date", None, "Date of the bar r_now is measured on.", "outcome"),
        Prop("mae20", "number", "fraction",
             "Worst point: lowest low of the first 20 bars / entry - 1. Null "
             "until 20 bars exist.", "outcome"),
        Prop("mfe20", "number", "fraction",
             "Best point: highest high of the first 20 bars / entry - 1. Null "
             "until 20 bars exist.", "outcome"),
        Prop("outcome_label", "string", None,
             "good | bad_stays_down | bad | pending — see label definitions.", "label"),
        Prop("memory_state", "string", None,
             "good | open | recovered | bad_down | pending — the LIVE outcome "
             "memory state; it changes as new bars arrive. Null when the rec "
             "has not been scored.", "label"),
        Prop("memory_entry_price", "number", "USD/share",
             "Entry the memory judges against: the actual fill when filled, "
             "else the first open after as_of_date.", "outcome"),
        Prop("memory_stop_hit_date", "date", None,
             "First bar after the rec whose low reached the stop level; null "
             "if never.", "outcome"),
        Prop("memory_return_to_date", "number", "fraction",
             "Latest close / memory_entry_price - 1.", "outcome"),
    ),
    rels=(
        Rel("symbol", "Symbol", "one", "The instrument recommended."),
        Rel("strategy", "Strategy", "one", "The strategy that produced it."),
        Rel("trade", "Trade", "zero_or_one", "The fill, when fill_status = filled."),
    ),
)

TRADE = ClassDef(
    name="Trade", folder="Trades", jsonl="trades.jsonl",
    description="A recommendation that was actually filled at the broker "
                "(mm_recs_ledger.fill_status = 'filled').",
    props=(
        Prop("rec_id", "string", None, "Ledger rec_id of the originating recommendation.", "key"),
        Prop("ticker", "string", None, "Symbol traded."),
        Prop("strategy_id", "integer", None, "Originating strategy."),
        Prop("entry_price", "number", "USD/share", "actual_fill_price."),
        Prop("entry_at", "datetime", None, "actual_fill_at, verbatim."),
        Prop("entry_date", "date", None, "Calendar date of entry_at (as recorded, UTC)."),
        Prop("qty", "integer", "shares", "fill_qty."),
        Prop("stop_price", "number", "USD/share", "Protective stop recorded by the ledger."),
        Prop("target_price", "number", "USD/share", "Profit target recorded by the ledger."),
        Prop("exit_status", "string", None, "open | stopped_out | target_hit | closed | operator_manual."),
        Prop("exit_price", "number", "USD/share", "Exit fill price.", "outcome"),
        Prop("exit_at", "datetime", None, "Exit timestamp, verbatim.", "outcome"),
        Prop("exit_date", "date", None, "Calendar date of exit_at.", "outcome"),
        Prop("exit_qty", "integer", "shares", "Shares exited.", "outcome"),
        Prop("realized_return_pct", "number", "fraction",
             "exit_price / entry_price - 1; null while either is missing. "
             "Before costs.", "outcome"),
        Prop("realized_pnl_usd", "number", "USD",
             "(exit_price - entry_price) x exit_qty (fill qty if exit qty is "
             "missing). Before costs.", "outcome"),
        Prop("days_held", "integer", "days",
             "Calendar days from entry_date to exit_date; null while open.", "outcome"),
    ),
    rels=(
        Rel("recommendation", "Recommendation", "one", "The rec this fill came from."),
        Rel("symbol", "Symbol", "one", "The instrument traded."),
        Rel("strategy", "Strategy", "one", "The originating strategy."),
    ),
)

POLICY_CHANGE = ClassDef(
    name="PolicyChange", folder="Policy", jsonl="policy_changes.jsonl",
    description="One entry of the policy document's change history.",
    props=(
        Prop("date", "date", None, "Ratification date.", "key"),
        Prop("version", "string", None, "Baseline version, e.g. 1.3.", "key"),
        Prop("text", "string", None, "The change-history entry, verbatim, unwrapped."),
    ),
    rels=(),
)

ANALYST_NOTE = ClassDef(
    name="AnalystNote", folder="AnalystNotes", jsonl="analyst_notes.jsonl",
    description="One Titan analyst note per (as_of_date, symbol) of "
                "mm_titan_notes. Context only — it explains, it never votes.",
    props=(
        Prop("as_of_date", "date", None, "Note date.", "key"),
        Prop("ticker", "string", None, "Symbol the note is about.", "key"),
        Prop("narrative_model", "string", None,
             "Who wrote the prose (source_column of the narrative rows, e.g. "
             "model=template)."),
        Prop("n_metrics", "integer", None, "Numeric / non-narrative rows in the note."),
        Prop("sections", "list[string]", None, "Sections present, sorted."),
    ),
    rels=(Rel("symbol", "Symbol", "one", "The instrument the note is about."),),
    extras=(
        Prop("summary", "string", None, "Narrative: summary."),
        Prop("detail", "string", None, "Narrative: detail."),
        Prop("wrong_if", "string", None, "Narrative: what would make this view wrong."),
        Prop("metrics", "list[object]", None,
             "Rows {section, metric, value, unit, source_table, source_column, "
             "source_asof, note}, in (section, metric) order."),
    ),
)

CLASSES: tuple[ClassDef, ...] = (
    SYMBOL, STRATEGY, RECOMMENDATION, TRADE, POLICY_CHANGE, ANALYST_NOTE)


@dataclass
class Entity:
    cls: ClassDef
    id: str
    stem: str
    label: str
    props: dict[str, Any]
    rels: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)

    @property
    def relpath(self) -> str:
        return f"{self.cls.folder}/{self.stem}.md"


@dataclass(frozen=True)
class Bars:
    ""

    dates: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)


def _clean(v: Any) -> Any:
    ""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        f = round(f, FLOAT_DECIMALS)
        return 0.0 if f == 0 else f
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return str(v)


def _day(v: Any) -> str | None:
    ""
    if v is None:
        return None
    s = str(v)[:10]
    return s if _ISO_DATE.match(s) else None


def safe_stem(text: str) -> str:
    ""
    s = text.replace("/", "-")
    s = _UNSAFE_CHARS.sub(" ", s).replace(".", "-")
    s = re.sub(r"\s+", " ", s).strip(" -")
    if not s:
        s = "_"
    if s.upper() in _WINDOWS_RESERVED:
        s += "_"
    return s


def _label(text: str) -> str:
    ""
    return re.sub(r"[\[\]|]", " ", text).strip()


def _quoted(s: str) -> str:
    ""
    return json.dumps(s, ensure_ascii=False)


def _yaml_scalar(v: Any, typ: str) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, float):
        r = repr(v)
        return r if "e" not in r.lower() else f"{v:.{FLOAT_DECIMALS}f}"
    s = str(v)
    if typ == "date" and _ISO_DATE.match(s):
        return s
    return _quoted(s)


def _yaml_field(name: str, v: Any, typ: str) -> list[str]:
    if isinstance(v, list):
        if not v:
            return [f"{name}: []"]
        return [f"{name}:"] + [f"  - {_yaml_scalar(x, 'string')}" for x in v]
    return [f"{name}: {_yaml_scalar(v, typ)}"]


def _pct(v: Any) -> str:
    return "—" if v is None else f"{v * 100:+.1f}%"


def _num(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.{decimals}f}"
    return str(v)


def _cell(v: Any) -> str:
    ""
    if v is None or v == "":
        return "—"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def connect_ro(db_path: Path | str) -> sqlite3.Connection:
    ""
    uri = f"file:{quote(str(Path(db_path).resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _rows(conn: sqlite3.Connection, table: str, wanted: tuple[str, ...],
          where: str = "", order: str = "") -> list[dict[str, Any]]:
    ""
    if not _table_exists(conn, table):
        return []
    have = _columns(conn, table)
    cols = [c for c in wanted if c in have]
    if not cols:
        return []
    sql = f"SELECT {', '.join(cols)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order:
        sql += f" ORDER BY {order}"
    return [{c: (r[c] if c in have else None) for c in wanted}
            for r in conn.execute(sql)]


def load_bars(conn: sqlite3.Connection, symbol: str) -> Bars | None:
    ""
    by_day: dict[str, tuple[float, float, float, float]] = {}
    for d, o, h, l, c in conn.execute(
            "SELECT date, open, high, low, close FROM mm_bars "
            "WHERE symbol = ? ORDER BY date", (symbol,)):
        day = _day(d)
        if day is None or c is None:
            continue
        c = float(c)
        by_day[day] = (float(o) if o else c, float(h) if h else c,
                       float(l) if l else c, c)
    if not by_day:
        return None
    days = tuple(sorted(by_day))
    arr = np.array([by_day[d] for d in days], dtype=float)
    return Bars(days, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


def read_symbols(path: Path) -> list[str]:
    ""
    out: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip().split(",")[0].strip()
        if not s or s.lower() == "symbol" or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_policy_changes(path: Path) -> list[dict[str, str]]:
    ""
    if not Path(path).exists():
        log.warning("policy file not found: %s", path)
        return []
    out: list[dict[str, str]] = []
    inside = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip().lower() == "## change history"
            continue
        if not inside:
            continue
        m = _POLICY_BULLET.match(line)
        if m:
            out.append({"date": m.group(1), "version": m.group(2),
                        "text": m.group(3).strip()})
        elif out and line.strip() and line[:1] in (" ", "\t"):
            out[-1]["text"] = f"{out[-1]['text']} {line.strip()}".strip()
    return out


MEMORY_COLUMNS = ("rec_id", "as_of_date", "symbol", "state", "entry_price",
                  "stop_hit_date", "worst_low", "last_close", "last_date",
                  "return_to_date", "filled")


def load_memory(conn: sqlite3.Connection,
                as_of: date | None = None) -> tuple[dict[str, dict[str, Any]], str | None]:
    ""
    if _table_exists(conn, "mm_outcome_memory"):
        have = _columns(conn, "mm_outcome_memory")
        if {"rec_id", "state"} <= have:
            rows = _rows(conn, "mm_outcome_memory", MEMORY_COLUMNS)
            if rows:
                return {str(r["rec_id"]): r for r in rows}, "table:mm_outcome_memory"
        else:
            log.warning("mm_outcome_memory lacks rec_id/state — ignoring the table")
    if not (_table_exists(conn, "mm_recs_ledger") and _table_exists(conn, "mm_bars")):
        return {}, None
    try:
        from macmini.recs.outcome_memory import evaluate
        rows = evaluate(conn, as_of or date.today())
    except Exception as exc:
        log.warning("outcome memory unavailable (%s: %s)", type(exc).__name__, exc)
        return {}, None
    out = {str(r["rec_id"]): {c: r.get(c) for c in MEMORY_COLUMNS} for r in rows}
    return out, "computed:macmini.recs.outcome_memory.evaluate"


def load_selector(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    ""
    if not _table_exists(conn, "mm_selector_scores"):
        return {}
    have = _columns(conn, "mm_selector_scores")
    if not {"as_of_date", "symbol", "score"} <= have:
        log.warning("mm_selector_scores lacks as_of_date/symbol/score — skipped")
        return {}
    wanted = ("as_of_date", "symbol", "score", "rank_pct", "eligible",
              "model_id", "created_at")
    cols = ", ".join(f"s.{c}" if c in have else f"NULL AS {c}" for c in wanted)
    rows = conn.execute(
        f"SELECT {cols} FROM mm_selector_scores s JOIN ("
        "  SELECT symbol, MAX(substr(as_of_date, 1, 10)) AS d "
        "  FROM mm_selector_scores GROUP BY symbol) m "
        "ON s.symbol = m.symbol AND substr(s.as_of_date, 1, 10) = m.d").fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: (str(r["symbol"]), str(r["created_at"] or ""),
                                         str(r["model_id"] or ""))):
        out[str(r["symbol"])] = {c: r[c] for c in wanted}
    return out


def discover_datasets(root: Path) -> list[dict[str, Any]]:
    ""
    out: list[dict[str, Any]] = []
    root = Path(root)
    for pattern, kind, description in DATASET_SPECS:
        for path in sorted(root.glob(pattern)):
            entry: dict[str, Any] = {
                "path": path.relative_to(root).as_posix(), "kind": kind,
                "rows": None, "files": None, "description": description}
            if kind == "parquet":
                try:
                    import pyarrow.parquet as pq
                    entry["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
                except Exception as exc:
                    log.warning("no row count for %s (%s)", path, exc)
            elif kind == "directory":
                if not path.is_dir():
                    continue
                entry["files"] = sum(1 for f in path.iterdir()
                                     if f.is_file() and not f.name.startswith("."))
            elif not path.is_file():
                continue
            out.append(entry)
    return out


def _blocked_reason(rows: list[dict[str, Any]]) -> str | None:
    ""
    bad = sorted((r for r in rows if r.get("state") == MEMORY_BLOCKING),
                 key=lambda r: (str(r.get("as_of_date")), str(r.get("rec_id"))))
    if not bad:
        return None
    r = bad[0]
    text = f"Recommended {_day(r.get('as_of_date')) or '?'}"
    if r.get("entry_price") is not None:
        text += f" at {float(r['entry_price']):.2f}"
    text += f"; stop level breached {_day(r.get('stop_hit_date')) or '?'}"
    if r.get("last_close") is not None and r.get("return_to_date") is not None:
        text += (f"; last close {float(r['last_close']):.2f} on "
                 f"{_day(r.get('last_date')) or '?'} is still "
                 f"{float(r['return_to_date']) * 100:+.1f}% vs entry")
    text += f". {len(bad)} rec{'s' if len(bad) != 1 else ''} on this symbol in bad_down."
    return text


FEATURE_KEYS = ("close", "vol60", "close_vs_sma50", "close_vs_sma200",
                "mom21", "mom63", "mom252", "off_252d_high")
OUTCOME_KEYS = ("outcome_entry_date", "outcome_entry_price", "bars_forward",
                "r5", "r10", "r20", "r40", "r_now", "r_now_date",
                "mae20", "mfe20")


def features_at(bars: Bars | None, as_of: str) -> dict[str, Any]:
    ""
    out: dict[str, Any] = {k: None for k in ("feature_bar_date", *FEATURE_KEYS)}
    if bars is None:
        return out
    i = bisect_right(bars.dates, as_of) - 1
    if i < 0:
        return out
    c = bars.close[: i + 1]
    last = float(c[-1])
    out["feature_bar_date"] = bars.dates[i]
    out["close"] = last
    if len(c) >= VOL_LOOKBACK + 1:
        w = c[-(VOL_LOOKBACK + 1):]
        rets = w[1:] / w[:-1] - 1.0
        out["vol60"] = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS))
    for n, key in ((50, "close_vs_sma50"), (200, "close_vs_sma200")):
        if len(c) >= n:
            out[key] = last / float(np.mean(c[-n:])) - 1.0
    for n in (21, 63, 252):
        if len(c) >= n + 1 and c[-(n + 1)] > 0:
            out[f"mom{n}"] = last / float(c[-(n + 1)]) - 1.0
    if len(c) >= TRADING_DAYS:
        out["off_252d_high"] = last / float(np.max(c[-TRADING_DAYS:])) - 1.0
    return out


def outcomes_after(bars: Bars | None, as_of: str) -> dict[str, Any]:
    ""
    out: dict[str, Any] = {k: None for k in OUTCOME_KEYS}
    if bars is None:
        return out
    e = bisect_right(bars.dates, as_of)
    n = len(bars)
    if e >= n:
        return out
    entry = float(bars.open[e])
    if not math.isfinite(entry) or entry <= 0:
        entry = float(bars.close[e])
    if not math.isfinite(entry) or entry <= 0:
        return out
    out["outcome_entry_date"] = bars.dates[e]
    out["outcome_entry_price"] = entry
    out["bars_forward"] = n - e
    for h in HORIZONS:
        if e + h < n:
            out[f"r{h}"] = float(bars.close[e + h]) / entry - 1.0
    out["r_now"] = float(bars.close[-1]) / entry - 1.0
    out["r_now_date"] = bars.dates[-1]
    if e + EXCURSION_BARS <= n:
        out["mae20"] = float(np.min(bars.low[e: e + EXCURSION_BARS])) / entry - 1.0
        out["mfe20"] = float(np.max(bars.high[e: e + EXCURSION_BARS])) / entry - 1.0
    return out


def outcome_label(r20: float | None, r40: float | None, r_now: float | None,
                  mae20: float | None, stop_pct: float | None) -> str:
    ""
    if r20 is None:
        return "pending"
    if r20 > 0:
        return "good"
    later = r40 if r40 is not None else r_now
    if (stop_pct is not None and stop_pct > 0 and mae20 is not None
            and mae20 <= -stop_pct and later is not None and later < 0):
        return "bad_stays_down"
    return "bad"


LEDGER_COLUMNS = (
    "rec_id", "as_of_date", "entry_date", "rank", "symbol", "strategy_id",
    "strategy_name", "sector", "suggested_size_pct", "suggested_stop_loss_pct",
    "suggested_target_pct", "rationale", "status", "stop_price", "target_price",
    "fill_status", "actual_fill_price", "actual_fill_at", "fill_qty",
    "exit_status", "exit_price", "exit_at", "exit_qty")


def _rec_sort_key(r: Mapping[str, Any]) -> tuple:
    rank = r["rank"]
    return (str(r["as_of_date"]), rank is None, rank if rank is not None else 0,
            str(r["symbol"]), str(r["rec_id"]))


def load_ledger(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = _rows(conn, "mm_recs_ledger", LEDGER_COLUMNS)
    kept = [r for r in rows
            if r["rec_id"] and not str(r["rec_id"]).startswith(FIXTURE_PREFIX)]
    if len(kept) != len(rows):
        log.info("ledger: dropped %d fixture/blank rows", len(rows) - len(kept))
    return sorted(kept, key=_rec_sort_key)


def _short_strategy_name(name: str) -> str:
    ""
    return re.sub(r"\s*\([^)]*\)", "", name).strip() or name


def _strategy_doc(cls: type) -> str:
    doc = (cls.__dict__.get("__doc__") or "").strip()
    if not doc:
        mod = sys.modules.get(cls.__module__)
        doc = ((mod.__doc__ if mod else "") or "").strip()
    return re.sub(r"\s+", " ", doc)


def _unique_stems(wanted: list[tuple[str, str, str]]) -> dict[str, str]:
    ""
    counts = Counter(p.lower() for _, p, _ in wanted)
    out: dict[str, str] = {}
    taken: set[str] = set()
    for key, preferred, fallback in wanted:
        stem = preferred if counts[preferred.lower()] == 1 else fallback
        if stem.lower() in taken:
            raise ValueError(f"file name collision on {stem!r} ({key})")
        taken.add(stem.lower())
        out[key] = stem
    return out


def build_entities(conn: sqlite3.Connection, symbols_csv: Path, policy_path: Path,
                   registry: Mapping[int, type] | None = None,
                   memory_as_of: date | None = None,
                   ) -> tuple[list[Entity], dict[str, Any]]:
    ""
    if registry is None:
        from macmini.strategies import STRATEGY_REGISTRY
        registry = STRATEGY_REGISTRY

    universe = read_symbols(symbols_csv)
    ledger = load_ledger(conn)
    titan = _rows(conn, "mm_titan_notes",
                  ("as_of_date", "symbol", "section", "metric", "value", "unit",
                   "source_table", "source_column", "source_asof", "narrative"),
                  order="as_of_date, symbol, section, metric")
    qualifying = sorted(
        {(int(r["strategy_id"]), str(r["symbol"]))
         for r in _rows(conn, "mm_leaderboard", ("strategy_id", "symbol", "qualifies"),
                        where="qualifies = 1")})
    sector_map = {r["symbol"]: r for r in _rows(
        conn, "mm_sector_map",
        ("symbol", "sector", "finnhub_sector", "market_cap_usd", "archetype"))}

    memory, memory_source = load_memory(conn, memory_as_of)
    selector = load_selector(conn)

    in_universe = set(universe)
    tickers = sorted(in_universe
                     | {str(r["symbol"]) for r in ledger}
                     | {str(r["symbol"]) for r in titan}
                     | {s for _, s in qualifying})
    strategy_ids = sorted(set(registry)
                          | {int(r["strategy_id"]) for r in ledger}
                          | {sid for sid, _ in qualifying})

    sym_id = {t: f"sym:{t}" for t in tickers}
    strat_id = {s: f"strat:{s}" for s in strategy_ids}

    has_bars = _table_exists(conn, "mm_bars")
    bars: dict[str, Bars | None] = {}
    for n, t in enumerate(tickers, 1):
        bars[t] = load_bars(conn, t) if has_bars else None
        if n % 100 == 0:
            log.info("bars loaded: %d / %d symbols", n, len(tickers))

    rec_stems = _unique_stems([
        (str(r["rec_id"]),
         safe_stem(f"{_day(r['as_of_date']) or r['as_of_date']} {safe_stem(str(r['symbol']))} "
                   f"{r['strategy_id']} ({safe_stem(str(r['rec_id'])[:8])})"),
         safe_stem(f"{_day(r['as_of_date']) or r['as_of_date']} {safe_stem(str(r['symbol']))} "
                   f"{r['strategy_id']} ({safe_stem(str(r['rec_id']))})"))
        for r in ledger])
    filled = [r for r in ledger if r["fill_status"] == "filled"]

    def _trade_day(r: Mapping[str, Any]) -> str:
        return (_day(r["actual_fill_at"]) or _day(r["entry_date"])
                or _day(r["as_of_date"]) or "undated")

    trade_stems = _unique_stems([
        (str(r["rec_id"]),
         safe_stem(f"{_trade_day(r)} {safe_stem(str(r['symbol']))} trade "
                   f"({safe_stem(str(r['rec_id'])[:8])})"),
         safe_stem(f"{_trade_day(r)} {safe_stem(str(r['symbol']))} trade "
                   f"({safe_stem(str(r['rec_id']))})"))
        for r in filled])
    trade_ids = {str(r["rec_id"]) for r in filled}

    recs: list[Entity] = []
    trades: list[Entity] = []
    for r in ledger:
        rid, ticker, sid = str(r["rec_id"]), str(r["symbol"]), int(r["strategy_id"])
        as_of = _day(r["as_of_date"]) or str(r["as_of_date"])
        feats = features_at(bars.get(ticker), as_of)
        outs = outcomes_after(bars.get(ticker), as_of)
        props = {
            "rec_id": rid, "as_of_date": as_of, "entry_date": _day(r["entry_date"]),
            "rank": r["rank"], "ticker": ticker, "strategy_id": sid,
            "strategy_name": r["strategy_name"], "sector": r["sector"],
            "status": r["status"], "fill_status": r["fill_status"],
            "rationale": r["rationale"],
            "suggested_size_pct": r["suggested_size_pct"],
            "suggested_stop_loss_pct": r["suggested_stop_loss_pct"],
            "suggested_target_pct": r["suggested_target_pct"],
            **feats, **outs,
        }
        props = _clean(props)
        props["outcome_label"] = outcome_label(
            props["r20"], props["r40"], props["r_now"], props["mae20"],
            props["suggested_stop_loss_pct"])
        mem = memory.get(rid) or {}
        props.update(_clean({
            "memory_state": mem.get("state"),
            "memory_entry_price": mem.get("entry_price"),
            "memory_stop_hit_date": _day(mem.get("stop_hit_date")),
            "memory_return_to_date": mem.get("return_to_date"),
        }))
        recs.append(Entity(
            RECOMMENDATION, f"rec:{rid}", rec_stems[rid],
            _label(f"{as_of} {ticker} #{sid}"), props,
            rels={"symbol": sym_id[ticker], "strategy": strat_id[sid],
                  "trade": f"trade:{rid}" if rid in trade_ids else None}))

        if rid not in trade_ids:
            continue
        entry, exit_ = r["actual_fill_price"], r["exit_price"]
        entry_day, exit_day = _day(r["actual_fill_at"]), _day(r["exit_at"])
        qty = r["exit_qty"] if r["exit_qty"] is not None else r["fill_qty"]
        realized = pnl = held = None
        if entry and exit_ is not None:
            realized = float(exit_) / float(entry) - 1.0
            if qty is not None:
                pnl = (float(exit_) - float(entry)) * float(qty)
        if entry_day and exit_day:
            held = (date.fromisoformat(exit_day) - date.fromisoformat(entry_day)).days
        trades.append(Entity(
            TRADE, f"trade:{rid}", trade_stems[rid],
            _label(f"{_trade_day(r)} {ticker} trade"),
            _clean({
                "rec_id": rid, "ticker": ticker, "strategy_id": sid,
                "entry_price": entry, "entry_at": r["actual_fill_at"],
                "entry_date": entry_day, "qty": r["fill_qty"],
                "stop_price": r["stop_price"], "target_price": r["target_price"],
                "exit_status": r["exit_status"], "exit_price": exit_,
                "exit_at": r["exit_at"], "exit_date": exit_day,
                "exit_qty": r["exit_qty"], "realized_return_pct": realized,
                "realized_pnl_usd": pnl, "days_held": held,
            }),
            rels={"recommendation": f"rec:{rid}", "symbol": sym_id[ticker],
                  "strategy": strat_id[sid]}))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in titan:
        key = (_day(row["as_of_date"]) or str(row["as_of_date"]), str(row["symbol"]))
        grouped.setdefault(key, []).append(row)
    notes: list[Entity] = []
    for (as_of, ticker), rows in sorted(grouped.items()):
        narrative = {r["metric"]: r for r in rows if r["section"] == "narrative"}
        metrics = [{
            "section": r["section"], "metric": r["metric"], "value": r["value"],
            "unit": r["unit"], "source_table": r["source_table"],
            "source_column": r["source_column"], "source_asof": r["source_asof"],
            "note": r["narrative"],
        } for r in rows if r["section"] != "narrative"]
        models = sorted({str(r["source_column"]) for r in narrative.values()
                         if r["source_column"]})
        notes.append(Entity(
            ANALYST_NOTE, f"note:{as_of}:{ticker}",
            safe_stem(f"{as_of} {safe_stem(ticker)} analyst note"),
            _label(f"{as_of} {ticker} note"),
            _clean({
                "as_of_date": as_of, "ticker": ticker,
                "narrative_model": ", ".join(models) or None,
                "n_metrics": len(metrics),
                "sections": sorted({str(r["section"]) for r in rows}),
            }),
            rels={"symbol": sym_id[ticker]},
            extras=_clean({
                "summary": (narrative.get("summary") or {}).get("narrative"),
                "detail": (narrative.get("detail") or {}).get("narrative"),
                "wrong_if": (narrative.get("wrong_if") or {}).get("narrative"),
                "metrics": metrics,
            })))

    symbol_stems = _unique_stems(
        [(t, safe_stem(t), safe_stem(f"{t} {t.encode().hex()}")) for t in tickers])
    memory_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for r in ledger:
        if str(r["rec_id"]) in memory:
            memory_by_ticker.setdefault(str(r["symbol"]), []).append(
                memory[str(r["rec_id"])])
    symbols: list[Entity] = []
    for t in tickers:
        b = bars.get(t)
        meta = sector_map.get(t) or {}
        f = features_at(b, b.dates[-1]) if b else {}
        stem = symbol_stems[t]
        reason = _blocked_reason(memory_by_ticker.get(t, []))
        sel = selector.get(t) or {}
        symbols.append(Entity(
            SYMBOL, sym_id[t], stem, _label(t),
            _clean({
                "ticker": t, "in_universe": t in in_universe,
                "sector": meta.get("sector"),
                "finnhub_sector": meta.get("finnhub_sector"),
                "archetype": meta.get("archetype"),
                "market_cap_usd": meta.get("market_cap_usd"),
                "n_bars": len(b) if b else 0,
                "first_bar_date": b.dates[0] if b else None,
                "last_close": f.get("close"),
                "last_close_date": f.get("feature_bar_date"),
                "vol60": f.get("vol60"), "off_252d_high": f.get("off_252d_high"),
                "mom21": f.get("mom21"), "mom63": f.get("mom63"),
                "mom252": f.get("mom252"),
                "memory_blocked": (reason is not None) if memory_source else None,
                "memory_blocked_reason": reason,
                "selector_score": sel.get("score"),
                "selector_rank_pct": sel.get("rank_pct"),
                "selector_eligible": (bool(sel["eligible"])
                                      if sel.get("eligible") is not None else None),
                "selector_as_of": _day(sel.get("as_of_date")),
                "selector_model_id": sel.get("model_id"),
            }),
            rels={
                "recommendations": [e.id for e in recs if e.props["ticker"] == t],
                "trades": [e.id for e in trades if e.props["ticker"] == t],
                "qualifying_strategies": [strat_id[s] for s, sym in qualifying if sym == t],
                "analyst_notes": [e.id for e in notes if e.props["ticker"] == t],
            },
            aliases=[t] if stem != t else []))

    ledger_names: dict[int, set[str]] = {}
    for r in ledger:
        if r["strategy_name"]:
            ledger_names.setdefault(int(r["strategy_id"]), set()).add(str(r["strategy_name"]))
    strategies: list[Entity] = []
    for sid in strategy_ids:
        cls = registry.get(sid)
        names = sorted(ledger_names.get(sid, ()))
        if cls is not None:
            name = str(cls.name)
            props = {
                "strategy_id": sid, "name": name,
                "category": getattr(cls, "category", None),
                "timeframe": getattr(cls, "timeframe", None),
                "data_required": [str(x) for x in getattr(cls, "data_required", ())],
                "in_registry": True, "module": cls.__module__,
                "description": _strategy_doc(cls) or None, "ledger_names": names,
            }
        else:
            name = "Non-registry origination"
            props = {
                "strategy_id": sid, "name": name, "category": None,
                "timeframe": None, "data_required": [], "in_registry": False,
                "module": None,
                "description": "Not in STRATEGY_REGISTRY: recs that reached the "
                               "ledger from outside the strategy library "
                               "(operator or external lanes).",
                "ledger_names": names,
            }
        stem = safe_stem(f"{sid} - {_short_strategy_name(name)}")
        strategies.append(Entity(
            STRATEGY, strat_id[sid], stem, _label(stem), _clean(props),
            rels={
                "recommendations": [e.id for e in recs if e.props["strategy_id"] == sid],
                "qualifying_symbols": [sym_id[sym] for s, sym in qualifying if s == sid],
            },
            aliases=[name] if name != stem else []))

    changes = parse_policy_changes(policy_path)
    policy_stems = _unique_stems([
        (f"{n}", safe_stem(f"{c['date']} v{c['version']}"),
         safe_stem(f"{c['date']} v{c['version']} entry {n + 1}"))
        for n, c in enumerate(changes)])
    seen_ids: Counter[str] = Counter()
    policies: list[Entity] = []
    for n, c in enumerate(changes):
        pid = f"policy:{c['date']}:v{c['version']}"
        seen_ids[pid] += 1
        if seen_ids[pid] > 1:
            pid = f"{pid}:{seen_ids[pid]}"
        policies.append(Entity(
            POLICY_CHANGE, pid, policy_stems[f"{n}"],
            _label(f"{c['date']} v{c['version']}"), _clean(dict(c))))

    out = symbols + strategies + recs + trades + policies + notes
    for e in out:
        unknown = set(e.props) - {p.name for p in e.cls.props}
        if unknown:
            raise KeyError(f"{e.cls.name}: undeclared properties {sorted(unknown)}")
    meta = {
        "memory_source": memory_source,
        "selector_symbols": sum(1 for e in symbols
                                if e.props["selector_score"] is not None),
        "selector_as_of": max((e.props["selector_as_of"] for e in symbols
                               if e.props["selector_as_of"]), default=None),
    }
    return out, meta


class Linker:
    ""

    def __init__(self, prefix: str, entities: list[Entity]) -> None:
        self.prefix = prefix
        self._by_id = {e.id: e for e in entities}
        if len(self._by_id) != len(entities):
            dup = [k for k, n in Counter(e.id for e in entities).items() if n > 1]
            raise ValueError(f"duplicate entity ids: {dup[:5]}")

    def entity(self, eid: str) -> Entity:
        return self._by_id[eid]

    def link(self, eid: str, label: str | None = None) -> str:
        e = self._by_id[eid]
        return f"[[{self.prefix}{e.cls.folder}/{e.stem}|{label or e.label}]]"

    def index(self, label: str = INDEX_STEM) -> str:
        return f"[[{self.prefix}{INDEX_STEM}|{label}]]"


def link_prefix(vault: Path) -> str:
    ""
    vault = vault.resolve()
    for parent in vault.parents:
        if (parent / ".obsidian").is_dir():
            return vault.relative_to(parent).as_posix() + "/"
    return ""


def render_frontmatter(e: Entity, linker: Linker) -> list[str]:
    lines = ["---", f"type: {e.cls.name}", f"id: {_quoted(e.id)}", MARKER_LINE]
    if e.aliases:
        lines += _yaml_field("aliases", list(e.aliases), "list[string]")
    lines += _yaml_field("tags", [f"vicuna/{e.cls.name.lower()}"], "list[string]")
    for p in e.cls.props:
        lines += _yaml_field(p.name, e.props.get(p.name), p.type)
    for r in e.cls.rels:
        v = e.rels.get(r.name)
        if r.cardinality == "many":
            lines += _yaml_field(r.name, [linker.link(x) for x in (v or [])], "list[string]")
        else:
            lines += _yaml_field(r.name, linker.link(v) if v else None, "string")
    lines.append("---")
    return lines


def _links_section(title: str, ids: list[str], linker: Linker,
                   describe: Callable[[Entity], str] | None = None) -> list[str]:
    out = ["", f"## {title} ({len(ids)})", ""]
    if not ids:
        return out + ["_None._"]
    for i in ids:
        extra = describe(linker.entity(i)) if describe else ""
        out.append(f"- {linker.link(i)}{' — ' + extra if extra else ''}")
    return out


def _describe_rec(e: Entity) -> str:
    p = e.props
    return (f"{p['status']}, {p['outcome_label']}, r20 {_pct(p['r20'])}, "
            f"now {_pct(p['r_now'])}")


def _describe_trade(e: Entity) -> str:
    p = e.props
    return f"{p['exit_status'] or 'open'}, realized {_pct(p['realized_return_pct'])}"


def _body_symbol(e: Entity, lk: Linker) -> list[str]:
    p = e.props
    out = [f"# {p['ticker']}", "",
           f"Class: {lk.index('Symbol')}"
           + ("" if p["in_universe"] else " · **not in the current universe**"), ""]
    out += _table(("Property", "Value"), [
        ("Sector", p["sector"]), ("Finnhub sector", p["finnhub_sector"]),
        ("Archetype", p["archetype"]),
        ("Market cap (USD)", _num(p["market_cap_usd"], 0)),
        ("Last close", f"{_num(p['last_close'])} on {p['last_close_date'] or '—'}"),
        ("Realized volatility, 60d annualized", _pct(p["vol60"])),
        ("Off 252-day high", _pct(p["off_252d_high"])),
        ("Momentum 21 / 63 / 252 bars",
         f"{_pct(p['mom21'])} / {_pct(p['mom63'])} / {_pct(p['mom252'])}"),
        ("Bars held", f"{_num(p['n_bars'])} since {p['first_bar_date'] or '—'}"),
        ("Outcome memory",
         "not available" if p["memory_blocked"] is None
         else f"BLOCKED — {p['memory_blocked_reason']}" if p["memory_blocked"]
         else "not blocked"),
        ("Selector (shadow, context only)",
         None if p["selector_score"] is None else
         f"score {_num(p['selector_score'], 4)}, rank "
         f"{_num(p['selector_rank_pct'], 3)}, "
         f"{'eligible' if p['selector_eligible'] else 'not eligible'}, "
         f"{p['selector_as_of'] or '—'}, {p['selector_model_id'] or '—'}"),
    ])
    out += _links_section("Recommendations", e.rels["recommendations"], lk, _describe_rec)
    out += _links_section("Trades", e.rels["trades"], lk, _describe_trade)
    out += _links_section("Qualifying strategies", e.rels["qualifying_strategies"], lk)
    out += _links_section("Analyst notes", e.rels["analyst_notes"], lk)
    return out


def _body_strategy(e: Entity, lk: Linker) -> list[str]:
    p = e.props
    out = [f"# {p['strategy_id']} — {p['name']}", "",
           f"Class: {lk.index('Strategy')}"
           + ("" if p["in_registry"] else " · **not in STRATEGY_REGISTRY**"), ""]
    if p["description"]:
        out += [f"> {p['description']}", ""]
    out += _table(("Property", "Value"), [
        ("Category", p["category"]), ("Timeframe", p["timeframe"]),
        ("Data required", ", ".join(p["data_required"]) or None),
        ("Module", f"`{p['module']}`" if p["module"] else None),
        ("Names used in the ledger", "; ".join(p["ledger_names"]) or None),
    ])
    out += _links_section("Recommendations", e.rels["recommendations"], lk, _describe_rec)
    out += _links_section("Qualifying symbols", e.rels["qualifying_symbols"], lk)
    return out


def _body_recommendation(e: Entity, lk: Linker) -> list[str]:
    p = e.props
    trade = e.rels.get("trade")
    out = [f"# {p['as_of_date']} · {p['ticker']} · #{p['strategy_id']} "
           f"{p['strategy_name'] or ''}".rstrip(), "",
           f"Class: {lk.index('Recommendation')}", "",
           f"- Symbol: {lk.link(e.rels['symbol'])}",
           f"- Strategy: {lk.link(e.rels['strategy'])}",
           f"- Trade: {lk.link(trade) if trade else 'none (not filled)'}",
           f"- Status: {p['status']} · fill: {p['fill_status'] or '—'} · "
           f"rank {_num(p['rank'])}",
           f"- Outcome label (fixed horizon, for training): **{p['outcome_label']}**",
           f"- Memory state (live, changes with price): "
           f"**{p['memory_state'] or 'not scored'}**",
           "", "## Rationale", "",
           "> " + (str(p["rationale"]).replace("\n", " ") if p["rationale"] else "—"),
           "", "## Suggested bracket", ""]
    out += _table(("Size (of equity)", "Stop (below entry)", "Target (above entry)"), [
        tuple(_pct(p[k]).lstrip("+") for k in (
            "suggested_size_pct", "suggested_stop_loss_pct", "suggested_target_pct"))])
    out += ["", f"## Features at as-of (bar of {p['feature_bar_date'] or '—'})", "",
            "Only bars dated on or before the as-of date were used.", ""]
    out += _table(("Feature", "Value"), [
        ("Close", _num(p["close"])),
        ("Realized volatility, 60d annualized", _pct(p["vol60"])),
        ("Close vs 50-day average", _pct(p["close_vs_sma50"])),
        ("Close vs 200-day average", _pct(p["close_vs_sma200"])),
        ("Momentum 21 bars", _pct(p["mom21"])),
        ("Momentum 63 bars", _pct(p["mom63"])),
        ("Momentum 252 bars", _pct(p["mom252"])),
        ("Off 252-day high", _pct(p["off_252d_high"])),
    ])
    out += ["", "## What happened next", "",
            f"Entry {_num(p['outcome_entry_price'])} on "
            f"{p['outcome_entry_date'] or '—'} (first bar after as-of); "
            f"{_num(p['bars_forward'])} bars since.", ""]
    out += _table(("Outcome", "Value"), [
        ("Return after 5 bars", _pct(p["r5"])),
        ("Return after 10 bars", _pct(p["r10"])),
        ("Return after 20 bars", _pct(p["r20"])),
        ("Return after 40 bars", _pct(p["r40"])),
        (f"Return to {p['r_now_date'] or 'now'}", _pct(p["r_now"])),
        ("Worst point, first 20 bars", _pct(p["mae20"])),
        ("Best point, first 20 bars", _pct(p["mfe20"])),
    ])
    out += ["", "## Outcome memory (live)", "",
            "Judged against the actual fill when there was one, over every bar "
            "since the rec. It clears itself when price closes back at or "
            "above the entry.", ""]
    out += _table(("Memory", "Value"), [
        ("State", p["memory_state"]),
        ("Entry judged against", _num(p["memory_entry_price"], 4)),
        ("Stop level first breached", p["memory_stop_hit_date"]),
        ("Return to date", _pct(p["memory_return_to_date"])),
    ])
    return out


def _body_trade(e: Entity, lk: Linker) -> list[str]:
    p = e.props
    out = [f"# Trade · {p['ticker']} · {p['entry_date'] or '—'}", "",
           f"Class: {lk.index('Trade')}", "",
           f"- Recommendation: {lk.link(e.rels['recommendation'])}",
           f"- Symbol: {lk.link(e.rels['symbol'])}",
           f"- Strategy: {lk.link(e.rels['strategy'])}", ""]
    out += _table(("Property", "Value"), [
        ("Entry", f"{_num(p['entry_price'], 4)} x {_num(p['qty'])} at {p['entry_at'] or '—'}"),
        ("Stop / target", f"{_num(p['stop_price'])} / {_num(p['target_price'])}"),
        ("Exit status", p["exit_status"]),
        ("Exit", f"{_num(p['exit_price'], 4)} at {p['exit_at'] or '—'}"),
        ("Realized return", _pct(p["realized_return_pct"])),
        ("Realized P&L (USD)", _num(p["realized_pnl_usd"])),
        ("Days held", _num(p["days_held"])),
    ])
    return out


def _body_policy(e: Entity, lk: Linker) -> list[str]:
    p = e.props
    return [f"# Policy baseline v{p['version']} — {p['date']}", "",
            f"Class: {lk.index('PolicyChange')}", "",
            "Mirror of one entry of the policy document's change history.", "",
            f"> {p['text']}"]


def _body_note(e: Entity, lk: Linker) -> list[str]:
    p, x = e.props, e.extras
    out = [f"# Analyst note · {p['ticker']} · {p['as_of_date']}", "",
           f"Class: {lk.index('AnalystNote')}", "",
           f"- Symbol: {lk.link(e.rels['symbol'])}",
           f"- Narrative written by: {p['narrative_model'] or '—'}", "",
           "Context only: the note explains, it never votes."]
    for title, key in (("Summary", "summary"), ("Detail", "detail"),
                       ("Wrong if", "wrong_if")):
        out += ["", f"## {title}", "", str(x.get(key) or "—").replace("\n", " ")]
    out += ["", f"## Numbers ({p['n_metrics']})", ""]
    out += _table(
        ("Section", "Metric", "Value", "Unit", "Source table", "Source column",
         "Source as-of", "Note"),
        [(m["section"], m["metric"],
          None if m["value"] is None else repr(m["value"]), m["unit"],
          m["source_table"], m["source_column"], m["source_asof"], m["note"])
         for m in x.get("metrics", [])])
    return out


_BODIES: dict[str, Callable[[Entity, Linker], list[str]]] = {
    "Symbol": _body_symbol, "Strategy": _body_strategy,
    "Recommendation": _body_recommendation, "Trade": _body_trade,
    "PolicyChange": _body_policy, "AnalystNote": _body_note,
}


def render_note(e: Entity, linker: Linker) -> str:
    lines = render_frontmatter(e, linker) + [""] + _BODIES[e.cls.name](e, linker)
    return "\n".join(lines).rstrip("\n") + "\n"


def jsonl_record(e: Entity) -> dict[str, Any]:
    rec: dict[str, Any] = {"type": e.cls.name, "id": e.id}
    for p in e.cls.props:
        rec[p.name] = e.props.get(p.name)
    for r in e.cls.rels:
        v = e.rels.get(r.name)
        rec[r.name] = list(v or []) if r.cardinality == "many" else v
    for p in e.cls.extras:
        rec[p.name] = e.extras.get(p.name)
    return rec


def ontology_schema(counts: Mapping[str, int], labels: Mapping[str, int],
                    generated_at: str, run_id: str | None, source_db: str,
                    memory: Mapping[str, Any] | None = None,
                    selector: Mapping[str, Any] | None = None,
                    datasets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    def prop(p: Prop) -> dict[str, Any]:
        return {"name": p.name, "type": p.type, "unit": p.unit,
                "role": p.role, "description": p.description}

    return {
        "name": "Vicuna Ontology",
        "generated_by": GENERATOR,
        "generated_at": generated_at,
        "source_db": source_db,
        "source_run_id": run_id,
        "conventions": {
            "ratios": "fractions (0.137 = 13.7%)", "money": "USD",
            "dates": "ISO 8601", "float_decimals": FLOAT_DECIMALS,
            "ids": "namespaced and globally unique: sym: strat: rec: trade: policy: note:",
            "roles": "key | attribute | feature (known at as_of) | outcome "
                     "(known only later) | label. Never train on outcome or "
                     "label columns as inputs.",
        },
        "classes": [{
            "name": c.name, "description": c.description, "folder": c.folder,
            "jsonl": c.jsonl, "count": counts.get(c.name, 0),
            "properties": [
                {"name": "type", "type": "string", "unit": None, "role": "key",
                 "description": "Class name."},
                {"name": "id", "type": "string", "unit": None, "role": "key",
                 "description": "Stable, globally unique id."},
                *[prop(p) for p in c.props]],
            "jsonl_only_properties": [prop(p) for p in c.extras],
        } for c in CLASSES],
        "relations": [{
            "name": r.name, "from": c.name, "to": r.to,
            "cardinality": r.cardinality, "description": r.description,
        } for c in CLASSES for r in c.rels],
        "label_definitions": {"outcome_label": dict(LABEL_DEFINITIONS),
                              "memory_state": dict(MEMORY_DEFINITIONS)},
        "label_systems": {
            "summary": "outcome_label is a fixed-horizon label for training "
                       "(r20 / mae20 / r40). memory_state is the live, "
                       "time-varying state the ranking exporter uses to hold "
                       "a name back; it clears itself when price closes back "
                       "at or above the entry. They answer different "
                       "questions and will disagree on some recs.",
            "comparison": [{"aspect": a, "outcome_label": o, "memory_state": m}
                           for a, o, m in LABEL_SYSTEMS],
        },
        "outcome_label_counts": dict(labels),
        "outcome_memory": dict(memory or {}),
        "selector": dict(selector or {}),
        "datasets": list(datasets or []),
    }


def render_index(schema: Mapping[str, Any], prefix: str) -> str:
    out = ["---", "type: Ontology", f"id: {_quoted('ontology:vicuna')}", MARKER_LINE,
           "tags:", '  - "vicuna/ontology"',
           f"generated_at: {_quoted(schema['generated_at'])}",
           f"source_run_id: {_yaml_scalar(schema['source_run_id'], 'string')}",
           "---", "", "# Vicuna Ontology", "",
           "The typed, linked map of the Vicuna research database. Generated "
           f"by `{GENERATOR}` — edits here are overwritten on the next run.", "",
           f"- Generated at: {schema['generated_at']}",
           f"- Source backtest run: `{schema['source_run_id'] or '—'}`",
           f"- How to use it: [[{prefix}{README_STEM}|README]]", "",
           "## Conventions", "",
           "- Ratios are fractions: `0.137` means 13.7%.",
           "- Money is in USD. Dates are ISO (`YYYY-MM-DD`).",
           "- **Feature** properties were knowable on the as-of date. "
           "**Outcome** and **label** properties were not — never feed them "
           "to a model as inputs.",
           "- Ledger text (rationale, strategy names) is historical and copied "
           "verbatim. Any decision number in it is historical "
           "and carries no authority.",
           "- Everything here is context. No ranking, sizing or order path "
           "reads it.", "",
           "## Counts", ""]
    out += _table(("Class", "Notes", "Folder", "Dataset"), [
        (c["name"], f"{c['count']:,}", f"`{c['folder']}/`", f"`{c['jsonl']}`")
        for c in schema["classes"]])
    out += ["", "## Outcome label — what good and bad mean", "",
            "The standard: a stock that goes up after it is added "
            "was a good decision; one that goes down and stays down was a bad "
            "one. Measured from the first bar after the as-of date.", ""]
    counts = schema["outcome_label_counts"]
    out += _table(("Label", "Recs", "Meaning"), [
        (k, f"{counts.get(k, 0):,}", v)
        for k, v in schema["label_definitions"]["outcome_label"].items()])
    mem = schema["outcome_memory"]
    out += ["", "## Memory state — is it a standing bad decision today", "",
            "Live outcome memory, one state per recommendation. A symbol is "
            "**blocked** while any of its recs is `bad_down`. Nothing is "
            "listed by hand: the block lifts by itself the day price closes "
            "back at or above the entry.", "",
            f"- Source: `{mem.get('source') or 'not available'}`",
            f"- Recs not scored: {mem.get('not_scored', 0):,}", ""]
    states = mem.get("state_counts", {})
    out += _table(("State", "Recs", "Meaning"), [
        (k, f"{states.get(k, 0):,}", v)
        for k, v in schema["label_definitions"]["memory_state"].items()])
    blocked = mem.get("blocked_symbols", [])
    out += ["", f"Blocked symbols ({len(blocked)}): "
            + (", ".join(f"[[{prefix}{SYMBOL.folder}/{b['stem']}|{_label(b['ticker'])}]]"
                         for b in blocked) or "none") + "."]
    out += ["", "## The two label systems, side by side", "",
            schema["label_systems"]["summary"], ""]
    out += _table(("", "`outcome_label`", "`memory_state`"), [
        (c["aspect"], c["outcome_label"], c["memory_state"])
        for c in schema["label_systems"]["comparison"]])
    agree = mem.get("outcome_label_by_memory_state", [])
    if agree:
        out += ["", "How the two line up on the current recs:", ""]
        out += _table(("outcome_label", "memory_state", "Recs"), [
            (a["outcome_label"], a["memory_state"] or "not scored", f"{a['recs']:,}")
            for a in agree])
    sel = schema["selector"]
    out += ["", "## Selector scores", "",
            "Latest `mm_selector_scores` row per symbol, carried on the Symbol "
            "notes. Shadow model output: context only.", "",
            f"- Symbols with a score: {sel.get('symbols', 0):,}",
            f"- Latest score date: {sel.get('as_of') or '—'}"]
    out += ["", "## Datasets", "",
            "Files other jobs produce, in the repo (`~/Projects/macmini`). "
            "Referenced here, never copied.", ""]
    if schema["datasets"]:
        out += _table(("Path", "Kind", "Rows", "Files", "What it is"), [
            (f"`{d['path']}`", d["kind"],
             None if d["rows"] is None else f"{d['rows']:,}",
             None if d["files"] is None else f"{d['files']:,}", d["description"])
            for d in schema["datasets"]])
    else:
        out += ["_None found on disk._"]
    out += ["", "## Relations", ""]
    out += _table(("From", "Relation", "To", "Cardinality", "Meaning"), [
        (r["from"], f"`{r['name']}`", r["to"], r["cardinality"], r["description"])
        for r in schema["relations"]])
    out += ["", "## Classes and properties"]
    for c in schema["classes"]:
        out += ["", f"### {c['name']}", "", c["description"], ""]
        out += _table(("Property", "Type", "Unit", "Role", "Meaning"), [
            (f"`{p['name']}`", p["type"], p["unit"], p["role"], p["description"])
            for p in c["properties"]])
        if c["jsonl_only_properties"]:
            out += ["", "In the dataset only (shown in the note body, not the "
                        "frontmatter):", ""]
            out += _table(("Property", "Type", "Unit", "Role", "Meaning"), [
                (f"`{p['name']}`", p["type"], p["unit"], p["role"], p["description"])
                for p in c["jsonl_only_properties"]])
    return "\n".join(out) + "\n"


def render_readme(prefix: str) -> str:
    out = ["---", "type: Readme", f"id: {_quoted('readme:vicuna')}", MARKER_LINE,
           "tags:", '  - "vicuna/ontology"', "---", "",
           "# Vicuna knowledge base", "",
           "Generated from the Vicuna research database (`macmini.db`). Start "
           f"at [[{prefix}{INDEX_STEM}|{INDEX_STEM}]].", "",
           "## Folders", ""]
    out += _table(("Folder", "Holds"), [(f"`{c.folder}/`", c.description) for c in CLASSES])
    out += ["", "## Regenerate", "", "```",
            "cd ~/Projects/macmini",
            f"PYTHONPATH=src venv/bin/python -m {GENERATOR}", "```", "",
            "Read-only against the database. The same data is written as "
            "JSONL under `data/ontology/` in the repo, with `ontology.json` "
            "describing every property.", "",
            "## Rules", "",
            f"- Every generated note carries `{MARKER_LINE}` in its "
            "frontmatter. Generated notes are rewritten on every run and "
            "removed when their source row disappears — do not edit them.",
            "- A note **without** that marker is never overwritten or deleted, "
            "so hand-written notes are safe in these folders.",
            "- To annotate a generated note, write a separate note and link "
            "to it."]
    return "\n".join(out) + "\n"


def has_marker(path: Path) -> bool:
    ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(65536)
    except OSError:
        return False
    if not head.startswith("---"):
        return False
    lines = head.splitlines()
    for line in lines[1:]:
        if line.strip() == "---":
            return False
        if line.strip() == MARKER_LINE:
            return True
    return False


@dataclass
class WriteStats:
    written: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped_unmarked: list[str] = field(default_factory=list)


def _inside(root: Path, target: Path) -> Path:
    ""
    resolved = target.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"refusing to write outside {root}: {target}")
    return resolved


def _write_note(root: Path, relpath: str, text: str, stats: WriteStats) -> Path:
    raw = root / relpath
    path = _inside(root, raw)
    if raw.is_symlink() or path.exists():
        if raw.is_symlink() or not has_marker(path):
            log.warning("not overwriting %s — it is not a generated note", path)
            stats.skipped_unmarked.append(relpath)
            return path
        if path.read_text(encoding="utf-8") == text:
            stats.unchanged += 1
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    stats.written += 1
    return path


def _remove_stale(root: Path, keep: set[Path], stats: WriteStats) -> None:
    ""
    for c in CLASSES:
        folder = root / c.folder
        if folder.is_symlink() or not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.md")):
            if p.is_symlink() or not p.is_file() or p.resolve() in keep:
                continue
            if has_marker(p):
                _inside(root, p)
                p.unlink()
                stats.removed += 1
                log.info("removed stale note %s", p)


def _write_data(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_id(conn: sqlite3.Connection) -> str | None:
    if not _table_exists(conn, "mm_leaderboard"):
        return None
    row = conn.execute("SELECT last_run_id FROM mm_leaderboard LIMIT 1").fetchone()
    return str(row[0]) if row and row[0] is not None else None


def export(db_path: Path | str = DB_PATH, vault: Path | str = DEFAULT_VAULT,
           data_dir: Path | str = DEFAULT_DATA_DIR,
           symbols_csv: Path | str = SYMBOLS_CSV,
           policy_path: Path | str = DEFAULT_POLICY,
           registry: Mapping[int, type] | None = None,
           generated_at: str | None = None,
           project_root: Path | str = PROJECT_ROOT,
           memory_as_of: date | None = None) -> dict[str, Any]:
    ""
    vault, data_dir = Path(vault).expanduser(), Path(data_dir).expanduser()
    if (vault / ".obsidian").exists():
        raise ValueError(
            f"{vault} is an Obsidian vault root — point --vault at a folder "
            "inside it, this exporter owns the whole folder it is given")
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = connect_ro(db_path)
    try:
        entities, meta = build_entities(conn, Path(symbols_csv), Path(policy_path),
                                        registry, memory_as_of)
        run_id = _run_id(conn)
    finally:
        conn.close()

    counts = {c.name: 0 for c in CLASSES}
    for e in entities:
        counts[e.cls.name] += 1
    seen = Counter(e.props["outcome_label"] for e in entities
                   if e.cls is RECOMMENDATION)
    labels = {k: seen.get(k, 0) for k in LABEL_DEFINITIONS}
    recs = [e for e in entities if e.cls is RECOMMENDATION]
    states = Counter(e.props["memory_state"] for e in recs)
    unknown = sorted(str(k) for k in states if k is not None and k not in MEMORY_DEFINITIONS)
    if unknown:
        log.warning("memory states outside the documented set: %s", unknown)
    pairs = Counter((e.props["outcome_label"], e.props["memory_state"]) for e in recs)
    memory = {
        "source": meta["memory_source"],
        "state_counts": {**{k: states.get(k, 0) for k in MEMORY_DEFINITIONS},
                         **{k: states[k] for k in unknown}},
        "not_scored": states.get(None, 0),
        "blocked_symbols": [
            {"ticker": e.props["ticker"], "stem": e.stem,
             "reason": e.props["memory_blocked_reason"]}
            for e in entities if e.cls is SYMBOL and e.props["memory_blocked"]],
        "outcome_label_by_memory_state": [
            {"outcome_label": o, "memory_state": m, "recs": n}
            for (o, m), n in sorted(pairs.items(),
                                    key=lambda kv: (kv[0][0], str(kv[0][1])))]
        if meta["memory_source"] else [],
    }
    selector = {"symbols": meta["selector_symbols"], "as_of": meta["selector_as_of"]}
    datasets = discover_datasets(Path(project_root))
    schema = ontology_schema(counts, labels, generated_at, run_id,
                             str(Path(db_path).resolve()), memory, selector, datasets)

    vault.mkdir(parents=True, exist_ok=True)
    prefix = link_prefix(vault)
    linker = Linker(prefix, entities)
    stats = WriteStats()
    keep: set[Path] = set()
    for e in entities:
        keep.add(_write_note(vault, e.relpath, render_note(e, linker), stats))
    _remove_stale(vault, keep, stats)
    _write_note(vault, f"{INDEX_STEM}.md", render_index(schema, prefix), stats)
    _write_note(vault, f"{README_STEM}.md", render_readme(prefix), stats)

    for c in CLASSES:
        lines = [json.dumps(jsonl_record(e), ensure_ascii=False, allow_nan=False)
                 for e in entities if e.cls is c]
        _write_data(data_dir / c.jsonl, "".join(f"{x}\n" for x in lines))
    _write_data(data_dir / "ontology.json",
                json.dumps(schema, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "counts": counts, "outcome_labels": labels,
        "memory_source": memory["source"], "memory_states": memory["state_counts"],
        "memory_not_scored": memory["not_scored"],
        "memory_blocked": [b["ticker"] for b in memory["blocked_symbols"]],
        "selector_symbols": selector["symbols"], "selector_as_of": selector["as_of"],
        "datasets": [d["path"] for d in datasets], "source_run_id": run_id,
        "generated_at": generated_at, "vault": str(vault), "data_dir": str(data_dir),
        "link_prefix": prefix, "notes_written": stats.written,
        "notes_unchanged": stats.unchanged, "notes_removed": stats.removed,
        "skipped_unmarked": list(stats.skipped_unmarked),
    }
    log.info("ontology export done: %s", json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export macmini.db as an Obsidian knowledge base + JSONL datasets.")
    p.add_argument("--db", default=str(DB_PATH), help="SQLite DB (opened read-only)")
    p.add_argument("--vault", default=str(DEFAULT_VAULT),
                   help="folder that receives the notes (owned by this exporter)")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="folder that receives the JSONL files and ontology.json")
    p.add_argument("--symbols-csv", default=str(SYMBOLS_CSV))
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = export(args.db, args.vault, args.data_dir, args.symbols_csv, args.policy)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
