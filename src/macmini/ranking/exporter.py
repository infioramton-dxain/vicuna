""
from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from statistics import median

import pandas as pd

from macmini.backtest.loader import load_aux, load_bars
from macmini.ranking import shadow_sizing
from macmini.recs import outcome_memory
from macmini.config import resolve_db
from macmini.data._db import connect
from macmini.data.schema import MM_PICK_FATES, MM_PICK_FUNNEL
from macmini.strategies import STRATEGY_REGISTRY
from macmini.strategies.personas import persona_of
from macmini.strategies.sentiment.fomc_seasonality import FOMC_DATES

log = logging.getLogger(__name__)

MIN_DEFLATED_SHARPE = 0.0
RESEARCH_MIN_DEFLATED_SHARPE = 0.10
MIN_PROFIT_FACTOR = 1.1
MAX_DRAWDOWN_FLOOR = -0.20
SURVIVAL_WINDOWS = 12

MIN_TOTAL_TRADES = 20
PF_WINSOR_CAP = 10.0


def _winsor_median(vals: list[float], cap: float = PF_WINSOR_CAP) -> float:
    ""
    clean = [min(float(v), cap) for v in vals
             if v is not None and float(v) == float(v)]
    if not clean:
        return 0.0
    return float(median(sorted(clean)))


RECENCY_LAMBDA = 0.8


def _recency_weight(age_windows) -> float:
    return float(RECENCY_LAMBDA ** float(age_windows))


def _winsor_wmedian(vals: list[float], weights: list[float],
                    cap: float = PF_WINSOR_CAP) -> float:
    ""
    pairs = [(min(float(v), cap), float(w)) for v, w in zip(vals, weights)
             if v is not None and float(v) == float(v) and w > 0]
    if not pairs:
        return 0.0
    pairs.sort(key=lambda p: p[0])
    half = sum(w for _, w in pairs) / 2.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return float(v)
    return float(pairs[-1][0])


STOP_ATR_MULT = 3.0
TARGET_R_MULT = 2.0
STOP_FRAC_MIN, STOP_FRAC_MAX = 0.05, 0.15
RISK_PER_TRADE = 0.005
SIZE_CAP, SIZE_FLOOR = 0.10, 0.02
FALLBACK_SIZE, FALLBACK_STOP, FALLBACK_TARGET = 0.05, 0.08, 0.20


def _atr_bracket(
    conn: sqlite3.Connection, symbol: str, as_of: date
) -> tuple[float, float, float]:
    ""
    row = conn.execute(
        "SELECT i.atr14, b.close FROM mm_indicators_daily i "
        "JOIN mm_bars b ON b.symbol = i.symbol AND b.date = i.date "
        "WHERE i.symbol = ? AND i.date <= ? "
        "  AND i.atr14 IS NOT NULL AND b.close > 0 "
        "ORDER BY i.date DESC LIMIT 1",
        (symbol, as_of.isoformat()),
    ).fetchone()
    if not row or not row[0] or not row[1]:
        return FALLBACK_SIZE, FALLBACK_STOP, FALLBACK_TARGET
    atr_frac = float(row[0]) / float(row[1])
    stop_frac = min(max(STOP_ATR_MULT * atr_frac, STOP_FRAC_MIN), STOP_FRAC_MAX)
    target_frac = TARGET_R_MULT * stop_frac
    size_pct = min(max(RISK_PER_TRADE / stop_frac, SIZE_FLOOR), SIZE_CAP)
    return round(size_pct, 4), round(stop_frac, 4), round(target_frac, 4)

MIN_DOLLAR_ADV = 10_000_000.0
MIN_PRICE = 10.0
ADV_LOOKBACK_DAYS = 60
ADV_MIN_ROWS = 20
KNIFE_ATR_MULT = 1.5


def _veto_veto(conn: sqlite3.Connection, symbol: str, as_of: date) -> str | None:
    ""
    rows = conn.execute(
        "SELECT close, volume FROM mm_bars WHERE symbol = ? AND date <= ? "
        "ORDER BY date DESC LIMIT ?",
        (symbol, as_of.isoformat(), ADV_LOOKBACK_DAYS),
    ).fetchall()
    if not rows:
        return "no bar history"
    last_close = rows[0][0]
    if not last_close or float(last_close) < MIN_PRICE:
        return f"price {last_close} below ${MIN_PRICE:g} floor"
    dollar = sorted(
        float(c) * float(v) for c, v in rows if c and v is not None
    )
    if len(dollar) < ADV_MIN_ROWS:
        return f"only {len(dollar)} volume-bearing bars in {ADV_LOOKBACK_DAYS}d"
    adv = float(median(dollar))
    if adv < MIN_DOLLAR_ADV:
        return f"median dollar ADV ${adv/1e6:.1f}M below ${MIN_DOLLAR_ADV/1e6:g}M floor"
    if len(rows) >= 3:
        atr_row = conn.execute(
            "SELECT atr14 FROM mm_indicators_daily "
            "WHERE symbol = ? AND date <= ? AND atr14 IS NOT NULL "
            "ORDER BY date DESC LIMIT 1",
            (symbol, as_of.isoformat()),
        ).fetchone()
        if atr_row and atr_row[0]:
            two_day_move = float(rows[0][0]) - float(rows[2][0])
            if two_day_move <= -KNIFE_ATR_MULT * float(atr_row[0]):
                return (
                    f"falling knife: 2-day move {two_day_move:+.2f} "
                    f"≤ -{KNIFE_ATR_MULT:g}×ATR14 ({float(atr_row[0]):.2f})"
                )
    return None


QS_VOL_MAX = 0.65
QS_VOL_LOOKBACK = 60
QS_MISS_STREAK_MAX = 2
QS_SENT_MAX = -0.75
QS_MAT_MIN = 0.70
QS_KRONOS_MAX_AGE_DAYS = 7
QS_SENT_WINDOW_DAYS = 7


def _model_stack(conn: sqlite3.Connection, symbol: str, as_of: date) -> dict:
    ""
    as_of_str = as_of.isoformat()
    stack: dict = {
        "kronos_5d": None, "kronos_30d": None,
        "sentiment_last7d": None, "sentiment_materiality": None,
        "realized_vol_60d_ann": None, "earnings_miss_streak": 0,
    }
    row = conn.execute(
        "SELECT predicted_5d_return, predicted_30d_return FROM mm_kronos_signals "
        "WHERE symbol=? AND date<=? AND date>=? AND quality_flag='ok' "
        "ORDER BY date DESC LIMIT 1",
        (symbol, as_of_str,
         (as_of - timedelta(days=QS_KRONOS_MAX_AGE_DAYS)).isoformat()),
    ).fetchone()
    if row:
        stack["kronos_5d"], stack["kronos_30d"] = float(row[0]), float(row[1])
    row = conn.execute(
        "SELECT sentiment, materiality FROM sentiment "
        "WHERE symbol=? AND window_end<=? AND window_end>=? AND sentiment IS NOT NULL "
        "ORDER BY sentiment ASC LIMIT 1",
        (symbol, as_of_str,
         (as_of - timedelta(days=QS_SENT_WINDOW_DAYS)).isoformat()),
    ).fetchone()
    if row:
        stack["sentiment_last7d"] = float(row[0])
        stack["sentiment_materiality"] = float(row[1]) if row[1] is not None else 0.0
    closes = [float(r[0]) for r in conn.execute(
        "SELECT close FROM mm_bars WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT ?",
        (symbol, as_of_str, QS_VOL_LOOKBACK + 1),
    ) if r[0]][::-1]
    if len(closes) >= 21:
        rets = pd.Series(closes).pct_change().dropna()
        stack["realized_vol_60d_ann"] = float(rets.std(ddof=1) * (252 ** 0.5))
    streak = 0
    for (s,) in conn.execute(
        "SELECT eps_surprise_pct FROM mm_earnings_events "
        "WHERE symbol=? AND report_date<=? AND eps_surprise_pct IS NOT NULL "
        "ORDER BY report_date DESC LIMIT 8",
        (symbol, as_of_str),
    ):
        if float(s) < 0:
            streak += 1
        else:
            break
    stack["earnings_miss_streak"] = streak
    return stack


def _qs_disqualify(stack: dict) -> str | None:
    ""
    vol = stack["realized_vol_60d_ann"]
    if vol is not None and vol > QS_VOL_MAX:
        return f"realized vol {vol:.0%} above {QS_VOL_MAX:.0%} ceiling"
    if stack["earnings_miss_streak"] >= QS_MISS_STREAK_MAX:
        return f"{stack['earnings_miss_streak']} consecutive earnings misses"
    sent, mat = stack["sentiment_last7d"], stack["sentiment_materiality"]
    if (sent is not None and sent <= QS_SENT_MAX
            and (mat or 0.0) >= QS_MAT_MIN):
        return f"sentiment {sent:+.2f} at materiality {mat:.2f}"
    return None


EXCLUDED_FINNHUB_SECTORS = frozenset({
    "Health Care", "Biotechnology", "Pharmaceuticals",
})


def _out_of_scope_symbols(conn: sqlite3.Connection) -> set[str]:
    ""
    rows = conn.execute(
        "SELECT symbol, finnhub_sector FROM mm_sector_map"
    ).fetchall()
    return {sym for sym, fs in rows if fs in EXCLUDED_FINNHUB_SECTORS}


RETIRED_STRATEGY_IDS = frozenset({
    1,
    13,
    16,
    21,
    23,
    43,
    56,
    57,
    58,
    59,
    60,
    63,
    65,
})

LIVE_MIN_DEFLATED_SHARPE = 0.5
LIVE_MIN_PROFIT_FACTOR = 1.3
LIVE_MAX_DRAWDOWN_FLOOR = -0.15

K_PER_SECTOR = 5
MAX_PICKS_TOTAL = 25
EXPORTER_MIN_DEFLATED_SHARPE = MIN_DEFLATED_SHARPE

EARNINGS_BLACKOUT_DAYS = 2

SCHEMA_VERSION = "3.0"
MACMINI_MIN_VERSION = "0.1.0"


@dataclass
class ExporterConfig:
    k_per_sector: int = K_PER_SECTOR
    max_picks: int = MAX_PICKS_TOTAL
    min_dsr_for_pick: float = EXPORTER_MIN_DEFLATED_SHARPE
    dl_soft_filter: bool = False
    dl_penalty: float = 0.5
    dl_model_id: str | None = None
    dl_rank_tilt: bool = False
    dl_tilt_weight: float = 0.10
    rl_annotate: bool = True
    rl_policy_id: str | None = None
    variant_types_enabled: bool = False
    selector_rank: bool = False
    selector_max_age_days: int = 5


def _refresh_leaderboard(conn: sqlite3.Connection,
                         funnel: list | None = None) -> int:
    ""
    candidates = conn.execute(
        """SELECT run_id, COUNT(*) AS n, MAX(created_at) AS last_write
           FROM mm_backtest_runs
           GROUP BY run_id ORDER BY last_write DESC LIMIT 5"""
    ).fetchall()
    if not candidates:
        log.warning("mm_backtest_runs is empty; leaderboard not refreshed")
        return 0
    quiet_cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    row_floor = max(n for _, n, _ in candidates) * 0.5
    run_id = None
    for cand_id, n, last_write in candidates:
        if last_write < quiet_cutoff and n >= row_floor:
            run_id = cand_id
            break
        log.warning("skipping run %s as incomplete (rows=%d, last_write=%s)",
                    cand_id, n, last_write)
    if run_id is None:
        log.warning("no complete backtest run among newest %d; leaderboard not refreshed",
                    len(candidates))
        return 0

    df = pd.read_sql_query(
        """SELECT strategy_id, symbol, sector, window_start,
                  deflated_sharpe, profit_factor, max_drawdown,
                  n_trades, hit_rate
           FROM mm_backtest_runs
           WHERE run_id = ?""",
        conn, params=[run_id],
    )
    if df.empty:
        return 0

    df = df.sort_values("window_start", kind="stable")
    df["row_num"] = df.groupby(["strategy_id", "symbol"]).cumcount(ascending=False)
    recent = df[df["row_num"] < SURVIVAL_WINDOWS]

    out_of_scope = _out_of_scope_symbols(conn)
    rows_out: list[dict] = []
    for (sid, sym, sector), g in recent.groupby(
        ["strategy_id", "symbol", "sector"], sort=False
    ):
        w = g["row_num"].map(_recency_weight)
        dsr_avg = float((g["deflated_sharpe"] * w).sum() / w.sum())
        dd_worst = float(g["max_drawdown"].min())
        traded = g[g["n_trades"] >= 1]
        total_trades = int(traded["n_trades"].sum())
        pf_median = _winsor_wmedian(
            traded["profit_factor"].tolist(),
            traded["row_num"].map(_recency_weight).tolist())
        qualifies = int(
            sym not in out_of_scope
            and sid not in RETIRED_STRATEGY_IDS
            and total_trades >= MIN_TOTAL_TRADES
            and dsr_avg >= MIN_DEFLATED_SHARPE
            and pf_median >= MIN_PROFIT_FACTOR
            and dd_worst >= MAX_DRAWDOWN_FLOOR
        )
        rows_out.append({
            "strategy_id": sid, "symbol": sym, "sector": sector,
            "deflated_sharpe_avg": dsr_avg, "profit_factor_avg": pf_median,
            "max_dd_worst": dd_worst, "n_windows_passed": int(len(traded)),
            "qualifies": qualifies,
            "total_trades": total_trades, "in_scope": sym not in out_of_scope,
            "retired": sid in RETIRED_STRATEGY_IDS,
        })
    agg = pd.DataFrame(rows_out)
    if agg.empty:
        return 0
    if funnel is not None:
        funnel.extend(_gate_counts(agg, run_id))

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM mm_leaderboard")
    for _, row in agg.iterrows():
        conn.execute(
            """INSERT INTO mm_leaderboard
               (strategy_id, symbol, sector, last_run_id,
                deflated_sharpe_avg, profit_factor_avg, max_dd_worst,
                n_windows_passed, qualifies, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(row["strategy_id"]), row["symbol"], row["sector"], run_id,
             float(row["deflated_sharpe_avg"]), float(row["profit_factor_avg"]),
             float(row["max_dd_worst"]), int(row["n_windows_passed"]),
             int(row["qualifies"]), now),
        )
    conn.commit()
    return len(agg)


def _gate_counts(agg: pd.DataFrame, run_id: str) -> list[tuple]:
    ""
    steps = [
        ("Every strategy-stock pair tested", None,
         f"backtest run {run_id}, last {SURVIVAL_WINDOWS} windows"),
        ("Traded at least once", agg["total_trades"] >= 1,
         "pairs whose strategy never fired carry no evidence"),
        (f"At least {MIN_TOTAL_TRADES} trades", agg["total_trades"] >= MIN_TOTAL_TRADES,
         "fewer trades than this is too small a sample to judge"),
        (f"Profit factor at least {MIN_PROFIT_FACTOR:g}",
         agg["profit_factor_avg"] >= MIN_PROFIT_FACTOR,
         "gross wins divided by gross losses"),
        (f"Worst drawdown no deeper than {MAX_DRAWDOWN_FLOOR:.0%}",
         agg["max_dd_worst"] >= MAX_DRAWDOWN_FLOOR, "loss control on the backtest"),
        *([(f"Deflated Sharpe at least {MIN_DEFLATED_SHARPE:g}",
            agg["deflated_sharpe_avg"] >= MIN_DEFLATED_SHARPE,
            "probability the result is not luck, given how many pairs were tried")]
          if MIN_DEFLATED_SHARPE > 0 else []),
        ("In sector scope", agg["in_scope"], "sector universe"),
        ("Strategy not retired", ~agg["retired"],
         f"{len(RETIRED_STRATEGY_IDS)} strategies retired for their "
         "live record"),
    ]
    out: list[tuple] = []
    mask = pd.Series(True, index=agg.index)
    for stage, cond, note in steps:
        if cond is not None:
            mask &= cond
        left = agg[mask]
        out.append((stage, int(len(left)), int(left["symbol"].nunique()), note))
    return out


def _write_funnel(conn: sqlite3.Connection, as_of: date, funnel: list[tuple],
                  fates: list[dict]) -> None:
    ""
    conn.execute(MM_PICK_FUNNEL)
    conn.execute(MM_PICK_FATES)
    as_of_str = as_of.isoformat()
    conn.execute("DELETE FROM mm_pick_funnel WHERE as_of_date = ?", (as_of_str,))
    conn.execute("DELETE FROM mm_pick_fates WHERE as_of_date = ?", (as_of_str,))
    conn.executemany(
        "INSERT INTO mm_pick_funnel VALUES (?, ?, ?, ?, ?, ?)",
        [(as_of_str, i, stage, cells, syms, note)
         for i, (stage, cells, syms, note) in enumerate(funnel, start=1)])
    conn.executemany(
        "INSERT OR REPLACE INTO mm_pick_fates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(as_of_str, f["strategy_id"], f["symbol"], f["strategy_name"],
          f["sector"], f["deflated_sharpe"], f["fate"], f["reason"])
         for f in fates])


def _compute_regime(conn: sqlite3.Connection, as_of: date) -> dict:
    as_of_str = as_of.isoformat()
    spy_rows = conn.execute(
        "SELECT date, close FROM mm_bars WHERE symbol='SPY' AND date <= ? "
        "ORDER BY date DESC LIMIT 250",
        (as_of_str,),
    ).fetchall()
    spy_50_vs_200 = None
    if len(spy_rows) >= 200:
        closes = pd.Series([r[1] for r in reversed(spy_rows)])
        sma50 = closes.tail(50).mean()
        sma200 = closes.tail(200).mean()
        spy_50_vs_200 = 1 if sma50 > sma200 else 0

    vix_close = None
    vix_regime = None
    vix_row = conn.execute(
        "SELECT close FROM mm_bars WHERE symbol='^VIX' AND date <= ? "
        "ORDER BY date DESC LIMIT 1",
        (as_of_str,),
    ).fetchone()
    if vix_row and vix_row[0] is not None:
        vix_close = float(vix_row[0])
        vix_regime = "low" if vix_close < 15 else ("high" if vix_close > 25 else "normal")

    fomc_within_3d = 0
    for f in FOMC_DATES:
        if abs((f - as_of).days) <= 3:
            fomc_within_3d = 1
            break

    horizon = (as_of + pd.Timedelta(days=EARNINGS_BLACKOUT_DAYS)).isoformat()
    earnings_rows = conn.execute(
        "SELECT DISTINCT symbol FROM mm_earnings_events "
        "WHERE report_date BETWEEN ? AND ?",
        (as_of_str, horizon),
    ).fetchall()
    blackout = ",".join(sorted(r[0] for r in earnings_rows))

    return {
        "as_of_date": as_of_str,
        "spy_50d_above_200d": spy_50_vs_200,
        "vix_close": vix_close,
        "vix_regime": vix_regime,
        "fomc_within_3d": fomc_within_3d,
        "earnings_blackout_symbols": blackout,
    }


def _refresh_regime(conn: sqlite3.Connection, as_of: date) -> None:
    r = _compute_regime(conn, as_of)
    conn.execute("DELETE FROM mm_regime WHERE as_of_date = ?", (r["as_of_date"],))
    conn.execute(
        """INSERT INTO mm_regime
           (as_of_date, spy_50d_above_200d, vix_close, vix_regime,
            fomc_within_3d, earnings_blackout_symbols)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (r["as_of_date"], r["spy_50d_above_200d"], r["vix_close"], r["vix_regime"],
         r["fomc_within_3d"], r["earnings_blackout_symbols"]),
    )
    conn.commit()


def _meta_one_liner(cls: type) -> str:
    ""
    import inspect

    mod = inspect.getmodule(cls)
    mod_doc = (mod.__doc__ or "").strip() if mod else ""
    own_doc = (cls.__dict__.get("__doc__") or "").strip()
    doc = mod_doc or own_doc or cls.name
    para = doc.split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())[:240]


def _refresh_strategy_meta(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM mm_strategy_meta")
    for sid, cls in sorted(STRATEGY_REGISTRY.items()):
        conn.execute(
            """INSERT INTO mm_strategy_meta
               (strategy_id, name, category, entry_rule, exit_rule, data_required)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sid, cls.name, cls.category, _meta_one_liner(cls),
             "see strategy docstring", ",".join(cls.data_required)),
        )
    conn.commit()


def compute_rec_id(as_of_date: str, symbol: str, strategy_id: int) -> str:
    ""
    payload = f"{as_of_date}|{symbol}|{strategy_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _today_signal_fires(
    db_path: Path, strategy_id: int, symbol: str, as_of: date
):
    ""
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        return None
    bars = load_bars(symbol, db_path=db_path)
    if bars.empty:
        return None
    bars = bars[bars.index <= pd.Timestamp(as_of)]
    if bars.empty:
        return None
    aux = load_aux(symbol, cls, db_path=db_path)
    try:
        sigs = cls().generate_signals(symbol, bars, **aux)
    except Exception as exc:
        log.warning("strategy #%d on %s raised %s", strategy_id, symbol, exc)
        return None
    if not sigs:
        return None
    cutoff = bars.index[-5] if len(bars) >= 5 else bars.index[0]
    enters = [s for s in sigs if s.action == "enter" and pd.Timestamp(s.date) >= cutoff]
    if not enters:
        return None
    return max(enters, key=lambda s: pd.Timestamp(s.date))


DL_MAX_STALENESS_DAYS = 7


def _load_dl_predictions(
    conn: sqlite3.Connection, as_of_date: str, model_id: str | None
) -> tuple[dict[str, float], str | None]:
    ""
    if model_id is None:
        row = conn.execute(
            "SELECT config_json FROM mm_dl_runs "
            "WHERE model_type = 'patchtst_inference' "
            "ORDER BY ended_at DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            import json
            try:
                model_id = json.loads(row[0]).get("model_id")
            except (ValueError, TypeError):
                model_id = None
        if model_id is None:
            row = conn.execute(
                "SELECT model_id FROM mm_dl_signals "
                "WHERE date <= ? GROUP BY model_id "
                "ORDER BY COUNT(*) DESC LIMIT 1",
                (as_of_date,),
            ).fetchone()
            if row:
                model_id = row[0]
    if model_id is None:
        return {}, None
    rows = conn.execute(
        """SELECT symbol, predicted_5d_return, date FROM (
             SELECT symbol, predicted_5d_return, date,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) rn
             FROM mm_dl_signals
             WHERE model_id = ? AND date <= ?
           ) WHERE rn = 1""",
        (model_id, as_of_date),
    ).fetchall()
    if not rows:
        return {}, model_id
    from datetime import date as _date, timedelta
    as_of_d = _date.fromisoformat(as_of_date)
    cutoff = (as_of_d - timedelta(days=DL_MAX_STALENESS_DAYS)).isoformat()
    return (
        {sym: float(pred) for sym, pred, d in rows
         if pred is not None and d >= cutoff},
        model_id,
    )


RL_MAX_STALENESS_DAYS = 7


def _load_rl_actions(
    conn: sqlite3.Connection, as_of_date: str, policy_id: str | None
) -> tuple[dict[str, int], str | None]:
    ""
    if policy_id is None:
        row = conn.execute(
            "SELECT config_json FROM mm_dl_runs "
            "WHERE model_type LIKE '%ppo_inference%' "
            "ORDER BY ended_at DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            import json
            try:
                policy_id = json.loads(row[0]).get("policy_id")
            except (ValueError, TypeError):
                policy_id = None
        if policy_id is None:
            row = conn.execute(
                "SELECT policy_id FROM mm_rl_signals "
                "WHERE date <= ? GROUP BY policy_id "
                "ORDER BY COUNT(*) DESC LIMIT 1",
                (as_of_date,),
            ).fetchone()
            if row:
                policy_id = row[0]
    if policy_id is None:
        return {}, None
    rows = conn.execute(
        """SELECT symbol, action, date FROM (
             SELECT symbol, action, date,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) rn
             FROM mm_rl_signals
             WHERE policy_id = ? AND date <= ?
           ) WHERE rn = 1""",
        (policy_id, as_of_date),
    ).fetchall()
    if not rows:
        return {}, policy_id
    from datetime import date as _date, timedelta
    as_of_d = _date.fromisoformat(as_of_date)
    cutoff = (as_of_d - timedelta(days=RL_MAX_STALENESS_DAYS)).isoformat()
    return (
        {sym: int(action) for sym, action, d in rows if d >= cutoff},
        policy_id,
    )


def _load_selector_ranks(
    conn: sqlite3.Connection, as_of: date, max_age_days: int
) -> dict[str, float]:
    ""
    try:
        row = conn.execute(
            "SELECT MAX(as_of_date) FROM mm_selector_scores WHERE as_of_date <= ?",
            (as_of.isoformat(),)).fetchone()
    except sqlite3.OperationalError:
        return {}
    if not row or not row[0]:
        return {}
    if (as_of - date.fromisoformat(row[0])).days > max_age_days:
        return {}
    return {s: float(r) for s, r in conn.execute(
        "SELECT symbol, rank_pct FROM mm_selector_scores "
        "WHERE as_of_date = ? AND rank_pct IS NOT NULL", (row[0],))}


def _refresh_active_picks(
    conn: sqlite3.Connection, db_path: Path, as_of: date, cfg: ExporterConfig,
    funnel: list | None = None,
) -> int:
    ""
    conn.execute("DELETE FROM mm_active_picks")

    candidates_df = pd.read_sql_query(
        """SELECT * FROM mm_leaderboard
           WHERE qualifies = 1 AND deflated_sharpe_avg >= ?
           ORDER BY deflated_sharpe_avg DESC""",
        conn, params=[cfg.min_dsr_for_pick],
    )
    funnel = list(funnel or [])
    if candidates_df.empty:
        log.info("no qualifying leaderboard rows; mm_active_picks left empty")
        funnel.append(("Picks shipped", 0, 0, "no pair cleared the gates above"))
        _write_funnel(conn, as_of, funnel, [])
        conn.commit()
        return 0

    blackout = set()
    blackout_row = conn.execute(
        "SELECT earnings_blackout_symbols FROM mm_regime WHERE as_of_date = ?",
        (as_of.isoformat(),),
    ).fetchone()
    if blackout_row and blackout_row[0]:
        blackout = set(blackout_row[0].split(","))

    dl_preds: dict[str, float] = {}
    dl_model_id: str | None = None
    if cfg.dl_soft_filter:
        dl_preds, dl_model_id = _load_dl_predictions(conn, as_of.isoformat(), cfg.dl_model_id)
        log.info("dl soft-filter: model_id=%s coverage=%d symbols",
                 dl_model_id, len(dl_preds))

    dl_ranks: dict[str, float] = {}
    if cfg.dl_rank_tilt:
        tilt_preds, tilt_model_id = _load_dl_predictions(
            conn, as_of.isoformat(), cfg.dl_model_id)
        if tilt_preds:
            dl_ranks = pd.Series(tilt_preds).rank(pct=True).to_dict()
        log.info("dl rank tilt: model_id=%s coverage=%d symbols weight=%g",
                 tilt_model_id, len(dl_ranks), cfg.dl_tilt_weight)

    rl_actions: dict[str, int] = {}
    rl_policy_id: str | None = None
    if cfg.rl_annotate:
        rl_actions, rl_policy_id = _load_rl_actions(conn, as_of.isoformat(), cfg.rl_policy_id)
        log.info("rl annotation: policy_id=%s coverage=%d symbols",
                 rl_policy_id, len(rl_actions))

    out_of_scope = _out_of_scope_symbols(conn)
    remembered = outcome_memory.blocked_symbols(conn)
    selector_ranks = _load_selector_ranks(conn, as_of, cfg.selector_max_age_days)
    held_back: dict[str, str] = {}
    picks: list[dict] = []
    picked: dict[str, dict] = {}
    sector_count: dict[str, int] = {}
    veto_cache: dict[str, str | None] = {}
    stack_cache: dict[str, dict] = {}
    qs_cache: dict[str, str | None] = {}
    fates: list[dict] = []

    def _fate(row, fate: str, reason: str) -> None:
        sid = int(row["strategy_id"])
        fates.append({
            "strategy_id": sid, "symbol": row["symbol"],
            "strategy_name": STRATEGY_REGISTRY[sid].name if sid in STRATEGY_REGISTRY else "",
            "sector": row["sector"] or "Other",
            "deflated_sharpe": float(row["deflated_sharpe_avg"]),
            "fate": fate, "reason": reason,
        })

    for _, row in candidates_df.iterrows():
        sector = row["sector"] or "Other"
        if sector_count.get(sector, 0) >= cfg.k_per_sector:
            _fate(row, "sector cap",
                  f"{sector} already has {cfg.k_per_sector} picks today")
            continue
        if row["symbol"] in blackout:
            _fate(row, "earnings blackout",
                  f"reports earnings within {EARNINGS_BLACKOUT_DAYS} days")
            continue
        if row["symbol"] in out_of_scope:
            _fate(row, "out of scope", "outside the sector universe")
            continue
        if row["symbol"] in remembered:
            held_back[row["symbol"]] = remembered[row["symbol"]]
            _fate(row, "outcome memory", remembered[row["symbol"]])
            continue
        if row["symbol"] not in veto_cache:
            veto_cache[row["symbol"]] = _veto_veto(conn, row["symbol"], as_of)
        if veto_cache[row["symbol"]] is not None:
            _fate(row, "tradability screen", veto_cache[row["symbol"]])
            continue
        if row["symbol"] not in stack_cache:
            stack_cache[row["symbol"]] = _model_stack(conn, row["symbol"], as_of)
            qs_cache[row["symbol"]] = _qs_disqualify(stack_cache[row["symbol"]])
        if qs_cache[row["symbol"]] is not None:
            _fate(row, "quality screen", qs_cache[row["symbol"]])
            continue
        stack = stack_cache[row["symbol"]]
        signal = _today_signal_fires(db_path, int(row["strategy_id"]), row["symbol"], as_of)
        if signal is None:
            _fate(row, "no entry signal",
                  "the strategy has not signalled a buy in the last 5 trading days")
            continue
        if row["symbol"] in picked:
            _fate(row, "merged", "second strategy on a stock already picked today")
            sid = int(row["strategy_id"])
            picked[row["symbol"]]["rationale"] += (
                f" | also fired: #{sid} {STRATEGY_REGISTRY[sid].name}")
            continue
        ar_row = conn.execute(
            "SELECT archetype FROM mm_sector_map WHERE symbol = ?", (row["symbol"],)
        ).fetchone()
        archetype = (ar_row[0] if ar_row else None)
        size_pct, stop_frac, target_frac = _atr_bracket(conn, row["symbol"], as_of)
        entry_type = exit_type = trail_pct = None
        if cfg.variant_types_enabled:
            vs = conn.execute(
                "SELECT entry_type, exit_type FROM mm_variant_selection "
                "WHERE strategy_id = ? AND symbol = ?",
                (int(row["strategy_id"]), row["symbol"]),
            ).fetchone()
            if vs is not None:
                entry_type, exit_type = vs
                if exit_type in ("trail", "trail_tp"):
                    trail_pct = stop_frac
        score = float(row["deflated_sharpe_avg"]) * float(signal.conviction)
        rationale = (f"{signal.reason} | persona: "
                     f"{persona_of(STRATEGY_REGISTRY[int(row['strategy_id'])])}")
        if entry_type or exit_type:
            rationale = f"{rationale} | exec={entry_type or 'market_open'}/{exit_type or 'bracket'}"
        dl_pred = dl_preds.get(row["symbol"]) if cfg.dl_soft_filter else None
        if dl_pred is not None:
            if dl_pred < 0:
                score *= cfg.dl_penalty
                rationale = f"{rationale} | DL={dl_pred*100:+.2f}% (×{cfg.dl_penalty:g})"
            else:
                rationale = f"{rationale} | DL={dl_pred*100:+.2f}%"
        dl_rank = dl_ranks.get(row["symbol"]) if cfg.dl_rank_tilt else None
        if dl_rank is not None:
            score *= 1.0 + cfg.dl_tilt_weight * (2.0 * dl_rank - 1.0)
            rationale = f"{rationale} | DLrank={dl_rank:.2f}"
        sel = selector_ranks.get(row["symbol"])
        if sel is not None:
            rationale = f"{rationale} | selector: ranks above {sel:.0%} of stocks"
            if cfg.selector_rank:
                score = sel
        rl_action = rl_actions.get(row["symbol"]) if cfg.rl_annotate else None
        if rl_action is not None:
            rl_label = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(rl_action, str(rl_action))
            rationale = f"{rationale} | RL={rl_label}"
        picks.append({
            "strategy_id": int(row["strategy_id"]),
            "strategy_name": STRATEGY_REGISTRY[int(row["strategy_id"])].name,
            "symbol": row["symbol"], "sector": sector, "archetype": archetype,
            "deflated_sharpe": float(row["deflated_sharpe_avg"]),
            "profit_factor": float(row["profit_factor_avg"]),
            "max_dd": float(row["max_dd_worst"]),
            "suggested_size_pct": size_pct,
            "suggested_stop_loss_pct": stop_frac,
            "suggested_target_pct": target_frac,
            "rationale": rationale,
            "score": score,
            "entry_type": entry_type,
            "exit_type": exit_type,
            "trail_pct": trail_pct,
            "kronos_5d": stack["kronos_5d"],
            "kronos_30d": stack["kronos_30d"],
            "sentiment_last7d": stack["sentiment_last7d"],
            "sentiment_materiality": stack["sentiment_materiality"],
            "realized_vol_60d_ann": stack["realized_vol_60d_ann"],
            "earnings_miss_streak": stack["earnings_miss_streak"],
        })
        picked[row["symbol"]] = picks[-1]
        sector_count[sector] = sector_count.get(sector, 0) + 1
        _fate(row, "picked", signal.reason)

    if held_back:
        log.info("outcome memory held back %d symbol(s): %s", len(held_back),
                 "; ".join(f"{s} ({r})" for s, r in sorted(held_back.items())))
    vetoed = {s: r for s, r in veto_cache.items() if r is not None}
    if vetoed:
        log.info("liquidity screens vetoed %d symbol(s): %s", len(vetoed),
                 "; ".join(f"{s} ({r})" for s, r in sorted(vetoed.items())))
    disqualified = {s: r for s, r in qs_cache.items() if r is not None}
    if disqualified:
        log.info("quality screen disqualified %d symbol(s): %s",
                 len(disqualified),
                 "; ".join(f"{s} ({r})" for s, r in sorted(disqualified.items())))

    picks.sort(key=lambda p: -p["score"])
    for p in picks[cfg.max_picks:]:
        for f in fates:
            if f["symbol"] == p["symbol"] and f["fate"] == "picked":
                f["fate"], f["reason"] = "over the daily limit", (
                    f"ranked below the top {cfg.max_picks}")
    picks = picks[: cfg.max_picks]

    left = fates
    for stage, drop, note in (
        ("Not in earnings blackout", {"earnings blackout", "out of scope"},
         f"no entries within {EARNINGS_BLACKOUT_DAYS} days of an earnings report"),
        ("Not held back by outcome memory", {"outcome memory"},
         "an earlier recommendation broke its stop and is still below entry"),
        ("Tradable (price and liquidity)", {"tradability screen"},
         f"price at least ${MIN_PRICE:g}, ${MIN_DOLLAR_ADV/1e6:g}M traded daily, "
         "not in free fall"),
        ("Passed the quality screen", {"quality screen"},
         f"volatility under {QS_VOL_MAX:.0%}, no earnings-miss streak, "
         "no severe negative news"),
        ("Strategy signalled a buy in the last 5 days", {"no entry signal"},
         "a pair with a good record still needs its entry condition to occur"),
        ("Picks shipped", {"merged", "sector cap", "over the daily limit"},
         f"one per stock, {cfg.k_per_sector} per sector, {cfg.max_picks} in total"),
    ):
        left = [f for f in left if f["fate"] not in drop]
        funnel.append((stage, len(left), len({f["symbol"] for f in left}), note))
    _write_funnel(conn, as_of, funnel, fates)
    as_of_str = as_of.isoformat()
    for i, p in enumerate(picks, start=1):
        rec_id = compute_rec_id(as_of_str, p["symbol"], p["strategy_id"])
        shadow_sizing.record_shadow_size(
            conn, rec_id=rec_id, as_of_date=as_of_str, symbol=p["symbol"],
            strategy_id=p["strategy_id"],
            deflated_sharpe=p["deflated_sharpe"],
            real_size_pct=p["suggested_size_pct"],
            stop_frac=p["suggested_stop_loss_pct"])
        conn.execute(
            """INSERT INTO mm_active_picks
               (rank, strategy_id, strategy_name, symbol, sector, archetype,
                deflated_sharpe, profit_factor, max_dd,
                suggested_size_pct, suggested_stop_loss_pct, suggested_target_pct,
                rationale, rec_id, as_of_date, entry_type, exit_type, trail_pct,
                kronos_5d, kronos_30d, sentiment_last7d, sentiment_materiality,
                realized_vol_60d_ann, earnings_miss_streak)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
            (i, p["strategy_id"], p["strategy_name"], p["symbol"], p["sector"],
             p["archetype"], p["deflated_sharpe"], p["profit_factor"], p["max_dd"],
             p["suggested_size_pct"], p["suggested_stop_loss_pct"],
             p["suggested_target_pct"], p["rationale"], rec_id, as_of_str,
             p["entry_type"], p["exit_type"], p["trail_pct"],
             p["kronos_5d"], p["kronos_30d"], p["sentiment_last7d"],
             p["sentiment_materiality"], p["realized_vol_60d_ann"],
             p["earnings_miss_streak"]),
        )
    conn.commit()
    return len(picks)


def _refresh_export_metadata(conn: sqlite3.Connection, as_of: date) -> None:
    latest_run = conn.execute(
        "SELECT run_id FROM mm_backtest_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    items = [
        ("as_of_date", as_of.isoformat()),
        ("generated_at", datetime.now(timezone.utc).isoformat()),
        ("backtest_run_id", latest_run[0] if latest_run else ""),
        ("schema_version", SCHEMA_VERSION),
        ("macmini_min_version", MACMINI_MIN_VERSION),
    ]
    for k, v in items:
        conn.execute("INSERT OR REPLACE INTO mm_export_metadata VALUES (?, ?)", (k, v))
    conn.commit()


def refresh_rankings(
    db_path: Path | str | None = None,
    as_of: date | None = None,
    cfg: ExporterConfig | None = None,
) -> dict:
    db = resolve_db(db_path)
    as_of = as_of or date.today()
    cfg = cfg or ExporterConfig()

    with connect(db) as conn:
        funnel: list[tuple] = []
        n_leader = _refresh_leaderboard(conn, funnel)
        _refresh_strategy_meta(conn)
        _refresh_regime(conn, as_of)
        try:
            outcome_memory.rebuild(conn, as_of)
        except Exception as exc:
            log.warning("outcome memory rebuild failed (%s); using the stored table", exc)
        n_picks = _refresh_active_picks(conn, db, as_of, cfg, funnel)
        _refresh_export_metadata(conn, as_of)
    return {
        "leaderboard_rows": n_leader,
        "active_picks": n_picks,
        "as_of_date": as_of.isoformat(),
        "dl_soft_filter": cfg.dl_soft_filter,
        "rl_annotate": cfg.rl_annotate,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="macmini ranking exporter")
    p.add_argument("--db", default=None)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    p.add_argument("--dl-filter", action="store_true",
                   help="ENABLE the PatchTST soft-filter (off by default; no OOS skill)")
    p.add_argument("--no-dl-filter", action="store_true",
                   help="(no-op; the filter is already off by default)")
    p.add_argument("--dl-penalty", type=float, default=None,
                   help="score multiplier for negative DL predictions (default 0.5)")
    p.add_argument("--dl-model-id", default=None,
                   help="pin a specific mm_dl_signals model_id; default = latest from mm_dl_runs")
    p.add_argument("--no-rl-annotation", action="store_true",
                   help="disable the RL annotation on rationale (informational only, no score effect)")
    p.add_argument("--rl-policy-id", default=None,
                   help="pin a specific mm_rl_signals policy_id; default = latest from mm_dl_runs")
    args = p.parse_args(argv)
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    cfg = ExporterConfig(
        dl_soft_filter=args.dl_filter,
        rl_annotate=not args.no_rl_annotation,
    )
    if args.dl_penalty is not None:
        cfg.dl_penalty = args.dl_penalty
    if args.dl_model_id is not None:
        cfg.dl_model_id = args.dl_model_id
    if args.rl_policy_id is not None:
        cfg.rl_policy_id = args.rl_policy_id
    out = refresh_rankings(db_path=args.db, as_of=as_of, cfg=cfg)
    print(f"Exporter done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
