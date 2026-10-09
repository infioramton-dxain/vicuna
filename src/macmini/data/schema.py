""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from macmini.config import DB_PATH, resolve_db

SENTIMENT_NATIVE_TABLES = {
    "sentiment",
    "term",
    "runs",
    "checkpoint",
    "dropped_articles",
    "term_stats",
}

MM_BARS = """
CREATE TABLE IF NOT EXISTS mm_bars (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      INTEGER,
    source      TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);
"""

MM_BARS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mm_bars_date        ON mm_bars(date);",
    "CREATE INDEX IF NOT EXISTS idx_mm_bars_symbol_date ON mm_bars(symbol, date DESC);",
]

MM_BARS_INGEST_LOG = """
CREATE TABLE IF NOT EXISTS mm_bars_ingest_log (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT,
    ended_at            TEXT,
    symbols_requested   INTEGER,
    symbols_ok          INTEGER,
    symbols_empty       INTEGER,
    symbols_failed      INTEGER,
    notes               TEXT
);
"""

MM_FUNDAMENTALS_QUARTERLY = """
CREATE TABLE IF NOT EXISTS mm_fundamentals_quarterly (
    symbol              TEXT NOT NULL,
    period_end          TEXT NOT NULL,
    revenue             REAL,
    earnings            REAL,
    eps                 REAL,
    operating_cash_flow REAL,
    free_cash_flow      REAL,
    total_debt          REAL,
    total_equity        REAL,
    ebitda              REAL,
    market_cap          REAL,
    pe_ratio            REAL,
    pb_ratio            REAL,
    ev_ebitda           REAL,
    roic                REAL,
    shares_outstanding  REAL,
    dividend_yield      REAL,
    buyback_yield       REAL,
    capex               REAL,
    f_score             INTEGER,
    source              TEXT NOT NULL,
    fetched_at          TEXT NOT NULL,
    PRIMARY KEY (symbol, period_end)
);
"""

MM_FUNDAMENTALS_ANNUAL = """
CREATE TABLE IF NOT EXISTS mm_fundamentals_annual (
    symbol              TEXT NOT NULL,
    period_end          TEXT NOT NULL,
    revenue             REAL,
    earnings            REAL,
    eps                 REAL,
    operating_cash_flow REAL,
    free_cash_flow      REAL,
    total_debt          REAL,
    total_equity        REAL,
    shares_outstanding  REAL,
    capex               REAL,
    source              TEXT NOT NULL,
    fetched_at          TEXT NOT NULL,
    PRIMARY KEY (symbol, period_end)
);
"""

MM_FUNDAMENTALS_GROWTH = """
CREATE TABLE IF NOT EXISTS mm_fundamentals_growth (
    symbol             TEXT NOT NULL,
    as_of_date         TEXT NOT NULL,
    latest_period_end  TEXT NOT NULL,
    n_quarters         INTEGER NOT NULL,
    rev_chg_mrq_yoy    REAL,
    rev_chg_ttm        REAL,
    rev_growth_5y_avg  REAL,
    eps_chg_mrq_yoy    REAL,
    eps_chg_ttm        REAL,
    eps_growth_5y_avg  REAL,
    computed_at        TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of_date)
);
"""

MM_ANALYST_ESTIMATES = """
CREATE TABLE IF NOT EXISTS mm_analyst_estimates (
    symbol              TEXT NOT NULL,
    as_of_date          TEXT NOT NULL,
    fy0_eps_consensus   REAL,
    fy1_eps_consensus   REAL,
    eps_revision_30d    REAL,
    rating_buy          INTEGER,
    rating_hold         INTEGER,
    rating_sell         INTEGER,
    PRIMARY KEY (symbol, as_of_date)
);
"""

MM_EARNINGS_EVENTS = """
CREATE TABLE IF NOT EXISTS mm_earnings_events (
    symbol                  TEXT NOT NULL,
    report_date             TEXT NOT NULL,
    eps_actual              REAL,
    eps_estimate            REAL,
    eps_surprise_pct        REAL,
    revenue_actual          REAL,
    revenue_estimate        REAL,
    revenue_surprise_pct    REAL,
    guidance_raised         INTEGER,
    PRIMARY KEY (symbol, report_date)
);
"""

MM_INSIDER_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS mm_insider_transactions (
    symbol              TEXT NOT NULL,
    transaction_date    TEXT NOT NULL,
    insider_name        TEXT,
    insider_title       TEXT,
    transaction_type    TEXT,
    shares              INTEGER,
    value_usd           REAL,
    source_url          TEXT,
    PRIMARY KEY (symbol, transaction_date, insider_name, transaction_type)
);
"""

MM_SEC_FACTS = """
CREATE TABLE IF NOT EXISTS mm_sec_facts (
    symbol          TEXT NOT NULL,
    cik             TEXT NOT NULL,
    tag             TEXT NOT NULL,
    unit            TEXT NOT NULL,
    period_end      TEXT NOT NULL,
    period_start    TEXT NOT NULL DEFAULT '',
    fy              INTEGER,
    fp              TEXT,
    form            TEXT,
    filed           TEXT NOT NULL,
    frame           TEXT,
    value           REAL NOT NULL,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (symbol, tag, unit, period_end, period_start, filed)
);
"""
MM_SEC_FACTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sec_facts_sym_tag_end ON mm_sec_facts (symbol, tag, period_end);",
    "CREATE INDEX IF NOT EXISTS idx_sec_facts_sym_filed ON mm_sec_facts (symbol, filed);",
]

MM_CFTC_COT_WEEKLY = """
CREATE TABLE IF NOT EXISTS mm_cftc_cot_weekly (
    commodity               TEXT NOT NULL,
    report_date             TEXT NOT NULL,
    commercial_long         INTEGER,
    commercial_short        INTEGER,
    commercial_net          INTEGER,
    noncommercial_long      INTEGER,
    noncommercial_short     INTEGER,
    PRIMARY KEY (commodity, report_date)
);
"""

MM_SECTOR_ETF_FLOWS_WEEKLY = """
CREATE TABLE IF NOT EXISTS mm_sector_etf_flows_weekly (
    etf_ticker      TEXT NOT NULL,
    week_end        TEXT NOT NULL,
    net_flow_usd    REAL,
    aum_usd         REAL,
    PRIMARY KEY (etf_ticker, week_end)
);
"""

MM_DIVIDENDS = """
CREATE TABLE IF NOT EXISTS mm_dividends (
    symbol      TEXT NOT NULL,
    ex_date     TEXT NOT NULL,
    rate        REAL NOT NULL,
    special     INTEGER NOT NULL DEFAULT 0,
    is_foreign  INTEGER NOT NULL DEFAULT 0,
    source      TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (symbol, ex_date, rate, special)
);
"""

MM_SECTOR_MAP = """
CREATE TABLE IF NOT EXISTS mm_sector_map (
    symbol          TEXT PRIMARY KEY,
    sector          TEXT NOT NULL,
    finnhub_sector  TEXT,
    market_cap_usd  REAL,
    archetype       TEXT
);
"""

MM_BACKTEST_RUNS = """
CREATE TABLE IF NOT EXISTS mm_backtest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    strategy_id         INTEGER NOT NULL,
    symbol              TEXT NOT NULL,
    sector              TEXT NOT NULL,
    window_start        TEXT NOT NULL,
    window_end          TEXT NOT NULL,
    n_trades            INTEGER,
    sharpe              REAL,
    deflated_sharpe     REAL,
    profit_factor       REAL,
    max_drawdown        REAL,
    hit_rate            REAL,
    total_return        REAL,
    avg_holding_days    REAL,
    skew                REAL,
    kurt                REAL,
    p_value             REAL,
    p_value_bonferroni  REAL,
    created_at          TEXT NOT NULL,
    entry_type          TEXT,
    exit_type           TEXT
);
"""

MM_BACKTEST_RUNS_MIGRATIONS: list[tuple[str, str]] = [
    ("entry_type", "TEXT"),
    ("exit_type",  "TEXT"),
]

MM_BACKTEST_RUNS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mm_runs_strategy_symbol ON mm_backtest_runs(strategy_id, symbol);",
    "CREATE INDEX IF NOT EXISTS idx_mm_runs_run_id          ON mm_backtest_runs(run_id);",
]

MM_VARIANT_SELECTION = """
CREATE TABLE IF NOT EXISTS mm_variant_selection (
    strategy_id     INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    entry_type      TEXT NOT NULL,
    exit_type       TEXT NOT NULL,
    dsr_ref         REAL,
    dsr_best        REAL,
    n_variants      INTEGER,
    run_id          TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (strategy_id, symbol)
);
"""

MM_LEADERBOARD = """
CREATE TABLE IF NOT EXISTS mm_leaderboard (
    strategy_id             INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    sector                  TEXT NOT NULL,
    last_run_id             TEXT NOT NULL,
    deflated_sharpe_avg     REAL,
    profit_factor_avg       REAL,
    max_dd_worst            REAL,
    n_windows_passed        INTEGER,
    qualifies               INTEGER,
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (strategy_id, symbol)
);
"""

MM_EXPORT_METADATA = """
CREATE TABLE IF NOT EXISTS mm_export_metadata (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""

MM_ACTIVE_PICKS = """
CREATE TABLE IF NOT EXISTS mm_active_picks (
    rank                    INTEGER PRIMARY KEY,
    strategy_id             INTEGER NOT NULL,
    strategy_name           TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    sector                  TEXT NOT NULL,
    archetype               TEXT,
    deflated_sharpe         REAL,
    profit_factor           REAL,
    max_dd                  REAL,
    suggested_size_pct      REAL,
    suggested_stop_loss_pct REAL,
    suggested_target_pct    REAL,
    rationale               TEXT,
    rec_id                  TEXT,
    as_of_date              DATE,
    entry_type              TEXT,
    exit_type               TEXT,
    trail_pct               REAL,
    kronos_5d               REAL,
    kronos_30d              REAL,
    sentiment_last7d        REAL,
    sentiment_materiality   REAL,
    realized_vol_60d_ann    REAL,
    earnings_miss_streak    INTEGER
);
"""

MM_ACTIVE_PICKS_MIGRATIONS: list[tuple[str, str]] = [
    ("rec_id",     "TEXT"),
    ("as_of_date", "DATE"),
    ("entry_type", "TEXT"),
    ("exit_type",  "TEXT"),
    ("trail_pct",  "REAL"),
    ("kronos_5d",             "REAL"),
    ("kronos_30d",            "REAL"),
    ("sentiment_last7d",      "REAL"),
    ("sentiment_materiality", "REAL"),
    ("realized_vol_60d_ann",  "REAL"),
    ("earnings_miss_streak",  "INTEGER"),
]

MM_ACTIVE_PICKS_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_picks_rec_id ON mm_active_picks(rec_id);",
    "CREATE INDEX        IF NOT EXISTS idx_picks_as_of  ON mm_active_picks(as_of_date);",
]

MM_REGIME = """
CREATE TABLE IF NOT EXISTS mm_regime (
    as_of_date                  TEXT PRIMARY KEY,
    spy_50d_above_200d          INTEGER,
    vix_close                   REAL,
    vix_regime                  TEXT,
    fomc_within_3d              INTEGER,
    earnings_blackout_symbols   TEXT
);
"""

MM_PICK_FUNNEL = """
CREATE TABLE IF NOT EXISTS mm_pick_funnel (
    as_of_date      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    stage           TEXT NOT NULL,
    cells           INTEGER,
    symbols         INTEGER,
    note            TEXT,
    PRIMARY KEY (as_of_date, seq)
);
"""

MM_PICK_FATES = """
CREATE TABLE IF NOT EXISTS mm_pick_fates (
    as_of_date      TEXT NOT NULL,
    strategy_id     INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    strategy_name   TEXT,
    sector          TEXT,
    deflated_sharpe REAL,
    fate            TEXT NOT NULL,
    reason          TEXT,
    PRIMARY KEY (as_of_date, strategy_id, symbol)
);
"""

MM_STRATEGY_META = """
CREATE TABLE IF NOT EXISTS mm_strategy_meta (
    strategy_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    entry_rule      TEXT NOT NULL,
    exit_rule       TEXT NOT NULL,
    data_required   TEXT
);
"""

MM_INDICATORS_DAILY = """
CREATE TABLE IF NOT EXISTS mm_indicators_daily (
    symbol              TEXT NOT NULL,
    date                TEXT NOT NULL,
    sma20               REAL,
    sma50               REAL,
    sma200              REAL,
    ema12               REAL,
    ema26               REAL,
    ema200              REAL,
    rsi14               REAL,
    rsi2                REAL,
    atr14               REAL,
    macd_line           REAL,
    macd_signal         REAL,
    macd_hist           REAL,
    bb_upper            REAL,
    bb_middle           REAL,
    bb_lower            REAL,
    bb_width            REAL,
    adx14               REAL,
    williams_r14        REAL,
    ibs                 REAL,
    volume_zscore_50    REAL,
    PRIMARY KEY (symbol, date)
);
"""

MM_INDICATORS_INDEX = "CREATE INDEX IF NOT EXISTS idx_mm_ind_symbol_date ON mm_indicators_daily(symbol, date DESC);"

MM_TRAINING_LABELS = """
CREATE TABLE IF NOT EXISTS mm_training_labels (
    symbol              TEXT NOT NULL,
    date                TEXT NOT NULL,
    forward_1d_return   REAL,
    forward_5d_return   REAL,
    forward_10d_return  REAL,
    forward_5d_class    INTEGER,
    forward_5d_sharpe   REAL,
    max_dd_5d           REAL,
    PRIMARY KEY (symbol, date)
);
"""

MM_DL_SIGNALS = """
CREATE TABLE IF NOT EXISTS mm_dl_signals (
    model_id                TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    date                    TEXT NOT NULL,
    predicted_5d_return     REAL,
    predicted_class         INTEGER,
    confidence              REAL,
    created_at              TEXT NOT NULL,
    PRIMARY KEY (model_id, symbol, date)
);
"""

MM_RL_SIGNALS = """
CREATE TABLE IF NOT EXISTS mm_rl_signals (
    policy_id           TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    date                TEXT NOT NULL,
    action              INTEGER,
    suggested_size_pct  REAL,
    policy_logits_json  TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (policy_id, symbol, date)
);
"""

MM_DL_RUNS = """
CREATE TABLE IF NOT EXISTS mm_dl_runs (
    run_id              TEXT PRIMARY KEY,
    model_type          TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    epochs_or_timesteps INTEGER,
    best_val_metric     REAL,
    best_metric_name    TEXT,
    config_json         TEXT,
    checkpoint_path     TEXT,
    notes               TEXT
);
"""


MM_KRONOS_SIGNALS = """
CREATE TABLE IF NOT EXISTS mm_kronos_signals (
    model_id                TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    date                    TEXT NOT NULL,
    predicted_5d_return     REAL,
    predicted_30d_return    REAL,
    predicted_5d_vol        REAL,
    sample_count            INTEGER,
    quality_flag            TEXT,
    created_at              TEXT NOT NULL,
    PRIMARY KEY (model_id, symbol, date)
);
"""

MM_KRONOS_SIGNALS_MIGRATIONS: list[tuple[str, str]] = [
    ("quality_flag", "TEXT"),
]

MM_KRONOS_PATHS = """
CREATE TABLE IF NOT EXISTS mm_kronos_paths (
    model_id            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    as_of_date          TEXT NOT NULL,
    step                INTEGER NOT NULL,
    date                TEXT NOT NULL,
    open                REAL,
    high                REAL,
    low                 REAL,
    close               REAL,
    volume              REAL,
    close_lo            REAL,
    close_hi            REAL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (model_id, symbol, as_of_date, step)
);
"""

MM_KRONOS_PATHS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_mm_kronos_paths_symbol "
    "ON mm_kronos_paths(symbol, as_of_date DESC);"
)


MM_EXIT_SIGNALS = """
CREATE TABLE IF NOT EXISTS mm_exit_signals (
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    entry_date      TEXT,
    avg_entry_price REAL,
    hwm_price       REAL,
    trail_stop      REAL,
    trail_pct       REAL,
    trail_source    TEXT,
    chandelier      REAL,
    kronos_path_peak_step  INTEGER,
    kronos_path_peak_price REAL,
    kronos_5d       REAL,
    rsi14           REAL,
    rsi_rollover    INTEGER,
    macd_hist_shrink INTEGER,
    roc5_cross_zero INTEGER,
    adx14           REAL,
    momentum_state  TEXT,
    suggested_action TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (as_of_date, symbol)
);
"""


MM_QUALITY_CANDIDATES = """
CREATE TABLE IF NOT EXISTS mm_quality_candidates (
    strategy_id     TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    sector          TEXT,
    close           REAL,
    market_cap_usd  REAL,
    rev_growth_ttm  REAL,
    net_margin      REAL,
    free_cash_flow  REAL,
    debt_to_equity  REAL,
    rating_buy      INTEGER,
    rating_hold     INTEGER,
    rating_sell     INTEGER,
    kronos_5d       REAL,
    kronos_30d      REAL,
    dip_triggers    TEXT,
    suggested_stop  REAL,
    held            INTEGER,
    in_picks        INTEGER,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (strategy_id, as_of_date, symbol)
);
"""


MM_TITAN_NOTES = """
CREATE TABLE IF NOT EXISTS mm_titan_notes (
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    section         TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL,
    unit            TEXT,
    source_table    TEXT,
    source_column   TEXT,
    source_asof     TEXT,
    narrative       TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (as_of_date, symbol, section, metric)
);
"""


MM_OUTCOME_MEMORY = """
CREATE TABLE IF NOT EXISTS mm_outcome_memory (
    rec_id          TEXT PRIMARY KEY,
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    strategy_id     INTEGER,
    rec_status      TEXT,
    filled          INTEGER NOT NULL,
    entry_price     REAL,
    stop_frac       REAL,
    state           TEXT NOT NULL,
    stop_hit_date   TEXT,
    worst_low       REAL,
    last_close      REAL,
    last_date       TEXT,
    return_to_date  REAL,
    bars_since      INTEGER,
    computed_at     TEXT NOT NULL
);
"""


MM_SELECTOR_SCORES = """
CREATE TABLE IF NOT EXISTS mm_selector_scores (
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    score           REAL NOT NULL,
    rank_pct        REAL,
    eligible        INTEGER NOT NULL,
    model_id        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (as_of_date, symbol, model_id)
);
"""


MM_SHADOW_SIZING = """
CREATE TABLE IF NOT EXISTS mm_shadow_sizing (
    rec_id          TEXT PRIMARY KEY,
    as_of_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    strategy_id     INTEGER NOT NULL,
    deflated_sharpe REAL,
    conviction_mult REAL,
    real_size_pct   REAL,
    shadow_size_pct REAL,
    stop_frac       REAL,
    created_at      TEXT NOT NULL
);
"""


MM_LLM_COMMENTARY = """
CREATE TABLE IF NOT EXISTS mm_llm_commentary (
    as_of_date          TEXT NOT NULL,
    kind                TEXT NOT NULL DEFAULT 'daily_briefing',
    generated_at        TEXT NOT NULL,
    model               TEXT NOT NULL,
    commentary_md       TEXT NOT NULL,
    context_json        TEXT,
    PRIMARY KEY (as_of_date, kind, generated_at)
);
"""


MM_RECOMMENDATION_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS mm_recommendation_performance (
    rec_id                    TEXT PRIMARY KEY,
    as_of_date                DATE NOT NULL,
    symbol                    TEXT NOT NULL,
    strategy_id               INTEGER NOT NULL,
    strategy_name             TEXT NOT NULL,
    decision                  TEXT NOT NULL,
    decided_at                TEXT,
    operator_reply            TEXT,
    entry_date                DATE,
    entry_price               REAL,
    stop_pct                  REAL NOT NULL,
    target_pct                REAL NOT NULL,
    size_pct                  REAL NOT NULL,
    stop_price                REAL,
    target_price              REAL,
    alpaca_order_id           TEXT,
    actual_fill_price         REAL,
    actual_fill_at            TEXT,
    fill_qty                  INTEGER,
    slippage_pct              REAL,
    exit_date                 DATE,
    exit_price                REAL,
    exit_reason               TEXT,
    days_held                 INTEGER,
    actual_exit_price         REAL,
    actual_exit_at            TEXT,
    actual_exit_qty           INTEGER,
    actual_exit_reason        TEXT,
    return_pct_counterfactual REAL,
    return_pct_actual         REAL,
    dividends_per_share       REAL,
    status                    TEXT NOT NULL,
    computed_at               TEXT NOT NULL
);
"""

MM_RECOMMENDATION_PERFORMANCE_MIGRATIONS: list[tuple[str, str]] = [
    ("actual_exit_price",  "REAL"),
    ("actual_exit_at",     "TEXT"),
    ("actual_exit_qty",    "INTEGER"),
    ("actual_exit_reason", "TEXT"),
    ("dividends_per_share", "REAL"),
]

MM_RECOMMENDATION_PERFORMANCE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_perf_as_of    ON mm_recommendation_performance(as_of_date);",
    "CREATE INDEX IF NOT EXISTS idx_perf_symbol   ON mm_recommendation_performance(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_perf_decision ON mm_recommendation_performance(decision);",
    "CREATE INDEX IF NOT EXISTS idx_perf_status   ON mm_recommendation_performance(status);",
    "CREATE INDEX IF NOT EXISTS idx_perf_entry_dt ON mm_recommendation_performance(entry_date);",
]

MM_RECOMMENDATION_SCOREBOARD = """
CREATE TABLE IF NOT EXISTS mm_recommendation_scoreboard (
    as_of_date        DATE NOT NULL,
    window            TEXT NOT NULL,
    decision_bucket   TEXT NOT NULL,
    n_recs            INTEGER NOT NULL,
    n_closed          INTEGER NOT NULL,
    win_rate          REAL,
    avg_return_pct    REAL,
    median_return_pct REAL,
    total_return_pct  REAL,
    fill_rate                     REAL,
    avg_slippage_pct              REAL,
    avg_return_pct_actual         REAL,
    implementation_shortfall_pct  REAL,
    computed_at       TEXT NOT NULL,
    PRIMARY KEY (as_of_date, window, decision_bucket)
);
"""

MM_RECOMMENDATION_SCOREBOARD_MIGRATIONS: list[tuple[str, str]] = [
    ("fill_rate",                    "REAL"),
    ("avg_slippage_pct",             "REAL"),
    ("avg_return_pct_actual",        "REAL"),
    ("implementation_shortfall_pct", "REAL"),
]

MM_RECOMMENDATION_SCOREBOARD_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_scb_as_of ON mm_recommendation_scoreboard(as_of_date);"
)


MM_RECS_LEDGER = """
CREATE TABLE IF NOT EXISTS mm_recs_ledger (
    rec_id                    TEXT PRIMARY KEY,
    as_of_date                TEXT NOT NULL,
    entry_date                TEXT NOT NULL,
    rank                      INTEGER,
    symbol                    TEXT NOT NULL,
    strategy_id               INTEGER NOT NULL,
    strategy_name             TEXT,
    sector                    TEXT,
    suggested_size_pct        REAL,
    suggested_stop_loss_pct   REAL,
    suggested_target_pct      REAL,
    rationale                 TEXT,
    status                    TEXT NOT NULL,
    decided_at                TEXT,
    operator_reply            TEXT,
    notes                     TEXT,
    alpaca_order_id           TEXT,
    order_qty                 INTEGER,
    entry_est_price           REAL,
    stop_price                REAL,
    target_price              REAL,
    fill_status               TEXT,
    actual_fill_price         REAL,
    actual_fill_at            TEXT,
    fill_qty                  INTEGER,
    last_polled_at            TEXT,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL,
    stop_order_id             TEXT,
    target_order_id           TEXT,
    exit_status               TEXT,
    exit_price                REAL,
    exit_at                   TEXT,
    exit_qty                  INTEGER,
    superseded_order_ids      TEXT,
    entry_type                TEXT,
    exit_type                 TEXT,
    trail_pct                 REAL,
    trail_order_id            TEXT
);
"""

MM_RECS_LEDGER_MIGRATIONS: list[tuple[str, str]] = [
    ("stop_order_id",        "TEXT"),
    ("target_order_id",      "TEXT"),
    ("exit_status",          "TEXT"),
    ("exit_price",           "REAL"),
    ("exit_at",              "TEXT"),
    ("exit_qty",             "INTEGER"),
    ("superseded_order_ids", "TEXT"),
    ("entry_type",           "TEXT"),
    ("exit_type",            "TEXT"),
    ("trail_pct",            "REAL"),
    ("trail_order_id",       "TEXT"),
]

MM_RECS_LEDGER_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_recs_ledger_as_of ON mm_recs_ledger(as_of_date);",
    "CREATE INDEX IF NOT EXISTS idx_recs_ledger_symbol ON mm_recs_ledger(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_recs_ledger_status ON mm_recs_ledger(status);",
)


MM_ALPACA_POSITIONS = """
CREATE TABLE IF NOT EXISTS mm_alpaca_positions (
    symbol            TEXT PRIMARY KEY,
    side              TEXT,
    qty               REAL,
    avg_entry_price   REAL,
    market_value      REAL,
    cost_basis        REAL,
    current_price     REAL,
    unrealized_pl     REAL,
    unrealized_plpc   REAL,
    change_today      REAL,
    exchange          TEXT,
    asset_class       TEXT,
    snapshot_at       TEXT NOT NULL
);
"""

MM_ALPACA_ORDERS = """
CREATE TABLE IF NOT EXISTS mm_alpaca_orders (
    id                TEXT PRIMARY KEY,
    client_order_id   TEXT,
    symbol            TEXT NOT NULL,
    side              TEXT,
    order_type        TEXT,
    order_class       TEXT,
    qty               REAL,
    filled_qty        REAL,
    filled_avg_price  REAL,
    limit_price       REAL,
    stop_price        REAL,
    time_in_force     TEXT,
    status            TEXT,
    parent_id         TEXT,
    replaced_by       TEXT,
    replaces          TEXT,
    extended_hours    INTEGER,
    submitted_at      TEXT,
    filled_at         TEXT,
    canceled_at       TEXT,
    expired_at        TEXT,
    replaced_at       TEXT,
    updated_at        TEXT,
    snapshot_at       TEXT NOT NULL
);
"""

MM_ALPACA_ORDERS_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_ord_symbol ON mm_alpaca_orders(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_ord_status ON mm_alpaca_orders(status);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_ord_client ON mm_alpaca_orders(client_order_id);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_ord_parent ON mm_alpaca_orders(parent_id);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_ord_submitted ON mm_alpaca_orders(submitted_at);",
)

MM_ALPACA_ACTIVITIES = """
CREATE TABLE IF NOT EXISTS mm_alpaca_activities (
    id                TEXT PRIMARY KEY,
    activity_type     TEXT NOT NULL,
    transaction_time  TEXT,
    symbol            TEXT,
    side              TEXT,
    qty               REAL,
    price             REAL,
    cum_qty           REAL,
    leaves_qty        REAL,
    order_id          TEXT,
    order_status      TEXT,
    net_amount        REAL,
    per_share_amount  REAL,
    description       TEXT,
    snapshot_at       TEXT NOT NULL
);
"""

MM_ALPACA_ACTIVITIES_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_act_symbol ON mm_alpaca_activities(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_act_type ON mm_alpaca_activities(activity_type);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_act_order ON mm_alpaca_activities(order_id);",
    "CREATE INDEX IF NOT EXISTS idx_mm_alp_act_time ON mm_alpaca_activities(transaction_time);",
)


MM_BROKER_DIVERGENCES = """
CREATE TABLE IF NOT EXISTS mm_broker_divergences (
    snapshot_at   TEXT NOT NULL,
    severity      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    symbol        TEXT,
    rec_id        TEXT,
    detail        TEXT
);
"""

_STATEMENTS: list[tuple[str, str]] = [
    ("mm_bars",                     MM_BARS),
    *[(f"mm_bars idx[{i}]", s) for i, s in enumerate(MM_BARS_INDEXES)],
    ("mm_bars_ingest_log",          MM_BARS_INGEST_LOG),
    ("mm_fundamentals_quarterly",   MM_FUNDAMENTALS_QUARTERLY),
    ("mm_fundamentals_annual",      MM_FUNDAMENTALS_ANNUAL),
    ("mm_fundamentals_growth",      MM_FUNDAMENTALS_GROWTH),
    ("mm_analyst_estimates",        MM_ANALYST_ESTIMATES),
    ("mm_earnings_events",          MM_EARNINGS_EVENTS),
    ("mm_insider_transactions",     MM_INSIDER_TRANSACTIONS),
    ("mm_sec_facts",                MM_SEC_FACTS),
    *[(f"mm_sec_facts idx[{i}]", s) for i, s in enumerate(MM_SEC_FACTS_INDEXES)],
    ("mm_cftc_cot_weekly",          MM_CFTC_COT_WEEKLY),
    ("mm_sector_etf_flows_weekly",  MM_SECTOR_ETF_FLOWS_WEEKLY),
    ("mm_dividends",                MM_DIVIDENDS),
    ("mm_sector_map",               MM_SECTOR_MAP),
    ("mm_backtest_runs",            MM_BACKTEST_RUNS),
    *[(f"mm_backtest_runs idx[{i}]", s) for i, s in enumerate(MM_BACKTEST_RUNS_INDEXES)],
    ("mm_variant_selection",        MM_VARIANT_SELECTION),
    ("mm_leaderboard",              MM_LEADERBOARD),
    ("mm_export_metadata",          MM_EXPORT_METADATA),
    ("mm_active_picks",             MM_ACTIVE_PICKS),
    ("mm_regime",                   MM_REGIME),
    ("mm_pick_funnel",              MM_PICK_FUNNEL),
    ("mm_pick_fates",               MM_PICK_FATES),
    ("mm_strategy_meta",           MM_STRATEGY_META),
    ("mm_indicators_daily",         MM_INDICATORS_DAILY),
    ("mm_indicators_daily idx",     MM_INDICATORS_INDEX),
    ("mm_training_labels",          MM_TRAINING_LABELS),
    ("mm_dl_signals",               MM_DL_SIGNALS),
    ("mm_rl_signals",               MM_RL_SIGNALS),
    ("mm_dl_runs",                  MM_DL_RUNS),
    ("mm_kronos_signals",           MM_KRONOS_SIGNALS),
    ("mm_kronos_paths",             MM_KRONOS_PATHS),
    ("mm_kronos_paths idx",         MM_KRONOS_PATHS_INDEX),
    ("mm_exit_signals",             MM_EXIT_SIGNALS),
    ("mm_quality_candidates",       MM_QUALITY_CANDIDATES),
    ("mm_titan_notes",              MM_TITAN_NOTES),
    ("mm_outcome_memory",           MM_OUTCOME_MEMORY),
    ("mm_selector_scores",          MM_SELECTOR_SCORES),
    ("mm_shadow_sizing",            MM_SHADOW_SIZING),
    ("mm_llm_commentary",           MM_LLM_COMMENTARY),
    ("mm_recommendation_performance", MM_RECOMMENDATION_PERFORMANCE),
    *[(f"mm_recommendation_performance idx[{i}]", s)
      for i, s in enumerate(MM_RECOMMENDATION_PERFORMANCE_INDEXES)],
    ("mm_recommendation_scoreboard", MM_RECOMMENDATION_SCOREBOARD),
    ("mm_recommendation_scoreboard idx", MM_RECOMMENDATION_SCOREBOARD_INDEX),
    ("mm_recs_ledger",              MM_RECS_LEDGER),
    *[(f"mm_recs_ledger idx[{i}]", s) for i, s in enumerate(MM_RECS_LEDGER_INDEXES)],
    ("mm_alpaca_positions",         MM_ALPACA_POSITIONS),
    ("mm_alpaca_orders",            MM_ALPACA_ORDERS),
    *[(f"mm_alpaca_orders idx[{i}]", s) for i, s in enumerate(MM_ALPACA_ORDERS_INDEXES)],
    ("mm_alpaca_activities",        MM_ALPACA_ACTIVITIES),
    *[(f"mm_alpaca_activities idx[{i}]", s) for i, s in enumerate(MM_ALPACA_ACTIVITIES_INDEXES)],
    ("mm_broker_divergences",       MM_BROKER_DIVERGENCES),
]


def init_schema(db_path: Path | str | None = None) -> dict:
    ""
    path = resolve_db(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Expected macmini.db at {path}; it should already exist.")

    pre_native = _snapshot_native_counts(path)

    with sqlite3.connect(path) as c:
        c.execute("PRAGMA journal_mode = WAL;")
        c.execute("PRAGMA foreign_keys = ON;")
        for label, sql in _STATEMENTS:
            try:
                c.execute(sql)
            except sqlite3.Error as e:
                raise RuntimeError(f"schema step '{label}' failed: {e}") from e
        migrations_applied = _apply_migrations(c)
        c.commit()

    post_native = _snapshot_native_counts(path)
    if pre_native != post_native:
        raise RuntimeError(
            f"SENTIMENT native table row counts changed during schema init. "
            f"before={pre_native} after={post_native}"
        )

    return {
        "db_path": str(path),
        "statements_run": len(_STATEMENTS),
        "migrations_applied": migrations_applied,
        "sentsrc_native_counts": post_native,
        "mm_tables_present": _list_mm_tables(path),
    }


def _ensure_column(c: sqlite3.Connection, table: str, col: str, decl: str) -> bool:
    ""
    existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    if col in existing:
        return False
    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    return True


def _apply_migrations(c: sqlite3.Connection) -> list[str]:
    ""
    applied: list[str] = []
    for col, decl in MM_BACKTEST_RUNS_MIGRATIONS:
        if _ensure_column(c, "mm_backtest_runs", col, decl):
            applied.append(f"mm_backtest_runs.{col}")
    for col, decl in MM_ACTIVE_PICKS_MIGRATIONS:
        if _ensure_column(c, "mm_active_picks", col, decl):
            applied.append(f"mm_active_picks.{col}")
    for stmt in MM_ACTIVE_PICKS_INDEXES:
        c.execute(stmt)
    for col, decl in MM_RECOMMENDATION_PERFORMANCE_MIGRATIONS:
        if _ensure_column(c, "mm_recommendation_performance", col, decl):
            applied.append(f"mm_recommendation_performance.{col}")
    for col, decl in MM_RECS_LEDGER_MIGRATIONS:
        if _ensure_column(c, "mm_recs_ledger", col, decl):
            applied.append(f"mm_recs_ledger.{col}")
    for col, decl in MM_RECOMMENDATION_SCOREBOARD_MIGRATIONS:
        if _ensure_column(c, "mm_recommendation_scoreboard", col, decl):
            applied.append(f"mm_recommendation_scoreboard.{col}")
    for col, decl in MM_KRONOS_SIGNALS_MIGRATIONS:
        if _ensure_column(c, "mm_kronos_signals", col, decl):
            applied.append(f"mm_kronos_signals.{col}")
    return applied


def _snapshot_native_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with sqlite3.connect(path) as c:
        for t in sorted(SENTIMENT_NATIVE_TABLES):
            try:
                out[t] = c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.OperationalError:
                pass
    return out


def _list_mm_tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as c:
        rows = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mm_%' ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Create all mm_* tables in macmini.db.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to macmini.db")
    args = parser.parse_args()
    report = init_schema(args.db)
    print("schema bootstrap OK")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
