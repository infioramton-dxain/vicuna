""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from macmini.data._db import connect
from macmini.data.sentiment import daily_sentiment

_KWARG_VISIBILITY = {
    "fundamentals": ("period_end", 45),
    "fundamentals_annual": ("period_end", 90),
    "estimates": ("as_of_date", 0),
    "earnings": ("report_date", 0),
    "insider_tx": ("transaction_date", 2),
    "dividends": ("ex_date", 0),
    "cot": ("report_date", 3),
    "etf_flows": ("week_end", 0),
    "sentiment_daily": ("date", 0),
    "market": ("date", 0),
}


def slice_aux(aux: dict, as_of) -> dict:
    ""
    as_of = pd.Timestamp(as_of).normalize()
    out: dict = {}
    for key, val in aux.items():
        vis = _KWARG_VISIBILITY.get(key)
        if (
            vis is None
            or not isinstance(val, pd.DataFrame)
            or val.empty
            or vis[0] not in val.columns
        ):
            out[key] = val
            continue
        col, lag = vis
        dates = pd.to_datetime(
            val[col].astype(str).str.slice(0, 10), errors="coerce"
        )
        visible = dates + pd.Timedelta(days=lag) <= as_of
        out[key] = val.loc[visible.fillna(False)]
    return out


_TABLE_TO_KWARG = {
    "mm_fundamentals_quarterly": "fundamentals",
    "mm_fundamentals_annual": "fundamentals_annual",
    "mm_analyst_estimates": "estimates",
    "mm_earnings_events": "earnings",
    "mm_insider_transactions": "insider_tx",
    "mm_dividends": "dividends",
    "mm_cftc_cot_weekly": "cot",
    "mm_sector_etf_flows_weekly": "etf_flows",
    "mm_short_interest": "short_interest",
    "mm_options_flow": "options_flow",
    "sentiment": "sentiment_daily",
}


def load_bars(symbol: str, db_path: Path | str | None = None) -> pd.DataFrame:
    with connect(db_path) as c:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM mm_bars WHERE symbol = ? ORDER BY date",
            c, params=[symbol], parse_dates=["date"],
        )
    return df.set_index("date") if not df.empty else df


def load_dividends(symbol: str, db_path: Path | str | None = None) -> pd.DataFrame:
    ""
    with connect(db_path) as c:
        return pd.read_sql_query(
            "SELECT ex_date, rate FROM mm_dividends WHERE symbol = ? "
            "ORDER BY ex_date",
            c, params=[symbol], parse_dates=["ex_date"],
        )


def load_aux(symbol: str, strategy_cls, db_path: Path | str | None = None) -> dict:
    ""
    required = set(getattr(strategy_cls, "data_required", ()))
    out: dict = {}
    with connect(db_path) as c:
        if "mm_fundamentals_quarterly" in required:
            out["fundamentals"] = pd.read_sql_query(
                "SELECT * FROM mm_fundamentals_quarterly WHERE symbol = ? ORDER BY period_end",
                c, params=[symbol],
            )
        if "mm_fundamentals_annual" in required:
            out["fundamentals_annual"] = pd.read_sql_query(
                "SELECT * FROM mm_fundamentals_annual WHERE symbol = ? ORDER BY period_end",
                c, params=[symbol],
            )
        if "mm_analyst_estimates" in required:
            out["estimates"] = pd.read_sql_query(
                "SELECT * FROM mm_analyst_estimates WHERE symbol = ? ORDER BY as_of_date",
                c, params=[symbol],
            )
        if "mm_earnings_events" in required:
            out["earnings"] = pd.read_sql_query(
                "SELECT * FROM mm_earnings_events WHERE symbol = ? ORDER BY report_date",
                c, params=[symbol],
            )
        if "mm_insider_transactions" in required:
            out["insider_tx"] = pd.read_sql_query(
                "SELECT * FROM mm_insider_transactions WHERE symbol = ? "
                "ORDER BY transaction_date",
                c, params=[symbol],
            )
        if "mm_dividends" in required:
            out["dividends"] = pd.read_sql_query(
                "SELECT * FROM mm_dividends WHERE symbol = ? ORDER BY ex_date",
                c, params=[symbol],
            )
        if "mm_cftc_cot_weekly" in required:
            from macmini.strategies.sentiment.cftc_cot import SYMBOL_TO_COMMODITY
            commodity = SYMBOL_TO_COMMODITY.get(symbol)
            if commodity:
                out["cot"] = pd.read_sql_query(
                    "SELECT * FROM mm_cftc_cot_weekly WHERE commodity = ? ORDER BY report_date",
                    c, params=[commodity],
                )
            else:
                out["cot"] = pd.DataFrame()
        if "mm_sector_etf_flows_weekly" in required:
            sector_row = c.execute(
                "SELECT sector FROM mm_sector_map WHERE symbol = ?", (symbol,)
            ).fetchone()
            sector = sector_row[0] if sector_row else None
            sector_to_etf = {"Software": "XLK", "Computers": "XLK", "Mining": "XME",
                             "Materials": "XLB", "Defense": "ITA", "Other": "SPY",
                             "Consumer Staples": "XLP"}
            etf = sector_to_etf.get(sector)
            if etf:
                out["etf_flows"] = pd.read_sql_query(
                    "SELECT * FROM mm_sector_etf_flows_weekly WHERE etf_ticker = ? ORDER BY week_end",
                    c, params=[etf],
                )
            else:
                out["etf_flows"] = pd.DataFrame()
        if "mm_short_interest" in required:
            out["short_interest"] = pd.DataFrame()
        if "mm_options_flow" in required:
            out["options_flow"] = pd.DataFrame()
        if "market" in required:
            out["market"] = pd.read_sql_query(
                "SELECT date, close AS vix_close FROM mm_bars "
                "WHERE symbol = '^VIX' ORDER BY date", c,
            )
        if "sentiment" in required:
            row = c.execute(
                "SELECT MIN(date), MAX(date) FROM mm_bars WHERE symbol = ?", (symbol,)
            ).fetchone()
            if row and row[0] and row[1]:
                out["sentiment_daily"] = daily_sentiment(symbol, row[0], row[1], db_path=db_path)
            else:
                out["sentiment_daily"] = pd.DataFrame()
    return out
