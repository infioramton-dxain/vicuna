""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.config import DATA_DIR, resolve_db
from macmini.data._db import connect
from macmini.data.sentiment import daily_sentiment
from macmini.strategies._indicators import (
    adx, atr, bollinger, ema, ibs, macd, rsi, sma, williams_r, zscore,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = DATA_DIR / "dl_cache"


def compute_indicators_for_bars(bars: pd.DataFrame) -> pd.DataFrame:
    ""
    if bars.empty or "close" not in bars.columns:
        return pd.DataFrame(index=bars.index)
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    open_ = bars["open"]
    volume = bars["volume"]

    bb_upper, bb_middle, bb_lower, bb_width = bollinger(close, 20, 2.0)
    macd_line, macd_signal, macd_hist = macd(close)

    return pd.DataFrame({
        "sma20": sma(close, 20),
        "sma50": sma(close, 50),
        "sma200": sma(close, 200),
        "ema12": ema(close, 12),
        "ema26": ema(close, 26),
        "ema200": ema(close, 200),
        "rsi14": rsi(close, 14),
        "rsi2": rsi(close, 2),
        "atr14": atr(high, low, close, 14),
        "adx14": adx(high, low, close, 14),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "williams_r14": williams_r(high, low, close, 14),
        "ibs": ibs(open_, high, low, close),
        "volume_zscore_50": zscore(volume.astype(float), 50),
    }, index=bars.index)


def _load_bars(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, adj_close, volume "
        "FROM mm_bars WHERE symbol = ? ORDER BY date",
        conn, params=[symbol], parse_dates=["date"],
    )
    return df.set_index("date") if not df.empty else df


def _load_fundamentals_asof(conn, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    ""
    f = pd.read_sql_query(
        "SELECT period_end, revenue, earnings, eps, free_cash_flow, "
        "total_debt, total_equity FROM mm_fundamentals_quarterly "
        "WHERE symbol = ? ORDER BY period_end",
        conn, params=[symbol], parse_dates=["period_end"],
    )
    if f.empty:
        return pd.DataFrame(index=dates).reindex(dates)
    left = pd.DataFrame({"date": dates}).sort_values("date")
    return pd.merge_asof(
        left, f.rename(columns={"period_end": "date"}).sort_values("date"),
        on="date", direction="backward",
    ).set_index("date")


def _load_estimates_asof(conn, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    e = pd.read_sql_query(
        "SELECT as_of_date, eps_revision_30d, rating_buy, rating_hold, rating_sell "
        "FROM mm_analyst_estimates WHERE symbol = ? ORDER BY as_of_date",
        conn, params=[symbol], parse_dates=["as_of_date"],
    )
    if e.empty:
        return pd.DataFrame(index=dates).reindex(dates)
    left = pd.DataFrame({"date": dates}).sort_values("date")
    return pd.merge_asof(
        left, e.rename(columns={"as_of_date": "date"}).sort_values("date"),
        on="date", direction="backward",
    ).set_index("date")


def _load_sentiment_panel(
    conn, symbol: str, dates: pd.DatetimeIndex, db_path,
) -> pd.DataFrame:
    if not len(dates):
        return pd.DataFrame()
    start = dates[0].strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    s = daily_sentiment(symbol, start, end, db_path=db_path)
    if s.empty:
        return pd.DataFrame(index=dates).reindex(dates)
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date").reindex(dates)


def _load_sector_archetype(conn, symbol: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT sector, archetype FROM mm_sector_map WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def _load_labels(conn, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, forward_1d_return, forward_5d_return, forward_10d_return, "
        "forward_5d_class, forward_5d_sharpe, max_dd_5d "
        "FROM mm_training_labels WHERE symbol = ? ORDER BY date",
        conn, params=[symbol], parse_dates=["date"],
    )
    return df.set_index("date") if not df.empty else df


def materialize_symbol(
    symbol: str,
    output_dir: Path,
    db_path: Path | str | None = None,
) -> Path | None:
    ""
    output_dir.mkdir(parents=True, exist_ok=True)
    db = resolve_db(db_path)
    with connect(db) as c:
        bars = _load_bars(c, symbol)
        if bars.empty:
            return None
        indicators = compute_indicators_for_bars(bars)
        fundamentals = _load_fundamentals_asof(c, symbol, bars.index)
        estimates = _load_estimates_asof(c, symbol, bars.index)
        sentiment_panel = _load_sentiment_panel(c, symbol, bars.index, db)
        sector, archetype = _load_sector_archetype(c, symbol)
        labels = _load_labels(c, symbol)

    panel = bars.join(indicators, how="left")
    panel = panel.join(fundamentals.add_suffix(""), how="left")
    panel = panel.join(estimates.add_suffix(""), how="left")
    if not sentiment_panel.empty:
        panel = panel.join(sentiment_panel.add_suffix(""), how="left")
    if not labels.empty:
        panel = panel.join(labels.add_suffix(""), how="left")
    panel["symbol"] = symbol
    panel["sector"] = sector
    panel["archetype"] = archetype
    panel = panel.reset_index().rename(columns={"index": "date"})

    out_path = output_dir / f"{symbol.replace('/', '_').replace('^', 'C')}.parquet"
    panel.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    return out_path


def materialize_all(
    symbols: list[str] | None = None,
    output_dir: Path | None = None,
    db_path: Path | str | None = None,
) -> dict:
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    db = resolve_db(db_path)
    counts = {"written": 0, "skipped": 0}
    started = datetime.now(timezone.utc)

    with connect(db) as c:
        if symbols is None:
            symbols = [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol"
            ).fetchall()]

    for sym in symbols:
        try:
            path = materialize_symbol(sym, out_dir, db_path=db)
        except Exception as exc:
            log.warning("%s: %s", sym, exc)
            counts["skipped"] += 1
            continue
        if path is None:
            counts["skipped"] += 1
        else:
            counts["written"] += 1
            log.info("  %s → %s", sym, path)

    counts["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    counts["output_dir"] = str(out_dir)
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Materialize DL training Parquet shards")
    p.add_argument("--db", default=None)
    p.add_argument("--symbols", help="comma-separated; default all in mm_bars")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    args = p.parse_args(argv)
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    out = materialize_all(
        symbols=syms,
        output_dir=Path(args.output),
        db_path=args.db,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
