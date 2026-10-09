""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.config import DB_PATH

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = REPO_ROOT / "data" / "selector" / "panel.parquet"

STEP = 5
MIN_HISTORY = 252
MAX_HOLD = 60
STOP_MULT, R_MULT, STOP_MIN, STOP_MAX = 3.0, 2.0, 0.05, 0.15

MIN_PRICE = 10.0
MAX_VOL60 = 0.65
MIN_DOLLAR_ADV = 10_000_000.0
EXCLUDED_FINNHUB_SECTORS = ("Health Care", "Biotechnology", "Pharmaceuticals")

PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "ret_252",
    "mom_12_1", "vs_sma20", "vs_sma50", "vs_sma200", "sma50_vs_200",
    "off_high_252", "off_low_252", "off_high_63", "vol_20", "vol_60",
    "vol_ratio", "downside_vol_60", "atr_pct", "maxdd_63", "skew_60",
    "rsi_14", "ibs", "up_ratio_20", "gap_abs_20", "range_pos_20",
    "volz_20", "dollar_adv_60", "adv_trend", "beta_60", "corr_60",
    "resid_mom_63", "ret_63_vs_sector",
]
MARKET_FEATURES = ["spy_ret_21", "spy_ret_63", "spy_vs_sma200", "spy_vol_20"]
LABEL_COLUMNS = ["br", "how", "days", "f5", "f20", "f40", "stop_frac", "next_open"]
OUTCOME = "r_mult"

_DB: str = str(DB_PATH)
_STEP: int = STEP
_SPY: pd.DataFrame | None = None


def _read_bars(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT substr(date, 1, 10) AS date, open, high, low, close, volume "
        "FROM mm_bars WHERE symbol = ? ORDER BY date", conn, params=[symbol])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date")


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(b: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    ""
    o, h, l, c, v = (b[k] for k in ("open", "high", "low", "close", "volume"))
    r = c.pct_change()
    f = pd.DataFrame(index=b.index)
    for n in (1, 5, 10, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    for n in (20, 50, 200):
        f[f"vs_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_vs_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    f["off_high_252"] = c / c.rolling(252).max() - 1
    f["off_low_252"] = c / c.rolling(252).min() - 1
    f["off_high_63"] = c / c.rolling(63).max() - 1
    f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = r.rolling(60).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["downside_vol_60"] = r.clip(upper=0).rolling(60).std() * np.sqrt(252)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr14"] / c
    f["maxdd_63"] = c / c.rolling(63).max() - 1
    f["maxdd_63"] = f["maxdd_63"].rolling(63).min()
    f["skew_60"] = r.rolling(60).skew()
    f["rsi_14"] = _rsi(c)
    f["ibs"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    f["up_ratio_20"] = (r > 0).rolling(20).mean()
    f["gap_abs_20"] = (o / pc - 1).abs().rolling(20).mean()
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    f["range_pos_20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    f["volz_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    dollar = c * v
    f["dollar_adv_60"] = dollar.rolling(60).median()
    f["adv_trend"] = dollar.rolling(20).median() / f["dollar_adv_60"] - 1
    sr = spy["close"].pct_change().reindex(b.index)
    cov = r.rolling(60).cov(sr)
    f["beta_60"] = cov / sr.rolling(60).var()
    f["corr_60"] = r.rolling(60).corr(sr)
    spy63 = (spy["close"] / spy["close"].shift(63) - 1).reindex(b.index)
    f["resid_mom_63"] = f["ret_63"] - f["beta_60"] * spy63
    f["close"] = c
    return f


def labels(b: pd.DataFrame, atr14: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    ""
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    n = len(b)
    rows = []
    for i in idx:
        row = dict.fromkeys(LABEL_COLUMNS, np.nan)
        row["how"] = None
        if i + 1 < n and np.isfinite(atr14[i]):
            e = i + 1
            p0 = o[e]
            sf = min(max(STOP_MULT * atr14[i] / c[i], STOP_MIN), STOP_MAX)
            stop, tgt = p0 * (1 - sf), p0 * (1 + R_MULT * sf)
            last = min(n - 1, e + MAX_HOLD)
            row.update(stop_frac=sf, next_open=p0, how="open")
            for j in range(e, last + 1):
                if l[j] <= stop:
                    row.update(br=min(o[j], stop) / p0 - 1, how="stop", days=j - e)
                    break
                if h[j] >= tgt:
                    row.update(br=max(o[j], tgt) / p0 - 1, how="target", days=j - e)
                    break
            else:
                if last == e + MAX_HOLD:
                    row.update(br=c[last] / p0 - 1, how="timeout", days=MAX_HOLD)
            for k in (5, 20, 40):
                if e + k < n:
                    row[f"f{k}"] = c[e + k] / p0 - 1
        rows.append(row)
    return pd.DataFrame(rows, index=b.index[idx])


def _init(db: str, step: int = STEP) -> None:
    global _DB, _SPY, _STEP
    _DB, _STEP = db, step
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    _SPY = _read_bars(conn, "SPY")
    conn.close()


def _one_symbol(symbol: str) -> pd.DataFrame | None:
    conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=60)
    b = _read_bars(conn, symbol)
    conn.close()
    if len(b) < MIN_HISTORY + 2:
        return None
    sample_dates = _SPY.index[::_STEP]
    f = features(b, _SPY)
    pos = np.flatnonzero(b.index.isin(sample_dates))
    pos = pos[pos >= MIN_HISTORY]
    if not len(pos):
        return None
    out = f.iloc[pos].join(labels(b, f["atr14"].values, pos))
    out.insert(0, "symbol", symbol)
    return out.reset_index()


def market_features(spy: pd.DataFrame) -> pd.DataFrame:
    c = spy["close"]
    m = pd.DataFrame(index=spy.index)
    m["spy_ret_21"] = c / c.shift(21) - 1
    m["spy_ret_63"] = c / c.shift(63) - 1
    m["spy_vs_sma200"] = c / c.rolling(200).mean() - 1
    m["spy_vol_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    return m


def build(db_path: Path | str | None = None, symbols: list[str] | None = None,
          workers: int = 12, fundamentals: bool = True,
          step: int = STEP) -> pd.DataFrame:
    db = str(db_path or DB_PATH)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    if symbols is None:
        symbols = [s for (s,) in conn.execute(
            "SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol")]
    sectors = pd.read_sql_query(
        "SELECT symbol, sector, finnhub_sector FROM mm_sector_map", conn)
    spy = _read_bars(conn, "SPY")
    conn.close()

    with Pool(workers, initializer=_init, initargs=(db, step)) as pool:
        parts = [p for p in pool.imap_unordered(_one_symbol, symbols, chunksize=4)
                 if p is not None]
    panel = pd.concat(parts, ignore_index=True).merge(sectors, on="symbol", how="left")
    panel = panel.merge(market_features(spy), left_on="date", right_index=True, how="left")
    sector_median = panel.groupby(["date", "sector"])["ret_63"].transform("median")
    panel["ret_63_vs_sector"] = panel["ret_63"] - sector_median.fillna(
        panel.groupby("date")["ret_63"].transform("median"))
    panel[OUTCOME] = panel["br"] / panel["stop_frac"]
    if fundamentals:
        from macmini.selector.fundamentals import attach
        panel = attach(panel, db, workers)
    panel["eligible"] = (
        (panel["close"] >= MIN_PRICE)
        & (panel["vol_60"] <= MAX_VOL60)
        & (panel["dollar_adv_60"] >= MIN_DOLLAR_ADV)
        & ~panel["finnhub_sector"].isin(EXCLUDED_FINNHUB_SECTORS)
        & panel["sector"].notna()
    )
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Build the selector panel.")
    p.add_argument("--db", default=None)
    p.add_argument("--out", default=str(PANEL_PATH))
    p.add_argument("--step", type=int, default=STEP,
                   help="sessions between sample dates (1 = every day, for training)")
    args = p.parse_args(argv)
    panel = build(args.db, step=args.step)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    lab = panel[panel["br"].notna()]
    fund = [c for c in panel.columns if c.startswith("fund_")]
    if fund:
        print("fundamentals coverage of eligible rows:",
              panel.loc[panel.eligible, fund].notna().mean().round(2).to_dict())
    print(f"panel: {len(panel)} rows, {panel.symbol.nunique()} symbols, "
          f"{panel.date.nunique()} dates ({panel.date.min().date()} → "
          f"{panel.date.max().date()}); labelled {len(lab)}; "
          f"eligible {int(panel.eligible.sum())} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
