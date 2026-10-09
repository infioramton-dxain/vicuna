""
import logging, sqlite3, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.path.insert(0, "src")
from macmini.strategies import STRATEGY_REGISTRY

DB = Path("macmini.db").resolve()
MAX_HOLD = 60
STOP_MULT, R_MULT, STOP_MIN, STOP_MAX = 3.0, 2.0, 0.05, 0.15
TECH = [sid for sid, cls in sorted(STRATEGY_REGISTRY.items())
        if not set(getattr(cls, "data_required", ())) - {"mm_bars", "mm_indicators_daily"}]


def one_symbol(sym: str) -> list[tuple]:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    b = pd.read_sql_query(
        "SELECT date, open, high, low, close, adj_close, volume FROM mm_bars "
        "WHERE symbol = ? ORDER BY date", c, params=[sym], parse_dates=["date"])
    c.close()
    b = b.dropna(subset=["open", "high", "low", "close"])
    if len(b) < 300:
        return []
    b = b.set_index("date")
    o, h, l, cl = (b[k].values for k in ("open", "high", "low", "close"))
    n = len(b)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    s = pd.Series(cl)
    ret = s.pct_change()
    feat = {
        "vol60": (ret.rolling(60).std() * np.sqrt(252)).values,
        "vs_sma50": (s / s.rolling(50).mean() - 1).values,
        "vs_sma200": (s / s.rolling(200).mean() - 1).values,
        "mom21": (s / s.shift(21) - 1).values,
        "mom63": (s / s.shift(63) - 1).values,
        "mom252": (s / s.shift(252) - 1).values,
        "off_high252": (s / s.rolling(252).max() - 1).values,
    }
    pos = {d: i for i, d in enumerate(b.index)}
    out = []
    for sid in TECH:
        try:
            sigs = STRATEGY_REGISTRY[sid]().generate_signals(sym, b)
        except Exception:
            continue
        sigs = sorted(sigs, key=lambda x: pd.Timestamp(x.date))
        for k, sg in enumerate(sigs):
            if sg.action != "enter":
                continue
            i = pos.get(pd.Timestamp(sg.date))
            if i is None or i < 252 or i + 1 >= n or not np.isfinite(atr[i]):
                continue
            e = i + 1
            p0 = o[e]
            nat, nat_days = np.nan, np.nan
            for sx in sigs[k + 1:]:
                if sx.action == "exit":
                    j = pos.get(pd.Timestamp(sx.date))
                    if j is not None and j + 1 < n:
                        nat, nat_days = o[j + 1] / p0 - 1, j + 1 - e
                    break
                if sx.action == "enter":
                    break
            sf = min(max(STOP_MULT * atr[i] / cl[i], STOP_MIN), STOP_MAX)
            stop, tgt = p0 * (1 - sf), p0 * (1 + R_MULT * sf)
            br, br_days, how = np.nan, np.nan, "open"
            last = min(n - 1, e + MAX_HOLD)
            for j in range(e, last + 1):
                if l[j] <= stop:
                    br, br_days, how = min(o[j], stop) / p0 - 1, j - e, "stop"
                    break
                if h[j] >= tgt:
                    br, br_days, how = max(o[j], tgt) / p0 - 1, j - e, "target"
                    break
            else:
                if last == e + MAX_HOLD:
                    br, br_days, how = cl[last] / p0 - 1, MAX_HOLD, "timeout"
            f40 = cl[e + 40] / p0 - 1 if e + 40 < n else np.nan
            f20 = cl[e + 20] / p0 - 1 if e + 20 < n else np.nan
            out.append((sym, sid, b.index[i], sf, nat, nat_days, br, br_days, how,
                        f20, f40, *(feat[key][i] for key in feat)))
    return out


if __name__ == "__main__":
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    syms = [l.strip() for l in open("symbols.csv") if l.strip() and l.strip().lower() != "symbol"]
    c.close()
    with Pool(12) as p:
        rows = [r for chunk in p.imap_unordered(one_symbol, syms, chunksize=4) for r in chunk]
    df = pd.DataFrame(rows, columns=[
        "sym", "sid", "date", "stop_frac", "nat", "nat_days", "br", "br_days", "how",
        "f20", "f40", "vol60", "vs_sma50", "vs_sma200", "mom21", "mom63", "mom252",
        "off_high252"])
    df.to_pickle(sys.argv[1])
    print(len(df), "events;", df.sym.nunique(), "symbols;", df.sid.nunique(), "strategies;",
          df.date.min().date(), "→", df.date.max().date())
