""
import sqlite3, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

DB = Path("macmini.db").resolve()
MAX_HOLD, STOP_MULT, R_MULT, STOP_MIN, STOP_MAX = 60, 3.0, 2.0, 0.05, 0.15

def one(sym):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    b = pd.read_sql_query("SELECT date, open, high, low, close FROM mm_bars WHERE symbol = ? ORDER BY date",
                          c, params=[sym], parse_dates=["date"]).dropna()
    c.close()
    if len(b) < 300: return []
    o, h, l, cl = (b[k].values for k in ("open", "high", "low", "close")); n = len(b)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(14).mean().values
    vol = (pd.Series(cl).pct_change().rolling(60).std() * np.sqrt(252)).values
    out = []
    for i in range(252, n - 1, 5):
        e = i + 1; p0 = o[e]
        sf = min(max(STOP_MULT * atr[i] / cl[i], STOP_MIN), STOP_MAX)
        stop, tgt = p0 * (1 - sf), p0 * (1 + R_MULT * sf)
        br, how, days = np.nan, "open", np.nan
        last = min(n - 1, e + MAX_HOLD)
        for j in range(e, last + 1):
            if l[j] <= stop: br, how, days = min(o[j], stop) / p0 - 1, "stop", j - e; break
            if h[j] >= tgt: br, how, days = max(o[j], tgt) / p0 - 1, "target", j - e; break
        else:
            if last == e + MAX_HOLD: br, how, days = cl[last] / p0 - 1, "timeout", MAX_HOLD
        out.append((sym, b.date.values[i], br, how, days,
                    cl[e + 3] / p0 - 1 if e + 3 < n else np.nan,
                    cl[e + 20] / p0 - 1 if e + 20 < n else np.nan,
                    cl[e + 40] / p0 - 1 if e + 40 < n else np.nan, vol[i]))
    return out

if __name__ == "__main__":
    syms = [x.strip() for x in open("symbols.csv") if x.strip() and x.strip().lower() != "symbol"]
    with Pool(12) as p:
        rows = [r for ch in p.imap_unordered(one, syms, chunksize=4) for r in ch]
    pd.DataFrame(rows, columns="sym date br how days f3 f20 f40 vol60".split()).to_pickle(sys.argv[1])
