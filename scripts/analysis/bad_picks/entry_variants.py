""
import sqlite3, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

DB = Path("macmini.db").resolve()
MAX_HOLD, WINDOW = 60, 10
STOP_MULT, R_MULT, STOP_MIN, STOP_MAX = 3.0, 2.0, 0.05, 0.15


def bracket(o, h, l, cl, e, fill, sf, n, same_bar_after_fill=True):
    ""
    stop, tgt = fill * (1 - sf), fill * (1 + R_MULT * sf)
    last = min(n - 1, e + MAX_HOLD)
    for j in range(e, last + 1):
        lo, hi = l[j], h[j]
        if j == e and not same_bar_after_fill:
            continue
        if lo <= stop:
            px = stop if j == e else min(o[j], stop)
            return px / fill - 1, "stop", j - e
        if hi >= tgt:
            px = tgt if j == e else max(o[j], tgt)
            return px / fill - 1, "target", j - e
    if last == e + MAX_HOLD:
        return cl[last] / fill - 1, "timeout", MAX_HOLD
    return np.nan, "open", np.nan


def one(sym):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    b = pd.read_sql_query("SELECT date, open, high, low, close FROM mm_bars WHERE symbol = ? ORDER BY date",
                          c, params=[sym], parse_dates=["date"]).dropna()
    c.close()
    if len(b) < 300:
        return []
    o, h, l, cl = (b[k].values for k in ("open", "high", "low", "close")); n = len(b)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(14).mean().values
    vol = (pd.Series(cl).pct_change().rolling(60).std() * np.sqrt(252)).values
    out = []
    for i in range(252, n - 2, 5):
        if not np.isfinite(atr[i]) or cl[i] < 10 or not vol[i] <= 0.65:
            continue
        a = atr[i]; sf = min(max(STOP_MULT * a / cl[i], STOP_MIN), STOP_MAX)
        ref = cl[i]; d = b.date.values[i]
        end = min(n - 1, i + WINDOW)
        f40 = lambda e, fill: cl[e + 40] / fill - 1 if e + 40 < n else np.nan

        def rec(variant, e, fill):
            if e is None:
                out.append((sym, d, variant, 0, np.nan, "nofill", np.nan, np.nan, np.nan))
                return
            r, how, days = bracket(o, h, l, cl, e, fill, sf, n)
            out.append((sym, d, variant, 1, r, how, days, f40(e, fill), fill / ref - 1))

        rec("market", i + 1, o[i + 1])
        for k in (0.5, 1.0, 1.5, 2.0):
            lim = ref - k * a; e = fill = None
            for j in range(i + 1, end + 1):
                if l[j] <= lim:
                    e, fill = j, min(o[j], lim); break
            rec(f"limit_{k}", e, fill)
        e = fill = None
        for j in range(i + 1, end + 1):
            if h[j] > h[j - 1]:
                e, fill = j, max(o[j], h[j - 1]); break
        rec("confirm", e, fill)
        for k in (1.0, 1.5, 2.0):
            lim = ref - k * a; e = fill = None; dipped = False
            for j in range(i + 1, end + 1):
                if dipped and h[j] > h[j - 1]:
                    e, fill = j, max(o[j], h[j - 1]); break
                if l[j] <= lim:
                    dipped = True
            rec(f"dip_confirm_{k}", e, fill)
    return out


if __name__ == "__main__":
    syms = [x.strip() for x in open("symbols.csv") if x.strip() and x.strip().lower() != "symbol"]
    with Pool(12) as p:
        rows = [r for ch in p.imap_unordered(one, syms, chunksize=4) for r in ch]
    df = pd.DataFrame(rows, columns="sym date variant filled br how days f40 entry_vs_ref".split())
    df.to_pickle(sys.argv[1])
    C = 0.0015
    pd.set_option("display.width", 220)
    f = df[df.filled == 1]
    f = f[f.br.notna()]
    lo, hi = f.br.quantile([.01, .99])
    f = f.assign(bw=f.br.clip(lo, hi))
    n_sig = df.groupby("variant").size()
    g = f.groupby("variant").agg(
        trades=("br", "size"), avg=("bw", lambda x: x.mean() - C), win=("br", lambda x: (x > C).mean()),
        stop=("how", lambda x: (x == "stop").mean()),
        entry_vs_close=("entry_vs_ref", "mean"), days=("days", "median"))
    g["stays_down"] = f.assign(sd=(f.how == "stop") & (f.f40 < 0)).groupby("variant").sd.mean()
    g["fill_rate"] = g.trades / n_sig
    g["value_per_candidate"] = g.avg * g.fill_rate
    print(g.round(4).to_string())
    f = f.assign(yr=pd.to_datetime(f.date).dt.year)
    print("\navg result by year:")
    print(f.pivot_table(index="yr", columns="variant", values="bw", aggfunc="mean").round(4).to_string())
    print("\nwin rate by year:")
    print(f.assign(w=f.br > C).pivot_table(index="yr", columns="variant", values="w", aggfunc="mean").round(3).to_string())
