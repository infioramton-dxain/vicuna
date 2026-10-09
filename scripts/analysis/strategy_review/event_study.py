""
import logging, sqlite3, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.path.insert(0, "src")
from macmini.backtest.loader import load_aux, load_bars, slice_aux
from macmini.strategies import STRATEGY_REGISTRY

DB = Path("macmini.db").resolve()
GO_LIVE = pd.Timestamp("2026-07-06")
STOP_MULT, R_MULT, STOP_MIN, STOP_MAX, COST = 3.0, 2.0, 0.05, 0.15, 0.0015
BARS_ONLY = {"mm_bars", "mm_indicators_daily"}


def one_symbol(sym: str) -> list[tuple]:
    b = load_bars(sym, db_path=DB)
    b = b.dropna(subset=["open", "high", "low", "close"])
    if len(b) < 300 or b.index[-1] < GO_LIVE:
        return []
    o, h, l, cl = (b[k].values for k in ("open", "high", "low", "close"))
    vol = b["volume"].fillna(0).values
    n = len(b)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    adv = pd.Series(cl * vol).rolling(60, min_periods=20).median().values
    pos = {d: i for i, d in enumerate(b.index)}
    cuts = list(pd.date_range(GO_LIVE, b.index[-1], freq="14D")) + [b.index[-1] + pd.Timedelta(days=1)]
    out = []
    for sid, cls in sorted(STRATEGY_REGISTRY.items()):
        try:
            aux = load_aux(sym, cls, db_path=DB)
            if not set(getattr(cls, "data_required", ())) - BARS_ONLY:
                sigs = cls().generate_signals(sym, b, **aux)
            else:
                sigs = []
                for a, z in zip(cuts[:-1], cuts[1:]):
                    part = cls().generate_signals(sym, b[b.index < z], **slice_aux(aux, a))
                    sigs += [s for s in (part or []) if a <= pd.Timestamp(s.date) < z]
        except Exception:
            out.append((sid, sym, None, None, None, np.nan, "error", np.nan, False))
            continue
        for s in sigs or []:
            d = pd.Timestamp(s.date)
            if s.action != "enter" or d < GO_LIVE or d not in pos:
                continue
            i = pos[d]
            tradable = bool(cl[i] >= 10 and adv[i] >= 10e6)
            if i + 1 >= n:
                out.append((sid, sym, d, None, None, np.nan, "pending", np.nan, tradable))
                continue
            stop_frac = float(np.clip(STOP_MULT * atr[i] / cl[i], STOP_MIN, STOP_MAX)) \
                if atr[i] == atr[i] else 0.08
            entry = o[i + 1]
            stop, target = entry * (1 - stop_frac), entry * (1 + R_MULT * stop_frac)
            status, px, j = "open", cl[-1], n - 1
            for k in range(i + 1, n):
                if l[k] <= stop:
                    status, px, j = "stopped", min(stop, o[k]), k
                    break
                if h[k] >= target:
                    status, px, j = "target", max(target, o[k]) if o[k] > target else target, k
                    break
            out.append((sid, sym, d, b.index[i + 1], b.index[j],
                        px / entry - 1 - COST, status, stop_frac, tradable))
    return out


def main(out_path: str) -> None:
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    syms = [r[0] for r in c.execute(
        "SELECT DISTINCT symbol FROM mm_bars WHERE symbol NOT LIKE '^%'")]
    spy = pd.read_sql_query("SELECT date, open, close FROM mm_bars WHERE symbol='SPY' "
                            "ORDER BY date", c, parse_dates=["date"]).set_index("date")
    c.close()
    with Pool(12) as p:
        rows = [r for part in p.imap_unordered(one_symbol, syms, chunksize=4) for r in part]
    df = pd.DataFrame(rows, columns=["sid", "sym", "signal", "entry", "exit", "ret",
                                     "status", "stop_frac", "tradable"])
    t = df.dropna(subset=["entry"])
    df.loc[t.index, "spy_ret"] = (spy["close"].reindex(t["exit"]).values
                                  / spy["open"].reindex(t["entry"]).values - 1)
    df.to_pickle(out_path)
    print(len(df), "rows;", df.status.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
