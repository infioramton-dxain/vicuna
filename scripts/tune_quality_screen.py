""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median

PROJ = Path(__file__).resolve().parent.parent
DB = PROJ / "macmini.db"

KRONOS_MAX_AGE_D = 7
SENT_WINDOW_D = 7
VOL_LOOKBACK = 60


def kronos_axes(c, symbol: str, as_of: str):
    row = c.execute(
        "SELECT predicted_5d_return, predicted_30d_return FROM mm_kronos_signals "
        "WHERE symbol=? AND date<=? AND date>=? AND quality_flag='ok' "
        "ORDER BY date DESC LIMIT 1",
        (symbol, as_of,
         (date.fromisoformat(as_of) - timedelta(days=KRONOS_MAX_AGE_D)).isoformat()),
    ).fetchone()
    return (float(row[0]), float(row[1])) if row else (None, None)


def sentiment_axes(c, symbol: str, as_of: str):
    ""
    row = c.execute(
        "SELECT sentiment, materiality FROM sentiment "
        "WHERE symbol=? AND window_end<=? AND window_end>=? "
        "  AND sentiment IS NOT NULL "
        "ORDER BY sentiment ASC LIMIT 1",
        (symbol, as_of,
         (date.fromisoformat(as_of) - timedelta(days=SENT_WINDOW_D)).isoformat()),
    ).fetchone()
    return (float(row[0]), float(row[1]) if row[1] is not None else 0.0) if row else (None, None)


def miss_streak(c, symbol: str, as_of: str) -> int:
    rows = c.execute(
        "SELECT eps_surprise_pct FROM mm_earnings_events "
        "WHERE symbol=? AND report_date<=? AND eps_surprise_pct IS NOT NULL "
        "ORDER BY report_date DESC LIMIT 8",
        (symbol, as_of),
    ).fetchall()
    n = 0
    for (s,) in rows:
        if float(s) < 0:
            n += 1
        else:
            break
    return n


def realized_vol(c, symbol: str, as_of: str):
    rows = c.execute(
        "SELECT close FROM mm_bars WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT ?",
        (symbol, as_of, VOL_LOOKBACK + 1),
    ).fetchall()
    closes = [float(r[0]) for r in reversed(rows) if r[0]]
    if len(closes) < 21:
        return None
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    mu = mean(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def load_recs(c):
    return c.execute(
        """SELECT rec_id, as_of_date, symbol, strategy_id, decision,
                  return_pct_counterfactual
           FROM mm_recommendation_performance
           WHERE rec_id NOT LIKE 'fixture-%'
             AND as_of_date IS NOT NULL AND symbol IS NOT NULL
           ORDER BY as_of_date""",
    ).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--json", default=None, help="write full results here")
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    recs = load_recs(c)
    print(f"replaying {len(recs)} recs from mm_recommendation_performance\n")

    enriched = []
    for rec_id, as_of, sym, sid, decision, fwd in recs:
        k5, k30 = kronos_axes(c, sym, as_of)
        s, m = sentiment_axes(c, sym, as_of)
        enriched.append({
            "rec_id": rec_id, "as_of": as_of, "symbol": sym,
            "strategy_id": sid, "decision": decision,
            "fwd": float(fwd) if fwd is not None else None,
            "kronos_5d": k5, "kronos_30d": k30,
            "sent": s, "mat": m,
            "miss_streak": miss_streak(c, sym, as_of),
            "vol60": realized_vol(c, sym, as_of),
        })

    def summarize(subset):
        fwds = [e["fwd"] for e in subset if e["fwd"] is not None]
        if not fwds:
            return {"n": len(subset), "n_scored": 0}
        return {"n": len(subset), "n_scored": len(fwds),
                "fwd_mean": round(mean(fwds), 4),
                "fwd_median": round(median(fwds), 4),
                "fwd_min": round(min(fwds), 4)}

    axes = {
        "kronos_5d": ([-0.03, -0.05, -0.08, -0.10],
                      lambda th: lambda e: e["kronos_5d"] is not None and e["kronos_5d"] < th),
        "kronos_30d": ([-0.10, -0.15, -0.20, -0.25],
                       lambda th: lambda e: e["kronos_30d"] is not None and e["kronos_30d"] < th),
        "vol60": ([0.55, 0.65, 0.75, 0.85],
                  lambda th: lambda e: e["vol60"] is not None and e["vol60"] > th),
        "miss_streak": ([2, 3, 4],
                        lambda th: lambda e: e["miss_streak"] >= th),
        "sent_neg": ([-0.5, -0.75],
                     lambda th: lambda e: e["sent"] is not None and e["sent"] <= th
                     and (e["mat"] or 0) >= 0.5),
    }

    results = {"replayed": len(enriched), "axes": {}, "recs": enriched}
    for name, (grid, factory) in axes.items():
        results["axes"][name] = {}
        print(f"── axis {name} ──")
        for th in grid:
            pred = factory(th)
            excl = [e for e in enriched if pred(e)]
            kept = [e for e in enriched if not pred(e)]
            r = {"threshold": th, "excluded": summarize(excl),
                 "kept": summarize(kept),
                 "excluded_recs": sorted({(e["symbol"], e["as_of"]) for e in excl})}
            results["axes"][name][str(th)] = r
            print(f"  th={th:>6}: excluded {r['excluded']} | kept {r['kept']}")
            for sym, d in r["excluded_recs"][:12]:
                print(f"      - {sym} {d}")
        print()

    lar = [e for e in enriched if e["symbol"] == "LAR" and e["as_of"] >= "2026-08-13"]
    meta = [e for e in enriched if e["symbol"] == "META"]
    print("── acceptance probes ──")
    for e in lar + meta:
        print(f"  {e['symbol']} {e['as_of']}: k5={e['kronos_5d']} k30={e['kronos_30d']} "
              f"sent={e['sent']}@{e['mat']} streak={e['miss_streak']} "
              f"vol={None if e['vol60'] is None else round(e['vol60'],3)} fwd={e['fwd']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1, default=str))
        print(f"\nfull results → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
