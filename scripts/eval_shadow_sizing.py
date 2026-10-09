#!/usr/bin/env python
""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macmini.config import resolve_db

DECISION_MIN_CLOSED = 60


def main() -> int:
    db = sqlite3.connect(resolve_db(None))
    rows = db.execute(
        """SELECT p.symbol, p.decision, p.return_pct_counterfactual,
                  p.return_pct_actual, s.real_size_pct, s.shadow_size_pct,
                  s.conviction_mult
           FROM mm_shadow_sizing s
           JOIN mm_recommendation_performance p ON p.rec_id = s.rec_id
           WHERE p.status = 'closed'
           ORDER BY p.exit_date"""
    ).fetchall()
    total = db.execute("SELECT COUNT(*) FROM mm_shadow_sizing").fetchone()[0]
    db.close()

    flat = conv = 0.0
    n = 0
    for sym, dec, cf, act, real_sz, shadow_sz, mult in rows:
        ret = act if (dec == "approve" and act is not None) else cf
        if ret is None or real_sz is None or shadow_sz is None:
            continue
        flat += real_sz * ret * 100.0
        conv += shadow_sz * ret * 100.0
        n += 1

    print(f"shadow-sizing rows: {total}  (closed & joined: {n})")
    if n == 0:
        print("no closed joined recs yet — experiment is accumulating")
        return 0
    print(f"flat-risk scheme        : {flat:+.2f}% portfolio contribution")
    print(f"conviction-weighted     : {conv:+.2f}% portfolio contribution")
    print(f"spread (conviction-flat): {conv - flat:+.2f}%")
    if n < DECISION_MIN_CLOSED:
        print(f"NOT decision-grade: {n}/{DECISION_MIN_CLOSED} closed recs "
              f"— keep accumulating")
    else:
        print(f"decision-grade sample reached ({n} >= {DECISION_MIN_CLOSED}) "
              f"— take result to the decision log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
