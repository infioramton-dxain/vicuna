#!/usr/bin/env python
""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macmini.config import resolve_db

GATE_VERSION = "gate_v1"
MIN_CLOSED_TOTAL = 60
MIN_CLOSED_APPROVED = 15


def main() -> int:
    db = sqlite3.connect(resolve_db(None))
    rows = db.execute(
        """SELECT decision, return_pct_counterfactual, return_pct_actual
           FROM mm_recommendation_performance
           WHERE status = 'closed'"""
    ).fetchall()
    db.close()

    approved: list[float] = []
    counterfactual: list[float] = []
    for dec, cf, act in rows:
        if dec == "approve":
            ret = act if act is not None else cf
            if ret is not None:
                approved.append(ret)
        elif dec in ("reject", "expired"):
            if cf is not None:
                counterfactual.append(cf)

    n_total = len(approved) + len(counterfactual)
    print(f"autonomy gate {GATE_VERSION}")
    print(f"closed recs: {n_total} total "
          f"({len(approved)} approved, {len(counterfactual)} rejected/expired)")

    if not approved or not counterfactual:
        print("no spread computable yet — accumulating")
        return 0

    mean_appr = sum(approved) / len(approved)
    mean_cf = sum(counterfactual) / len(counterfactual)
    spread = mean_cf - mean_appr
    print(f"approved mean return          : {mean_appr * 100:+.2f}%")
    print(f"rejected/expired counterfactual: {mean_cf * 100:+.2f}%")
    print(f"spread (counterfactual-approved): {spread * 100:+.2f}%")

    bars = []
    if n_total < MIN_CLOSED_TOTAL:
        bars.append(f"total {n_total}/{MIN_CLOSED_TOTAL}")
    if len(approved) < MIN_CLOSED_APPROVED:
        bars.append(f"approved {len(approved)}/{MIN_CLOSED_APPROVED}")
    if bars:
        print(f"NOT decision-grade: {', '.join(bars)} — keep accumulating")
        return 0

    if spread > 0:
        print("GATE MET: manual filter still subtracting value at the "
              "evidence bar — auto-approve is justified.")
    else:
        print("GATE CLOSED: manual approval adds value at the evidence "
              "bar — keep manual approval in the loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
