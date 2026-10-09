""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone

from macmini.config import DB_PATH
from macmini.data.schema import MM_FUNDAMENTALS_GROWTH

log = logging.getLogger(__name__)

_QTR_GAP = (70, 110)
_YOY_GAP = (325, 405)
_FY_BEFORE_ANCHOR_MAX = 400
_5Y_GAP = (1650, 2010)
_SPLIT_RATIO = 1.8


def _d(s) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _pct_change(cur: float | None, base: float | None) -> float | None:
    ""
    if cur is None or base is None or base <= 0:
        return None
    return (cur - base) / base * 100.0


def _cagr(cur: float | None, base: float | None, days: int) -> float | None:
    ""
    if cur is None or base is None or cur <= 0 or base <= 0 or days <= 0:
        return None
    return ((cur / base) ** (365.25 / days) - 1.0) * 100.0


def _nearest(rows: list[dict], target: date, lo: int, hi: int) -> dict | None:
    ""
    best, best_off = None, None
    for r in rows:
        off = (target - r["pe"]).days
        if lo <= off <= hi:
            mid = (lo + hi) / 2
            score = abs(off - mid)
            if best is None or score < best_off:
                best, best_off = r, score
    return best


class _SymbolHistory:
    ""

    def __init__(self, qrows: list[tuple], arows: list[tuple]):
        self.q = [
            {"pe": pe, "rev": rev, "eps": eps}
            for (pe_s, rev, eps) in qrows
            if (pe := _d(pe_s)) is not None
        ]
        self.q.sort(key=lambda r: r["pe"], reverse=True)
        self.a = [
            {"pe": pe, "rev": rev, "eps": eps, "earn": earn}
            for (pe_s, rev, eps, earn) in arows
            if (pe := _d(pe_s)) is not None
        ]
        self.a.sort(key=lambda r: r["pe"], reverse=True)
        self._split_suspects = self._find_split_suspects()

    def _find_split_suspects(self) -> list[date]:
        ""
        out: list[date] = []
        prev_shares = prev_pe = None
        for r in sorted(self.a, key=lambda r: r["pe"]):
            eps, earn = r["eps"], r["earn"]
            if eps in (None, 0) or earn is None:
                continue
            shares = earn / eps
            if prev_shares is not None and shares > 0 and prev_shares > 0:
                ratio = shares / prev_shares
                if ratio >= _SPLIT_RATIO or ratio <= 1.0 / _SPLIT_RATIO:
                    out.append(r["pe"])
            prev_shares, prev_pe = shares, r["pe"]
        return out

    def _eps_window_clean(self, start: date, end: date) -> bool:
        return not any(start < s <= end for s in self._split_suspects)

    def _boundaries(self) -> list[date]:
        ""
        bs = sorted(a["pe"] for a in self.a)
        if bs:
            last = bs[-1]
            bs += [last + timedelta(days=round(365.25 * k)) for k in (1, 2, 3)]
        return bs

    def discrete_quarters(self, col: str) -> list[dict]:
        out: list[dict] = []
        asc = sorted(self.q, key=lambda r: r["pe"])
        bounds = self._boundaries()

        def same_fy(d1: date, d2: date) -> bool:
            return not any(d1 <= b < d2 for b in bounds)

        for i, r in enumerate(asc):
            if r[col] is None:
                continue
            prev = asc[i - 1] if i > 0 else None
            gap = (r["pe"] - prev["pe"]).days if prev else None
            if (
                prev is not None
                and gap is not None
                and _QTR_GAP[0] <= gap <= _QTR_GAP[1]
                and same_fy(prev["pe"], r["pe"])
                and prev[col] is not None
            ):
                out.append({"pe": r["pe"], "val": r[col] - prev[col]})
            else:
                if any(_QTR_GAP[0] <= (r["pe"] - b).days <= _QTR_GAP[1] for b in bounds):
                    out.append({"pe": r["pe"], "val": r[col]})
        for a in self.a:
            if a[col] is None:
                continue
            ytd3 = _nearest(self.q, a["pe"], *_QTR_GAP)
            if ytd3 is not None and ytd3[col] is not None:
                out.append({"pe": a["pe"], "val": a[col] - ytd3[col]})
        out.sort(key=lambda r: r["pe"], reverse=True)
        return out

    def ttm_at(self, col: str, anchor: dict | None) -> float | None:
        ""
        if anchor is None or anchor[col] is None:
            return None
        fy_prev = None
        for a in self.a:
            off = (anchor["pe"] - a["pe"]).days
            if 0 < off <= _FY_BEFORE_ANCHOR_MAX and a[col] is not None:
                fy_prev = a
                break
        if fy_prev is None:
            return None
        prior_ytd = _nearest(self.q, anchor["pe"], *_YOY_GAP)
        if (
            prior_ytd is None
            or prior_ytd[col] is None
            or not (0 <= (fy_prev["pe"] - prior_ytd["pe"]).days < 360)
        ):
            return None
        return anchor[col] + fy_prev[col] - prior_ytd[col]

    def latest_annual(self, col: str) -> dict | None:
        for a in self.a:
            if a[col] is not None:
                return a
        return None


def compute_growth(qrows: list[tuple], arows: list[tuple]) -> dict | None:
    ""
    if not qrows and not arows:
        return None
    h = _SymbolHistory(qrows, arows)

    latest_pe = max(
        [r["pe"] for r in h.q] + [a["pe"] for a in h.a]
    )
    out = {
        "latest_period_end": latest_pe.isoformat(),
        "n_quarters": len(h.q),
        "rev_chg_mrq_yoy": None, "rev_chg_ttm": None, "rev_growth_5y_avg": None,
        "eps_chg_mrq_yoy": None, "eps_chg_ttm": None, "eps_growth_5y_avg": None,
    }

    for col, mrq_key, ttm_key, avg_key in (
        ("rev", "rev_chg_mrq_yoy", "rev_chg_ttm", "rev_growth_5y_avg"),
        ("eps", "eps_chg_mrq_yoy", "eps_chg_ttm", "eps_growth_5y_avg"),
    ):
        dq = h.discrete_quarters(col)
        if dq:
            cur = dq[0]
            base = _nearest(dq[1:], cur["pe"], *_YOY_GAP)
            if base is not None and (
                col != "eps" or h._eps_window_clean(base["pe"], cur["pe"])
            ):
                out[mrq_key] = _pct_change(cur["val"], base["val"])

        anchor = h.q[0] if h.q else None
        cur_ttm = h.ttm_at(col, anchor)
        prior_anchor = _nearest(h.q, anchor["pe"], *_YOY_GAP) if anchor else None
        prior_ttm = h.ttm_at(col, prior_anchor)
        la = h.latest_annual(col)
        if anchor is None or (la is not None and la["pe"] > anchor["pe"]):
            cur_ttm = la[col] if la else None
            prior_anchor_a = _nearest(h.a, la["pe"], *_YOY_GAP) if la else None
            prior_ttm = prior_anchor_a[col] if prior_anchor_a else None
            span_start = prior_anchor_a["pe"] if prior_anchor_a else None
            span_end = la["pe"] if la else None
        else:
            span_start = prior_anchor["pe"] if prior_anchor else None
            span_end = anchor["pe"] if anchor else None
        if cur_ttm is not None and prior_ttm is not None:
            if col != "eps" or (
                span_start and span_end and h._eps_window_clean(span_start - (span_end - span_start), span_end)
            ):
                out[ttm_key] = _pct_change(cur_ttm, prior_ttm)

        a0 = h.latest_annual(col)
        if a0 is not None:
            a5 = _nearest([a for a in h.a if a[col] is not None], a0["pe"], *_5Y_GAP)
            if a5 is not None and (
                col != "eps" or h._eps_window_clean(a5["pe"], a0["pe"])
            ):
                out[avg_key] = _cagr(a0[col], a5[col], (a0["pe"] - a5["pe"]).days)

    return out


def refresh(db_path=None, as_of: str | None = None) -> dict:
    ""
    from macmini.data._db import connect

    as_of = as_of or datetime.now(timezone.utc).date().isoformat()
    computed_at = datetime.now(timezone.utc).isoformat()
    written = skipped = 0

    with connect(db_path) as conn:
        conn.execute(MM_FUNDAMENTALS_GROWTH)
        symbols = [r[0] for r in conn.execute(
            "SELECT symbol FROM mm_fundamentals_quarterly "
            "UNION SELECT symbol FROM mm_fundamentals_annual ORDER BY 1"
        )]
        for sym in symbols:
            qrows = conn.execute(
                "SELECT period_end, revenue, eps FROM mm_fundamentals_quarterly "
                "WHERE symbol = ? ORDER BY period_end DESC LIMIT 40",
                (sym,),
            ).fetchall()
            arows = conn.execute(
                "SELECT period_end, revenue, eps, earnings FROM mm_fundamentals_annual "
                "WHERE symbol = ? ORDER BY period_end DESC LIMIT 10",
                (sym,),
            ).fetchall()
            fields = compute_growth(qrows, arows)
            if fields is None:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO mm_fundamentals_growth
                    (symbol, as_of_date, latest_period_end, n_quarters,
                     rev_chg_mrq_yoy, rev_chg_ttm, rev_growth_5y_avg,
                     eps_chg_mrq_yoy, eps_chg_ttm, eps_growth_5y_avg,
                     computed_at)
                VALUES (:symbol, :as_of_date, :latest_period_end, :n_quarters,
                        :rev_chg_mrq_yoy, :rev_chg_ttm, :rev_growth_5y_avg,
                        :eps_chg_mrq_yoy, :eps_chg_ttm, :eps_growth_5y_avg,
                        :computed_at)
                ON CONFLICT (symbol, as_of_date) DO UPDATE SET
                    latest_period_end = excluded.latest_period_end,
                    n_quarters        = excluded.n_quarters,
                    rev_chg_mrq_yoy   = excluded.rev_chg_mrq_yoy,
                    rev_chg_ttm       = excluded.rev_chg_ttm,
                    rev_growth_5y_avg = excluded.rev_growth_5y_avg,
                    eps_chg_mrq_yoy   = excluded.eps_chg_mrq_yoy,
                    eps_chg_ttm       = excluded.eps_chg_ttm,
                    eps_growth_5y_avg = excluded.eps_growth_5y_avg,
                    computed_at       = excluded.computed_at
                """,
                {"symbol": sym, "as_of_date": as_of, "computed_at": computed_at, **fields},
            )
            conn.commit()
            written += 1

    return {"as_of_date": as_of, "symbols_written": written, "symbols_skipped": skipped}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Derive growth rates into mm_fundamentals_growth.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--as-of", default=None, help="Override as_of_date (YYYY-MM-DD, default today UTC)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = refresh(db_path=args.db, as_of=args.as_of)
    print(f"fundamentals_growth refresh complete: {report}")


if __name__ == "__main__":
    _cli()
