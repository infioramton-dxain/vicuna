""
from __future__ import annotations

import pandas as pd

STAPLES: frozenset[str] = frozenset({
    "WMT", "COST", "TGT", "KR", "DG", "DLTR", "BJ", "SFM", "SYY", "USFD",
    "PFGC", "ACI",
    "KO", "PEP", "MNST", "KDP", "STZ", "COKE", "TAP", "CELH", "FIZZ", "SAM",
    "PG", "CL", "KMB", "KVUE", "CHD", "CLX", "EL", "COTY", "EPC", "NWL",
    "PM", "MO", "BTI", "TPB",
    "MDLZ", "KHC", "HSY", "GIS", "CAG", "CPB", "SJM", "HRL", "MKC", "LW",
    "FLO", "JJSF", "POST", "BRBR",
    "ADM", "BG", "INGR", "DAR", "DOLE", "TSN", "CALM", "ANDE", "PPC", "SFD",
})


def bar_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    ""
    if isinstance(bars.index, pd.DatetimeIndex):
        return bars.index
    return pd.DatetimeIndex(pd.to_datetime(bars["date"]))


def latest_fundamental(fundamentals: pd.DataFrame | None) -> pd.Series | None:
    ""
    if fundamentals is None or fundamentals.empty:
        return None
    f = fundamentals.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    return f.sort_values("period_end").iloc[-1]


def ttm_yield_series(
    dividends: pd.DataFrame | None,
    close: pd.Series,
    dts: pd.DatetimeIndex,
) -> pd.Series | None:
    ""
    if dividends is None or dividends.empty:
        return None
    d = dividends[dividends["special"] == 0] if "special" in dividends else dividends
    if d.empty:
        return None
    paid = pd.Series(
        d["rate"].astype(float).values,
        index=pd.to_datetime(d["ex_date"]),
    ).sort_index()
    daily = paid.groupby(level=0).sum().reindex(
        pd.date_range(min(paid.index.min(), dts.min()), dts.max(), freq="D"),
        fill_value=0.0,
    )
    ttm = daily.rolling("365D").sum().reindex(dts)
    return 100.0 * ttm / close.set_axis(dts)


def yoy_pairs(
    fundamentals: pd.DataFrame | None,
    col: str,
    n_pairs: int = 4,
) -> list[float]:
    ""
    if fundamentals is None or fundamentals.empty or col not in fundamentals:
        return []
    f = fundamentals.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = (f.sort_values("period_end").drop_duplicates("period_end", keep="last")
          .dropna(subset=[col]))
    if len(f) < 2:
        return []
    dates = f["period_end"].reset_index(drop=True)
    vals = f[col].astype(float).reset_index(drop=True)
    out: list[float] = []
    for i in range(len(f) - 1, 0, -1):
        target = dates[i] - pd.Timedelta(days=365)
        deltas = (dates[:i] - target).abs()
        j = int(deltas.idxmin())
        if deltas[j] > pd.Timedelta(days=45) or vals[j] == 0:
            continue
        out.append(float(vals[i] / vals[j]) - 1.0)
        if len(out) == n_pairs:
            break
    return list(reversed(out))


def yoy_growth(fundamentals: pd.DataFrame | None, col: str) -> float | None:
    ""
    pairs = yoy_pairs(fundamentals, col, n_pairs=1)
    return pairs[-1] if pairs else None


def trailing_eps_annual(fundamentals_annual: pd.DataFrame | None) -> float | None:
    ""
    if fundamentals_annual is None or fundamentals_annual.empty \
            or "eps" not in fundamentals_annual:
        return None
    f = fundamentals_annual.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    eps = f.sort_values("period_end")["eps"].dropna()
    return float(eps.iloc[-1]) if len(eps) else None


def quarterly_history(fundamentals: pd.DataFrame | None, n: int) -> pd.DataFrame | None:
    ""
    if fundamentals is None or fundamentals.empty:
        return None
    f = fundamentals.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f.sort_values("period_end").drop_duplicates("period_end", keep="last")
    if len(f) < n:
        return None
    return f.tail(n)
