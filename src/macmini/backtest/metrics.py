""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    ""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float((r.mean() / std) * math.sqrt(periods_per_year))


def profit_factor(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / peak
    return float(dd.min())


def deflated_sharpe_ratio(
    sharpe: float,
    n_returns: int,
    n_trials: int,
    skew: float,
    kurt: float,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int | None = 252,
) -> float:
    ""
    if n_returns < 2 or n_trials < 1:
        return 0.0
    if periods_per_year is not None and periods_per_year > 1:
        scale = math.sqrt(periods_per_year)
        sharpe = sharpe / scale
        benchmark_sharpe = benchmark_sharpe / scale
    e = math.e
    emc = 0.5772156649015329

    var_sr = (1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_returns - 1)
    if var_sr <= 0:
        return 0.0
    sd_sr = math.sqrt(var_sr)

    if n_trials > 1:
        z_1 = stats.norm.ppf(1 - 1 / n_trials)
        z_2 = stats.norm.ppf(1 - 1 / (n_trials * e))
        sharpe_0 = sd_sr * ((1 - emc) * z_1 + emc * z_2)
    else:
        sharpe_0 = 0.0

    sharpe_diff = sharpe - benchmark_sharpe - sharpe_0
    z = sharpe_diff / sd_sr
    return float(stats.norm.cdf(z))


def bonferroni_pvalue(p_value: float, n_trials: int) -> float:
    ""
    return float(min(1.0, max(0.0, p_value) * max(1, n_trials)))


def sharpe_pvalue(sharpe: float, n_returns: int) -> float:
    ""
    if n_returns < 2:
        return 1.0
    t = sharpe * math.sqrt(n_returns)
    return float(1 - stats.norm.cdf(t))
