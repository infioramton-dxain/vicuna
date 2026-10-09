""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def ibs(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    ""
    rng = (high - low).replace(0, np.nan)
    return (close - low) / rng


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n).mean()
    std = s.rolling(n).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    ""
    if len(close) < 2 * n + 1:
        return pd.Series(np.nan, index=close.index)
    from ta.trend import ADXIndicator

    return ADXIndicator(high=high, low=low, close=close, window=n, fillna=False).adx()


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    ""
    nine_high = high.rolling(9).max()
    nine_low = low.rolling(9).min()
    tenkan = (nine_high + nine_low) / 2

    p26_high = high.rolling(26).max()
    p26_low = low.rolling(26).min()
    kijun = (p26_high + p26_low) / 2

    senkou_a = (tenkan + kijun) / 2
    p52_high = high.rolling(52).max()
    p52_low = low.rolling(52).min()
    senkou_b = (p52_high + p52_low) / 2

    return tenkan, kijun, senkou_a, senkou_b


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10, mult: float = 3.0):
    ""
    a = atr(high, low, close, n)
    hl2 = (high + low) / 2
    upperband = hl2 + mult * a
    lowerband = hl2 - mult * a

    direction = pd.Series(index=close.index, dtype="float64")
    st = pd.Series(index=close.index, dtype="float64")

    for i in range(len(close)):
        if i == 0 or pd.isna(a.iloc[i]):
            direction.iloc[i] = 1.0
            st.iloc[i] = lowerband.iloc[i]
            continue
        prev_st = st.iloc[i - 1]
        if close.iloc[i] > prev_st:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < prev_st:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        if direction.iloc[i] == 1.0:
            st.iloc[i] = max(lowerband.iloc[i], prev_st) if direction.iloc[i - 1] == 1.0 else lowerband.iloc[i]
        else:
            st.iloc[i] = min(upperband.iloc[i], prev_st) if direction.iloc[i - 1] == -1.0 else upperband.iloc[i]

    return st, direction


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    from ta.trend import PSARIndicator

    p = PSARIndicator(high=high, low=low, close=close, step=step, max_step=max_step, fillna=False)
    psar = p.psar()
    return psar


def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    ha_close = (open_ + high + low + close) / 4
    ha_open = pd.Series(index=close.index, dtype="float64")
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close})


def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    return high.rolling(n).max(), low.rolling(n).min()
