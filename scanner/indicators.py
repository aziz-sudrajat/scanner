"""Technical indicator calculations for the Signal Scanner.

Pure computations on a pandas Series of closing prices (and volume).
All functions are pure - no side effects - so results are reproducible.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Ambil nilai terakhir series; fallback ke `default` jika NaN/inf.

    Mencegah NaN dari indikator (data < period) meracuni skor secara diam-diam.
    """
    if series is None or len(series) == 0:
        return default
    v = series.iloc[-1]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if np.isnan(f) or np.isinf(f):
        return default
    return f


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with span=`period`."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def moving_mavg(prices: pd.Series, periods: Tuple[int, ...] = (20, 50, 200)) -> pd.DataFrame:
    """Return a DataFrame of EMAs for several periods (labeled by period)."""
    return pd.DataFrame({p: ema(prices, p) for p in periods}, index=prices.index)


# --------------------------------------------------------------------------- #
# RSI (Relative Strength Index)
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI on `series`."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta).clip(lower=0).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


# --------------------------------------------------------------------------- #
# MACD (12, 26, 9)
# --------------------------------------------------------------------------- #
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by its trailing average (`window`).

    NaN/inf (volume nol) diganti 1.0 = netral.
    """
    avg = volume.rolling(window=window, min_periods=window).mean()
    ratio = volume / avg
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)


# --------------------------------------------------------------------------- #
# ATR (Average True Range) — volatilitas
# --------------------------------------------------------------------------- #
def atr(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR dari close saja (|close - prev_close| sebagai true range).

    CoinGecko /market_chart tidak memberi H/L terpisah (O=H=L=C),
    jadi true range = |close - close_prev|. Cukup untuk filter volatilitas.
    """
    tr = close.diff().abs()
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_pct(close: pd.Series, period: int = 14) -> pd.Series:
    """ATR sebagai % dari harga — bisa dibandingkan antar koin."""
    a = atr(close, period)
    return (a / close.replace(0, np.nan) * 100.0).fillna(0.0)


# --------------------------------------------------------------------------- #
# Support / resistance (simple swing levels)
# --------------------------------------------------------------------------- #
def swing_levels(prices: pd.Series, lookback: int = 100) -> Tuple[float, float]:
    """Return (recent swing low, recent swing high) over the trailing window."""
    if prices is None or len(prices) == 0:
        raise ValueError("swing_levels: price series kosong")
    window = prices.values if len(prices) < lookback else prices.values[-lookback:]
    return float(np.min(window)), float(np.max(window))


def pct_from_support(prices: pd.Series, lookback: int = 100) -> pd.Series:
    """Current price as % above the recent swing low."""
    low = swing_levels(prices, lookback)[0]
    if low <= 0:
        return pd.Series(0.0, index=prices.index)
    return (prices / low - 1.0) * 100.0


def pct_from_resistance(prices: pd.Series, lookback: int = 100) -> pd.Series:
    """Current price as % below the recent swing high."""
    high = swing_levels(prices, lookback)[1]
    if high <= 0:
        return pd.Series(0.0, index=prices.index)
    return (prices / high - 1.0) * 100.0
