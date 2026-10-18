"""Multi-factor signal generation for the Signal Scanner.

Two paths:

1) ``score_ohclv``  -- from historical candles (technical indicators:
   EMA, RSI, MACD, volume trend). Produces a full ``SignalResult``.
2) ``score_snapshot`` -- from the live CoinGecko markets snapshot (single
   HTTP call, reliable and rate-limit friendly). Produces a signal from
   momentum (24h %), volume activity, volatility and liquidity.

Both feed the same :class:`SignalResult` so the report layer is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config as _cfg
from .indicators import (ema, rsi, macd, volume_ratio, pct_from_support,
                         pct_from_resistance, atr_pct, safe_last)

# Signal tier labels
SIGNAL_TIERS = ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]

# Net score after clipping (signed, -100..100) -> used for classification only
STRONG_BUY_THRESHOLD = 35
BUY_THRESHOLD = 15
NEUTRAL_RANGE = (-15, 15)
SELL_THRESHOLD = -15
STRONG_SELL_THRESHOLD = -35


def to_score0100(net: float) -> float:
    """Map signed net (-100..100) to a 0..100 display score (higher = bullish)."""
    return round(max(0.0, min(100.0, net / 2.0 + 50.0)), 1)


@dataclass
class SignalResult:
    coin_id: str
    name: str
    symbol: str
    price: float
    timestamp: str
    # technical (OHLCV) fields
    ema50: float
    ema200: float
    ema20: float
    ema12: float
    ema24: float
    rsi14: float
    macd_line: float
    macd_hist: float
    volume_ratio: float
    pct_from_support: float
    pct_from_resistance: float
    # snapshot fields (populated when using snapshot mode)
    pct_24h: float = 0.0
    volume_to_mcap: float = 0.0
    volatility_24h: float = 0.0
    # scoring
    score: float = 0.0
    net: float = 0.0
    signal: str = "NEUTRAL"
    confidence: int = 50
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal in ("BUY", "STRONG_BUY")

    @property
    def is_sell(self) -> bool:
        return self.signal in ("SELL", "STRONG_SELL")


# --------------------------------------------------------------------------- #
# Snapshot scoring (single HTTP call)
# --------------------------------------------------------------------------- #
def score_snapshot(snap: Dict[str, Any], name: str, symbol: str) -> SignalResult:
    """Score one coin from a ``/coins/markets`` snapshot row."""
    price = float(snap.get("current_price", 0) or 0)
    market_cap = float(snap.get("market_cap", 0) or 0)
    total_volume = float(snap.get("total_volume", 0) or 0)
    pct_24h = float(snap.get("price_change_percentage_24h", 0) or 0)
    high = float(snap.get("high_24h", price) or price)
    low = float(snap.get("low_24h", price) or price)
    rank = int(snap.get("market_cap_rank", 0) or 0)

    # ---- factors ----
    # momentum: 24h change (scale +/-30)
    momentum = max(-30.0, min(30.0, pct_24h))
    # volume activity: volume / market cap (proxy for "how hot the coin is").
    # normal range ~0.05, hot > 0.2; mild scoring so low volume isn't harshly
    # punished (many legit coins trade below market cap).
    vol_activity = total_volume / market_cap if market_cap > 0 else 0.0
    vol_score = max(-5.0, min(15.0, (vol_activity - 0.05) * 10.0))
    # volatility: 24h range percentage; moderate volatility is opportunity
    vol_pct = (high - low) / low * 100.0 if low > 0 else 0.0
    vol_obs = max(-10.0, min(10.0, (vol_pct - 3.0) / 3.0))
    # liquidity: larger cap = less risky anchor (small cap penalty)
    liquidity = max(-10.0, min(10.0, (1.0 - min(rank, 250) / 250.0)) * 15.0)

    raw = momentum + vol_score + vol_obs + liquidity
    net = max(-100.0, min(100.0, raw))
    score = to_score0100(net)

    # ---- classification (on signed net) ----
    if net >= _cfg.STRONG_BUY_THRESHOLD:
        signal = "STRONG_BUY"
    elif net >= _cfg.BUY_THRESHOLD:
        signal = "BUY"
    elif net <= _cfg.STRONG_SELL_THRESHOLD:
        signal = "STRONG_SELL"
    elif net <= _cfg.SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    # confidence: from how strongly the factors agree
    active = sum(1 for v in (momentum, vol_score, vol_obs, liquidity) if v > 0)
    conf = min(95, max(5, round(active * 14 + abs(net) // 6)))

    factors = {
        "24h_momentum": momentum,
        "volume_activity": vol_score,
        "volatility": vol_obs,
        "liquidity": liquidity,
    }

    return SignalResult(
        coin_id=snap.get("id", ""),
        name=name, symbol=symbol.upper(), price=price, timestamp="",
        ema50=0.0, ema200=0.0, ema20=0.0, ema12=0.0, ema24=0.0,
        rsi14=0.0, macd_line=0.0, macd_hist=0.0, volume_ratio=0.0,
        pct_from_support=0.0, pct_from_resistance=0.0,
        pct_24h=pct_24h, volume_to_mcap=vol_activity,
        volatility_24h=vol_pct, score=score, net=net, signal=signal,
        confidence=conf, factors=factors,
    )


# --------------------------------------------------------------------------- #
# OHLCV (technical) scoring -- already implemented, kept for completeness
# --------------------------------------------------------------------------- #
def analyze(candle_df: pd.DataFrame, cfg: Any, coin_id: str, name: str, symbol: str, timestamp: str) -> SignalResult:
    close = candle_df["close"]
    volume = candle_df["total_volume"] if "total_volume" in candle_df.columns else candle_df["volume"]

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    ema20 = ema(close, 20)
    ema12 = ema(close, 12)
    ema24 = ema(close, 24)
    rsi14 = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close, 12, 26, 9)
    vol_ratio = volume_ratio(volume, 20)
    pct_sup = pct_from_support(close)
    pct_res = pct_from_resistance(close)
    atrp = atr_pct(close, 14)

    # safe_last: cegah NaN (data < period) meracuni skor diam-diam.
    # Default netral: ema→harga terakhir, rsi→50, sisanya→0.
    last_close = safe_last(close, 0.0)
    p = {
        "close": last_close,
        "ema50": safe_last(ema50, last_close),
        "ema200": safe_last(ema200, last_close),
        "ema20": safe_last(ema20, last_close),
        "ema12": safe_last(ema12, last_close),
        "ema24": safe_last(ema24, last_close),
        "rsi14": safe_last(rsi14, 50.0),
        "macd_hist": safe_last(macd_hist, 0.0),
        "macd": safe_last(macd_line, 0.0),
        "vol_ratio": safe_last(vol_ratio, 1.0),
        "pct_sup": safe_last(pct_sup, 0.0),
        "pct_res": safe_last(pct_res, 0.0),
        "atr_pct": safe_last(atrp, 0.0),
    }

    # ---- Trend sub-scores ----
    trend_price_above_ema200 = 25.0 if p["close"] > p["ema200"] else 0.0
    aligned_uptrend = 20.0 if (p["ema20"] > p["ema50"] > p["ema200"]) else 0.0
    short_momentum = 20.0 if (p["ema12"] > p["ema24"] > p["ema50"]) else 0.0
    aligned_downtrend = 20.0 if (p["ema20"] < p["ema50"] < p["ema200"]) else 0.0
    in_downtrend = aligned_downtrend > 0

    # ---- Momentum sub-scores ----
    rsi_val = p["rsi14"]
    if rsi_val < 30:
        # Oversold di downtrend kuat = falling knife, bukan bounce.
        # Redam sinyal mean-reversion agar tidak mengalahkan trend.
        rsi_bull = 5.0 if in_downtrend else 20.0
    elif rsi_val < 50:
        rsi_bull = 12.0 if rsi_val >= 35 else (5.0 if in_downtrend else 20.0)
    elif rsi_val < 70:
        rsi_bull = 4.0
    else:
        rsi_bull = -15.0

    macd_bull = 15.0 if p["macd_hist"] > 0 else 0.0
    vol_ratio = p["vol_ratio"]
    if vol_ratio > 1.5:
        vol_bull = 15.0
    elif vol_ratio >= 1.0:
        vol_bull = 10.0
    else:
        vol_bull = 0.0
    near_support = 0.0 if in_downtrend else (
        15.0 if (p["pct_sup"] >= 0 and p["pct_sup"] <= 3.0) else 0.0)
    # ATR%: volatilitas sehat = peluang; terlalu flat atau terlalu liar = risiko
    atrv = p["atr_pct"]
    if atrv < 0.5:
        atr_score = -5.0    # koin mati, tidak ada peluang
    elif atrv <= 8.0:
        atr_score = 5.0     # volatilitas sehat
    elif atrv <= 15.0:
        atr_score = 0.0     # agak liar, netral
    else:
        atr_score = -10.0   # terlalu liar, berisiko

    bullish = (trend_price_above_ema200 + aligned_uptrend + short_momentum +
               rsi_bull + macd_bull + vol_bull + near_support + atr_score)

    bearish = (aligned_downtrend +
               (15.0 if p["macd_hist"] < 0 else 0.0) +
               (15.0 if vol_ratio > 1.5 else 0.0) +
               (15.0 if vol_ratio > 1.2 else 0.0))

    net_raw = bullish - bearish
    net = max(-100.0, min(100.0, net_raw))

    if net >= _cfg.STRONG_BUY_THRESHOLD:
        signal = "STRONG_BUY"
    elif net >= _cfg.BUY_THRESHOLD:
        signal = "BUY"
    elif net <= _cfg.STRONG_SELL_THRESHOLD:
        signal = "STRONG_SELL"
    elif net <= _cfg.SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    active_bull = sum(1 for v in (trend_price_above_ema200, aligned_uptrend, short_momentum,
                                   rsi_bull, macd_bull, vol_bull, near_support) if v > 0)
    active_bear = sum(1 for v in (aligned_downtrend, (15.0 if p["macd_hist"] < 0 else 0.0),
                                   (15.0 if vol_ratio > 1.5 else 0.0), (15.0 if vol_ratio > 1.2 else 0.0)) if v > 0)
    n_agree = max(active_bull, active_bear)
    conf = min(95, max(5, round(n_agree * 11 + abs(net) // 4)))

    factors = {
        "price>EMA200": trend_price_above_ema200,
        "EMA20>EMA50>EMA200": aligned_uptrend,
        "EMA12>EMA24>EMA50": short_momentum,
        "RSI_30-50": rsi_bull,
        "MACD_hist>0": macd_bull,
        "vol>1.5x": vol_bull,
        "near_support": near_support,
        "atr_volatility": atr_score,
        "aligned_downtrend": aligned_downtrend,
        "macd_hist<0": (15.0 if p["macd_hist"] < 0 else 0.0),
        "vol>1.5x_dist": (15.0 if vol_ratio > 1.5 else 0.0),
        "vol>1.2x": (15.0 if vol_ratio > 1.2 else 0.0),
    }

    return SignalResult(
        coin_id=coin_id, name=name, symbol=symbol.upper(), price=p["close"], timestamp=timestamp,
        ema50=p["ema50"], ema200=p["ema200"], ema20=p["ema20"], ema12=p["ema12"], ema24=p["ema24"],
        rsi14=rsi_val, macd_line=p["macd"], macd_hist=p["macd_hist"],
        volume_ratio=vol_ratio, pct_from_support=pct_sup, pct_from_resistance=pct_res,
        pct_24h=0.0, volume_to_mcap=0.0, volatility_24h=0.0,
        net=net, score=to_score0100(net), signal=signal, confidence=conf, factors=factors,
    )
