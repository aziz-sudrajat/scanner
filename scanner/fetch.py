"""Data fetching helpers.

Primary source: CoinGecko public API (free, no API key required).

Endpoints used:
  GET /coins/markets        -> snapshot: price, 24h change, volume, rank
  GET /coins/{id}/market_chart -> historical candles for indicator math
"""

import time
from typing import Iterable, List, Dict, Any, Optional

import requests

from .config import COINGECKO_BASE, VS_CURRENCY, REQUEST_DELAY


class CoinGeckoError(Exception):
    """Raised when the CoinGecko API returns a non-success status."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _get(url: str, params: Optional[Dict[str, Any]] = None, tries: int = 3) -> Dict[str, Any]:
    """GET helper with rate-limit (429) backoff and retry."""
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            last_exc = exc
            if i == tries - 1:
                break
            time.sleep(1)
            continue
        if resp.status_code == 429:
            # rate limited - backoff dengan jeda menurun, lalu coba lagi
            if i == tries - 1:
                break
            time.sleep(min(15.0, 5.0 * (i + 1)))   # 5s, 10s, 15s
            continue
        if resp.status_code >= 400:
            last_exc = CoinGeckoError(f"HTTP {resp.status_code}", status_code=resp.status_code)
            break
        return resp.json()
    if isinstance(last_exc, Exception):
        raise CoinGeckoError(f"request failed after {tries} tries: {last_exc}") from last_exc
    raise CoinGeckoError("unexpected exit from request loop")


def fetch_snapshot(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch the top `limit` coins with live price snapshot.

    Returns a list of dicts with keys: id, symbol, name, current_price,
    market_cap, market_cap_rank, total_volume, price_change_percentage_24h.
    """
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {"vs_currency": VS_CURRENCY, "order": "market_cap_desc",
              "per_page": limit, "page": 1}
    data = _get(url, params=params)
    if not isinstance(data, list):
        raise CoinGeckoError("unexpected response shape from /coins/markets")
    return data


def fetch_ohlcv(coin_id: str, interval: str = "daily", days: int = 200) -> List[Dict[str, Any]]:
    """Fetch OHLCV history for one coin.

    Returns a list of {time, open, high, low, close, volume} sorted oldest->newest.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": days, "interval": f"{interval}"}
    data = _get(url, params=params)
    prices = data.get("prices", [])
    if not prices:
        raise CoinGeckoError(f"no OHLCV data returned for coin '{coin_id}'")
    # normalize: CoinGecko returns [timestamp_ms, price]
    out = []
    for ts_ms, close in prices:
        out.append({
            "time": ts_ms,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0.0,          # /coins/{id}/market_chart does not give volume split
        })
    # CoinGecko also returns market_caps and total_volumes; align them by timestamp
    caps = data.get("market_caps", [])
    vol = data.get("total_volumes", [])
    cap_by_ts = {t: c for t, c in caps}
    vol_by_ts = {t: v for t, v in vol}
    for row in out:
        row["market_cap"] = cap_by_ts.get(row["time"], 0.0)
        row["total_volume"] = vol_by_ts.get(row["time"], 0.0)
    # sort oldest->newest
    out.sort(key=lambda r: r["time"])
    return out


def fetch_coins_ids(snapshot: List[Dict[str, Any]]) -> List[str]:
    """Convert a /coins/markets snapshot into a list of coin ids."""
    return [c.get("id") for c in snapshot if c.get("id")]
