"""Data fetching helpers.

Primary source: CoinGecko public API (free, no API key required).

Endpoints used:
  GET /coins/markets        -> snapshot: price, 24h change, volume, rank
  GET /coins/{id}/market_chart -> historical candles for indicator math
"""

import random
import time
from typing import List, Dict, Any, Optional

import requests

from .config import (COINGECKO_BASE, VS_CURRENCY,
                     REQUEST_TIMEOUT, REQUEST_RETRIES, REQUEST_BACKOFF_BASE,
                     USER_AGENT)
from .cache import cached, cache_peek


class CoinGeckoError(Exception):
    """Raised when the CoinGecko API returns a non-success status."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _retry_after_seconds(resp: Any) -> Optional[float]:
    """Parse header Retry-After (detik). Return None bila tidak ada/invalid."""
    try:
        raw = resp.headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - headers bisa bukan mapping
        return None
    if raw is None:
        return None
    try:
        return max(0.0, min(120.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _sleep_backoff(attempt: int, retry_after: Optional[float] = None) -> float:
    """Backoff eksponensial + jitter (pola azzx/providers.py: never infinite).

    - Bila server memberi Retry-After: hormati, tambah jitter kecil.
    - Selain itu: full jitter dalam [0, min(30, base * 2**attempt)] agar
      retry berbarengan tidak "thundering herd".
    Return lama tidur (untuk test/logging).
    """
    if retry_after is not None:
        delay = min(retry_after + random.uniform(0, 1.0), 120.0)
    else:
        cap = min(30.0, REQUEST_BACKOFF_BASE * (2 ** attempt))
        delay = random.uniform(0, cap)
    time.sleep(delay)
    return delay


def _get(url: str, params: Optional[Dict[str, Any]] = None,
         tries: int = REQUEST_RETRIES) -> Any:
    """GET helper dengan retry + backoff.

    Kebijakan retry:
      - 429            -> retry, hormati header Retry-After
      - 5xx            -> retry (transient), bounded oleh `tries`
      - 4xx lain       -> raise langsung (tidak retry; mis. 404 = coin tak ada)
      - network error  -> retry, kecuali URL-nya sendiri invalid (bug)
      - JSON invalid   -> raise langsung

    Returns parsed JSON (dict or list). Raises CoinGeckoError on failure.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=REQUEST_TIMEOUT)
        except (requests.exceptions.InvalidURL,
                requests.exceptions.MissingSchema) as exc:
            # Bug di URL/base config — retry tidak akan membantu.
            raise CoinGeckoError(f"bad request URL: {exc}") from exc
        except requests.RequestException as exc:
            last_exc = exc
            if i < tries - 1:
                _sleep_backoff(i)
            continue
        if resp.status_code == 429:
            if i < tries - 1:
                _sleep_backoff(i, _retry_after_seconds(resp))
            else:
                last_exc = CoinGeckoError(
                    f"rate limited (429) after {tries} tries",
                    status_code=429)
            continue
        if 500 <= resp.status_code < 600:
            # Server error transient (502/503/504) — layak dicoba lagi.
            last_exc = CoinGeckoError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code)
            if i < tries - 1:
                _sleep_backoff(i)
            continue
        if resp.status_code >= 400:
            raise CoinGeckoError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code)
        try:
            return resp.json()
        except ValueError as exc:
            raise CoinGeckoError(f"invalid JSON response: {exc}") from exc
    if isinstance(last_exc, Exception):
        # Pertahankan status_code jika sudah CoinGeckoError (mis. 429)
        sc = getattr(last_exc, "status_code", None)
        raise CoinGeckoError(f"request failed after {tries} tries: {last_exc}",
                             status_code=sc) from last_exc
    raise CoinGeckoError("unexpected exit from request loop")


def _fetch_snapshot_page(limit: int, page: int) -> list:
    """Satu halaman /coins/markets (di-cache per limit+page, TTL 5 mnt)."""
    @cached("snapshot-page", {"limit": limit, "page": page, "v": 1}, ttl=300)
    def _go():
        url = f"{COINGECKO_BASE}/coins/markets"
        params = {"vs_currency": VS_CURRENCY, "order": "market_cap_desc",
                  "per_page": min(limit, 250), "page": page}
        data = _get(url, params=params)
        if not isinstance(data, list):
            raise CoinGeckoError("unexpected response shape from /coins/markets")
        return data
    return _go()


def fetch_snapshot(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch the top `limit` coins with live price snapshot.

    Returns a list of dicts with keys: id, symbol, name, current_price,
    market_cap, market_cap_rank, total_volume, price_change_percentage_24h.

    Hasil di-cache ke disk 5 menit (bisa dimatikan via env SCAN_NO_CACHE=1).
    Pagination otomatis jika limit > 250.
    """
    # cache key per limit agar tidak tercampur
    @cached(f"snapshot-l{limit}", {"v": 1}, ttl=300)
    def _all():
        out: List[Dict[str, Any]] = []
        page = 1
        # Guard anti infinite-loop: bila API terus mengembalikan halaman penuh
        # (bug server), berhenti setelah halaman yang dibutuhkan + 1 cadangan.
        max_pages = (limit // 250) + 2
        while len(out) < limit and page <= max_pages:
            chunk = _fetch_snapshot_page(limit, page)
            if not chunk:
                break
            out.extend(chunk)
            if len(chunk) < 250:
                break
            page += 1
        return out[:limit]
    return _all()


_OHLCV_TTL = 3600  # detik; samakan dengan cache_peek agar konsisten


def _ohlcv_cache_params(coin_id: str, interval: str, days: int) -> Dict[str, Any]:
    return {"coin": coin_id, "interval": interval, "days": days}


def fetch_ohlcv(coin_id: str, interval: str = "daily", days: int = 200) -> List[Dict[str, Any]]:
    """Fetch OHLCV history for one coin.

    Returns a list of {time, open, high, low, close, volume} sorted oldest->newest.
    Di-cache ke disk 1 jam (env SCAN_NO_CACHE=1 untuk matikan).
    """
    @cached("ohlcv", _ohlcv_cache_params(coin_id, interval, days), ttl=_OHLCV_TTL)
    def _go():
        return _fetch_ohlcv_uncached(coin_id, interval, days)
    return _go()


def ohlcv_cached(coin_id: str, interval: str = "daily",
                 days: int = 200) -> tuple:
    """Ambil OHLCV dari cache disk bila masih fresh, TANPA network.

    Return (True, rows) bila hit, (False, None) bila miss/expired.
    Dipakai worker agar tidak sleep rate-limiter sia-sia saat data sudah ada.
    """
    return cache_peek("ohlcv", _ohlcv_cache_params(coin_id, interval, days),
                      _OHLCV_TTL)


def _fetch_ohlcv_uncached(coin_id: str, interval: str, days: int) -> List[Dict[str, Any]]:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": days, "interval": f"{interval}"}
    data = _get(url, params=params)
    if not isinstance(data, dict):
        raise CoinGeckoError(f"unexpected response shape for coin '{coin_id}': "
                             f"expected object, got {type(data).__name__}")
    prices = data.get("prices", [])
    if not isinstance(prices, list):
        raise CoinGeckoError(f"unexpected 'prices' shape for coin '{coin_id}': "
                             f"expected list, got {type(prices).__name__}")
    if not prices:
        raise CoinGeckoError(f"no OHLCV data returned for coin '{coin_id}'")
    # normalize: CoinGecko returns [timestamp_ms, price]
    out = []
    skipped = 0
    for row in prices:
        try:
            ts_ms, close = row
            ts_ms, close = int(ts_ms), float(close)
        except (ValueError, TypeError):
            skipped += 1
            continue
        out.append({
            "time": ts_ms,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0.0,          # /coins/{id}/market_chart does not give volume split
        })
    if not out:
        raise CoinGeckoError(
            f"no valid OHLCV rows for coin '{coin_id}' ({skipped} malformed rows skipped)")
    # CoinGecko also returns market_caps and total_volumes; align them by timestamp
    # (baris malformed di sini di-skip diam-diam — bukan data utama).
    def _ts_map(key: str) -> Dict[Any, Any]:
        m: Dict[Any, Any] = {}
        rows = data.get(key, [])
        if not isinstance(rows, list):
            return m
        for r in rows:
            try:
                t, v = r
                m[int(t)] = float(v)
            except (ValueError, TypeError):
                continue
        return m
    cap_by_ts = _ts_map("market_caps")
    vol_by_ts = _ts_map("total_volumes")
    for row in out:
        row["market_cap"] = cap_by_ts.get(row["time"], 0.0)
        row["total_volume"] = vol_by_ts.get(row["time"], 0.0)
    # sort oldest->newest
    out.sort(key=lambda r: r["time"])
    return out


def fetch_coins_ids(snapshot: List[Dict[str, Any]]) -> List[str]:
    """Convert a /coins/markets snapshot into a list of coin ids."""
    return [c["id"] for c in snapshot if isinstance(c, dict) and c.get("id")]
