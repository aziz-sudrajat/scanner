"""Disk cache & rate limiter untuk Signal Scanner.

- ``disk_cache``: cache JSON berbasis file dengan TTL. Menghemat API calls
  saat run diulang (dev iteration, tuning threshold).
- ``RateLimiter``: jaga jarak minimum antar request (pengganti sleep buta).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


def _cache_dir() -> Path:
    d = Path(os.path.expanduser("~/.cache/signal-scanner"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(name: str, params: dict) -> str:
    raw = name + "|" + json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def cached(name: str, params: dict, ttl: int = 300, allow_stale: bool = True):
    """Decorator: cache hasil fungsi ke disk (JSON) selama `ttl` detik.

    `allow_stale=True`: bila fetch GAGAL (API down / rate limited habis retry)
    tetapi ada cache basi, kembalikan cache basi dengan peringatan ke stderr
    alih-alih melempar error. Untuk API rate-limit agresif seperti CoinGecko,
    data 1 jam lalu jauh lebih berguna daripada crash. Bisa dimatikan per
    panggilan bila kesegaran mutlak diperlukan.
    """
    def deco(fn: Callable):
        def wrapper(*args, **kwargs):
            if os.environ.get("SCAN_NO_CACHE"):
                return fn(*args, **kwargs)
            cdir = _cache_dir()
            fp = cdir / f"{_key(name, params)}.json"
            stale = None
            stale_age = 0.0
            if fp.exists():
                try:
                    stale_age = time.time() - fp.stat().st_mtime
                    stale = json.loads(fp.read_text(encoding="utf-8"))
                    if stale_age < ttl:
                        return stale
                except (OSError, ValueError):
                    stale = None
            try:
                result = fn(*args, **kwargs)
            except Exception:
                if allow_stale and stale is not None:
                    print(f"[warn] fetch '{name}' gagal; memakai cache basi "
                          f"({stale_age / 60:.0f} mnt)",
                          file=sys.stderr)
                    return stale
                raise
            try:
                fd, tmp = tempfile.mkstemp(dir=str(cdir), prefix=".cache-")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(result, fh)
                os.replace(tmp, fp)
            except (OSError, TypeError):
                pass  # cache gagal -> tetap return hasil
            return result
        return wrapper
    return deco


def cache_peek(name: str, params: dict, ttl: int = 300):
    """Cek cache disk TANPA network: return (True, value) bila fresh.

    Untuk menghindari kerja sia-sia (mis. rate-limiter sleep) saat data
    sudah ada di cache. Hormati SCAN_NO_CACHE=1 -> selalu (False, None).
    """
    if os.environ.get("SCAN_NO_CACHE"):
        return False, None
    fp = _cache_dir() / f"{_key(name, params)}.json"
    try:
        if fp.exists() and time.time() - fp.stat().st_mtime < ttl:
            return True, json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return False, None


def clear_cache() -> int:
    """Hapus semua cache disk. Return jumlah file dihapus."""
    cdir = _cache_dir()
    n = 0
    for f in cdir.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


class RateLimiter:
    """Jaga jarak minimum `min_interval` detik antar request.

    Thread-safe. Pengganti `time.sleep(REQUEST_DELAY)` buta per task —
    hanya sleep selama yang dibutuhkan sejak request terakhir.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> float:
        """Tunggu bila perlu. Return lama sleep (0 jika tidak perlu)."""
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            return max(0.0, wait)
