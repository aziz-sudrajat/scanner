#!/usr/bin/env python3
"""Signal Scanner - CLI entry point (dual mode).

Both modes consume the same scoring report, so output columns are shared.

  snapshot   (default) -- uses CoinGecko ``/coins/markets`` snapshot.
                         Fast, 1 HTTP call, rate-limit friendly.
  technical  -- fetches per-coin OHLCV history, runs EMA/RSI/MACD etc.
                         Slower; needs CoinGecko rate limit to be clear.

Usage:
  python3 main.py                       # snapshot scan of top 30
  python3 main.py --mode technical --limit 10
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from scanner import __version__
from scanner.config import (CANDLE_INTERVAL, DEFAULT_EXCLUDE_TOP, DEFAULT_LIMIT,
                     FETCH_WORKERS, MAX_CANDLES, REQUEST_DELAY, VS_CURRENCY)
from scanner.cache import RateLimiter, clear_cache
from scanner.fetch import fetch_snapshot, fetch_ohlcv, fetch_coins_ids, CoinGeckoError
from scanner.report import export, render_table, count_by_signal
from scanner.signals import SIGNAL_TIERS, analyze, score_snapshot
from scanner import config as _cfg

# module-level snapshot cache (technical mode reuses the ranked list)
_snap_cache: Dict[tuple, List[dict]] = {}

# Rate limiter global: jaga jarak antar request CoinGecko
_limiter = RateLimiter(REQUEST_DELAY)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signal-scanner",
        description="Multi-factor crypto signal scanner (CoinGecko)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""contoh:
  python3 main.py                                   # snapshot top 30 (cepat)
  python3 main.py --limit 50 --exclude-top 10       # top 50, skip 10 besar
  python3 main.py --mode technical --limit 10       # analisa teknikal 10 koin
  python3 main.py --signal STRONG_BUY --export csv   # filter + simpan CSV
  python3 main.py --no-cache                        # fetch fresh (abaikan cache)
  python3 main.py --clear-cache                     # hapus cache lalu keluar

mode:
  snapshot  = 1 API call, cepat, ramah rate-limit (default)
  technical = OHLCV per koin + EMA/RSI/MACD/ATR, lambat (~10 dtk/koin)

config:
  ~/.config/signal-scanner/config.json dapat meng-override threshold
  (lihat --config). Contoh: {"STRONG_BUY_THRESHOLD": 50}""",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-n", "--limit", type=int, default=DEFAULT_LIMIT,
                   help="number of coins to scan (ranked by market cap)")
    p.add_argument("--exclude-top", type=int, default=DEFAULT_EXCLUDE_TOP,
                   help="skip the top N coins (usually BTC/ETH)")
    p.add_argument("--mode", choices=["snapshot", "technical"], default="snapshot",
                   help="snapshot = live price snapshot (fast, reliable); "
                        "technical = OHLCV + EMA/RSI/MACD (slower)")
    p.add_argument("--interval", choices=["daily", "hourly"], default=CANDLE_INTERVAL,
                   help="candle granularity for technical scoring")
    p.add_argument("--days", type=int, default=MAX_CANDLES,
                   help="history days per coin (technical mode)")
    p.add_argument("--workers", type=int, default=FETCH_WORKERS,
                   help="parallel API workers (technical mode)")
    p.add_argument("--signal", choices=SIGNAL_TIERS, default="all",
                   help="only show this signal tier")
    p.add_argument("--export", choices=["csv", "json", "csv,json"], default=None,
                   help="also write results to file(s)")
    p.add_argument("--out", default="results/snapshot",
                   help="output base path")
    p.add_argument("--no-color", action="store_true", dest="color",
                   default=True, help="disable ANSI colors")
    p.add_argument("--quiet", action="store_true", help="suppress progress logs")
    p.add_argument("--config", default=None,
                   help="path ke JSON config user (override config.py); "
                        "default: ~/.config/signal-scanner/config.json bila ada")
    p.add_argument("--no-cache", action="store_true",
                   help="matikan disk cache (fetch fresh dari API)")
    p.add_argument("--clear-cache", action="store_true",
                   help="hapus semua cache disk lalu keluar")
    return p


def _format_error(exc: Exception) -> str:
    """Ubah exception jadi pesan yang actionable untuk user."""
    if isinstance(exc, CoinGeckoError):
        if exc.status_code == 429:
            return ("Rate limited (429) oleh CoinGecko. "
                    "Tunggu ~1 menit, kecilkan --limit/--workers, "
                    "atau coba lagi nanti.")
        if exc.status_code and exc.status_code >= 500:
            return (f"Server CoinGecko error ({exc.status_code}). "
                    "Coba lagi beberapa menit.")
    msg = str(exc)
    if "Failed to establish" in msg or "Max retries" in msg or "timeout" in msg.lower():
        return ("Koneksi ke CoinGecko gagal. Cek internet/proxy, "
                "lalu coba lagi.")
    return msg[:150]


class _Progress:
    """Progress bar ASCII sederhana ke stderr (tanpa dependency)."""

    def __init__(self, total: int, quiet: bool, width: int = 20):
        self.total = total
        self.quiet = quiet
        self.width = width
        self.done = 0
        self._last_len = 0

    def tick(self, label: str = ""):
        if self.quiet or self.total <= 0:
            return
        self.done += 1
        pct = self.done / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        msg = f"\r[{self.done}/{self.total}] {bar} {pct:>3.0%} {label[:24]}"
        # hapus sisa baris sebelumnya
        msg = msg.ljust(self._last_len)
        self._last_len = len(msg)
        print(msg, end="", file=sys.stderr, flush=True)

    def finish(self):
        if not self.quiet and self.total > 0:
            print(file=sys.stderr)


def _fetch_one(args: tuple) -> Optional[dict]:
    coin_id, interval, days = args
    _limiter.acquire()                 # rate limit adaptif, bukan sleep buta
    try:
        df = fetch_ohlcv(coin_id, interval=interval, days=days)
        return {"coin_id": coin_id, "rows": df}
    except CoinGeckoError as exc:
        return {"coin_id": coin_id, "error": str(exc)}


def _run_technical(limit: int, exclude_top: int, interval: str,
                    days: int, workers: int, quiet: bool) -> List:
    """Fetch OHLCV per coin (after excluding top `exclude_top`), then run."""
    ranked = _ranked(limit, exclude_top)
    ids = fetch_coins_ids(ranked[exclude_top:exclude_top + limit])
    by_id = {c["id"]: c for c in ranked}
    results, fail = [], 0
    t0 = time.time()
    prog = _Progress(len(ids), quiet)
    if not quiet:
        print(f"[1/3] fetch OHLCV ({len(ids)} koin, mode technical)", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, (cid, interval, days)): cid for cid in ids}
        for fut in as_completed(futs):
            res = fut.result()
            prog.tick(res["coin_id"])
            if "error" in res:
                fail += 1
                if not quiet:
                    print(f"\n   [skip] {res['coin_id']}: "
                          f"{_format_error(Exception(res['error']))}", file=sys.stderr)
            else:
                results.append(res)
    prog.finish()
    if not quiet:
        print(f"[2/3] fetched {len(results)}/{len(ids)} coins "
              f"({fail} gagal, {time.time()-t0:.1f}s)", file=sys.stderr)
    return _analyze(results, by_id, quiet)


def _analyze(results, by_id: Dict[str, dict], quiet: bool) -> List:
    """Run ``analyze()`` on the fetched OHLCV data."""
    import pandas as pd
    out = []
    for res in results:
        cid = res["coin_id"]
        rows = res["rows"]
        if len(rows) < 60:
            continue
        # Bangun DataFrame langsung dari rows (1x, bukan 2x)
        dfp = pd.DataFrame(rows)[["close", "total_volume"]]
        dfp.index = pd.to_datetime([r["time"] for r in rows], unit="ms")
        coin = by_id.get(cid, {})
        sig = analyze(dfp, _cfg, cid,
                      coin.get("name", ""), coin.get("symbol", ""), "")
        out.append(sig)
    return out


def _ranked(limit: int, exclude_top: int) -> List[dict]:
    """Return the ranked coin list, cached so technical mode doesn't refetch."""
    key = (limit, exclude_top)
    if key not in _snap_cache:
        # Fetch secukupnya saja, bukan 250 fixed — hemat rate limit.
        want = max(limit + exclude_top, 50)
        snap = fetch_snapshot(limit=want)
        _snap_cache[key] = [c for c in snap if c.get("id")]
    return _snap_cache[key]


def _run_snapshot(limit: int, exclude_top: int) -> List:
    """One /coins/markets call, score every eligible coin."""
    ranked = _ranked(limit, exclude_top)
    return [score_snapshot(c, c.get("name", ""), c.get("symbol", ""))
            for c in ranked[exclude_top:limit + exclude_top]]


def run_scanner(args: Optional[argparse.Namespace] = None) -> int:
    ns = args or _build_parser().parse_args()
    if ns.clear_cache:
        n = clear_cache()
        print(f"cache dibersihkan ({n} file)")
        return 0
    if ns.no_cache:
        import os as _os
        _os.environ["SCAN_NO_CACHE"] = "1"
    # User config override (pola azzx): JSON -> override konstanta config.py
    from scanner.config import apply_user_config, USER_CONFIG_PATH
    cfg_path = ns.config or USER_CONFIG_PATH
    applied = apply_user_config(cfg_path)
    if applied and not ns.quiet:
        print(f"[config] override dari {cfg_path}: {sorted(applied)}",
              file=sys.stderr)
    out_path = Path(ns.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if ns.mode == "technical":
        try:
            data = _run_technical(ns.limit, ns.exclude_top, ns.interval, ns.days, ns.workers, ns.quiet)
            if not data:
                raise CoinGeckoError("technical scan returned no coins")
        except Exception as exc:  # rate limit / transient -> fall back to snapshot
            if not ns.quiet:
                print(f"[warn] {_format_error(exc)}",
                      file=sys.stderr)
                print("[warn] fallback ke snapshot mode", file=sys.stderr)
            data = _run_snapshot(ns.limit, ns.exclude_top)
    else:
        if not ns.quiet:
            print("[1/3] fetch snapshot", file=sys.stderr)
        data = _run_snapshot(ns.limit, ns.exclude_top)

    data.sort(key=lambda r: r.score, reverse=True)

    if not ns.quiet:
        print("[3/3] report", file=sys.stderr)
    if not ns.color:
        for line in render_table(data).splitlines():
            print(line)
    else:
        print(render_table(data))

    counts = count_by_signal(data)
    if not ns.quiet:
        print(f"\nSignal mix: {counts}")
        if data:
            s = max(data, key=lambda r: r.score)
            print(f"Strongest signal: {s.symbol.upper()} ({s.name}) -> {s.signal} "
                  f"(score {s.score:+.1f}, confidence {s.confidence}%)")

    if ns.export:
        out_path.mkdir(parents=True, exist_ok=True)
        for fmt in ns.export.split(","):
            export(data, out_path / f"snapshot.{fmt}", fmt=fmt.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(run_scanner())
