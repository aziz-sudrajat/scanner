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
from scanner.fetch import fetch_snapshot, fetch_ohlcv, fetch_coins_ids, CoinGeckoError
from scanner.report import export, render_table, count_by_signal
from scanner.signals import SIGNAL_TIERS, analyze, score_snapshot
from scanner import config as _cfg

# module-level snapshot cache (technical mode reuses the ranked list)
_snap_cache: Dict[tuple, List[dict]] = {}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signal-scanner",
        description="Multi-factor crypto signal scanner (CoinGecko)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    return p


def _fetch_one(args: tuple) -> Optional[dict]:
    coin_id, interval, days = args
    time.sleep(REQUEST_DELAY)            # jaga agar tidak diejek rate limit
    try:
        df = fetch_ohlcv(coin_id, interval=interval, days=days)
        return {"coin_id": coin_id, "rows": df}
    except CoinGeckoError as exc:
        return {"coin_id": coin_id, "error": str(exc)}


def _run_technical(limit: int, exclude_top: int, interval: str,
                    days: int, workers: int, quiet: bool) -> List:
    """Fetch OHLCV per coin (after excluding top `exclude_top`), then run."""
    ids = fetch_coins_ids(_ranked(limit, exclude_top)[exclude_top:])
    results, fail = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, (cid, interval, days)): cid for cid in ids}
        for fut in as_completed(futs):
            res = fut.result()
            if "error" in res:
                fail += 1
                if not quiet:
                    print(f"   [skip] {res['coin_id']}: {res['error']}", file=sys.stderr)
            else:
                results.append(res)
    if not quiet:
        print(f"[technical] fetched {len(results)}/{len(ids)} coins ({time.time()-t0:.1f}s)",
              file=sys.stderr)
    return _analyze(results, ids, interval, days, limit, exclude_top, quiet)


def _analyze(results, ids, interval, days, limit, exclude_top, quiet: bool) -> List:
    """Run ``analyze()`` on the fetched OHLCV data."""
    import pandas as pd
    ranked = _ranked(limit, exclude_top)
    out = []
    for res in results:
        cid = res["coin_id"]
        df = res["rows"]
        ts = pd.to_datetime([r["time"] for r in df], unit="ms")
        dfp = pd.DataFrame([{"close": r["close"], "total_volume": r["total_volume"]} for r in df])\
                  .set_index(ts)
        if len(dfp) < 60:
            continue
        sig = analyze(dfp, _cfg, cid, "", "")
        for c in ranked:
            if c["id"] == cid:
                sig.name = c.get("name", "")
                sig.symbol = c.get("symbol", "")
                break
        out.append(sig)
    return out


def _ranked(limit: int, exclude_top: int) -> List[dict]:
    """Return the ranked coin list, cached so technical mode doesn't refetch."""
    key = (limit, exclude_top)
    if key not in _snap_cache:
        snap = fetch_snapshot(limit=250)
        _snap_cache[key] = [c for c in snap if c.get("id")]
    return _snap_cache[key]


def _run_snapshot(limit: int, exclude_top: int) -> List:
    """One /coins/markets call, score every eligible coin."""
    ranked = _ranked(limit, exclude_top)
    return [score_snapshot(c, c.get("name", ""), c.get("symbol", ""))
            for c in ranked[exclude_top:limit + exclude_top]]


def run_scanner(args: Optional[argparse.Namespace] = None) -> int:
    ns = args or _build_parser().parse_args()
    out_path = Path(ns.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if ns.mode == "technical":
        try:
            data = _run_technical(ns.limit, ns.exclude_top, ns.interval, ns.days, ns.workers, ns.quiet)
            if not data:
                raise CoinGeckoError("technical scan returned no coins")
        except Exception as exc:  # rate limit / transient -> fall back to snapshot
            if not ns.quiet:
                print(f"[warn] technical mode failed ({exc}); falling back to snapshot mode",
                      file=sys.stderr)
            data = _run_snapshot(ns.limit, ns.exclude_top)
    else:
        data = _run_snapshot(ns.limit, ns.exclude_top)

    data.sort(key=lambda r: r.score, reverse=True)

    if not ns.quiet:
        print("[2/3] report", file=sys.stderr)
    if not ns.color:
        for line in render_table(data).splitlines():
            print(line)
    else:
        print(render_table(data))

    counts = count_by_signal(data)
    if not ns.quiet:
        print(f"\nSignal mix: {counts}")
        if data:
            strongest = max(data, key=lambda r: r.score if r.is_buy else -(r.score))
            s = strongest
            print(f"Strongest signal: {s.symbol.upper()} ({s.name}) -> {s.signal} "
                  f"(score {s.score:+.1f}, confidence {s.confidence}%)")

    if ns.export:
        out_path.mkdir(parents=True, exist_ok=True)
        for fmt in ns.export.split(","):
            export(data, out_path / f"snapshot.{fmt}", fmt=fmt.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(run_scanner())
