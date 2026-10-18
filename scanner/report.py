"""Output formatting for the Signal Scanner.

Prints an aligned console table and exports results to CSV and/or JSON.
Columns adapt to the scoring mode (snapshot vs OHLCV technical).
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

from .signals import SignalResult


def _atomic_write_text(path: str, content: str) -> None:
    """Tulis file secara atomik (pola dari azzx/core.py).

    Tulis ke temp file dulu, lalu rename — file tidak pernah setengah jadi
    walau proses crash di tengah jalan.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".scan-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _fmt_price(v: float) -> str:
    """Format harga adaptif — koin murah tidak jadi $0.00."""
    if v is None:
        return "$-"
    v = float(v)
    if v >= 1000:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    if v >= 0.01:
        return f"${v:,.4f}"
    # koin sangat murah: tampilkan hingga 8 digit signifikan
    s = f"${v:.8f}".rstrip("0").rstrip(".")
    return s if s != "$" else "$0"


def _fmt_num(v: float, dec: int = 2) -> str:
    return f"{v:,.{dec}f}"


def _fmt_pct(v: float, dec: int = 2) -> str:
    return f"{v:+,.{dec}f}%"


def _signal_badge(signal: str) -> str:
    return {"STRONG_BUY": "STRONG BUY",
            "BUY": "BUY",
            "NEUTRAL": "NEUTRAL",
            "SELL": "SELL",
            "STRONG_SELL": "STRONG SELL"}.get(signal, signal)


def _signal_colored(signal: str, width: int = 11) -> str:
    """Badge berwarna dengan padding manual agar alignment tidak rusak ANSI."""
    colors = {"STRONG_BUY": "\033[1;32m", "BUY": "\033[1;36m",
              "NEUTRAL": "\033[1;33m", "SELL": "\033[1;31m",
              "STRONG_SELL": "\033[1;31m"}
    plain = _signal_badge(signal)
    padded = plain.ljust(width)
    c = colors.get(signal, "")
    return f"{c}{padded}\033[0m" if c else padded


def _use_tech(results: Iterable[SignalResult]) -> bool:
    results = list(results)
    return any(r.ema50 != 0 or r.rsi14 != 0 for r in results)


def _score(r: SignalResult) -> float:
    # net score: prefer technical score, else snapshot score
    return getattr(r, "score", 0.0)


def render_table(results: Iterable[SignalResult]) -> str:
    """Build a aligned, color-coded console table."""
    results = list(results)
    tech = _use_tech(results)
    hdr = f"{'#':>3} | {'Code':>6} {'Name':<12} {'Price':>12} {'Score':>6} {'Signal':<11} {'Conf':>4}"
    if tech:
        hdr += f" | {'EMA200':>8} {'RSI':>5} {'MACD':>7}"
    else:
        hdr += f" | {'24h%':>7} {'Vol/Mc':>7} {'Volty%':>6}"
    lines = [hdr, "-" * len(hdr)]
    for i, r in enumerate(results, start=1):
        row = (f"{i:>3} | {r.symbol:>6} {r.name:<12} {_fmt_price(r.price):>12} "
               f"{_score(r):>6.1f} {_signal_colored(r.signal)} {r.confidence:>4}")
        if tech:
            row += f" | {_fmt_num(r.ema200):>8} {_fmt_num(r.rsi14):>5} {_fmt_num(r.macd_hist):>7}"
        else:
            row += f" | {_fmt_pct(r.pct_24h):>7} {_fmt_num(r.volume_to_mcap):>7} {_fmt_num(r.volatility_24h):>6}"
        lines.append(row)
    return "\n".join(lines)


def export(results: Iterable[SignalResult], path: str, fmt: str = "csv") -> None:
    """Write results to `path` in csv or json format (atomic write)."""
    results = list(results)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "rank", "coin_id", "name", "symbol", "price",
            "ema50", "ema200", "ema20", "ema12", "ema24",
            "rsi14", "macd_hist", "volume_ratio",
            "pct_from_support", "pct_from_resistance",
            "pct_24h", "volume_to_mcap", "volatility_24h",
            "score", "net", "signal", "confidence", "factors",
        ])
        for i, r in enumerate(results, start=1):
            writer.writerow([
                i, r.coin_id, r.name, r.symbol, r.price,
                r.ema50, r.ema200, r.ema20, r.ema12, r.ema24,
                r.rsi14, r.macd_hist, r.volume_ratio,
                r.pct_from_support, r.pct_from_resistance,
                r.pct_24h, r.volume_to_mcap, r.volatility_24h,
                r.score, r.net, r.signal, r.confidence, json.dumps(r.factors),
            ])
        _atomic_write_text(path, buf.getvalue())
    elif fmt == "json":
        _atomic_write_text(path,
                           json.dumps([r.__dict__ for r in results],
                                      indent=2, default=str))
    else:
        raise ValueError(f"unknown format: {fmt}")


def count_by_signal(results: Iterable[SignalResult]) -> dict:
    from collections import Counter
    return dict(Counter(r.signal for r in results))
