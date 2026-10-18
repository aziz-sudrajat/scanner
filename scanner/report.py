"""Output formatting for the Signal Scanner.

Prints an aligned console table and exports results to CSV and/or JSON.
Columns adapt to the scoring mode (snapshot vs OHLCV technical).
"""

from __future__ import annotations

import csv
import json
from typing import Iterable, List

from .signals import SignalResult


def _fmt_price(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_num(v: float, dec: int = 2) -> str:
    return f"{v:,.{dec}f}"


def _fmt_pct(v: float, dec: int = 2) -> str:
    return f"{v:+,.{dec}f}%"


def _signal_badge(signal: str) -> str:
    return {"STRONG_BUY": "\033[1;32mSTRONG BUY\033[0m",
            "BUY": "\033[1;36mBUY\033[0m",
            "NEUTRAL": "\033[1;33mNEUTRAL\033[0m",
            "SELL": "\033[1;31mSELL\033[0m",
            "STRONG_SELL": "\033[1;31mSTRONG SELL\033[0m"}.get(signal, signal)


def _use_tech(results: Iterable[SignalResult]) -> bool:
    return any(getattr(r, "ema50", 0) or getattr(r, "rsi14", 0) for r in results)


def _score(r: SignalResult) -> float:
    # net score: prefer technical score, else snapshot score
    return getattr(r, "score", 0.0)


def render_table(results: Iterable[SignalResult]) -> str:
    """Build a aligned, color-coded console table."""
    results = list(results)
    tech = _use_tech(results)
    hdr = f"{'#':>3} | {'Code':>6} {'Name':<12} {'Price':>12} {'Score':>6} {'Signal':<12} {'Conf':>4}"
    if tech:
        hdr += f" | {'EMA200':>8} {'RSI':>5} {'MACD':>7}"
    lines = [hdr, "-" * len(hdr)]
    for i, r in enumerate(results, start=1):
        row = f"{i:>3} | {r.symbol:>6} {r.name:<12} {_fmt_price(r.price):>12} "
        row += f"{_score(r):>6.1f} {_signal_badge(r.signal):<12} {r.confidence:>4}"
        if tech:
            row += f" | {_fmt_num(r.ema200):>8} {_fmt_num(r.rsi14):>5} {_fmt_num(r.macd_hist):>7}"
        else:
            row += f" | {_fmt_pct(r.pct_24h):>7} {_fmt_num(r.volume_to_mcap):>7} {_fmt_num(r.volatility_24h):>6}"
        lines.append(row)
    return "\n".join(lines)


def export(results: Iterable[SignalResult], path: str, fmt: str = "csv") -> None:
    """Write results to `path` in csv or json format."""
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "rank", "coin_id", "name", "symbol", "price",
                "ema50", "ema200", "ema20", "ema12", "ema24",
                "rsi14", "macd_hist", "volume_ratio",
                "pct_from_support", "pct_from_resistance",
                "pct_24h", "volume_to_mcap", "volatility_24h",
                "score", "signal", "confidence", "factors",
            ])
            for i, r in enumerate(results, start=1):
                writer.writerow([
                    i, r.coin_id, r.name, r.symbol, r.price,
                    r.ema50, r.ema200, r.ema20, r.ema12, r.ema24,
                    r.rsi14, r.macd_hist, r.volume_ratio,
                    r.pct_from_support, r.pct_from_resistance,
                    r.pct_24h, r.volume_to_mcap, r.volatility_24h,
                    r.score, r.signal, r.confidence, json.dumps(r.factors),
                ])
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in results], fh, indent=2, default=str)
    else:
        raise ValueError(f"unknown format: {fmt}")


def count_by_signal(results: Iterable[SignalResult]) -> dict:
    from collections import Counter
    return dict(Counter(r.signal for r in results))
