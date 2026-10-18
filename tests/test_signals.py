"""Tests untuk scanner/signals.py — klasifikasi & scoring."""
import unittest

import numpy as np
import pandas as pd

from scanner.signals import (
    analyze, score_snapshot, to_score0100, SignalResult,
    STRONG_BUY_THRESHOLD, BUY_THRESHOLD, SELL_THRESHOLD, STRONG_SELL_THRESHOLD,
)
from scanner import config as cfg


def snap(**kw):
    base = {"id": "test", "current_price": 100.0, "market_cap": 1_000_000_000,
            "total_volume": 50_000_000, "price_change_percentage_24h": 0.0,
            "high_24h": 105.0, "low_24h": 95.0, "market_cap_rank": 50}
    base.update(kw)
    return base


class TestToScore0100(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(to_score0100(100), 100.0)
        self.assertEqual(to_score0100(-100), 0.0)
        self.assertEqual(to_score0100(0), 50.0)

    def test_clamp(self):
        self.assertEqual(to_score0100(500), 100.0)
        self.assertEqual(to_score0100(-500), 0.0)


class TestScoreSnapshot(unittest.TestCase):
    def test_strong_pump(self):
        r = score_snapshot(snap(price_change_percentage_24h=25.0,
                                 total_volume=500_000_000), "T", "t")
        self.assertIn(r.signal, ("BUY", "STRONG_BUY"))
        self.assertGreater(r.score, 50)

    def test_strong_dump(self):
        # rank 200: liquidity bonus kecil, momentum -25% harus dominan -> SELL
        r = score_snapshot(snap(price_change_percentage_24h=-25.0,
                                 market_cap_rank=200), "T", "t")
        self.assertIn(r.signal, ("SELL", "STRONG_SELL"))
        self.assertLess(r.score, 50)

    def test_neutral(self):
        r = score_snapshot(snap(), "T", "t")
        # dengan rank 50 & vol normal, harusnya tidak ekstrem
        self.assertNotIn(r.signal, ("STRONG_BUY", "STRONG_SELL"))

    def test_missing_fields(self):
        r = score_snapshot({"id": "x"}, "X", "x")
        self.assertIsInstance(r, SignalResult)
        self.assertFalse(np.isnan(r.score))

    def test_no_nan(self):
        r = score_snapshot(snap(market_cap=0, total_volume=0,
                                 high_24h=0, low_24h=0), "T", "t")
        for v in (r.score, r.net, r.pct_24h, r.volatility_24h):
            self.assertFalse(np.isnan(v), f"NaN di {v}")


class TestAnalyze(unittest.TestCase):
    def _df(self, n=250, trend=0.1):
        return pd.DataFrame({
            "close": [100.0 + i * trend + (i % 5) for i in range(n)],
            "total_volume": [1000.0 + i for i in range(n)],
        })

    def test_uptrend_bullish(self):
        sig = analyze(self._df(250, 0.5), cfg, "c", "C", "c", "")
        self.assertIn(sig.signal, ("BUY", "STRONG_BUY", "NEUTRAL"))
        self.assertFalse(np.isnan(sig.score))

    def test_downtrend_bearish(self):
        sig = analyze(self._df(250, -0.5), cfg, "c", "C", "c", "")
        self.assertIn(sig.signal, ("SELL", "STRONG_SELL", "NEUTRAL"))

    def test_short_data_no_nan(self):
        # 70 candle < EMA200 period — tidak boleh NaN
        sig = analyze(self._df(70, 0.2), cfg, "c", "C", "c", "")
        self.assertFalse(np.isnan(sig.score))
        self.assertFalse(np.isnan(sig.rsi14))
        self.assertIn("atr_volatility", sig.factors)

    def test_score_range(self):
        sig = analyze(self._df(250, 0.0), cfg, "c", "C", "c", "")
        self.assertTrue(0 <= sig.score <= 100)
        self.assertTrue(-100 <= sig.net <= 100)


if __name__ == "__main__":
    unittest.main()
