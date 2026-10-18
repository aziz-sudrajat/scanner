"""Tests untuk scanner/indicators.py — edge cases & kebenaran matematis."""
import unittest

import numpy as np
import pandas as pd

from scanner.indicators import (
    ema, sma, rsi, macd, volume_ratio, atr, atr_pct,
    swing_levels, pct_from_support, pct_from_resistance, safe_last,
)


def series(vals):
    return pd.Series([float(v) for v in vals])


class TestSafeLast(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(safe_last(series([1, 2, 3]), -1), 3.0)

    def test_nan_fallback(self):
        self.assertEqual(safe_last(series([np.nan, np.nan]), 7.0), 7.0)

    def test_inf_fallback(self):
        self.assertEqual(safe_last(series([1, np.inf]), 7.0), 7.0)

    def test_empty(self):
        self.assertEqual(safe_last(series([]), 5.0), 5.0)


class TestEma(unittest.TestCase):
    def test_short_data_all_nan(self):
        # period > len -> semua NaN (min_periods=period)
        self.assertTrue(ema(series(range(10)), 200).isna().all())

    def test_converges_to_price(self):
        s = series([100.0] * 300)
        self.assertAlmostEqual(safe_last(ema(s, 50), -1), 100.0, places=1)


class TestRsi(unittest.TestCase):
    def test_flat_price_nan_safe(self):
        r = rsi(series([100.0] * 30), 14)
        # gain=loss=0 -> NaN; safe_last harus fallback
        self.assertEqual(safe_last(r, 50.0), 50.0)

    def test_all_gains_high_rsi(self):
        s = series([100 + i for i in range(30)])
        r = rsi(s, 14)
        self.assertGreater(safe_last(r, -1), 70)

    def test_all_losses_low_rsi(self):
        s = series([200 - i for i in range(30)])
        r = rsi(s, 14)
        self.assertLess(safe_last(r, -1), 30)


class TestVolumeRatio(unittest.TestCase):
    def test_zero_volume_neutral(self):
        vr = volume_ratio(series([0.0] * 30), 20)
        self.assertEqual(safe_last(vr, -1), 1.0)

    def test_spike(self):
        v = series([100.0] * 20 + [500.0])
        self.assertGreater(safe_last(volume_ratio(v, 20), -1), 2.0)


class TestAtr(unittest.TestCase):
    def test_steady_1pct(self):
        prices = series([100.0 * (1.01 ** i) for i in range(30)])
        av = safe_last(atr_pct(prices, 14), -1)
        self.assertTrue(0.5 < av < 2.0, f"ATR%={av}")

    def test_flat_zero(self):
        av = safe_last(atr_pct(series([100.0] * 30), 14), -1)
        self.assertEqual(av, 0.0)


class TestSwing(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            swing_levels(series([]))

    def test_pct_guards(self):
        # harga nol -> tidak division by zero
        z = series([0.0] * 10)
        self.assertTrue((pct_from_support(z) == 0.0).all())


if __name__ == "__main__":
    unittest.main()
