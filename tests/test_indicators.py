"""Tests untuk scanner/indicators.py — edge cases & kebenaran matematis."""
import unittest

import numpy as np
import pandas as pd

from scanner.indicators import (
    ema, sma, rsi, macd, volume_ratio, atr, atr_pct, moving_mavg,
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

    def test_all_nan_neutral(self):
        n = pd.Series([np.nan] * 10)
        self.assertEqual(swing_levels(n), (0.0, 0.0))
        self.assertTrue((pct_from_support(n) == 0.0).all())
        self.assertTrue((pct_from_resistance(n) == 0.0).all())

    def test_partial_nan_ignored(self):
        s = pd.Series([np.nan, 90.0, 100.0, 110.0, np.nan])
        lo, hi = swing_levels(s)
        self.assertEqual(lo, 90.0)
        self.assertEqual(hi, 110.0)

    def test_nan_prices_no_nan_output(self):
        s = pd.Series([100.0, np.nan, 110.0])
        out = pct_from_support(s)
        self.assertFalse(out.isna().any())


class TestPeriodValidation(unittest.TestCase):
    def test_invalid_periods_rejected(self):
        s = series(range(30))
        for fn in (lambda: ema(s, 0), lambda: ema(s, -3),
                   lambda: sma(s, 0), lambda: rsi(s, 0),
                   lambda: macd(s, 12, 0, 9), lambda: atr(s, -1),
                   lambda: atr_pct(s, 0), lambda: volume_ratio(s, 0),
                   lambda: ema(s, 2.5), lambda: rsi(s, "14")):
            with self.assertRaises(ValueError):
                fn()

    def test_valid_periods_ok(self):
        s = series([100.0 + i * 0.1 for i in range(50)])
        self.assertFalse(ema(s, 1).isna().all())
        self.assertFalse(sma(s, 1).isna().all())
        ml, sg, hh = macd(s, 12, 26, 9)
        self.assertFalse(ml.isna().all())


class TestRsiFlat(unittest.TestCase):
    def test_flat_is_50_not_nan(self):
        r = rsi(series([100.0] * 30), 14)
        tail = r.dropna()
        self.assertTrue((tail == 50.0).all())

    def test_single_spike_then_flat(self):
        # naik sekali lalu flat: RSI harus terdefinisi (tidak NaN di ekor)
        s = series([100.0] * 20 + [110.0] + [110.0] * 20)
        self.assertFalse(np.isnan(safe_last(rsi(s, 14), np.nan)))


class TestAtrGuards(unittest.TestCase):
    def test_zero_and_inf_safe(self):
        s = pd.Series([0.0] * 30)
        self.assertTrue((atr_pct(s, 14) == 0.0).all())
        s2 = pd.Series([100.0] * 29 + [np.inf])
        out = atr_pct(s2, 14)
        self.assertFalse(np.isinf(safe_last(out, 0.0)))


class TestMovingMavg(unittest.TestCase):
    def test_labels_and_shape(self):
        s = series([100.0 + i for i in range(60)])
        df = moving_mavg(s, (20, 50))
        self.assertEqual(list(df.columns), [20, 50])
        self.assertEqual(len(df), 60)
        self.assertFalse(np.isnan(safe_last(df[20], np.nan)))


class TestSafeLastEdges(unittest.TestCase):
    def test_pd_na(self):
        self.assertEqual(safe_last(pd.Series([1.0, pd.NA]), 9.0), 9.0)

    def test_non_numeric_string(self):
        self.assertEqual(safe_last(pd.Series(["a", "b"]), 9.0), 9.0)

    def test_none_series(self):
        self.assertEqual(safe_last(None, 9.0), 9.0)


class TestSmaShort(unittest.TestCase):
    def test_short_all_nan(self):
        self.assertTrue(sma(series(range(5)), 20).isna().all())


if __name__ == "__main__":
    unittest.main()
