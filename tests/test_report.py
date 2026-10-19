"""Tests untuk scanner/report.py — format & tabel."""
import json
import math
import os
import tempfile
import unittest

from scanner.report import (_fmt_price, _fmt_num, _fmt_pct, _fmt_sig, _fmt_int,
                            render_table, export, _atomic_write_text)
from scanner.signals import SignalResult


def mk(symbol="TST", signal="BUY", conf=70, price=1.5, tech=False, **kw):
    base = dict(coin_id="x", name="Test", symbol=symbol, price=price,
                timestamp="", ema50=0, ema200=0, ema20=0, ema12=0, ema24=0,
                rsi14=0, macd_line=0, macd_hist=0, volume_ratio=0,
                pct_from_support=0, pct_from_resistance=0,
                pct_24h=5.5, volume_to_mcap=0.12, volatility_24h=3.2,
                score=65.0, net=30.0, signal=signal, confidence=conf)
    if tech:
        base.update(ema50=100, ema200=90, rsi14=55, macd_hist=0.5)
    base.update(kw)
    return SignalResult(**base)


class TestFmtPrice(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(_fmt_price(81234.567), "$81,235")
        self.assertEqual(_fmt_price(1.23456), "$1.23")
        self.assertEqual(_fmt_price(0.123456), "$0.1235")
        self.assertEqual(_fmt_price(0.00001234), "$0.00001234")
        self.assertEqual(_fmt_price(0.00000001), "$0.00000001")

    def test_zero_none(self):
        self.assertEqual(_fmt_price(0), "$0")
        self.assertEqual(_fmt_price(None), "$-")

    def test_nan_inf(self):
        self.assertEqual(_fmt_price(float("nan")), "$-")
        self.assertEqual(_fmt_price(float("inf")), "$-")


class TestFmtGuards(unittest.TestCase):
    def test_num_guards(self):
        self.assertEqual(_fmt_num(None), "-")
        self.assertEqual(_fmt_num(float("nan")), "-")
        self.assertEqual(_fmt_num(float("inf")), "-")
        self.assertEqual(_fmt_num(1.23456), "1.23")

    def test_pct_guards(self):
        self.assertEqual(_fmt_pct(None), "-")
        self.assertEqual(_fmt_pct(float("nan")), "-")
        self.assertEqual(_fmt_pct(5.5), "+5.50%")

    def test_sig_strips_zeros(self):
        self.assertEqual(_fmt_sig(0.0035), "0.0035")   # dulu "0.00"
        self.assertEqual(_fmt_sig(0.12), "0.12")
        self.assertEqual(_fmt_sig(0.0), "0")
        self.assertEqual(_fmt_sig(-0.00001), "0")
        self.assertEqual(_fmt_sig(-0.5), "-0.5")       # jangan jadi "0.5"
        self.assertEqual(_fmt_sig(None), "-")

    def test_int_guard(self):
        self.assertEqual(_fmt_int(60.7), "61")
        self.assertEqual(_fmt_int(None), "-")


class TestRenderTable(unittest.TestCase):
    def test_snapshot_header(self):
        t = render_table([mk()])
        hdr = t.split("\n")[0]
        self.assertIn("Vol/Mc", hdr)
        self.assertIn("Volty%", hdr)
        self.assertIn("24h%", hdr)

    def test_tech_header(self):
        t = render_table([mk(tech=True)])
        self.assertIn("EMA200", t.split("\n")[0])
        self.assertIn("RSI", t.split("\n")[0])

    def test_cheap_price_visible(self):
        t = render_table([mk(price=0.00001234)])
        self.assertIn("0.00001234", t)

    def test_empty(self):
        t = render_table([])
        self.assertIn("#", t.split("\n")[0])
        self.assertIn("(no results)", t)

    def test_explicit_tech_param(self):
        # mode eksplisit menang atas heuristik
        t = render_table([mk()], tech=True)
        self.assertIn("EMA200", t.split("\n")[0])
        t = render_table([mk(tech=True)], tech=False)
        self.assertIn("24h%", t.split("\n")[0])

    def test_no_color_no_ansi(self):
        t = render_table([mk()], color=False)
        self.assertNotIn("\033[", t)
        self.assertIn("BUY", t)

    def test_color_has_ansi(self):
        t = render_table([mk()], color=True)
        self.assertIn("\033[", t)

    def test_cheap_ema_visible(self):
        # EMA koin murah tidak boleh jadi 0.00
        t = render_table([mk(tech=True, price=0.00001234, ema200=0.00001111)],
                         tech=True)
        self.assertIn("0.00001111", t)

    def test_small_vol_ratio_visible(self):
        t = render_table([mk(volume_to_mcap=0.0035)])
        self.assertIn("0.0035", t)

    def test_none_fields_no_crash(self):
        r = mk(price=None, pct_24h=None, volume_to_mcap=None,
               volatility_24h=None, score=None, confidence=None)
        t = render_table([r])  # tidak boleh raise
        self.assertIn("$-", t)


class TestExport(unittest.TestCase):
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "o.csv")
            export([mk()], p, "csv")
            with open(p) as fh:
                content = fh.read()
            self.assertIn("net", content.split("\n")[0])
            self.assertIn("TST", content)

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "o.json")
            export([mk()], p, "json")
            with open(p) as fh:
                data = json.load(fh)
            self.assertEqual(data[0]["symbol"], "TST")

    def test_no_temp_leftovers(self):
        with tempfile.TemporaryDirectory() as d:
            _atomic_write_text(os.path.join(d, "f.txt"), "hi")
            left = [f for f in os.listdir(d) if f.startswith(".scan-")]
            self.assertEqual(left, [])

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            export([mk()], "/tmp/x", "xml")


class TestMisc(unittest.TestCase):
    def test_signal_badge_unknown_passthrough(self):
        from scanner.report import _signal_badge
        self.assertEqual(_signal_badge("???"), "???")
        self.assertEqual(_signal_badge("STRONG_BUY"), "STRONG BUY")

    def test_count_by_signal_empty(self):
        from scanner.report import count_by_signal
        self.assertEqual(count_by_signal([]), {})

    def test_count_by_signal(self):
        from scanner.report import count_by_signal
        out = count_by_signal([mk(signal="BUY"), mk(signal="BUY"),
                               mk(signal="SELL")])
        self.assertEqual(out, {"BUY": 2, "SELL": 1})

    def test_csv_header_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "o.csv")
            export([mk()], p, "csv")
            with open(p) as fh:
                header = fh.readline().strip().split(",")
        for col in ("rank", "coin_id", "score", "net", "signal",
                    "confidence", "factors"):
            self.assertIn(col, header)

    def test_atomic_write_creates_parents(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "dir", "f.txt")
            _atomic_write_text(p, "hello")
            with open(p) as fh:
                self.assertEqual(fh.read(), "hello")


class TestPrintJson(unittest.TestCase):
    def test_valid_json(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            from scanner.report import print_json
            print_json([mk(symbol="TST", signal="BUY")])
        data = json.loads(buf.getvalue())
        self.assertEqual(data[0]["symbol"], "TST")
        self.assertEqual(data[0]["signal"], "BUY")
        self.assertIn("factors", data[0])

    def test_empty_list(self):
        import io
        from contextlib import redirect_stdout
        from scanner.report import print_json
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_json([])
        self.assertEqual(json.loads(buf.getvalue()), [])


if __name__ == "__main__":
    unittest.main()
