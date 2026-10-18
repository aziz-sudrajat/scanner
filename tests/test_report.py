"""Tests untuk scanner/report.py — format & tabel."""
import json
import os
import tempfile
import unittest

from scanner.report import _fmt_price, render_table, export, _atomic_write_text
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


if __name__ == "__main__":
    unittest.main()
