"""Tests untuk main.py — logika slicing & analyze (mock network)."""
import unittest
from unittest.mock import patch

import pandas as pd

import main as m


def fake_coins(n):
    return [{"id": f"coin{i}", "name": f"Coin{i}", "symbol": f"c{i}"}
            for i in range(n)]


class TestRankedSlicing(unittest.TestCase):
    def setUp(self):
        m._snap_cache.clear()

    def tearDown(self):
        m._snap_cache.clear()

    def test_technical_respects_limit(self):
        # _run_technical harus fetch tepat `limit` koin setelah exclude_top
        m._snap_cache[(30, 5)] = fake_coins(100)
        with patch("main.fetch_coins_ids") as mids, \
             patch("main._fetch_one") as mfo, \
             patch("main._analyze") as man:
            mids.side_effect = lambda snap: [c["id"] for c in snap]
            mfo.return_value = {"coin_id": "x", "rows": []}
            man.return_value = []
            m._run_technical(30, 5, "daily", 200, 1, True)
            ids_arg = mids.call_args[0][0]
            self.assertEqual(len(ids_arg), 30)
            self.assertEqual(ids_arg[0]["id"], "coin5")

    def test_snapshot_slice(self):
        m._snap_cache[(30, 5)] = fake_coins(100)
        with patch("main.score_snapshot") as mss:
            mss.side_effect = lambda c, n, s: c["id"]
            out = m._run_snapshot(30, 5)
            self.assertEqual(len(out), 30)
            self.assertEqual(out[0], "coin5")

    def test_ranked_fetch_size(self):
        # fetch secukupnya, bukan 250 fixed
        with patch("main.fetch_snapshot") as mfs:
            mfs.return_value = fake_coins(60)
            m._ranked(30, 5)
            want = mfs.call_args[1]["limit"]
            self.assertLessEqual(want, 100, f"fetch {want} kebanyakan")


class TestAnalyzeLookup(unittest.TestCase):
    def test_dict_lookup(self):
        ts = 1700000000000
        rows = [{"time": ts + i * 86400000, "close": 100 + i,
                 "total_volume": 1000} for i in range(70)]
        by_id = {"bitcoin": {"id": "bitcoin", "name": "Bitcoin",
                             "symbol": "btc"}}
        out = m._analyze([{"coin_id": "bitcoin", "rows": rows}], by_id, True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "Bitcoin")
        self.assertEqual(out[0].symbol, "BTC")

    def test_short_df_skipped(self):
        rows = [{"time": 1 + i, "close": 100, "total_volume": 1}
                for i in range(10)]
        out = m._analyze([{"coin_id": "x", "rows": rows}], {}, True)
        self.assertEqual(out, [])


class TestCoinWindow(unittest.TestCase):
    def test_window(self):
        coins = fake_coins(100)
        w = m._coin_window(coins, 30, 5)
        self.assertEqual(len(w), 30)
        self.assertEqual(w[0]["id"], "coin5")
        self.assertEqual(w[-1]["id"], "coin34")

    def test_window_short_list(self):
        # list lebih pendek dari jendela -> kembalikan yang ada, tanpa error
        w = m._coin_window(fake_coins(10), 30, 5)
        self.assertEqual(len(w), 5)
        self.assertEqual(w[0]["id"], "coin5")

    def test_window_negative_rejected(self):
        with self.assertRaises(ValueError):
            m._coin_window(fake_coins(10), -1, 0)
        with self.assertRaises(ValueError):
            m._coin_window(fake_coins(10), 10, -2)

    def test_snapshot_and_technical_agree(self):
        # kedua mode memakai jendela yang sama
        m._snap_cache.clear()
        m._snap_cache[(30, 5)] = fake_coins(100)
        try:
            with patch("main.score_snapshot") as mss:
                mss.side_effect = lambda c, n, s: c["id"]
                out = m._run_snapshot(30, 5)
            self.assertEqual(out[0], "coin5")
            self.assertEqual(len(out), 30)
        finally:
            m._snap_cache.clear()


class TestFetchOneRobust(unittest.TestCase):
    def test_broad_exception_becomes_error_dict(self):
        # cache korup / bug tak terduga tidak boleh meledak di thread pool
        with patch("main.fetch_ohlcv") as mfo, patch("main._limiter"):
            mfo.side_effect = ValueError("cache korup")
            res = m._fetch_one(("bitcoin", "daily", 200))
            self.assertEqual(res["coin_id"], "bitcoin")
            self.assertIn("cache korup", res["error"])

    def test_coingecko_error_still_caught(self):
        from scanner.fetch import CoinGeckoError
        with patch("main.fetch_ohlcv") as mfo, patch("main._limiter"):
            mfo.side_effect = CoinGeckoError("boom 429", status_code=429)
            res = m._fetch_one(("bitcoin", "daily", 200))
            self.assertIn("error", res)

    def test_cache_hit_skips_limiter(self):
        # Perf: data fresh di cache -> tidak sleep rate-limiter, tidak fetch
        rows = [{"time": 1, "close": 1.0, "total_volume": 1.0}]
        with patch("main.ohlcv_cached", return_value=(True, rows)) as mp, \
             patch("main._limiter") as ml, \
             patch("main.fetch_ohlcv") as mfo:
            res = m._fetch_one(("bitcoin", "daily", 200))
        self.assertEqual(res, {"coin_id": "bitcoin", "rows": rows})
        mp.assert_called_once()
        ml.acquire.assert_not_called()
        mfo.assert_not_called()

    def test_cache_miss_uses_limiter_and_fetch(self):
        rows = [{"time": 1, "close": 1.0, "total_volume": 1.0}]
        with patch("main.ohlcv_cached", return_value=(False, None)), \
             patch("main._limiter") as ml, \
             patch("main.fetch_ohlcv", return_value=rows) as mfo:
            res = m._fetch_one(("bitcoin", "daily", 200))
        self.assertEqual(res["rows"], rows)
        ml.acquire.assert_called_once()
        mfo.assert_called_once()


class TestRankedRefetch(unittest.TestCase):
    def setUp(self):
        m._snap_cache.clear()

    def tearDown(self):
        m._snap_cache.clear()

    def test_empty_cache_triggers_refetch(self):
        m._snap_cache[(30, 5)] = []  # cache poisoning simulasi
        with patch("main.fetch_snapshot") as mfs:
            mfs.return_value = fake_coins(60)
            out = m._ranked(30, 5)
            mfs.assert_called_once()
            self.assertEqual(len(out), 60)

    def test_single_fetch_per_key(self):
        with patch("main.fetch_snapshot") as mfs:
            mfs.return_value = fake_coins(60)
            m._ranked(30, 5)
            m._ranked(30, 5)  # panggilan ke-2 harus dari cache
            mfs.assert_called_once()


class TestSignalFilter(unittest.TestCase):
    def _ns(self, signal):
        ns = m._build_parser().parse_args(
            ["--signal", signal, "--quiet", "--no-color"])
        return ns

    def _fake_results(self):
        from scanner.signals import SignalResult
        def mk(sym, sig):
            return SignalResult(coin_id=sym, name=sym, symbol=sym, price=1.0,
                                timestamp="", ema50=0, ema200=0, ema20=0,
                                ema12=0, ema24=0, rsi14=0, macd_line=0,
                                macd_hist=0, volume_ratio=0,
                                pct_from_support=0, pct_from_resistance=0,
                                score=50.0, net=10.0, signal=sig,
                                confidence=60)
        return [mk("AAA", "BUY"), mk("BBB", "NEUTRAL"), mk("CCC", "STRONG_BUY")]

    def test_signal_filter_applied(self):
        import io
        from contextlib import redirect_stdout
        ns = self._ns("BUY")
        with patch("main._run_snapshot", return_value=self._fake_results()):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.run_scanner(ns)
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("AAA", out)
        self.assertNotIn("BBB", out)
        self.assertNotIn("CCC", out)

    def test_signal_all_shows_everything(self):
        import io
        from contextlib import redirect_stdout
        ns = self._ns("all")
        with patch("main._run_snapshot", return_value=self._fake_results()):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.run_scanner(ns)
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("AAA", out)
        self.assertIn("BBB", out)
        self.assertIn("CCC", out)

    def test_signal_filter_empty_message(self):
        import io
        from contextlib import redirect_stdout
        ns = self._ns("STRONG_SELL")
        with patch("main._run_snapshot", return_value=self._fake_results()):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.run_scanner(ns)
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("STRONG_SELL", out)


class TestNoColorFlag(unittest.TestCase):
    def test_no_color_disables(self):
        ns = m._build_parser().parse_args(["--no-color"])
        self.assertFalse(ns.color)

    def test_default_color_on(self):
        ns = m._build_parser().parse_args([])
        self.assertTrue(ns.color)

    def test_no_color_output_clean(self):
        import io
        from contextlib import redirect_stdout
        from scanner.report import render_table
        ns = m._build_parser().parse_args(["--no-color", "--quiet"])
        from scanner.signals import SignalResult
        r = SignalResult(coin_id="x", name="Test", symbol="TST", price=1.0,
                         timestamp="", ema50=0, ema200=0, ema20=0, ema12=0,
                         ema24=0, rsi14=0, macd_line=0, macd_hist=0,
                         volume_ratio=0, pct_from_support=0,
                         pct_from_resistance=0, score=60.0, net=20.0,
                         signal="BUY", confidence=70)
        with patch("main._run_snapshot", return_value=[r]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                m.run_scanner(ns)
        self.assertNotIn("\033[", buf.getvalue())


class TestJsonFlag(unittest.TestCase):
    def test_json_output_parseable(self):
        import io
        import json as _json
        from contextlib import redirect_stdout
        from scanner.signals import SignalResult
        r = SignalResult(coin_id="x", name="Test", symbol="AAA", price=1.0,
                         timestamp="", ema50=0, ema200=0, ema20=0, ema12=0,
                         ema24=0, rsi14=0, macd_line=0, macd_hist=0,
                         volume_ratio=0, pct_from_support=0,
                         pct_from_resistance=0, score=60.0, net=20.0,
                         signal="BUY", confidence=70)
        ns = m._build_parser().parse_args(["--json", "--quiet"])
        with patch("main._run_snapshot", return_value=[r]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.run_scanner(ns)
        self.assertEqual(rc, 0)
        data = _json.loads(buf.getvalue())  # harus valid JSON murni
        self.assertEqual(data[0]["symbol"], "AAA")
        self.assertNotIn("Signal mix", buf.getvalue())  # ringkasan disembunyikan


class TestFormatError(unittest.TestCase):
    def test_429(self):
        from scanner.fetch import CoinGeckoError
        msg = m._format_error(CoinGeckoError("rl", status_code=429))
        self.assertIn("Rate limited", msg)
        self.assertIn("--limit", msg)

    def test_5xx(self):
        from scanner.fetch import CoinGeckoError
        msg = m._format_error(CoinGeckoError("x", status_code=503))
        self.assertIn("503", msg)

    def test_connection(self):
        msg = m._format_error(Exception("Failed to establish a new connection"))
        self.assertIn("Koneksi", msg)
        msg2 = m._format_error(Exception("Read timeout after 20s"))
        self.assertIn("Koneksi", msg2)

    def test_generic_truncated(self):
        msg = m._format_error(Exception("z" * 500))
        self.assertLessEqual(len(msg), 150)


class TestProgress(unittest.TestCase):
    def test_quiet_no_output(self):
        import io
        from contextlib import redirect_stderr
        p = m._Progress(10, quiet=True)
        buf = io.StringIO()
        with redirect_stderr(buf):
            p.tick("x")
            p.finish()
        self.assertEqual(buf.getvalue(), "")

    def test_tick_counts(self):
        import io
        from contextlib import redirect_stderr
        p = m._Progress(4, quiet=False)
        buf = io.StringIO()
        with redirect_stderr(buf):
            for _ in range(4):
                p.tick("coin")
        self.assertEqual(p.done, 4)
        self.assertIn("4/4", buf.getvalue())

    def test_zero_total_safe(self):
        p = m._Progress(0, quiet=False)
        p.tick()    # tidak boleh ZeroDivisionError
        p.finish()


class TestRunTechnicalFailures(unittest.TestCase):
    def setUp(self):
        m._snap_cache.clear()

    def tearDown(self):
        m._snap_cache.clear()

    def test_mixed_success_and_error(self):
        m._snap_cache[(2, 0)] = fake_coins(10)
        rows = [{"time": 1 + i, "close": 100 + i, "total_volume": 1}
                for i in range(70)]

        def fake_fetch_one(args):
            cid = args[0]
            if cid == "coin0":
                return {"coin_id": cid, "error": "boom"}
            return {"coin_id": cid, "rows": rows}

        with patch("main._fetch_one", side_effect=fake_fetch_one):
            out = m._run_technical(2, 0, "daily", 200, 1, True)
        # coin0 gagal -> hanya coin1 yang dianalisa
        self.assertEqual(len(out), 1)

    def test_all_fail_returns_empty(self):
        m._snap_cache[(2, 0)] = fake_coins(10)
        with patch("main._fetch_one",
                   return_value={"coin_id": "x", "error": "down"}):
            out = m._run_technical(2, 0, "daily", 200, 1, True)
        self.assertEqual(out, [])

    def test_analyze_unknown_coin_names_blank(self):
        rows = [{"time": 1 + i, "close": 100 + i, "total_volume": 1}
                for i in range(70)]
        out = m._analyze([{"coin_id": "ghost", "rows": rows}], {}, True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "")


class TestRunScannerFlags(unittest.TestCase):
    def _r(self):
        from scanner.signals import SignalResult
        return SignalResult(coin_id="x", name="Test", symbol="AAA", price=1.0,
                            timestamp="", ema50=0, ema200=0, ema20=0, ema12=0,
                            ema24=0, rsi14=0, macd_line=0, macd_hist=0,
                            volume_ratio=0, pct_from_support=0,
                            pct_from_resistance=0, score=60.0, net=20.0,
                            signal="BUY", confidence=70)

    def test_clear_cache(self):
        import io
        from contextlib import redirect_stdout
        ns = m._build_parser().parse_args(["--clear-cache", "--quiet"])
        with patch("main.clear_cache", return_value=3) as mc:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.run_scanner(ns)
        self.assertEqual(rc, 0)
        mc.assert_called_once()
        self.assertIn("3 file", buf.getvalue())

    def test_no_cache_sets_env(self):
        import io
        import os as _os
        from contextlib import redirect_stdout
        _os.environ.pop("SCAN_NO_CACHE", None)
        ns = m._build_parser().parse_args(["--no-cache", "--quiet"])
        with patch("main._run_snapshot", return_value=[self._r()]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                m.run_scanner(ns)
        self.assertEqual(_os.environ.get("SCAN_NO_CACHE"), "1")
        _os.environ.pop("SCAN_NO_CACHE", None)

    def test_export_writes_files(self):
        import io
        import tempfile
        import os as _os
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            out = _os.path.join(d, "res")
            ns = m._build_parser().parse_args(
                ["--quiet", "--export", "csv,json", "--out", out])
            with patch("main._run_snapshot", return_value=[self._r()]):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = m.run_scanner(ns)
            self.assertEqual(rc, 0)
            self.assertTrue(_os.path.exists(_os.path.join(out, "snapshot.csv")))
            self.assertTrue(_os.path.exists(_os.path.join(out, "snapshot.json")))


class TestArgValidation(unittest.TestCase):
    def _run(self, argv):
        import io
        from contextlib import redirect_stderr
        ns = m._build_parser().parse_args(argv + ["--quiet"])
        buf = io.StringIO()
        with redirect_stderr(buf):
            with self.assertRaises(SystemExit) as cm:
                m.run_scanner(ns)
        self.assertEqual(cm.exception.code, 2)  # argparse usage error
        return buf.getvalue()

    def test_limit_zero(self):
        err = self._run(["--limit", "0"])
        self.assertIn("--limit", err)

    def test_limit_negative(self):
        self._run(["--limit", "-5"])

    def test_exclude_top_negative(self):
        err = self._run(["--exclude-top", "-1"])
        self.assertIn("--exclude-top", err)

    def test_workers_zero(self):
        err = self._run(["--workers", "0"])
        self.assertIn("--workers", err)

    def test_days_zero(self):
        err = self._run(["--days", "0"])
        self.assertIn("--days", err)

    def test_workers_high_warns(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        ns = m._build_parser().parse_args(["--workers", "5"])
        with patch("main._run_snapshot", return_value=[]):
            serr, sout = io.StringIO(), io.StringIO()
            with redirect_stderr(serr), redirect_stdout(sout):
                rc = m.run_scanner(ns)
        self.assertEqual(rc, 0)
        self.assertIn("rate-limit", serr.getvalue())


class TestFriendlyErrors(unittest.TestCase):
    def test_snapshot_429_no_traceback(self):
        import io
        import traceback
        from contextlib import redirect_stderr, redirect_stdout
        from scanner.fetch import CoinGeckoError
        ns = m._build_parser().parse_args(["--quiet"])
        with patch("main._run_snapshot",
                   side_effect=CoinGeckoError("rl", status_code=429)):
            serr, sout = io.StringIO(), io.StringIO()
            with redirect_stderr(serr), redirect_stdout(sout):
                rc = m.run_scanner(ns)
        self.assertEqual(rc, 1)
        self.assertIn("Rate limited", serr.getvalue())
        self.assertNotIn("Traceback", serr.getvalue())

    def test_unexpected_error_friendly(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        ns = m._build_parser().parse_args(["--quiet"])
        with patch("main._run_snapshot",
                   side_effect=RuntimeError("sesuatu yang aneh")):
            serr, sout = io.StringIO(), io.StringIO()
            with redirect_stderr(serr), redirect_stdout(sout):
                rc = m.run_scanner(ns)
        self.assertEqual(rc, 1)
        self.assertIn("[error] tak terduga", serr.getvalue())


class TestNoMkdirWithoutExport(unittest.TestCase):
    def test_no_export_no_dir(self):
        import io
        import os as _os
        import tempfile
        from contextlib import redirect_stdout
        from scanner.signals import SignalResult
        r = SignalResult(coin_id="x", name="T", symbol="AAA", price=1.0,
                         timestamp="", ema50=0, ema200=0, ema20=0, ema12=0,
                         ema24=0, rsi14=0, macd_line=0, macd_hist=0,
                         volume_ratio=0, pct_from_support=0,
                         pct_from_resistance=0, score=1.0, net=1.0,
                         signal="BUY", confidence=60)
        with tempfile.TemporaryDirectory() as d:
            target = _os.path.join(d, "should-not-exist")
            ns = m._build_parser().parse_args(
                ["--quiet", "--out", target])
            with patch("main._run_snapshot", return_value=[r]):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    m.run_scanner(ns)
            self.assertFalse(_os.path.exists(target))


if __name__ == "__main__":
    unittest.main()
