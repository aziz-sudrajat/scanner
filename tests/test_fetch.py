"""Tests untuk scanner/fetch.py — retry, validasi (mock requests)."""
import unittest
from unittest.mock import MagicMock, patch

from scanner import fetch
from scanner.fetch import CoinGeckoError
from scanner.cache import clear_cache


def resp(status=200, json_data=None, text=""):
    m = MagicMock()
    m.status_code = status
    m.text = text
    m.json.return_value = json_data
    return m


class TestGet(unittest.TestCase):
    def setUp(self):
        clear_cache()
    def test_user_agent_and_timeout(self):
        with patch("scanner.fetch.requests.get") as mg:
            mg.return_value = resp(200, [])
            fetch._get("http://x", tries=1)
            _, kw = mg.call_args
            self.assertIn("signal-scanner", kw["headers"]["User-Agent"])
            self.assertEqual(kw["timeout"], 20)

    def test_http_error_detail(self):
        with patch("scanner.fetch.requests.get") as mg:
            mg.return_value = resp(404, text="not here")
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=1)
            self.assertEqual(cm.exception.status_code, 404)
            self.assertIn("not here", str(cm.exception))

    def test_429_gives_up(self):
        import requests as _rq  # noqa
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.return_value = resp(429)
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=2)
            self.assertEqual(cm.exception.status_code, 429)

    def test_bad_json(self):
        with patch("scanner.fetch.requests.get") as mg:
            m = resp(200)
            m.json.side_effect = ValueError("no json")
            mg.return_value = m
            with self.assertRaises(CoinGeckoError):
                fetch._get("http://x", tries=1)


class TestFetchOhlcv(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _mk(self, prices):
        return {"prices": prices, "market_caps": [], "total_volumes": []}

    def test_malformed_rows_skipped(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk([[1, 100.0], "rusak", [3, "xx"],
                                        [4, 105.0]])
            out = fetch.fetch_ohlcv("c")
            self.assertEqual(len(out), 2)
            self.assertEqual(out[1]["close"], 105.0)

    def test_all_bad_raises(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk(["x", None])
            with self.assertRaises(CoinGeckoError) as cm:
                fetch.fetch_ohlcv("c")
            self.assertIn("malformed", str(cm.exception))

    def test_non_dict_raises(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = [1, 2]
            with self.assertRaises(CoinGeckoError):
                fetch.fetch_ohlcv("c")


class TestFetchCoinsIds(unittest.TestCase):
    def test_filters(self):
        snap = [{"id": "a"}, {"id": ""}, {"x": 1}, {"id": "b"}]
        self.assertEqual(fetch.fetch_coins_ids(snap), ["a", "b"])


def resp_headers(status=200, json_data=None, text="", headers=None):
    m = resp(status, json_data, text)
    m.headers = headers or {}
    return m


class TestRetryPolicy(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_5xx_retried_then_succeeds(self):
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.side_effect = [resp_headers(503, text="overload"),
                              resp_headers(200, [])]
            out = fetch._get("http://x", tries=3)
            self.assertEqual(out, [])
            self.assertEqual(mg.call_count, 2)

    def test_5xx_gives_up(self):
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.return_value = resp_headers(502, text="bad gateway")
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=2)
            self.assertEqual(cm.exception.status_code, 502)
            self.assertEqual(mg.call_count, 2)

    def test_404_not_retried(self):
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep") as ms:
            mg.return_value = resp_headers(404, text="not found")
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=3)
            self.assertEqual(cm.exception.status_code, 404)
            self.assertEqual(mg.call_count, 1)  # langsung raise, tanpa retry
            ms.assert_not_called()

    def test_429_honors_retry_after(self):
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep") as ms:
            mg.side_effect = [resp_headers(429, headers={"Retry-After": "7"}),
                              resp_headers(200, [])]
            fetch._get("http://x", tries=3)
            slept = ms.call_args[0][0]
            self.assertGreaterEqual(slept, 7.0)
            self.assertLess(slept, 9.0)

    def test_429_invalid_retry_after_ignored(self):
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.side_effect = [resp_headers(429, headers={"Retry-After": "nonsense"}),
                              resp_headers(200, {"ok": True})]
            out = fetch._get("http://x", tries=2)
            self.assertEqual(out, {"ok": True})

    def test_invalid_url_not_retried(self):
        import requests as _rq
        with patch("scanner.fetch.requests.get") as mg:
            mg.side_effect = _rq.exceptions.InvalidURL("bad schema")
            with self.assertRaises(CoinGeckoError) as cm:
                fetch._get("http://x", tries=3)
            self.assertIn("bad request URL", str(cm.exception))
            self.assertEqual(mg.call_count, 1)

    def test_timeout_retried(self):
        import requests as _rq
        with patch("scanner.fetch.requests.get") as mg, \
             patch("scanner.fetch.time.sleep"):
            mg.side_effect = [_rq.exceptions.ConnectTimeout("t"),
                              resp_headers(200, [])]
            out = fetch._get("http://x", tries=2)
            self.assertEqual(out, [])
            self.assertEqual(mg.call_count, 2)

    def test_backoff_jitter_bounds(self):
        with patch("scanner.fetch.time.sleep") as ms, \
             patch("scanner.fetch.random.uniform", return_value=1.5):
            d = fetch._sleep_backoff(0)
            self.assertEqual(d, 1.5)
            d = fetch._sleep_backoff(2, retry_after=7.0)
            self.assertGreaterEqual(d, 7.0)
            self.assertLessEqual(d, 8.5)  # retry_after + jitter(uniform->1.5)


class TestOhlcvValidation(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _mk(self, prices, caps="default", vols="default"):
        d = {"prices": prices}
        d["market_caps"] = [] if caps == "default" else caps
        d["total_volumes"] = [] if vols == "default" else vols
        return d

    def test_prices_not_list_raises(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk({"x": 1})
            with self.assertRaises(CoinGeckoError) as cm:
                fetch.fetch_ohlcv("c")
            self.assertIn("'prices' shape", str(cm.exception))

    def test_malformed_cap_rows_skipped(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk(
                [[1, 100.0], [2, 101.0]],
                caps=[[1, 500.0], "rusak", [2, None], [None]],
                vols=[[1, 10.0], [2, 20.0]])
            out = fetch.fetch_ohlcv("c")
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0]["market_cap"], 500.0)
            self.assertEqual(out[0]["total_volume"], 10.0)
            self.assertEqual(out[1]["market_cap"], 0.0)  # None -> skip

    def test_unsorted_input_sorted(self):
        with patch("scanner.fetch._get") as mg:
            mg.return_value = self._mk([[3, 103.0], [1, 101.0], [2, 102.0]])
            out = fetch.fetch_ohlcv("c")
            self.assertEqual([r["time"] for r in out], [1, 2, 3])


class TestSnapshotPagination(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_pagination_guard_no_infinite_loop(self):
        # API nakal: selalu kembalikan halaman penuh -> harus berhenti juga
        full = [{"id": f"c{i}"} for i in range(250)]
        with patch("scanner.fetch._fetch_snapshot_page", return_value=full):
            out = fetch.fetch_snapshot(limit=300)
        self.assertEqual(len(out), 300)  # dipotong ke limit, tidak hang

    def test_empty_snapshot_returns_empty(self):
        with patch("scanner.fetch._fetch_snapshot_page", return_value=[]):
            self.assertEqual(fetch.fetch_snapshot(limit=50), [])


class TestRetryAfterParse(unittest.TestCase):
    def test_valid(self):
        m = resp_headers(429, headers={"Retry-After": "12"})
        self.assertEqual(fetch._retry_after_seconds(m), 12.0)

    def test_missing(self):
        self.assertIsNone(fetch._retry_after_seconds(resp_headers(429)))

    def test_negative_clamped(self):
        m = resp_headers(429, headers={"Retry-After": "-5"})
        self.assertEqual(fetch._retry_after_seconds(m), 0.0)

    def test_huge_clamped(self):
        m = resp_headers(429, headers={"Retry-After": "9999"})
        self.assertEqual(fetch._retry_after_seconds(m), 120.0)

    def test_non_numeric(self):
        m = resp_headers(429, headers={"Retry-After": "soon"})
        self.assertIsNone(fetch._retry_after_seconds(m))


class TestSnapshotPageShape(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_non_list_raises(self):
        with patch("scanner.fetch._get", return_value={"error": 1}):
            with self.assertRaises(CoinGeckoError):
                fetch._fetch_snapshot_page(10, 1)

    def test_list_ok(self):
        with patch("scanner.fetch._get", return_value=[{"id": "a"}]):
            self.assertEqual(fetch._fetch_snapshot_page(10, 1), [{"id": "a"}])


class TestOhlcvCached(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def _rows(self):
        return {"prices": [[1, 100.0], [2, 101.0]],
                "market_caps": [], "total_volumes": []}

    def test_miss_then_hit(self):
        with patch("scanner.fetch._get", return_value=self._rows()):
            hit, _ = fetch.ohlcv_cached("peek-coin")
            self.assertFalse(hit)
            rows = fetch.fetch_ohlcv("peek-coin")
            self.assertEqual(len(rows), 2)
            hit, val = fetch.ohlcv_cached("peek-coin")
            self.assertTrue(hit)
            self.assertEqual(val, rows)

    def test_params_must_match(self):
        # interval/days berbeda -> key cache berbeda -> miss
        with patch("scanner.fetch._get", return_value=self._rows()):
            fetch.fetch_ohlcv("peek-coin-2", interval="daily", days=200)
            hit, _ = fetch.ohlcv_cached("peek-coin-2", interval="hourly",
                                        days=200)
            self.assertFalse(hit)


if __name__ == "__main__":
    unittest.main()
